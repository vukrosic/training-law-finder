"""Round-7 residual ideas (search: residual; O33 wild round 4). Standard init, stacked on NAG via rs=nag.
Knob rk. forward(x, subs): subs alternate attention, FFN. 28 new mechanisms + 2 from tried families here;
10 changed variants of r5 ideas reuse inventions_r5 classes at new strengths (tools/mk_r7.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inventions_r4 import Base, rms, rmsv


def _orthm(d, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


class _BwdNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g + ctx.k * rmsv(g) * torch.randn_like(g), None


class _BwdSmooth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        nb = torch.cat([g[:, 1:], g[:, -1:]], 1)
        return (1 - ctx.k) * g + ctx.k * 0.5 * (g + nb), None


class _AuxGrad(torch.autograd.Function):
    """returns x; adds a local regression gradient (pred - target) to pred."""
    @staticmethod
    def forward(ctx, x, pred, target):
        ctx.save_for_backward(pred, target)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        pred, target = ctx.saved_tensors
        n = pred.numel() / pred.shape[-1]
        return g, (pred - target) / n, None


# ---------------- stochastic / gradient-path (6) ----------------
class StochDepth(Base):
    """stochastic depth: train-only, sublayer j>0 dropped per sequence with prob rk j/(L-1), kept writes rescaled."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            if s.training and j:
                q = s.k * j / max(1, s.L - 1)
                m = (torch.rand(x.shape[0], 1, 1, device=x.device) > q).to(x.dtype) / (1 - q)
                y = y * m
            x = x + y
        return x


class Shake(Base):
    """shake-drop on writes: train-only, forward scale a ~ U(1-rk, 1+rk), backward scale b ~ U(1-rk, 1+rk), per token."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            if s.training:
                sh = x.shape[:2] + (1,)
                a = 1 + s.k * (2 * torch.rand(sh, device=x.device) - 1)
                b = 1 + s.k * (2 * torch.rand(sh, device=x.device) - 1)
                y = y * b + (y * a - y * b).detach()
            x = x + y
        return x


class ManifoldMix(Base):
    """cross-example stream mixing (train-only): after one random sublayer, x <- (1-l) x + l x[perm], l ~ U(0, rk);
    targets unchanged (so it acts as structured stream noise)."""
    def forward(s, x, subs):
        jm = int(torch.randint(len(subs), ())) if s.training else -1
        for j, f in enumerate(subs):
            x = x + f(x)
            if j == jm and x.shape[0] > 1:
                lam = s.k * float(torch.rand(()))
                x = (1 - lam) * x + lam * x[torch.randperm(x.shape[0], device=x.device)]
        return x


class Nested(Base):
    """nested (matryoshka) read: train-only, each sequence's reads keep only the first c channels,
    c ~ U(d (1-rk), d), rescaled d/c; eval full."""
    def forward(s, x, subs):
        m = None
        if s.training:
            keep = torch.randint(int(s.d * (1 - s.k)), s.d + 1, (x.shape[0], 1, 1), device=x.device)
            ar = torch.arange(s.d, device=x.device).view(1, 1, -1)
            m = (ar < keep).to(x.dtype) * (s.d / keep.to(x.dtype))
        for f in subs:
            x = x + f(x * m if m is not None else x)
        return x


class GradNoise(Base):
    """backward-only noise: the gradient into each write gets + rk RMS(g) eps (forward unchanged)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            x = x + (_BwdNoise.apply(y, s.k) if s.training else y)
        return x


class AdvRead(Base):
    """adversarial read: train-only, sublayer j reads x + rk RMS(x) u_j, u_j = unit-RMS batch-mean direction of the
    gradient at that read on the previous step (loss-increasing direction)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("u", torch.zeros(L, cfg.d_model))

    def _upd(s, j, g):
        gm = g.detach().mean((0, 1))
        s.u[j] = gm / rmsv(gm)

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            r = x
            if s.training:
                r = x + s.k * rmsv(x) * s.u[j].clone()
                if r.requires_grad:
                    r.register_hook(lambda g, j=j: s._upd(j, g))
            x = x + f(r)
        return x


class GradSmooth(Base):
    """backward-only smoothing over positions: gradient into each write mixed with the next position's gradient
    (weight rk/2); forward unchanged, no forward leak."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            x = x + (_BwdSmooth.apply(y, s.k) if s.training else y)
        return x


# ---------------- composition algebra (2) ----------------
class Mobius(Base):
    """hyperbolic (Moebius) addition instead of +: x (+)_c y with c = rk exp(g_j) / (|x|^2 + |y|^2), g init 0."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            xx, yy = x.pow(2).sum(-1, keepdim=True), y.pow(2).sum(-1, keepdim=True)
            c = s.k * torch.exp(s.g[j]) / (xx + yy + 1e-6).detach()
            xy = (x * y).sum(-1, keepdim=True)
            num = (1 + 2 * c * xy + c * yy) * x + (1 - c * xx) * y
            x = num / (1 + 2 * c * xy + c * c * xx * yy).clamp_min(1e-3)
        return x


class Mirror(Base):
    """mirror-map stream update: x <- (grad phi)^-1(grad phi(x) + y), grad phi(z) = z + b z^3,
    b = rk exp(g_j) / RMS(x)^2 (3 Newton steps)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            b = s.k * torch.exp(s.g[j]) / x.pow(2).mean(-1, keepdim=True).add(1e-6).detach()
            w = x + b * x.pow(3) + y
            z = x + y
            for _ in range(3):
                z = z - (z + b * z.pow(3) - w) / (1 + 3 * b * z.pow(2))
            x = z
        return x


# ---------------- integration / write extrapolation (3) ----------------
class Heun(Base):
    """[tried family: iter2] Heun (RK2) write: y1 = F(x), y2 = F(x + y1); write y1 + rk (y2 - y1) / 2 (2x compute)."""
    def forward(s, x, subs):
        for f in subs:
            y1 = f(x)
            x = x + y1 + s.k * 0.5 * (f(x + y1) - y1)
        return x


class Secant(Base):
    """write extrapolation: y_j + rk a_j (y_j - y_{j-2}) vs the previous same-type write, a per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            y = f(x)
            if j >= 2:
                y = y + s.k * s.a[j] * (y - ys[j - 2])
            ys.append(y)
            x = x + y
        return x


class Fractal(Base):
    """fractal path averaging per block: x + ya + yf(x+ya) + rk g_b (yf(x) - yf(x+ya)) (FFN also on the shallow path),
    g init 0."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j in range(0, len(subs), 2):
            fa, ff = subs[j], subs[j + 1]
            ya = fa(x)
            yd = ff(x + ya)
            ysh = ff(x)
            x = x + ya + yd + s.k * s.g[j] * (ysh - yd)
        return x


# ---------------- read transforms (8) ----------------
class Antithetic(Base):
    """symmetrised read: write 0.5 (F(x + e) + F(x - e)), e = rk RMS(x) a_j, a random small init (2x compute)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.randn(L, cfg.d_model) * 0.1)

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            e = s.k * rmsv(x) * s.a[j]
            x = x + 0.5 * (f(x + e) + f(x - e))
        return x


class Hopfield(Base):
    """associative cleanup read: r = softmax(beta rms(x) M^T) M over 32 learned patterns (shared);
    sublayer j reads x + rk a_j RMS(x) rms(r), a per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.M = nn.Parameter(torch.randn(32, cfg.d_model) / math.sqrt(cfg.d_model))
        s.beta = nn.Parameter(torch.tensor(4.0))
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            r = torch.softmax(s.beta * rms(x) @ s.M.t() / math.sqrt(s.d) * 4, -1) @ s.M
            x = x + f(x + s.k * s.a[j] * rmsv(x) * rms(r))
        return x


class RFFRead(Base):
    """random-feature read: phi = sqrt2 cos(rms(x) W + b) (W, b fixed random); reads x + rk a_j RMS(x) phi, a zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(7)
        s.register_buffer("W", torch.randn(cfg.d_model, cfg.d_model, generator=g) / math.sqrt(cfg.d_model))
        s.register_buffer("b", torch.rand(cfg.d_model, generator=g) * 2 * math.pi)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            phi = math.sqrt(2) * torch.cos(rms(x) @ s.W + s.b)
            x = x + f(x + s.k * s.a[j] * rmsv(x) * phi)
        return x


class SortRead(Base):
    """order-statistic read: channel c also sees the c-th smallest stream value of the token:
    x + rk a_j (sort(x) - x), a zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * (torch.sort(x, -1).values - x))
        return x


class DepthEmb(Base):
    """depth embedding at read: sublayer j reads x + rk RMS(x) e_j (e zero init): a sublayer id seen by its own read."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.e = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * rmsv(x) * s.e[j])
        return x


class _Route(Base):
    G = 4

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("Q", _orthm(cfg.d_model, 11))
        s.W = nn.Parameter(torch.zeros(L, cfg.d_model, s.G))

    def route(s, v, j):
        p = torch.softmax(rms(v) @ s.W[j], -1) * s.G           # (B,T,G), 1 each at init
        c = (v @ s.Q).unflatten(-1, (s.G, -1))                  # coordinates per subspace
        return (c * p.unsqueeze(-1)).flatten(-2) @ s.Q.t()


class WriteRoute(_Route):
    """routed write subspaces: write split over 4 fixed orthogonal subspaces, per-token softmax gains from rms(y) W_j
    (W zero init = plain); write y + rk (routed - y)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * (s.route(y, j) - y)
        return x


class ReadRoute(_Route):
    """routed read subspaces: read split over 4 fixed orthogonal subspaces with per-token softmax gains
    (W zero init = plain); read x + rk (routed - x)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * (s.route(x, j) - x))
        return x


class ChunkMem(Base):
    """[tried family: causal pooling] previous-chunk memory: reads x + rk a_j (m - x), m = mean of the stream over the
    previous complete 8-token chunk (0 for the first), a zero init."""
    C = 8

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            B, T, D = x.shape
            n = T // s.C
            cm = x[:, :n * s.C].unflatten(1, (n, s.C)).mean(2)
            cm = torch.cat([torch.zeros_like(cm[:, :1]), cm[:, :-1]], 1).repeat_interleave(s.C, 1)
            if cm.shape[1] < T:
                cm = torch.cat([cm, cm[:, -1:].expand(B, T - cm.shape[1], D)], 1)
            x = x + f(x + s.k * s.a[j] * (cm - x))
        return x


# ---------------- write transforms (8) ----------------
class HRR(Base):
    """holographic binding write: y + rk a_j (y (*) rms(x)) / sqrt(d), (*) = circular convolution over channels,
    a per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            b = torch.fft.irfft(torch.fft.rfft(y, dim=-1) * torch.fft.rfft(rms(x), dim=-1), n=s.d, dim=-1)
            x = x + y + s.k * s.a[j] * b / math.sqrt(s.d)
        return x


class PredCode(Base):
    """predictive-coding write: a linear predictor P_j (zero init, trained only by its own local regression on the
    normalised write) guesses y from rms(x); the stream gets y - rk RMS(x) pred (the surprise)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.P = nn.Parameter(torch.zeros(L, cfg.d_model, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            sc = rmsv(x).detach()
            pred = rms(x).detach() @ s.P[j]
            y2 = y - s.k * sc * pred.detach()
            if s.training:
                y2 = _AuxGrad.apply(y2, pred, (y / sc).detach())
            x = x + y2
        return x


class Stagger(Base):
    """channel depth staggering: channel group g(c) = floor(c L / d); sublayer j writes channel c at
    1 - rk relu(g(c) - j) / L (late channels filled by late sublayers). No parameters."""
    def forward(s, x, subs):
        g = (torch.arange(s.d, device=x.device) * s.L // s.d).to(x.dtype)
        for j, f in enumerate(subs):
            x = x + f(x) * (1 - s.k * F.relu(g - j) / s.L)
        return x


class GroupAvg(Base):
    """symmetry averaging: write y + rk (P^T F(P x) - y) / 2, P a fixed random channel permutation (2x compute)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(5)
        p = torch.randperm(cfg.d_model, generator=g)
        s.register_buffer("p", p)
        s.register_buffer("pi", torch.argsort(p))

    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            yp = f(x[..., s.p])[..., s.pi]
            x = x + y + s.k * 0.5 * (yp - y)
        return x


class BagWrite(Base):
    """dropout-bagged read: train-only, write 0.5 (F(x m1) + F(x m2)) with two channel-dropout masks (p = rk,
    rescaled); eval F(x) (2x compute in training)."""
    def forward(s, x, subs):
        for f in subs:
            if s.training:
                m1 = (torch.rand_like(x) > s.k).to(x.dtype) / (1 - s.k)
                m2 = (torch.rand_like(x) > s.k).to(x.dtype) / (1 - s.k)
                x = x + 0.5 * (f(x * m1) + f(x * m2))
            else:
                x = x + f(x)
        return x


class Energy(Base):
    """energy-gradient write: extra write -rk RMS(x) grad_z E_j(z) at z = rms(x), E_j = sum softplus(z W_j),
    i.e. -rk RMS(x) sigmoid(z W_j) W_j^T (tied-weight gradient flow), W (d x 16) small init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.randn(L, cfg.d_model, 16) * 0.02)

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            e = torch.sigmoid(rms(x) @ s.W[j]) @ s.W[j].t()
            x = x + f(x) - s.k * rmsv(x) * e
        return x


class CausalSE(Base):
    """causal squeeze-excite over the sequence: write y (1 + rk tanh(cummean_t(y) W_j)), W zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.zeros(L, cfg.d_model, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            t = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            cm = y.cumsum(1) / t
            x = x + y * (1 + s.k * torch.tanh(rms(cm) @ s.W[j]))
        return x


class Expire(Base):
    """transient writes: write j is fully in the stream for the next two reads; then rk of it is removed
    (pending removals applied before the head too)."""
    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            if j >= 2:
                x = x - s.k * ys[j - 2]
            y = f(x)
            ys.append(y)
            x = x + y
        for y in ys[-2:]:
            x = x - s.k * y
        return x


# ---------------- state / conditioning (3) ----------------
class Reservoir(Base):
    """depth reservoir: fixed random state r_{j+1} = tanh(0.9 A r_j + rms(y_j)) (A fixed orthogonal);
    sublayer j reads x + rk c_j RMS(x) r_j, c per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("A", 0.9 * _orthm(cfg.d_model, 13))
        s.c = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        r = torch.zeros_like(x)
        for j, f in enumerate(subs):
            y = f(x + s.k * s.c[j] * rmsv(x) * r)
            r = torch.tanh(r @ s.A + rms(y))
            x = x + y
        return x


class ClusterGain(Base):
    """cluster-conditioned depth gains: soft assignment p = softmax(2 rms(x0) C^T) to 4 learned centroids;
    sublayer j writes at 1 + rk (p G)_j, G zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.C = nn.Parameter(torch.randn(4, cfg.d_model) / math.sqrt(cfg.d_model))
        s.G = nn.Parameter(torch.zeros(4, L))

    def forward(s, x, subs):
        p = torch.softmax(2 * rms(x) @ s.C.t(), -1)
        gain = 1 + s.k * p @ s.G                                # (B,T,L)
        for j, f in enumerate(subs):
            x = x + f(x) * gain[..., j:j + 1]
        return x


class AlphaCapInit(Base):
    """CONTROL for trustwrite (not an idea): writes scaled by a learnable per-sublayer gain c_j initialised at
    min(1, rk / (0.95 / sqrt(j+1))) = trustwrite's cap applied only at init (NAG alpha init 1/sqrt(l), gate m ~ 0.95)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.tensor([min(1.0, s.k / (0.95 / math.sqrt(j + 1))) for j in range(L)]))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x) * s.c[j]
        return x


FAMILY = dict(stochdepth="stochastic depth", shake="stochastic write scaling", manifoldmix="cross-example mixing",
              nested="nested read width", gradnoise="backward-only noise", advread="adversarial read",
              gradsmooth="backward-only smoothing", mobius="hyperbolic composition", mirror="mirror-map composition",
              heun="ODE integration (tried: iter2)", secant="write extrapolation", fractal="path averaging",
              antithetic="symmetrised read", hopfield="associative cleanup", rffread="random-feature read",
              sortread="order-statistic read", depthemb="depth embedding", writeroute="routed write subspaces",
              readroute="routed read subspaces", chunkmem="causal pooling (tried: causalavg)", hrr="holographic binding",
              predcode="predictive coding", stagger="channel depth staggering", groupavg="symmetry averaging",
              bagwrite="dropout bagging", energy="energy-gradient write", causalse="causal squeeze-excite",
              expire="transient writes", reservoir="depth reservoir", clustergain="cluster-conditioned gains", alphacapinit="control (trustwrite)")

IDEAS = dict(stochdepth=StochDepth, shake=Shake, manifoldmix=ManifoldMix, nested=Nested, gradnoise=GradNoise,
             advread=AdvRead, gradsmooth=GradSmooth, mobius=Mobius, mirror=Mirror, heun=Heun, secant=Secant,
             fractal=Fractal, antithetic=Antithetic, hopfield=Hopfield, rffread=RFFRead, sortread=SortRead,
             depthemb=DepthEmb, writeroute=WriteRoute, readroute=ReadRoute, chunkmem=ChunkMem, hrr=HRR,
             predcode=PredCode, stagger=Stagger, groupavg=GroupAvg, bagwrite=BagWrite, energy=Energy,
             causalse=CausalSE, expire=Expire, reservoir=Reservoir, clustergain=ClusterGain,
             alphacapinit=AlphaCapInit)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
