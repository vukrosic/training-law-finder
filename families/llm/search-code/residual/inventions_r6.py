"""Round-6 residual ideas (search: residual; O33 wild round 3). Standard init, stacked on NAG via rs=nag.
Knob rk. forward(x, subs): subs alternate attention, FFN. 28 new mechanisms here; 12 changed variants of
round-4 losers are added after r4 lands.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inventions_r4 import Base, rms, rmsv, rot, _orth

STEPS = 150


class Stepper(Base):
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("step", torch.zeros(()))

    def tick(s):
        if s.training:
            s.step += 1
        return float(s.step) if s.training else float("inf")


class Kalman(Base):
    """uncertainty-weighted stream: per-channel variance v (init 1); write gain K = v / (v + r_j), r = rk softplus(c),
    normalised so the first write has gain 1; v <- (1 - K) v + (y / RMS(x))^2 (no grad)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        v = torch.ones_like(x)
        n0 = 1 + s.k * math.log(2)
        for j, f in enumerate(subs):
            r = s.k * F.softplus(s.c[j])
            K = v / (v + r)
            y = f(x)
            x = x + y * K * n0
            v = ((1 - K) * v + (y / rmsv(x)).pow(2)).detach()
        return x


class PosRot(Base):
    """position-dependent stream rotation: sublayer j reads R(t w_j rk) x and writes R(-t w_j rk) y, w zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.w = nn.Parameter(torch.zeros(L, cfg.d_model // 2))

    def forward(s, x, subs):
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        for j, f in enumerate(subs):
            th = t * s.w[j] * s.k * 0.01
            x = x + rot(f(rot(x, th)), -th)
        return x


class DualView(Base):
    """two views with shared weights: a copy whose second half channels are sign-flipped runs the same sublayers;
    output (1 - a) x1 + a m x2, a = rk sigmoid(c), c init -2."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.tensor(-2.0))

    def forward(s, x, subs):
        m = torch.ones(s.d, device=x.device); m[s.d // 2:] = -1
        x1, x2 = x, x * m
        for f in subs:
            x1 = x1 + f(x1)
            x2 = x2 + f(x2)
        a = s.k * torch.sigmoid(s.c)
        return (1 - a) * x1 + a * m * x2


class WhitenRead(Base):
    """stream whitening read: h = x + rk (W x - x), W = scaled inverse sqrt of an EMA (no grad) of the stream
    covariance per sublayer (train statistics, fixed at eval)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("C", torch.eye(cfg.d_model).repeat(L, 1, 1))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            if s.training:
                with torch.no_grad():
                    z = rms(x).reshape(-1, s.d)
                    s.C[j] = 0.9 * s.C[j] + 0.1 * (z.t() @ z) / z.shape[0]
            e, V = torch.linalg.eigh(s.C[j] + 1e-3 * torch.eye(s.d, device=x.device))
            W = (V * e.clamp_min(1e-3).rsqrt()) @ V.t()
            W = W / W.diagonal().mean()
            x = x + f(x + s.k * (x @ W - x))
        return x


class OutlierDir(Base):
    """outlier-direction removal on read: u = top eigenvector of an EMA stream covariance (one power step per
    forward, no grad); sublayers read x - rk (x . u) u."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("C", torch.eye(cfg.d_model).repeat(L, 1, 1))
        s.register_buffer("u", F.normalize(torch.randn(L, cfg.d_model), dim=-1))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            if s.training:
                with torch.no_grad():
                    z = x.reshape(-1, s.d)
                    s.C[j] = 0.9 * s.C[j] + 0.1 * (z.t() @ z) / z.shape[0]
                    s.u[j] = F.normalize(s.C[j] @ s.u[j], dim=0)
            u = s.u[j].clone()
            x = x + f(x - s.k * (x @ u).unsqueeze(-1) * u)
        return x


class SinkChan(Base):
    """sink channel on read: channel 0 of every read gets + rk 4 RMS(x) (a constant large feature sublayers can use
    as a bias / attention sink)."""
    def forward(s, x, subs):
        e = torch.zeros(s.d, device=x.device); e[0] = 4.0
        for f in subs:
            x = x + f(x + s.k * rmsv(x) * e)
        return x


class FourierRead(Base):
    """random-Fourier read: sublayer j reads x + rk a_j RMS(x) sin(rms(x) W) (W fixed random, a per channel zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(13)
        s.register_buffer("W", torch.randn(cfg.d_model, cfg.d_model, generator=g))
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * rmsv(x) * torch.sin(rms(x) @ s.W))
        return x


class VQWrite(Base):
    """vector-quantised write: write mixed (rk) with its nearest of 64 learned codes (cosine, scaled to |y|),
    soft-assignment gradient, hard forward."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.E = nn.Parameter(torch.randn(64, cfg.d_model))

    def forward(s, x, subs):
        En = F.normalize(s.E, dim=-1)
        for f in subs:
            y = f(x)
            sim = F.normalize(y, dim=-1) @ En.t()
            soft = torch.softmax(10 * sim, -1) @ En
            hard = En[sim.argmax(-1)]
            q = (soft + (hard - soft).detach()) * y.norm(dim=-1, keepdim=True)
            x = x + (1 - s.k) * y + s.k * q
        return x


class WidthCurr(Stepper):
    """width curriculum: writes reach only the first max(d/4, d t / (rk T)) channels during the first rk of
    training (t = step, T = 150); full width after."""
    def forward(s, x, subs):
        t = s.tick()
        n = s.d if t == float("inf") else int(min(s.d, max(s.d // 4, s.d * t / (s.k * STEPS))))
        m = torch.zeros(s.d, device=x.device); m[:n] = 1
        for f in subs:
            x = x + f(x) * m
        return x


class DepthCurr(Stepper):
    """depth curriculum: sublayer j's write ramps in over training, gain clamp(L t / (rk T) - j + 1, 0, 1)."""
    def forward(s, x, subs):
        t = s.tick()
        for j, f in enumerate(subs):
            g = 1.0 if t == float("inf") else min(1.0, max(0.0, s.L * t / (s.k * STEPS) - j + 1))
            x = x + f(x) * g if g > 0 else x
        return x


class NormFeat(Base):
    """norm feature: pre-norm erases the stream scale, so sublayer j reads x + rk RMS(x) log(RMS(x)) v_j (v zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.v = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            r = rmsv(x)
            x = x + f(x + s.k * r * torch.log(r) * s.v[j])
        return x


class StreamAdapter(Base):
    """inter-block stream adapter: after each block x += rk RMS(x) gelu(rms(x) A_b) B_b (rank 8, B zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        s.A = nn.Parameter(torch.randn(L // 2, d, 8) / math.sqrt(d))
        s.B = nn.Parameter(torch.zeros(L // 2, 8, d))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                x = x + s.k * rmsv(x) * F.gelu(rms(x) @ s.A[j // 2]) @ s.B[j // 2]
        return x


class Blend(Base):
    """cross-depth sublayer blending: write j = f_j(x) + rk a_j (f_{j+2}(x) - f_j(x)) for j < L - 2 (a zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        n = len(subs)
        for j, f in enumerate(subs):
            y = f(x)
            if j + 2 < n:
                y = y + s.k * s.a[j] * (subs[j + 2](x) - y)
            x = x + y
        return x


class Lookahead(Base):
    """lookahead read: sublayer j reads x + rk a_j f_{j+1}(x) (what the next sublayer would write now; a zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        n = len(subs)
        for j, f in enumerate(subs):
            h = x + s.k * s.a[j] * subs[j + 1](x) if j + 1 < n else x
            x = x + f(h)
        return x


class SpeedSpec(Base):
    """channel speed spectrum: fixed per-channel write gain from 1 (channel 0) down to 1 - rk (last channel), same
    for all sublayers (slow-memory and fast channels)."""
    def forward(s, x, subs):
        g = 1 - s.k * torch.linspace(0, 1, s.d, device=x.device)
        for f in subs:
            x = x + f(x) * g
        return x


class ParityChains(Base):
    """interleaved chains: block b reads the stream minus rk times the writes of blocks of the other parity;
    head reads everything."""
    def forward(s, x, subs):
        W = [0, 0]
        for b in range(len(subs) // 2):
            p = b % 2
            r = x - s.k * W[1 - p]
            ya = subs[2 * b](r)
            yf = subs[2 * b + 1](r + ya)
            W[p] = W[p] + ya + yf
            x = x + ya + yf
        return x


class PhaseLag(Base):
    """phase-lag write: the write is turned by angle rk tanh(a_j) toward the stream direction (in the plane of y and
    x), keeping |y| (a zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            yn = y.norm(dim=-1, keepdim=True) + 1e-6
            yh = y / yn
            xp = x - (x * yh).sum(-1, keepdim=True) * yh
            xp = xp / (xp.norm(dim=-1, keepdim=True) + 1e-6)
            th = s.k * torch.tanh(s.a[j])
            x = x + yn * (torch.cos(th) * yh + torch.sin(th) * xp)
        return x


class ReadReg(Base):
    """read register: sublayer j reads x + rk RMS(x) r_j (learned constant read vector, zero init, never stored)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.r = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * rmsv(x) * s.r[j])
        return x


class DirDrop(Base):
    """train-only subspace dropout of writes: each step a random subspace of rank rk d is removed from every write."""
    def forward(s, x, subs):
        P = None
        if s.training and s.k > 0:
            Q = torch.linalg.qr(torch.randn(s.d, max(1, int(s.k * s.d)), device=x.device))[0]
            P = Q
        for f in subs:
            y = f(x)
            if P is not None:
                y = (y - (y @ P) @ P.t()) / (1 - s.k)
            x = x + y
        return x


class Logistic(Base):
    """logistic channel capacity: write scaled by 1 - rk tanh(|x| / (c RMS(x)))^2, c = softplus(q) + 1 (q init 2):
    channels near capacity grow slower."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.full((L,), 2.0))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            c = F.softplus(s.q[j]) + 1
            x = x + y * (1 - s.k * torch.tanh(x.abs() / (c * rmsv(x))).pow(2))
        return x


class X0Contract(Base):
    """contraction toward the embedding: after each block x <- x0 + (1 - rk sigmoid(c_b)) (x - x0), c init -4."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L // 2,), -4.0))

    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                x = x0 + (1 - s.k * torch.sigmoid(s.c[j // 2])) * (x - x0)
        return x


class FrozenCarrier(Base):
    """frozen-carrier channels: the identity path carries no gradient on the last rk d channels (they still carry
    values; their gradient reaches earlier layers only through the sublayers)."""
    def forward(s, x, subs):
        n = int(round(s.k * s.d))
        m = torch.ones(s.d, device=x.device); m[s.d - n:] = 0
        for f in subs:
            x = f(x) + x * m + (x * (1 - m)).detach()
        return x


class SideMem(Base):
    """side memory: a 16-dim per-token memory s += rms(y) P_j (P fixed random); sublayers read x + rk RMS(x) s Q_j
    (Q zero init); never seen by the head."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        g = torch.Generator().manual_seed(17)
        s.register_buffer("P", torch.randn(L, d, 16, generator=g) / math.sqrt(d))
        s.Q = nn.Parameter(torch.zeros(L, 16, d))

    def forward(s, x, subs):
        m = 0
        for j, f in enumerate(subs):
            h = x + s.k * rmsv(x) * (m @ s.Q[j]) if j else x
            y = f(h)
            m = m + rms(y) @ s.P[j]
            x = x + y
        return x


class SignSplit(Base):
    """rectified read: sublayer j reads x + rk a_j relu(x) (a per channel zero init; positive and negative parts
    read with different weights)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * F.relu(x))
        return x


class RankWrite(Base):
    """rank-transformed write: y mixed (rk) with the standardised channel ranks of y scaled to RMS(y)
    (straight-through)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            r = y.argsort(-1).argsort(-1).to(y.dtype)
            r = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6) * rmsv(y)
            q = y + (r - y).detach()
            x = x + (1 - s.k) * y + s.k * q
        return x


class OrderSym(Base):
    """order-symmetric block: block write = (1 - rk/2) [attn then FFN] + rk/2 [FFN then attn] (2x compute)."""
    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            a, f = subs[2 * b], subs[2 * b + 1]
            ya = a(x); w1 = ya + f(x + ya)
            if s.k > 0:
                yf = f(x); w2 = yf + a(x + yf)
                x = x + (1 - s.k / 2) * w1 + s.k / 2 * w2
            else:
                x = x + w1
        return x


class LoanWrite(Base):
    """transient writes: each write is taken back (rk sigmoid(c_j), c init -2) two sublayers later; only nearby
    sublayers see it at full weight."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L,), -2.0))

    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            if j >= 2:
                x = x - s.k * torch.sigmoid(s.c[j - 2]) * ys[j - 2]
            y = f(x)
            ys.append(y)
            x = x + y
        return x


class FFNExclude(Base):
    """read exclusion by type: FFN reads the stream minus rk times all earlier FFN writes (sees attention writes and
    embedding fully)."""
    def forward(s, x, subs):
        acc = 0
        for j, f in enumerate(subs):
            if j % 2 == 1:
                y = f(x - s.k * acc)
                acc = acc + y
            else:
                y = f(x)
            x = x + y
        return x


class HistRead(Base):
    """write-history read: per channel, h = mean over earlier writes of |y| / RMS(y) (how often a channel was written);
    sublayer j reads x + rk a_j RMS(x) (h - mean_c h), a per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        H = 0
        for j, f in enumerate(subs):
            r = x
            if j:
                h = H / j
                r = x + s.k * s.a[j] * rmsv(x) * (h - h.mean(-1, keepdim=True))
            y = f(r)
            H = H + y.abs() / rmsv(y)
            x = x + y
        return x


FAMILY = dict(kalman="uncertainty-weighted stream", posrot="position-dependent stream rotation",
              dualview="two-view shared-weight ensemble", whitenread="stream whitening read",
              outlierdir="outlier-direction removal", sinkchan="sink channel", fourierread="random-Fourier read",
              vqwrite="vector-quantised write", widthcurr="width curriculum", depthcurr="depth curriculum",
              normfeat="norm feature read", streamadapter="inter-block stream adapter",
              blend="cross-depth sublayer blending", lookahead="lookahead read", speedspec="channel speed spectrum",
              paritychains="interleaved parity chains", phaselag="phase-lag write", readreg="read register",
              dirdrop="subspace write dropout", logistic="logistic channel capacity",
              x0contract="contraction toward embedding", frozencarrier="frozen-carrier channels",
              sidemem="side memory", signsplit="rectified read", rankwrite="rank-transformed write",
              ordersym="order-symmetric block", loanwrite="transient writes", ffnexclude="read exclusion by type", histread="write-history read")

IDEAS = dict(kalman=Kalman, posrot=PosRot, dualview=DualView, whitenread=WhitenRead, outlierdir=OutlierDir,
             sinkchan=SinkChan, fourierread=FourierRead, vqwrite=VQWrite, widthcurr=WidthCurr, depthcurr=DepthCurr,
             normfeat=NormFeat, streamadapter=StreamAdapter, blend=Blend, lookahead=Lookahead, speedspec=SpeedSpec,
             paritychains=ParityChains, phaselag=PhaseLag, readreg=ReadReg, dirdrop=DirDrop, logistic=Logistic,
             x0contract=X0Contract, frozencarrier=FrozenCarrier, sidemem=SideMem, signsplit=SignSplit,
             rankwrite=RankWrite, ordersym=OrderSym, loanwrite=LoanWrite, ffnexclude=FFNExclude, histread=HistRead)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
