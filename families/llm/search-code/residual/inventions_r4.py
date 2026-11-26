"""Round-4 residual ideas (search: residual; O33 wild round). Standard init, stacked on NAG via rs=nag.
Knob rk (strength / param-scale multiplier). forward(x, subs): subs alternate attention, FFN.
Most ideas start exactly as the plain stream (pre-equivalent) and learn away from it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rmsv(x, eps=1e-6):
    return x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def rot(x, th):
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


class Base(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        self.cfg, self.L, self.d = cfg, L, cfg.d_model
        self.k = float(cfg.rk)


# ---------------- untried mechanisms (28) ----------------
class HalfWrite(Base):
    """width routing: attention writes channels [0, d/2) fully and the other half at (1 - rk); FFN the reverse."""
    def forward(s, x, subs):
        h = s.d // 2
        ma = torch.ones(s.d, device=x.device); ma[h:] = 1 - s.k
        mf = torch.ones(s.d, device=x.device); mf[:h] = 1 - s.k
        for j, f in enumerate(subs):
            x = x + f(x) * (ma if j % 2 == 0 else mf)
        return x


class RotGroups(Base):
    """width routing, 4 groups: sublayer j writes channel group j mod 4 fully, other groups at (1 - rk)."""
    def forward(s, x, subs):
        g = s.d // 4
        for j, f in enumerate(subs):
            m = torch.full((s.d,), 1 - s.k, device=x.device)
            q = j % 4
            m[q * g:(q + 1) * g] = 1.0
            x = x + f(x) * m
        return x


class Givens(Base):
    """stream transport by rotation: before each write, channel pairs of the stream rotate by learned angles
    (zero init, scale rk)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.th = nn.Parameter(torch.zeros(L, cfg.d_model // 2))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = rot(x, s.th[j] * s.k) + f(x)
        return x


class RotWrite(Base):
    """input-dependent rotation write: x <- R(theta(y)) x + y, theta = rk * tanh(y W_j) per channel pair, W zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.zeros(L, cfg.d_model, cfg.d_model // 2))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = rot(x, s.k * torch.tanh(rms(y) @ s.W[j])) + y
        return x


class RevCouple(Base):
    """reversible-style coupling: streams A, B (both start as the embedding); attention reads B + rk*A, writes A;
    FFN reads A + rk*B, writes B; readout (A + B) / 2."""
    def forward(s, x, subs):
        A, B = x, x
        for j, f in enumerate(subs):
            if j % 2 == 0:
                A = A + f(B + s.k * A)
            else:
                B = B + f(A + s.k * B)
        return (A + B) / 2


class OrthX0(Base):
    """embedding-orthogonal writes: each write loses rk times its component along the token's embedding x0."""
    def forward(s, x, subs):
        u = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        for f in subs:
            y = f(x)
            x = x + y - s.k * (y * u).sum(-1, keepdim=True) * u
        return x


class Permute(Base):
    """channel transport: after each block, x <- (1 - rk) x + rk x[perm_b] (fixed random permutation per block)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(11)
        s.register_buffer("P", torch.stack([torch.randperm(cfg.d_model, generator=g) for _ in range(L // 2)]))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                x = (1 - s.k) * x + s.k * x[..., s.P[j // 2]]
        return x


class Bilinear(Base):
    """multiplicative write: x <- x + y + rk * g_j * (rms(x) * y), g per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * s.g[j] * rms(x) * y
        return x


class GluStream(Base):
    """stream-conditioned per-channel write gate: x += y * 2 sigmoid(rk * (rms(x) U_j) V_j), V zero (gate 1)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        s.U = nn.Parameter(torch.randn(L, d, 8) / math.sqrt(d))
        s.V = nn.Parameter(torch.zeros(L, 8, d))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y * 2 * torch.sigmoid(s.k * (rms(x) @ s.U[j]) @ s.V[j])
        return x


class SoftCap(Base):
    """bounded stream: after each write x <- c tanh(x / c), c = rk * RMS(x) per token (outliers squashed)."""
    def forward(s, x, subs):
        for f in subs:
            x = x + f(x)
            c = s.k * rmsv(x)
            x = c * torch.tanh(x / c)
        return x


class Sandwich(Base):
    """sublayer order: rk = 0 every block runs FFN before attention; rk = 1 only the first block does."""
    def forward(s, x, subs):
        nb = len(subs) // 2
        for b in range(nb):
            a, f = subs[2 * b], subs[2 * b + 1]
            order = (f, a) if (s.k == 0 or b == 0) else (a, f)
            for g in order:
                x = x + g(x)
        return x


class SharedProj(Base):
    """shared write basis: x += y (I + rk A), one d x d matrix A for all sublayers, zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.A = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_model))

    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            x = x + y + s.k * (y @ s.A)
        return x


class ReadProj(Base):
    """learned read transform: sublayer j reads x + rk (x U_j) V_j (rank 8, V zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        s.U = nn.Parameter(torch.randn(L, d, 8) / math.sqrt(d))
        s.V = nn.Parameter(torch.zeros(L, 8, d))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * (x @ s.U[j]) @ s.V[j])
        return x


class SparseWrite(Base):
    """sparse writes: each write keeps only its top rk*d channels by magnitude per token."""
    def forward(s, x, subs):
        kk = max(1, int(round(s.k * s.d)))
        for f in subs:
            y = f(x)
            thr = y.abs().topk(kk, dim=-1).values[..., -1:]
            x = x + y * (y.abs() >= thr).to(y.dtype)
        return x


class ConflictMask(Base):
    """write conflict masking: channels where the write's sign disagrees with the stream's are scaled by (1 - rk)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            x = x + torch.where((y * x) < 0, (1 - s.k) * y, y)
        return x


class AgreeGate(Base):
    """inter-sublayer agreement: FFN write scaled by 1 + rk * a_j * cos(y_ffn, y_attn) (a zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        ya = None
        for j, f in enumerate(subs):
            y = f(x)
            if j % 2 == 0:
                ya = y
            else:
                y = y * (1 + s.k * s.a[j] * F.cosine_similarity(y, ya, dim=-1, eps=1e-6).unsqueeze(-1))
            x = x + y
        return x


class X0Inject(Base):
    """embedding re-injection: after each block x += rk * beta_b * x0 (beta zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L // 2))

    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                x = x + s.k * s.b[j // 2] * x0
        return x


class Echo(Base):
    """echo writes: sublayer j writes y_j + rk * c_j * y_{j-2} (same-type previous output re-written), c zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            y = f(x)
            w = y + (s.k * s.c[j] * ys[j - 2] if j >= 2 else 0)
            ys.append(y)
            x = x + w
        return x


class ChanConv(Base):
    """channel-axis convolution of each write: kernel 3 = [0, 1, 0] + rk * delta_j (delta zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.dk = nn.Parameter(torch.zeros(L, 3))

    def forward(s, x, subs):
        base = torch.tensor([0.0, 1.0, 0.0], device=x.device)
        for j, f in enumerate(subs):
            y = f(x)
            k = base + s.k * s.dk[j]
            yp = F.pad(y, (1, 1))
            y = k[0] * yp[..., :-2] + k[1] * yp[..., 1:-1] + k[2] * yp[..., 2:]
            x = x + y
        return x


class DualWrite(Base):
    """write routing to the readout: a fraction f_j = sigmoid(rk p_j - 3) of each write goes to an accumulator
    only the head reads; the rest to the stream later layers read."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.p = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        acc = torch.zeros_like(x)
        for j, f in enumerate(subs):
            y = f(x)
            fr = torch.sigmoid(s.k * s.p[j] - 3)
            x = x + (1 - fr) * y
            acc = acc + fr * y
        return x + acc


def _orth(L, d, r, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.linalg.qr(torch.randn(d, r, generator=g))[0] for _ in range(L)])


class WriteSub(Base):
    """write bottleneck: y <- y - rk (y - y U U^T), U learned d x 16 per sublayer (orthonormal init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.U = nn.Parameter(_orth(L, cfg.d_model, 16, 5))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y - s.k * (y - (y @ s.U[j]) @ s.U[j].t())
        return x


class ReadSub(Base):
    """read subspace: sublayer j reads x - rk (x - x V V^T), V learned d x 16 (orthonormal init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.V = nn.Parameter(_orth(L, cfg.d_model, 16, 6))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            h = x - s.k * (x - (x @ s.V[j]) @ s.V[j].t())
            x = x + f(h)
        return x


class EntGate(Base):
    """entropy gate: write scaled by 1 + rk a_j H(softmax(rms(x))) / log d (a zero init): tokens with a flat
    stream (low certainty) can be written harder."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        ld = math.log(s.d)
        for j, f in enumerate(subs):
            p = rms(x).softmax(-1)
            H = -(p * p.clamp_min(1e-9).log()).sum(-1, keepdim=True) / ld
            x = x + f(x) * (1 + s.k * s.a[j] * H)
        return x


class AttnLocal(Base):
    """block-local attention output: FFN reads x + y_attn; the stream gets beta_b y_attn + y_ffn,
    beta = 1 + rk q_b (q zero init), so attention output can stay inside its block."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L // 2))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            yf = subs[2 * b + 1](x + ya)
            x = x + (1 + s.k * s.q[b]) * ya + yf
        return x


class EraseWrite(Base):
    """delta rule over depth: before writing y, erase beta_j of the stream's component along y's direction:
    x <- x - rk b_j (x . yhat) yhat + y (b zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            u = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
            x = x - s.k * s.b[j] * (x * u).sum(-1, keepdim=True) * u + y
        return x


class SignWrite(Base):
    """quantised writes: y <- (1 - rk) y + rk RMS(y) sign(y), straight-through gradient."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            q = y + (rmsv(y) * torch.sign(y) - y).detach()
            x = x + (1 - s.k) * y + s.k * q
        return x


class UNetSkip(Base):
    """U-Net skips across depth: block nb-1-b also reads + rk beta_b * (stream after block b), b < nb/2 (beta zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L // 2))

    def forward(s, x, subs):
        nb = len(subs) // 2
        saved = {}
        for b in range(nb):
            src = nb - 1 - b
            extra = s.k * s.b[b] * saved[src] if (src < b and src in saved) else 0
            for g in (subs[2 * b], subs[2 * b + 1]):
                x = x + g(x + extra)
            saved[b] = x
        return x


class XCouple(Base):
    """cross-sublayer write coupling: FFN write += rk (y_attn U_b) V_b (rank 8, V zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        s.U = nn.Parameter(torch.randn(L // 2, d, 8) / math.sqrt(d))
        s.V = nn.Parameter(torch.zeros(L // 2, 8, d))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            x = x + ya
            yf = subs[2 * b + 1](x)
            x = x + yf + s.k * (ya @ s.U[b]) @ s.V[b]
        return x


# ---------------- changed variants of tried mechanisms ----------------
class ChanDecay(Base):
    """per-channel stream decay (variant of r1 wdecay, scalar -> per channel): x <- (1 + rk p_j) x + y, p zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.p = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = (1 + s.k * s.p[j]) * x + f(x)
        return x


class ChanMom(Base):
    """per-channel depth momentum (variant of r1 wmom): v <- rk b_j * v + y; x += v (b zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        v = torch.zeros_like(x)
        for j, f in enumerate(subs):
            v = s.k * s.b[j] * v + f(x)
            x = x + v
        return x


class GradScale(Base):
    """gradient path shaping: forward is the plain stream; backward through each skip is scaled by rk
    (x_new = y + rk x + (1 - rk) x.detach())."""
    def forward(s, x, subs):
        for f in subs:
            x = f(x) + s.k * x + (1 - s.k) * x.detach()
        return x


class StreamNoise(Base):
    """train-only stream noise (variant of init's normnoise, on the stream): x += rk RMS(x) eps after each write."""
    def forward(s, x, subs):
        for f in subs:
            x = x + f(x)
            if s.training:
                x = x + s.k * rmsv(x).detach() * torch.randn_like(x)
        return x


class WriteDrop(Base):
    """train-only write dropout per token (stochastic depth per token, prob rk, rescaled)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            if s.training and s.k > 0:
                keep = (torch.rand(*y.shape[:-1], 1, device=y.device) > s.k).to(y.dtype)
                y = y * keep / (1 - s.k)
            x = x + y
        return x


class Householder(Base):
    """stream transport by reflection: x <- x - 2 rk (x . v_j) v_j / |v_j|^2 * t_j before each write,
    t = sigmoid(c_j), c init -4 (near identity)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.v = nn.Parameter(torch.randn(L, cfg.d_model) / math.sqrt(cfg.d_model))
        s.c = nn.Parameter(torch.full((L,), -4.0))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            v = s.v[j]
            x = x - 2 * s.k * torch.sigmoid(s.c[j]) * (x @ v).unsqueeze(-1) * v / (v @ v + 1e-6)
            x = x + f(x)
        return x


class Compete(Base):
    """write competition per block: attention and FFN outputs share a per-token budget,
    weights 2 softmax(rk (rms(x) W_b)), W zero init (both 1)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.zeros(L // 2, cfg.d_model, 2))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            w = 2 * torch.softmax(s.k * rms(x) @ s.W[b], -1)
            ya = subs[2 * b](x)
            x1 = x + w[..., :1] * ya
            yf = subs[2 * b + 1](x1)
            x = x1 + w[..., 1:] * yf
        return x


class PolyRead(Base):
    """nonlinear read: sublayer j reads x + rk c_j x^3 / RMS(x)^2 (c per channel, zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            r2 = x.pow(2).mean(-1, keepdim=True) + 1e-6
            x = x + f(x + s.k * s.c[j] * x.pow(3) / r2)
        return x


def _hadamard(d):
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


class Hadamard(Base):
    """basis change: odd blocks read and write in a fixed Walsh-Hadamard basis, mixed by rk:
    read Q x, write Q^T y with Q = orth((1 - rk) I + rk H)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        M = (1 - float(cfg.rk)) * torch.eye(d) + float(cfg.rk) * _hadamard(d)
        U, _, Vt = torch.linalg.svd(M)
        s.register_buffer("Q", U @ Vt)

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            if (j // 2) % 2 == 1:
                x = x + f(x @ s.Q.t()) @ s.Q
            else:
                x = x + f(x)
        return x


class Loop(Base):
    """depth recurrence: the last block runs a second time, its writes scaled by rk g (g learned, init 1)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.ones(2))

    def forward(s, x, subs):
        for f in subs:
            x = x + f(x)
        for i, f in enumerate(subs[-2:]):
            x = x + s.k * s.g[i] * f(x)
        return x


class GSDiv(Base):
    """write diversity over depth: each write loses rk times its projection on the previous write of the same
    sublayer type (attention vs attention, FFN vs FFN)."""
    def forward(s, x, subs):
        prev = [None, None]
        for j, f in enumerate(subs):
            y = f(x)
            p = prev[j % 2]
            if p is not None:
                u = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
                y = y - s.k * (y * u).sum(-1, keepdim=True) * u
            prev[j % 2] = y
            x = x + y
        return x


class RollRead(Base):
    """channel transport on read: sublayer j reads x rolled by j * round(rk d / L) channels and writes rolled back."""
    def forward(s, x, subs):
        st = int(round(s.k * s.d / s.L))
        for j, f in enumerate(subs):
            sh = j * st
            x = x + torch.roll(f(torch.roll(x, sh, -1)), -sh, -1)
        return x


FAMILY = dict(halfwrite="width routing", rotgroups="width routing", givens="stream transport",
              rotwrite="stream transport", permute="stream transport", householder="stream transport",
              rollread="stream transport", hadamard="stream transport", revcouple="two coupled streams",
              dualwrite="two coupled streams", orthx0="write geometry", erasewrite="write geometry",
              writesub="write geometry", gsdiv="write geometry", conflictmask="write geometry",
              sparsewrite="write geometry", signwrite="write geometry", chanconv="write geometry",
              bilinear="conditional write", glustream="conditional write", entgate="conditional write",
              agreegate="conditional write", compete="conditional write", softcap="bounded stream",
              sandwich="sublayer order", attnlocal="sublayer order", loop="depth recurrence",
              readproj="read transform", readsub="read transform", polyread="read transform",
              sharedproj="shared write basis", xcouple="shared write basis", gradscale="gradient path",
              x0inject="depth skips (tried: dense/readx0)", echo="depth skips (tried: dense)",
              unetskip="depth skips (tried: dense)", streamnoise="stochastic stream (tried: init normnoise)",
              writedrop="stochastic stream (tried: stochastic depth)", chandecay="tried variant: r1 wdecay",
              chanmom="tried variant: r1 wmom")

IDEAS = dict(gradscale=GradScale, streamnoise=StreamNoise, writedrop=WriteDrop, householder=Householder,
             compete=Compete, polyread=PolyRead, hadamard=Hadamard, loop=Loop, gsdiv=GSDiv, rollread=RollRead,
             halfwrite=HalfWrite, rotgroups=RotGroups, givens=Givens, rotwrite=RotWrite, revcouple=RevCouple,
             orthx0=OrthX0, permute=Permute, bilinear=Bilinear, glustream=GluStream, softcap=SoftCap,
             sandwich=Sandwich, sharedproj=SharedProj, readproj=ReadProj, sparsewrite=SparseWrite,
             conflictmask=ConflictMask, agreegate=AgreeGate, x0inject=X0Inject, echo=Echo, chanconv=ChanConv,
             dualwrite=DualWrite, writesub=WriteSub, readsub=ReadSub, entgate=EntGate, attnlocal=AttnLocal,
             erasewrite=EraseWrite, signwrite=SignWrite, unetskip=UNetSkip, xcouple=XCouple,
             chandecay=ChanDecay, chanmom=ChanMom)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
