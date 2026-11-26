"""Round-8 residual ideas (search: residual; first 600-step round, XS 6 layers). Standard init, stacked on NAG via
rs=nag. Knob rk. forward(x, subs): subs alternate attention, FFN. 28 new mechanisms + 2 variants here
(rffwrite, unbind); other variants reuse inventions_r5 / r7 classes at new strengths (tools/mk_r8.py).
Most ideas start exactly at the NAG function (zero-init mixing parameter) and learn away from it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inventions_r4 import Base, rms, rmsv


def _blocks(v, m=8):
    """(..., d) -> (..., g, m, m) square blocks (d must be a multiple of m*m)."""
    return v.unflatten(-1, (-1, m, m))


def _cos(a, b):
    return (a * b).sum(-1, keepdim=True) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True) + 1e-6)


class _Zero(Base):
    """helper: per-sublayer per-channel mixing vector a (zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))


# ---------------- depth wiring (read from earlier stream states) ----------------
class Fenwick(_Zero):
    """Fenwick-tree depth read: sublayer j also reads the stream state at its Fenwick parent j & (j-1):
    x + rk a_j (S_parent - x), a zero init (log-depth structured skip reads)."""
    def forward(s, x, subs):
        S = []
        for j, f in enumerate(subs):
            S.append(x)
            r = x + s.k * s.a[j] * (S[j & (j - 1)] - x) if j else x
            x = x + f(r)
        return x


class ChanLag(Base):
    """per-channel read latency: channel c reads a soft mix of earlier stream states centred at lag
    4 sigmoid(l_c) (l init -4: lag ~0); read = x + rk (mix - x)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.l = nn.Parameter(torch.full((L, cfg.d_model), -4.0))

    def forward(s, x, subs):
        S = []
        for j, f in enumerate(subs):
            S.append(x)
            r = x
            if j:
                lag = 4 * torch.sigmoid(s.l[j])                             # (d,)
                idx = torch.arange(j + 1, device=x.device, dtype=x.dtype).view(-1, 1)
                w = torch.softmax(-2 * (idx - (j - lag)).abs(), 0)          # (j+1, d)
                mix = (torch.stack(S, 0) * w.view(j + 1, 1, 1, -1)).sum(0)
                r = x + s.k * (mix - x)
            x = x + f(r)
        return x


class DepthProd(_Zero):
    """multiplicative depth interaction: read x + rk a_j (x * S_{j-2}) / RMS(S_{j-2}), a zero init."""
    def forward(s, x, subs):
        S = []
        for j, f in enumerate(subs):
            S.append(x)
            r = x + s.k * s.a[j] * x * S[j - 2] / rmsv(S[j - 2]) if j >= 2 else x
            x = x + f(r)
        return x


class MedianRead(_Zero):
    """robust depth pooling: read x + rk a_j (median over earlier stream states - x), a zero init."""
    def forward(s, x, subs):
        S = []
        for j, f in enumerate(subs):
            S.append(x)
            r = x + s.k * s.a[j] * (torch.stack(S, 0).median(0).values - x) if j >= 2 else x
            x = x + f(r)
        return x


class SpanRead(_Zero):
    """write-span read: read x + rk a_j (P x - x), P = orthogonal projector (per token) onto the span of all
    earlier writes."""
    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            r = x
            if ys:
                Q, _ = torch.linalg.qr(torch.stack(ys, -1))                 # (B,T,d,k)
                px = (Q * (Q * x.unsqueeze(-1)).sum(-2, keepdim=True)).sum(-1)
                r = x + s.k * s.a[j] * (px - x)
            y = f(r)
            ys.append(y)
            x = x + y
        return x


class Deembed(Base):
    """de-embedding read: read x - rk a_j proj_{x0}(x) (a scalar zero init): sublayers can hide the token
    embedding direction from their read."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        x0 = x
        u = x0 / (x0.norm(dim=-1, keepdim=True) + 1e-6)
        for j, f in enumerate(subs):
            x = x + f(x - s.k * s.a[j] * (x * u).sum(-1, keepdim=True) * u)
        return x


# ---------------- read feature maps ----------------
class ChanCA(_Zero):
    """cellular-automaton read: z = rms(x); read x + rk a_j RMS(x) tanh(z_{c-1} + z_{c+1} - 2 z_c) (ring)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            z = rms(x)
            ca = torch.tanh(z.roll(1, -1) + z.roll(-1, -1) - 2 * z)
            x = x + f(x + s.k * s.a[j] * rmsv(x) * ca)
        return x


class ChanPrefix(_Zero):
    """channel prefix read: read x + rk a_j (prefix mean over channels 0..c - x)."""
    def forward(s, x, subs):
        n = torch.arange(1, s.d + 1, device=x.device, dtype=x.dtype)
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * (x.cumsum(-1) / n - x))
        return x


class KronRead(Base):
    """Kronecker channel mixing on read: X = x as (8, d/8); read x + rk vec(A_j X B_j^T), A zero init, B identity."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        n = cfg.d_model // 8
        s.A = nn.Parameter(torch.zeros(L, 8, 8))
        s.B = nn.Parameter(torch.eye(n).repeat(L, 1, 1))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            X = x.unflatten(-1, (8, -1))
            x = x + f(x + s.k * (s.A[j] @ X @ s.B[j].t()).flatten(-2))
        return x


class SignRead(_Zero):
    """sign-feature read: read x + rk a_j RMS(x) sign(x)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * rmsv(x) * torch.sign(x))
        return x


class BucketRead(Base):
    """magnitude-bucket read: per channel, bucket b = clamp(floor(log2(|x|/RMS(x))), -4, 2) (7 buckets);
    read x + rk RMS(x) E_j[b, c], E zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.E = nn.Parameter(torch.zeros(L, 7, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            b = torch.clamp(torch.floor(torch.log2(x.abs() / rmsv(x) + 1e-4)), -4, 2).long() + 4
            e = torch.gather(s.E[j].t().expand(*x.shape[:2], -1, -1), -1, b.unsqueeze(-1)).squeeze(-1)
            x = x + f(x + s.k * rmsv(x) * e)
        return x


class LSHRead(Base):
    """LSH memory read: 8 fixed random hyperplanes hash rms(x) to one of 256 buckets; a shared learned table
    T (zero init) is read: x + rk g_j RMS(x) T[code], g init 1."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(17)
        s.register_buffer("R", torch.randn(cfg.d_model, 8, generator=g))
        s.register_buffer("p2", 2 ** torch.arange(8))
        s.T = nn.Parameter(torch.zeros(256, cfg.d_model))
        s.g = nn.Parameter(torch.ones(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            code = ((rms(x) @ s.R > 0).long() * s.p2).sum(-1)
            x = x + f(x + s.k * s.g[j] * rmsv(x) * s.T[code])
        return x


class SoftmaxRead(_Zero):
    """softmax-feature read: read x + rk a_j RMS(x) (d softmax(rms(x)) - 1)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * rmsv(x) * (s.d * torch.softmax(rms(x), -1) - 1))
        return x


# ---------------- write shaping ----------------
class SignAgree(_Zero):
    """momentum-sign agreement: H = sum of earlier writes; write y (1 + rk a_j tanh(y H / (RMS(y) RMS(H))))."""
    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            y = f(x)
            H = x - x0
            if j:
                y = y * (1 + s.k * s.a[j] * torch.tanh(y * H / (rmsv(y) * rmsv(H) + 1e-6)))
            x = x + y
        return x


class CurvGate(Base):
    """depth curvature gate: write y_j (1 + rk a_j cos(y_j, y_{j-1})) per token, a scalar zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        yp = None
        for j, f in enumerate(subs):
            y = f(x)
            y2 = y * (1 + s.k * s.a[j] * _cos(y, yp)) if yp is not None else y
            yp = y
            x = x + y2
        return x


class SpanWrite(Base):
    """full-span write orthogonalisation: y - rk a_j P_prev y, P_prev = projector on the span of all earlier
    writes (per token), a scalar zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        ys = []
        for j, f in enumerate(subs):
            y = f(x)
            if ys:
                Q, _ = torch.linalg.qr(torch.stack(ys, -1))
                py = (Q * (Q * y.unsqueeze(-1)).sum(-2, keepdim=True)).sum(-1)
                y = y - s.k * s.a[j] * py
            ys.append(y)
            x = x + y
        return x


class Ownership(Base):
    """soft channel ownership: 4 channel groups; sublayer j writes group g at m_jg = 4 softmax_g(O_j), columns
    rebalanced to mean 1 over sublayers (Sinkhorn step); write y (1 + rk (m - 1)), O zero init."""
    G = 4

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.O = nn.Parameter(torch.zeros(L, s.G))

    def forward(s, x, subs):
        m = s.G * torch.softmax(s.O, -1)
        m = m / m.mean(0, keepdim=True)
        m = m.repeat_interleave(s.d // s.G, -1)                            # (L, d)
        for j, f in enumerate(subs):
            x = x + f(x) * (1 + s.k * (m[j] - 1))
        return x


class WaterFill(Base):
    """water-filling write: channels where the stream is already large get less: write y d softmax_c(-rk a_j z^2),
    z = rms(x), a scalar zero init (uniform = plain)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            w = s.d * torch.softmax(-s.k * s.a[j] * rms(x).pow(2), -1)
            x = x + f(x) * w
        return x


class Saturate(_Zero):
    """saturating accumulator: a write pushing a channel further the way it already points is damped:
    y (1 - rk a_j relu(tanh(x y / (RMS(x) RMS(y))))), a zero init."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y * (1 - s.k * s.a[j] * F.relu(torch.tanh(x * y / (rmsv(x) * rmsv(y)))))
        return x


class X0Novel(Base):
    """embedding-novelty gain: write y (1 + rk a_j (1 - cos^2(y, x0))), a scalar zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y * (1 + s.k * s.a[j] * (1 - _cos(y, x0).pow(2)))
        return x


class SrcTag(Base):
    """source tags: every write carries a fixed random unit-RMS tag t_j of its sublayer, scaled by the write's RMS:
    y + rk g_j RMS(y) t_j (g init 1), so later reads can tell where content came from."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(23)
        t = torch.randn(L, cfg.d_model, generator=g)
        s.register_buffer("t", t / rmsv(t))
        s.g = nn.Parameter(torch.ones(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * s.g[j] * rmsv(y) * s.t[j]
        return x


class ContrastSub(Base):
    """contrastive sublayer: write F(x) - rk a_j F(x0) (what the sublayer would add for the bare embedding is
    removed), a scalar zero init (2x compute)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            y = f(x)
            if j:
                y = y - s.k * s.a[j] * f(x0)
            x = x + y
        return x


# ---------------- block fusion / composition ----------------
class Auction(Base):
    """per-channel auction between a block's attention and FFN writes: weights 2 softmax(rk (B + C |y|/RMS(y)))
    over the two writes, B, C zero init (1 each = plain)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.B = nn.Parameter(torch.zeros(L // 2, 2, cfg.d_model))
        s.C = nn.Parameter(torch.zeros(L // 2, 2, cfg.d_model))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            yf = subs[2 * b + 1](x + ya)
            la = s.k * (s.B[b, 0] + s.C[b, 0] * ya.abs() / rmsv(ya))
            lf = s.k * (s.B[b, 1] + s.C[b, 1] * yf.abs() / rmsv(yf))
            w = 2 * torch.softmax(torch.stack([la, lf], 0), 0)
            x = x + w[0] * ya + w[1] * yf
        return x


class SkewCouple(Base):
    """skew coupling between blocks: ya' = ya + rk yf_prev S_b, yf' = yf - rk ya S_b^T (energy-exchanging pair),
    S zero init per block."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.S = nn.Parameter(torch.zeros(L // 2, cfg.d_model, cfg.d_model))

    def forward(s, x, subs):
        yfp = torch.zeros_like(x)
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            ya = ya + s.k * yfp @ s.S[b]
            x = x + ya
            yf = subs[2 * b + 1](x)
            yf2 = yf - s.k * ya @ s.S[b].t()
            x = x + yf2
            yfp = yf
        return x


class MedFuse(Base):
    """median fusion: block write ya + yf + rk a_b (2 med(ya, yf, 0) - ya - yf), a per channel zero init
    (robust agreement of the two writes)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L // 2, cfg.d_model))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            yf = subs[2 * b + 1](x + ya)
            med = torch.maximum(torch.minimum(ya, yf), torch.minimum(torch.maximum(ya, yf), torch.zeros_like(ya)))
            x = x + ya + yf + s.k * s.a[b] * (2 * med - ya - yf)
        return x


class MatmulStream(Base):
    """matrix-product composition: stream channels as 8x8 blocks X, Y; x + y + rk a_j vec(X Y) / (sqrt8 RMS(x)),
    a scalar zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            p = (_blocks(x) @ _blocks(y)).flatten(-3)
            x = x + y + s.k * s.a[j] * p / (math.sqrt(8) * rmsv(x))
        return x


class LieBracket(Base):
    """Lie-bracket composition (second-order BCH term): x + y + rk a_j vec([X, Y]) / (2 sqrt8 RMS(x)),
    X, Y = 8x8 blocks, a scalar zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            X, Y = _blocks(x), _blocks(y)
            c = (X @ Y - Y @ X).flatten(-3)
            x = x + y + s.k * s.a[j] * c / (2 * math.sqrt(8) * rmsv(x))
        return x


class GeoCompose(_Zero):
    """geometric (multiplicative) composition: x + y + rk a_j (x (exp(tanh(y / RMS(x))) - 1) - y), a zero init
    (a = 1/rk: x exp(tanh(y/RMS)))."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * s.a[j] * (x * (torch.exp(torch.tanh(y / rmsv(x))) - 1) - y)
        return x


# ---------------- changed variants (2) ----------------
class RFFWrite(_Zero):
    """[variant of rffread, write side] write y + rk a_j RMS(y) sqrt2 cos(rms(y) W + b), W, b fixed random."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(7)
        s.register_buffer("W", torch.randn(cfg.d_model, cfg.d_model, generator=g) / math.sqrt(cfg.d_model))
        s.register_buffer("b", torch.rand(cfg.d_model, generator=g) * 2 * math.pi)

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * s.a[j] * rmsv(y) * math.sqrt(2) * torch.cos(rms(y) @ s.W + s.b)
        return x


class Unbind(_Zero):
    """[variant of hrr, read side] holographic unbinding with the embedding: read x + rk a_j RMS(x)
    (rms(x) (circ-corr) rms(x0)) / sqrt(d)."""
    def forward(s, x, subs):
        fx0 = torch.fft.rfft(rms(x), dim=-1).conj()
        for j, f in enumerate(subs):
            u = torch.fft.irfft(torch.fft.rfft(rms(x), dim=-1) * fx0, n=s.d, dim=-1) / math.sqrt(s.d)
            x = x + f(x + s.k * s.a[j] * rmsv(x) * u)
        return x


FAMILY = dict(fenwick="structured depth skip read", chanlag="per-channel read latency",
              depthprod="multiplicative depth interaction", medianread="robust depth pooling",
              spanread="write-span projection read", deembed="embedding-direction removal at read",
              chanca="cellular-automaton channel read", chanprefix="channel prefix read",
              kronread="Kronecker channel mixing read", signread="sign-feature read",
              bucketread="magnitude-bucket read", lshread="LSH memory read", softmaxread="softmax-feature read",
              signagree="write/history sign agreement", curvgate="depth curvature gate",
              spanwrite="full-span write orthogonalisation", ownership="soft channel ownership",
              waterfill="water-filling write", saturate="saturating accumulator", x0novel="embedding-novelty gain",
              srctag="source tags", contrastsub="contrastive sublayer", auction="per-channel write auction",
              skewcouple="skew block coupling", medfuse="median fusion", matmulstream="matrix-product composition",
              liebracket="Lie-bracket composition", geocompose="geometric composition",
              rffwrite="random features (variant of rffread)", unbind="holographic (variant of hrr)")

IDEAS = dict(fenwick=Fenwick, chanlag=ChanLag, depthprod=DepthProd, medianread=MedianRead, spanread=SpanRead,
             deembed=Deembed, chanca=ChanCA, chanprefix=ChanPrefix, kronread=KronRead, signread=SignRead,
             bucketread=BucketRead, lshread=LSHRead, softmaxread=SoftmaxRead, signagree=SignAgree,
             curvgate=CurvGate, spanwrite=SpanWrite, ownership=Ownership, waterfill=WaterFill, saturate=Saturate,
             x0novel=X0Novel, srctag=SrcTag, contrastsub=ContrastSub, auction=Auction, skewcouple=SkewCouple,
             medfuse=MedFuse, matmulstream=MatmulStream, liebracket=LieBracket, geocompose=GeoCompose,
             rffwrite=RFFWrite, unbind=Unbind)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
