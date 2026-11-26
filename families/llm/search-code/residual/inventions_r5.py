"""Round-5 residual ideas (search: residual; O33 wild round 2). Standard init, stacked on NAG via rs=nag.
Knob rk. forward(x, subs): subs alternate attention, FFN. Most start pre-equivalent and learn away from it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inventions_r4 import Base, rms, rmsv, _orth


class GradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


# ---------------- untried mechanisms (30) ----------------
class DelayWrite(Base):
    """delayed write: write j joins the stream that sublayers >= j+2 read; sublayer j+1 reads it at weight (1 - rk)."""
    def forward(s, x, subs):
        pend = None
        for f in subs:
            r = x + (1 - s.k) * pend if pend is not None else x
            y = f(r)
            if pend is not None:
                x = x + pend
            pend = y
        return x + pend


class TwoTimescale(Base):
    """multi-timescale stream: fast stream holds the current block's writes; at block end the slow stream keeps
    (1 - rk + rk sigmoid(c_b)) of it (c init 4: about all), the rest is forgotten for later blocks but the head
    still sees all writes."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L // 2,), 4.0))

    def forward(s, x, subs):
        slow, fast, lost = x, 0, 0
        for j, f in enumerate(subs):
            fast = fast + f(slow + fast)
            if j % 2 == 1:
                keep = 1 - s.k + s.k * torch.sigmoid(s.c[j // 2])
                slow, lost, fast = slow + keep * fast, lost + (1 - keep) * fast, 0
        return slow + lost


class DerivRead(Base):
    """depth-derivative read: sublayer j reads x + rk c_j (x - x_at_block_start_prev), c per channel zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        prev = x
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.c[j] * (x - prev))
            if j % 2 == 1:
                prev = x
        return x


class InputCarry(Base):
    """input carry: sublayer j reads x + rk a_j (h_{j-1} - x), h_{j-1} the previous sublayer's read (a zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        hp = x
        for j, f in enumerate(subs):
            h = x + s.k * s.a[j] * (hp - x)
            x = x + f(h)
            hp = h
        return x


class TrustWrite(Base):
    """trust-ratio write: per token, a write is capped at rk times the stream's RMS."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            x = x + y * torch.clamp(s.k * rmsv(x) / rmsv(y), max=1.0)
        return x


class SparseRead(Base):
    """sparse read: each sublayer reads only the top rk*d channels of the stream by magnitude (others zero)."""
    def forward(s, x, subs):
        kk = max(1, int(round(s.k * s.d)))
        for f in subs:
            thr = x.abs().topk(kk, dim=-1).values[..., -1:]
            x = x + f(x * (x.abs() >= thr).to(x.dtype))
        return x


class FastWeight(Base):
    """depth fast weights: per token M = sum_i yhat_i yhat_i^T over earlier writes; sublayer j reads
    x + rk a_j M rms(x) / (i count) (a zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        Y = []
        for j, f in enumerate(subs):
            h = x
            if Y:
                U = torch.stack(Y, -2)                                   # [B,T,i,D]
                h = x + s.k * s.a[j] * rmsv(x) * torch.einsum("btid,bti->btd", U,
                                                              torch.einsum("btid,btd->bti", U, rms(x))) / len(Y)
            y = f(h)
            Y.append(y / (y.norm(dim=-1, keepdim=True) + 1e-6))
            x = x + y
        return x


class ErrFeed(Base):
    """error feedback: each write is projected on a learned rank-16 basis; the dropped part (rk of it) is carried
    and added to the next write; the head gets the remainder."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.U = nn.Parameter(_orth(L, cfg.d_model, 16, 7))

    def forward(s, x, subs):
        e = 0
        for j, f in enumerate(subs):
            y = f(x) + e
            p = (y @ s.U[j]) @ s.U[j].t()
            e = s.k * (y - p)
            x = x + y - e
        return x + e


class HaltSoft(Base):
    """soft halting over depth: per-token halting mass h accumulates sigmoid(rms(x) w_j + b_j) (w 0, b -4);
    writes scaled by 1 - rk clamp(h, 0, 1)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.w = nn.Parameter(torch.zeros(L, cfg.d_model))
        s.b = nn.Parameter(torch.full((L,), -4.0))

    def forward(s, x, subs):
        h = 0
        for j, f in enumerate(subs):
            x = x + f(x) * (1 - s.k * torch.clamp(h, 0, 1)) if j else x + f(x)
            h = h + torch.sigmoid(rms(x) @ s.w[j].unsqueeze(-1) + s.b[j])
        return x


class ClipRead(Base):
    """clipped read: sublayers read clamp(x, +-rk RMS(x)) (stream outliers hidden from the read, kept in the stream)."""
    def forward(s, x, subs):
        for f in subs:
            c = s.k * rmsv(x)
            x = x + f(torch.maximum(torch.minimum(x, c), -c))
        return x


class Scratchpad(Base):
    """scratchpad writes: attention writes are read by later layers, but the head sees x - rk g * (sum of attention
    writes), g = sigmoid(c), c init -4."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.tensor(-4.0))

    def forward(s, x, subs):
        acc = 0
        for j, f in enumerate(subs):
            y = f(x)
            if j % 2 == 0:
                acc = acc + y
            x = x + y
        return x - s.k * torch.sigmoid(s.c) * acc


class Feedback(Base):
    """top-down feedback: after the full pass, block 0 runs again reading x0 + rk g (final - x0) (g init 0),
    its writes added to the final stream."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.zeros(1))

    def forward(s, x, subs):
        x0 = x
        for f in subs:
            x = x + f(x)
        h = x0 + s.k * s.g * (x - x0)
        for f in subs[:2]:
            y = f(h)
            h = h + y
            x = x + s.k * s.g * y
        return x


class Bottleneck(Base):
    """stream bottleneck: after the middle block the stream is pulled toward its rank-16 projection by rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.U = nn.Parameter(_orth(1, cfg.d_model, 16, 8)[0])

    def forward(s, x, subs):
        mid = len(subs) // 2 - 1
        mid = mid if mid % 2 else mid + 1
        for j, f in enumerate(subs):
            x = x + f(x)
            if j == mid:
                x = x - s.k * (x - (x @ s.U) @ s.U.t())
        return x


class RandOrder(Base):
    """train-only stochastic sublayer order: each block swaps attention/FFN order with probability rk (eval: normal)."""
    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            a, f = subs[2 * b], subs[2 * b + 1]
            order = (f, a) if (s.training and torch.rand(()) < s.k) else (a, f)
            for g in order:
                x = x + g(x)
        return x


class PosScale(Base):
    """position-conditioned writes: write j scaled by 1 + rk p_j[t] (p learned per position, zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.p = nn.Parameter(torch.zeros(L, cfg.seq_len))

    def forward(s, x, subs):
        T = x.shape[1]
        for j, f in enumerate(subs):
            x = x + f(x) * (1 + s.k * s.p[j, :T]).view(1, T, 1)
        return x


class GapFill(Base):
    """gap-filling write: channels where the stream is small get boosted writes: y (1 + rk a_j exp(-x^2/RMS^2))."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y * (1 + s.k * s.a[j] * torch.exp(-x.pow(2) / rmsv(x).pow(2)))
        return x


class QuantStream(Base):
    """quantised stream: after every block x is rounded to steps of RMS(x)/rk (straight-through)."""
    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                st = rmsv(x).detach() / s.k
                x = x + (torch.round(x / st) * st - x).detach()
        return x


class StreamNonlin(Base):
    """stream nonlinearity across depth: after each block x <- x + rk c (RMS gelu(x/RMS) - x), c per channel zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L // 2, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                r = rmsv(x)
                x = x + s.k * s.c[j // 2] * (r * F.gelu(x / r) - x)
        return x


class Shrink(Base):
    """stream soft-threshold: after each block x <- sign(x) max(|x| - rk softplus(c_b) RMS, 0), c init -6."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L // 2,), -6.0))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                t = s.k * F.softplus(s.c[j // 2]) * rmsv(x)
                x = torch.sign(x) * F.relu(x.abs() - t)
        return x


class WriteBias(Base):
    """stream offset writes: each sublayer also writes rk RMS(x) b_j (constant direction per sublayer, zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x) + s.k * rmsv(x) * s.b[j]
        return x


class EmbModRead(Base):
    """embedding-modulated read: sublayer j reads x + rk a_j rms(x) * rms(x0) * RMS(x) (a per channel zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        e = rms(x)
        for j, f in enumerate(subs):
            x = x + f(x + s.k * s.a[j] * rms(x) * e * rmsv(x))
        return x


class GroupPool(Base):
    """multi-resolution write: x += y + rk a_j * (y averaged over groups of 8 channels, broadcast), a zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            c = y.view(*y.shape[:-1], -1, 8).mean(-1, keepdim=True).expand(*y.shape[:-1], -1, 8).reshape(y.shape)
            x = x + y + s.k * s.a[j] * c
        return x


class PolyWrite(Base):
    """polynomial write: x += y + rk c_j y^2 / RMS(y) (c per channel zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.k * s.c[j] * y.pow(2) / rmsv(y)
        return x


class IdentDrop(Base):
    """train-only identity dropout: with prob rk per token, a block's read loses its input except the embedding
    (block must rebuild from x0 + own writes); eval normal."""
    def forward(s, x, subs):
        x0 = x
        for b in range(len(subs) // 2):
            if s.training and b > 0 and s.k > 0:
                m = (torch.rand(*x.shape[:-1], 1, device=x.device) < s.k).to(x.dtype)
                r = m * x0 + (1 - m) * x
            else:
                r = x
            ya = subs[2 * b](r)
            yf = subs[2 * b + 1](r + ya)
            x = x + ya + yf
        return x


class Hysteresis(Base):
    """thresholded write with carry: per channel, write only the part above rk RMS(y); the rest is carried
    into the next write; the head gets the remainder."""
    def forward(s, x, subs):
        c = 0
        for f in subs:
            y = f(x) + c
            m = (y.abs() >= s.k * rmsv(y)).to(y.dtype)
            x, c = x + m * y, (1 - m) * y
        return x + c


class TypePrivate(Base):
    """type-private streams: a fraction rk sigmoid(c_j) (c init -2) of each write goes to a private stream only
    the same sublayer type reads; the rest to the shared stream; head reads all."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L,), -2.0))

    def forward(s, x, subs):
        P = [0, 0]
        for j, f in enumerate(subs):
            t = j % 2
            y = f(x + P[t])
            fr = s.k * torch.sigmoid(s.c[j])
            x, P[t] = x + (1 - fr) * y, P[t] + fr * y
        return x + P[0] + P[1]


class SoftmaxWrite(Base):
    """peaky write: y <- (1 - rk) y + rk sign(y) RMS(y) d softmax(|y| / RMS(y)) / RMS(d softmax) (mass on few channels)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            p = s.d * torch.softmax(y.abs() / rmsv(y), -1)
            q = torch.sign(y) * p / rmsv(p) * rmsv(y)
            x = x + (1 - s.k) * y + s.k * q
        return x


class MulFuse(Base):
    """multiplicative block fusion: block writes ya + yf + rk c_b (ya * yf) / RMS(ya) (c per channel zero)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L // 2, cfg.d_model))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            x = x + ya
            yf = subs[2 * b + 1](x)
            x = x + yf + s.k * s.c[b] * ya * yf / rmsv(ya)
        return x


class Rectified(Base):
    """rectified write: negative write channels scaled by 1 - rk sigmoid(c_j) (c init -4: about plain)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L, cfg.d_model), -4.0))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + torch.where(y < 0, (1 - s.k * torch.sigmoid(s.c[j])) * y, y)
        return x


class Budget(Base):
    """write budget: write j scaled 1 / (1 + rk softplus(c) E_j), E_j = cumulative write energy / x0 energy
    (c init -4)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.tensor(-4.0))

    def forward(s, x, subs):
        e0 = x.pow(2).mean(-1, keepdim=True) + 1e-6
        E = 0
        for f in subs:
            y = f(x) / (1 + s.k * F.softplus(s.c) * E)
            E = E + y.pow(2).mean(-1, keepdim=True) / e0
            x = x + y
        return x


# ---------------- changed variants of tried mechanisms (10) ----------------
class GradAmp(Base):
    """gradient path variant (r4 gradscale): backward through early writes (first half of depth) scaled by 1 + rk."""
    def forward(s, x, subs):
        n = len(subs)
        for j, f in enumerate(subs):
            y = f(x)
            x = x + (GradFn.apply(y, 1 + s.k) if j < n // 2 else y)
        return x


class CrossGLU(Base):
    """conditional write variant (r4 agreegate): FFN write * 2 sigmoid(rk a_b * rms(y_attn)), a per channel zero."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L // 2, cfg.d_model))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            x = x + ya
            x = x + subs[2 * b + 1](x) * 2 * torch.sigmoid(s.k * s.a[b] * rms(ya))
        return x


class PolarWrite(Base):
    """write geometry variant (r4 erasewrite): write split into the part along x and the part across it, scaled
    1 + rk a_j and 1 + rk b_j (zero init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L))
        s.b = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            u = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
            yp = (y * u).sum(-1, keepdim=True) * u
            x = x + (1 + s.k * s.a[j]) * yp + (1 + s.k * s.b[j]) * (y - yp)
        return x


class ExpMap(Base):
    """conditional write variant (r4 bilinear): multiplicative stream update x <- x exp(rk c_j y / RMS(x)) + y,
    c per channel zero init, exponent clamped to +-3."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x * torch.exp(torch.clamp(s.k * s.c[j] * y / rmsv(x), -3, 3)) + y
        return x


class NoisyRead(Base):
    """stochastic stream variant (r4 streamnoise): train-only noise on the read only (stream kept clean),
    std rk RMS(x)."""
    def forward(s, x, subs):
        for f in subs:
            h = x + s.k * rmsv(x).detach() * torch.randn_like(x) if s.training else x
            x = x + f(h)
        return x


class StaleRead(Base):
    """sublayer-order variant (r2 parallel): sublayer j reads x - rk g_j y_{j-1} (g init 1: rk = 1 reads the stream
    before the previous write)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.ones(L))

    def forward(s, x, subs):
        yp = 0
        for j, f in enumerate(subs):
            y = f(x - s.k * s.g[j] * yp)
            x, yp = x + y, y
        return x


class TiedSub(Base):
    """read-transform variant (r4 readsub/writesub): sublayer j reads and writes through the same learned rank-32
    subspace, mixed by rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.U = nn.Parameter(_orth(L, cfg.d_model, 32, 9))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            P = lambda v: v - s.k * (v - (v @ s.U[j]) @ s.U[j].t())
            x = x + P(f(P(x)))
        return x


class DynForget(Base):
    """depth-decay variant (r4 chandecay): token-dependent per-channel forget gate x <- (1 - rk sigmoid(rms(x) W_j - 4)
    + rk sigmoid(-4)) x + y, W zero init (start = plain)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.zeros(L, cfg.d_model, cfg.d_model))

    def forward(s, x, subs):
        g0 = torch.sigmoid(torch.tensor(-4.0))
        for j, f in enumerate(subs):
            y = f(x)
            x = (1 - s.k * torch.sigmoid(rms(x) @ s.W[j] - 4) + s.k * g0) * x + y
        return x


class AttnPrivate(Base):
    """sublayer-order variant (r4 attnlocal): attention writes reach the FFN of their block fully but the stream
    only at weight 1 - rk sigmoid(c_b) (c init -2)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L // 2,), -2.0))

    def forward(s, x, subs):
        for b in range(len(subs) // 2):
            ya = subs[2 * b](x)
            yf = subs[2 * b + 1](x + ya)
            x = x + (1 - s.k * torch.sigmoid(s.c[b])) * ya + yf
        return x


class ReadoutMid(Base):
    """readout variant (r1 readout): head reads final + rk a (stream after block 0), a zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(1))

    def forward(s, x, subs):
        mid = None
        for j, f in enumerate(subs):
            x = x + f(x)
            if j == 1:
                mid = x
        return x + s.k * s.a * mid


FAMILY = dict(delaywrite="delayed write", twotimescale="multi-timescale stream", derivread="depth-derivative read",
              inputcarry="input carry", trustwrite="trust-ratio write", sparseread="sparse read",
              fastweight="depth fast weights", errfeed="error feedback", haltsoft="soft halting",
              clipread="clipped read", scratchpad="scratchpad writes", feedback="top-down feedback",
              bottleneck="stream bottleneck", randorder="stochastic sublayer order", posscale="position-scaled write",
              gapfill="gap-filling write", quantstream="quantised stream", streamnonlin="stream nonlinearity",
              shrink="stream soft-threshold", writebias="stream offset write", embmodread="embedding-modulated read",
              grouppool="multi-resolution write", polywrite="polynomial write", identdrop="identity dropout",
              hysteresis="thresholded write with carry", typeprivate="type-private streams",
              softmaxwrite="peaky write", mulfuse="multiplicative block fusion", rectified="rectified write",
              budget="write budget",
              gradamp="tried variant: r4 gradscale", crossglu="tried variant: r4 agreegate",
              polarwrite="tried variant: r4 erasewrite", expmap="tried variant: r4 bilinear",
              noisyread="tried variant: r4 streamnoise", staleread="tried variant: r2 parallel",
              tiedsub="tried variant: r4 readsub/writesub", dynforget="tried variant: r4 chandecay",
              attnprivate="tried variant: r4 attnlocal", readoutmid="tried variant: r1 readout")

IDEAS = dict(delaywrite=DelayWrite, twotimescale=TwoTimescale, derivread=DerivRead, inputcarry=InputCarry,
             trustwrite=TrustWrite, sparseread=SparseRead, fastweight=FastWeight, errfeed=ErrFeed, haltsoft=HaltSoft,
             clipread=ClipRead, scratchpad=Scratchpad, feedback=Feedback, bottleneck=Bottleneck,
             randorder=RandOrder, posscale=PosScale, gapfill=GapFill, quantstream=QuantStream,
             streamnonlin=StreamNonlin, shrink=Shrink, writebias=WriteBias, embmodread=EmbModRead,
             grouppool=GroupPool, polywrite=PolyWrite, identdrop=IdentDrop, hysteresis=Hysteresis,
             typeprivate=TypePrivate, softmaxwrite=SoftmaxWrite, mulfuse=MulFuse, rectified=Rectified,
             budget=Budget, gradamp=GradAmp, crossglu=CrossGLU, polarwrite=PolarWrite, expmap=ExpMap,
             noisyread=NoisyRead, staleread=StaleRead, tiedsub=TiedSub, dynforget=DynForget,
             attnprivate=AttnPrivate, readoutmid=ReadoutMid)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
