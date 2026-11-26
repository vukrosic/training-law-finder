"""Round-1 residual-stream ideas (search: residual). Selected by cfg.res=<name>; free knobs rk, rk2, rs.

Every scheme: forward(x, subs) where x = token embedding [B,T,D], subs = list of 2L sublayers F_j(h)
(attn, ffn alternating, each with its own pre-norm). self.ids = token ids [B,T]; self.blocks_ref, self.rope
are set by GPT.forward. Return the final stream (GPT applies the final norm and head).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def hrow(ids, rows):
    return (ids * 40503 + 7) % rows


class Base(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        self.cfg, self.L, self.d = cfg, L, cfg.d_model
        self.k, self.k2 = cfg.rk, cfg.rk2


# ---------------- A. write-side ----------------
class WNorm(Base):
    """x += g_j * rms_normalize(F(x)); g learned per sublayer, init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + s.g[j] * rms(f(x))
        return x


class WProj(Base):
    """orthogonal-ish write: remove rk * (component of F along x) before adding."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            u = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
            x = x + y - s.k * (y * u).sum(-1, keepdim=True) * u
        return x


class WSign(Base):
    """write gate from agreement: x += F * 2 sigmoid(a_j cos(F,x)), a init rk (gate 1 at cos 0)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            c = F.cosine_similarity(y, x, dim=-1, eps=1e-6).unsqueeze(-1)
            x = x + y * 2 * torch.sigmoid(s.a[j] * c)
        return x


class WTok(Base):
    """per-token-id scalar write gain: x += F * (1 + G[id, j]); G zero init, scaled by rk (lr boost)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.G = nn.Parameter(torch.zeros(cfg.vocab, L))

    def forward(s, x, subs):
        g = s.G[s.ids] * s.k                                          # [B,T,L]
        for j, f in enumerate(subs):
            x = x + f(x) * (1 + g[..., j:j + 1])
        return x


class WTokVec(Base):
    """per-id (hashed to rk2 rows) vector write gain per sublayer: x += F * (1 + G[h(id), j, :])."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.rows = int(cfg.rk2 or 512)
        s.G = nn.Parameter(torch.zeros(s.rows, L, cfg.d_model))

    def forward(s, x, subs):
        g = s.G[hrow(s.ids, s.rows)] * s.k                            # [B,T,L,D]
        for j, f in enumerate(subs):
            x = x + f(x) * (1 + g[:, :, j])
        return x


class WDecay(Base):
    """leaky stream: x <- (1 - e_j) x + F, e_j = sigmoid(p_j), init e = rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.p = nn.Parameter(torch.full((L,), math.log(cfg.rk / (1 - cfg.rk))))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = (1 - torch.sigmoid(s.p[j])) * x + f(x)
        return x


class WMom(Base):
    """momentum residual: v <- b v + F(x); x <- x + v. b = rk fixed."""
    def forward(s, x, subs):
        v = 0
        for f in subs:
            v = s.k * v + f(x)
            x = x + v
        return x


class WRms(Base):
    """relative write: x += k_j * rms(x) * F / rms(F), k init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.g = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            r = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            x = x + s.g[j] * r * rms(y)
        return x


class WTanh(Base):
    """bounded write: x += c tanh(F / c), c = rk."""
    def forward(s, x, subs):
        for f in subs:
            x = x + s.k * torch.tanh(f(x) / s.k)
        return x


class Poly(Base):
    """bilinear write: x += F + c_j (F * rms(x)); c init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.c[j] * y * rms(x)
        return x


class CausalAvg(Base):
    """writes also smear forward in time: x += F + a_j * causal_mean_{t'<=t} F; a init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        T = x.shape[1]
        n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[None, :, None]
        for j, f in enumerate(subs):
            y = f(x)
            x = x + y + s.a[j] * y.cumsum(1) / n
        return x


class LowRank(Base):
    """learned linear residual map: x <- x + F + U_j V_j^T x, rank rk, zero-init U."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        r = int(cfg.rk)
        s.U = nn.Parameter(torch.zeros(L, cfg.d_model, r))
        s.V = nn.Parameter(torch.randn(L, cfg.d_model, r) / math.sqrt(cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x) + (x @ s.V[j]) @ s.U[j].t()
        return x


# ---------------- B. read-side ----------------
class RLast(Base):
    """read boosts the last write: h = x + a_j y_{j-1}; a init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        y = torch.zeros_like(x)
        for j, f in enumerate(subs):
            y = f(x + s.a[j] * y)
            x = x + y
        return x


class RDiff(Base):
    """read removes the causal running mean of the stream: h = x - c cummean(x); c = rk."""
    def forward(s, x, subs):
        T = x.shape[1]
        n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[None, :, None]
        for f in subs:
            x = x + f(x - s.k * x.cumsum(1) / n)
        return x


class TDiff(Base):
    """read adds the temporal change of the stream: h = x + a_j (x_t - x_{t-1}); a init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            d = x - F.pad(x, (0, 0, 1, 0))[:, :-1]
            x = x + f(x + s.a[j] * d)
        return x


class ChMask(Base):
    """soft channel masks: sublayer j reads x * sig(m_j) and writes F * sig(w_j); logits init rk,
    random +-rk2 so sublayers start on different channel subsets."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        g = torch.Generator().manual_seed(11)
        s.m = nn.Parameter(cfg.rk + cfg.rk2 * torch.randn(L, cfg.d_model, generator=g))
        s.w = nn.Parameter(cfg.rk + cfg.rk2 * torch.randn(L, cfg.d_model, generator=g))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x * 2 * torch.sigmoid(s.m[j])) * 2 * torch.sigmoid(s.w[j])
        return x


class IdScale(Base):
    """per-id scalar on each read: h = x * (1 + S[id, j]), zero init, scaled rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.S = nn.Parameter(torch.zeros(cfg.vocab, L))

    def forward(s, x, subs):
        g = s.S[s.ids] * s.k
        for j, f in enumerate(subs):
            x = x + f(x * (1 + g[..., j:j + 1]))
        return x


class IdGateRead(Base):
    """per-id hashed vector gate on reads: h = x * (1 + G[h(id), j]); rows rk2 (512), scale rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.rows = int(cfg.rk2 or 512)
        s.G = nn.Parameter(torch.zeros(s.rows, L, cfg.d_model))

    def forward(s, x, subs):
        g = s.G[hrow(s.ids, s.rows)] * s.k
        for j, f in enumerate(subs):
            x = x + f(x * (1 + g[:, :, j]))
        return x


class PrevEmb(Base):
    """stream gets the previous token's embedding, gated per sublayer read: h = x + a_j e_{t-1};
    a init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        e = F.pad(x, (0, 0, 1, 0))[:, :-1]
        for j, f in enumerate(subs):
            x = x + f(x + s.a[j] * e)
        return x


class SelfEx(Base):
    """self-excluding read: sublayer j reads x - c * (its own type's last output y_{j-2}); c = rk."""
    def forward(s, x, subs):
        last = [None, None]
        for j, f in enumerate(subs):
            h = x if last[j % 2] is None else x - s.k * last[j % 2]
            y = f(h)
            last[j % 2] = y
            x = x + y
        return x


class FfnReadsAttnIn(Base):
    """FFN reads the stream with part of the fresh attention output removed: h = x - c a; c = rk
    (c = 1 is a parallel block)."""
    def forward(s, x, subs):
        a = None
        for j, f in enumerate(subs):
            if j % 2 == 0:
                a = f(x)
                x = x + a
            else:
                x = x + f(x - s.k * a)
        return x


class KVLag(Base):
    """attention takes keys/values from a mix of the current stream and the stream before the previous
    block: kv = (1 - c) x + c x_{prev block input}; c = rk."""
    def forward(s, x, subs):
        cos, sin = s.rope
        prev = x
        for i, b in enumerate(s.blocks_ref):
            kv = (1 - s.k) * x + s.k * prev
            prev = x
            x = x + b.attn(b.n1(x), cos, sin, kv_x=b.n1(kv))
            x = x + subs[2 * i + 1](x)
        return x


class ReadX0(Base):
    """final readout subtracts (c>0) or adds (c<0) the token embedding: out = x - c x0; c = rk."""
    def forward(s, x, subs):
        x0 = x
        for f in subs:
            x = x + f(x)
        return x - s.k * x0


# ---------------- C. stream geometry ----------------
class CapNorm(Base):
    """cap stream rms at c: x <- x * min(1, c / rms(x)) after each write; c = rk."""
    def forward(s, x, subs):
        for f in subs:
            x = x + f(x)
            r = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            x = x * torch.clamp(s.k / r, max=1.0)
        return x


class Centered(Base):
    """remove c * channel mean from the stream after each write; c = rk."""
    def forward(s, x, subs):
        for f in subs:
            x = x + f(x)
            x = x - s.k * x.mean(-1, keepdim=True)
        return x


class EWhiten(Base):
    """subtract an EMA (rate rk) of the per-channel batch mean of the stream after each write."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.register_buffer("mu", torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if s.training and torch.is_grad_enabled():
                with torch.no_grad():
                    s.mu[j].lerp_(x.mean((0, 1)), s.k)
            x = x - s.mu[j]
        return x


# ---------------- D. multi-stream ----------------
class HCId(Base):
    """static Hyper-Connections (n = rk) whose read and write weights get per-id hashed offsets
    (rows rk2 = 512), zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        n = s.n = int(cfg.rk)
        s.rows = int(cfg.rk2 or 512)
        am = torch.zeros(L, n)
        am[torch.arange(L), torch.arange(L) % n] = 1.0
        s.am = nn.Parameter(am)
        s.ar = nn.Parameter(torch.eye(n).repeat(L, 1, 1))
        s.b = nn.Parameter(torch.ones(L, n))
        s.G = nn.Parameter(torch.zeros(s.rows, L, 2 * n))

    def forward(s, x, subs):
        n = s.n
        g = s.G[hrow(s.ids, s.rows)]                                  # [B,T,L,2n]
        H = x.unsqueeze(0).expand(n, *x.shape)
        for j, f in enumerate(subs):
            am = s.am[j][:, None, None] + g[:, :, j, :n].permute(2, 0, 1)
            b = s.b[j][:, None, None] + g[:, :, j, n:].permute(2, 0, 1)
            y = f((am.unsqueeze(-1) * H).sum(0))
            H = torch.einsum("ik,ibtd->kbtd", s.ar[j], H) + b.unsqueeze(-1) * y
        return H.sum(0)


class HCPerm(Base):
    """n = rk streams; sublayer j writes only stream j mod n, reads a learned softmax mix of streams
    (init uniform); no stream-to-stream mixing."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.n = int(cfg.rk)
        s.r = nn.Parameter(torch.zeros(L, s.n))

    def forward(s, x, subs):
        n = s.n
        H = [x] + [torch.zeros_like(x) for _ in range(n - 1)]
        for j, f in enumerate(subs):
            w = s.r[j].softmax(0) * n
            y = f(sum(w[i] * H[i] for i in range(n)))
            H[j % n] = H[j % n] + y
        return sum(H)


class HCOrth(Base):
    """Hyper-Connections whose stream-mixing matrix is orthogonal (Cayley of a learned skew matrix,
    init identity) instead of doubly stochastic; n = rk; static read/write."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        n = s.n = int(cfg.rk)
        am = torch.zeros(L, n)
        am[torch.arange(L), torch.arange(L) % n] = 1.0
        s.am = nn.Parameter(am)
        s.A = nn.Parameter(torch.zeros(L, n, n))
        s.b = nn.Parameter(torch.ones(L, n))

    def forward(s, x, subs):
        n = s.n
        I = torch.eye(n, device=x.device)
        H = x.unsqueeze(0).expand(n, *x.shape)
        for j, f in enumerate(subs):
            K = s.A[j] - s.A[j].t()
            Q = torch.linalg.solve(I + K, I - K)
            y = f(torch.einsum("n,nbtd->btd", s.am[j], H))
            H = torch.einsum("ik,ibtd->kbtd", Q, H) + s.b[j][:, None, None, None] * y
        return H.sum(0)


class HCExp(Base):
    """multi-timescale streams: stream k decays by r_k = rk^(k) per sublayer (stream 0 never decays);
    every sublayer writes all streams (learned weights, init 1) and reads a learned mix (init 1/n)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        n = s.n = int(cfg.rk2 or 3)
        s.register_buffer("r", torch.tensor([cfg.rk ** k for k in range(n)]))
        s.rd = nn.Parameter(torch.full((L, n), 1.0 / n))
        s.wr = nn.Parameter(torch.ones(L, n))

    def forward(s, x, subs):
        H = x.unsqueeze(0).expand(s.n, *x.shape)
        for j, f in enumerate(subs):
            y = f(torch.einsum("n,nbtd->btd", s.rd[j] * s.n, H) / s.n)
            H = s.r[:, None, None, None] * H + s.wr[j][:, None, None, None] * y
        return H[0]


class DualSum(Base):
    """two streams by writer type: A = sum of attn outputs (+emb), M = sum of FFN outputs (+emb);
    attn reads A + c M, FFN reads c A + M; readout A + M. c = rk learned per sublayer."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        A, M = x, x
        for j, f in enumerate(subs):
            if j % 2 == 0:
                A = A + f(A + s.c[j] * M)
            else:
                M = M + f(s.c[j] * A + M)
        return A + M


class DynGateMix(Base):
    """DualSum with a per-token read mix: h = A + M + tanh(w_j . rms(A+M)) * rk * (A - M)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.w = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        A, M = x, torch.zeros_like(x)
        for j, f in enumerate(subs):
            S = A + M
            g = torch.tanh((rms(S) * s.w[j]).sum(-1, keepdim=True)) * s.k
            y = f(S + g * (A - M))
            if j % 2 == 0:
                A = A + y
            else:
                M = M + y
        return A + M


# ---------------- E. depth attention variants ----------------
class AResTok(Base):
    """Attention Residuals with a per-id (hashed, rk2 rows) pseudo-query added to the learned one."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.rows = int(cfg.rk2 or 512)
        s.q = nn.Parameter(torch.zeros(L + 1, cfg.d_model))
        s.Q = nn.Parameter(torch.zeros(s.rows, L + 1, cfg.d_model))

    def mix(s, j, vals, qt):
        V = torch.stack(vals, 0)
        a = (rms(V) * (s.q[j] + s.k * qt[:, :, j])).sum(-1).softmax(0)
        return (a.unsqueeze(-1) * V).sum(0)

    def forward(s, x, subs):
        qt = s.Q[hrow(s.ids, s.rows)]
        vals = [x]
        for j, f in enumerate(subs):
            vals.append(f(s.mix(j, vals, qt)))
        return s.mix(len(subs), vals, qt)


class AResDyn(Base):
    """depth attention with a query projected from the running sum (control: depth-wise attention);
    h = sum softmax(rk * W_j rms(sum v) . rms(v_i)) v_i, W zero init."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.W = nn.Parameter(torch.zeros(L + 1, cfg.d_model, cfg.d_model))

    def mix(s, j, vals):
        V = torch.stack(vals, 0)
        q = rms(V.sum(0)) @ s.W[j]
        a = (rms(V) * q).sum(-1).softmax(0)
        return (a.unsqueeze(-1) * V).sum(0) * len(vals) ** s.k
    def forward(s, x, subs):
        vals = [x]
        for j, f in enumerate(subs):
            vals.append(f(s.mix(j, vals)))
        return s.mix(len(subs), vals)


class AResPlus(Base):
    """plain residual sum plus a gated Attention-Residual read: h = x + g_j AttnRes_j; g init rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L, cfg.d_model))
        s.g = nn.Parameter(torch.full((L,), float(cfg.rk)))

    def forward(s, x, subs):
        vals = [x]
        for j, f in enumerate(subs):
            V = torch.stack(vals, 0)
            a = (rms(V) * s.q[j]).sum(-1).softmax(0)
            y = f(x + s.g[j] * (a.unsqueeze(-1) * V).sum(0))
            vals.append(y)
            x = x + y
        return x


class AResSigned(Base):
    """per-token signed depth weights without normalization: h = sum_i (1 + rk tanh(w_j . rms(v_i))) v_i."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L + 1, cfg.d_model))

    def mix(s, j, vals):
        V = torch.stack(vals, 0)
        a = 1 + s.k * torch.tanh((rms(V) * s.q[j]).sum(-1))
        return (a.unsqueeze(-1) * V).sum(0)

    def forward(s, x, subs):
        vals = [x]
        for j, f in enumerate(subs):
            vals.append(f(s.mix(j, vals)))
        return s.mix(len(subs), vals)


class AResGeo(Base):
    """fixed geometric depth decay: h_j = x0 + sum_{i<j} g^(j-1-i) v_i (no learning); g = rk."""
    def forward(s, x, subs):
        vals = []
        x0 = x
        for f in subs:
            h = x0 + sum(s.k ** (len(vals) - 1 - i) * v for i, v in enumerate(vals)) if vals else x0
            vals.append(f(h))
        return x0 + sum(vals)


class Readout(Base):
    """plain residual, but the final readout is sum_i w_i v_i with learned scalars (init 1, emb init rk)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        w = torch.ones(L + 1)
        w[0] = cfg.rk
        s.w = nn.Parameter(w)

    def forward(s, x, subs):
        vals = [x]
        for f in subs:
            y = f(x)
            vals.append(y)
            x = x + y
        return sum(s.w[i] * v for i, v in enumerate(vals))


# ---------------- F. gradient routing ----------------
class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.k, None


class GradAmp(Base):
    """forward = pre-norm; gradient through the skip path scaled by rk at each sublayer."""
    def forward(s, x, subs):
        for f in subs:
            x = _GradScale.apply(x, s.k) + f(x)
        return x


class BranchGrad(Base):
    """forward = pre-norm; gradient into each sublayer branch scaled by rk."""
    def forward(s, x, subs):
        for f in subs:
            x = x + _GradScale.apply(f(x), s.k)
        return x


IDEAS = dict(wnorm=WNorm, wproj=WProj, wsign=WSign, wtok=WTok, wtokvec=WTokVec, wdecay=WDecay, wmom=WMom,
             wrms=WRms, wtanh=WTanh, poly=Poly, causalavg=CausalAvg, lowrank=LowRank, rlast=RLast,
             rdiff=RDiff, tdiff=TDiff, chmask=ChMask, idscale=IdScale, idgateread=IdGateRead,
             prevemb=PrevEmb, selfex=SelfEx, ffnattnin=FfnReadsAttnIn, kvlag=KVLag, readx0=ReadX0,
             capnorm=CapNorm, centered=Centered, ewhiten=EWhiten, hcid=HCId, hcperm=HCPerm, hcorth=HCOrth,
             hcexp=HCExp, dualsum=DualSum, dyngate=DynGateMix, arestok=AResTok, aresdyn=AResDyn,
             aresplus=AResPlus, aressigned=AResSigned, aresgeo=AResGeo, readout=Readout, gradamp=GradAmp,
             branchgrad=BranchGrad)


# ---------------- G. replacements for the dropped id/n-gram ideas (CLAIMS.md, 13:35) ----------------
class Wide(Base):
    """wide residual stream of m*d channels (m = rk): sublayer j reads R_j S (R init [I,0]), writes
    S += W_j F (W init [I; 0.1 randn/sqrt d]); readout P S (init [I,0]). Matrix-valued hyper-connections."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d, m = cfg.d_model, int(cfg.rk)
        s.m = m
        I = torch.eye(d)
        R = torch.zeros(L + 1, d, m * d)
        R[:, :, :d] = I
        s.R = nn.Parameter(R)
        g = torch.Generator().manual_seed(5)
        W = 0.1 * torch.randn(L, m * d, d, generator=g) / math.sqrt(d)
        W[:, :d] = I
        s.W = nn.Parameter(W)

    def forward(s, x, subs):
        S = torch.cat([x] + [torch.zeros_like(x)] * (s.m - 1), -1)
        for j, f in enumerate(subs):
            S = S + f(S @ s.R[j].t()) @ s.W[j].t()
        return S @ s.R[-1].t()


class LagRead(Base):
    """stale read: h_j = (1 - c) x_j + c x_{j-2} (the stream two sublayers ago); c = rk."""
    def forward(s, x, subs):
        hist = [x, x]
        for f in subs:
            h = (1 - s.k) * x + s.k * hist[-2]
            x = x + f(h)
            hist.append(x)
        return x


class Drift(Base):
    """reads remove c * (the stream's component along the token's own embedding); c = rk."""
    def forward(s, x, subs):
        e = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        for f in subs:
            x = x + f(x - s.k * (x * e).sum(-1, keepdim=True) * e)
        return x


class KVDepth(Base):
    """attention keys/values read a depth-attention mix of all earlier outputs (AttnRes-style learned
    query, zero init = mean), queries read the plain stream: kv = (1-c) x + c mix; c = rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        cos, sin = s.rope
        vals = [x]
        for i, b in enumerate(s.blocks_ref):
            V = torch.stack(vals, 0)
            a = (rms(V) * s.q[2 * i]).sum(-1).softmax(0)
            mix = (a.unsqueeze(-1) * V).sum(0)
            kv = (1 - s.k) * x + s.k * mix
            y = b.attn(b.n1(x), cos, sin, kv_x=b.n1(kv))
            vals.append(y)
            x = x + y
            y = subs[2 * i + 1](x)
            vals.append(y)
            x = x + y
        return x


class WCenter(Base):
    """writes lose c * their causal running mean over positions: x += y - c cummean_t(y); c = rk."""
    def forward(s, x, subs):
        T = x.shape[1]
        n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[None, :, None]
        for f in subs:
            y = f(x)
            x = x + y - s.k * y.cumsum(1) / n
        return x


class LowRankRead(Base):
    """learned read transform: h = x + U_j V_j^T x (rank rk, zero-init U); the stream itself is untouched."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        r = int(cfg.rk)
        s.U = nn.Parameter(torch.zeros(L, cfg.d_model, r))
        s.V = nn.Parameter(torch.randn(L, cfg.d_model, r) / math.sqrt(cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x + (x @ s.V[j]) @ s.U[j].t())
        return x


class Delayed(Base):
    """delayed writes: a fraction c of each sublayer's output enters the stream only after the next
    sublayer has read it (that sublayer never sees it); c = rk."""
    def forward(s, x, subs):
        pend = 0
        for f in subs:
            y = f(x)
            x = x + pend + (1 - s.k) * y
            pend = s.k * y
        return x + pend


IDEAS.update(wide=Wide, lagread=LagRead, drift=Drift, kvdepth=KVDepth, wcenter=WCenter,
             lowrankread=LowRankRead, delayed=Delayed)


# ---------------- H. fill-ins after the audit drop (depth-only, no cross-position mixing) ----------------
class DiffStream(Base):
    """second stream D accumulates attn writes minus FFN writes; reads h = x + c_j D (c init 0, x = plain sum);
    rk scales c's learning (c_eff = rk * c)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.c = nn.Parameter(torch.zeros(L))

    def forward(s, x, subs):
        D = torch.zeros_like(x)
        for j, f in enumerate(subs):
            y = f(x + s.k * s.c[j] * D)
            x = x + y
            D = D + (y if j % 2 == 0 else -y)
        return x


class ReadMax(Base):
    """read adds the elementwise max over depth of earlier outputs: h = x + c max_i v_i; c = rk."""
    def forward(s, x, subs):
        m = None
        for f in subs:
            y = f(x if m is None else x + s.k * m)
            m = y if m is None else torch.maximum(m, y)
            x = x + y
        return x


class Iter2(Base):
    """each sublayer looks ahead at its own write: y1 = F(x), y = F(x + c y1) (2x sublayer compute); c = rk."""
    def forward(s, x, subs):
        for f in subs:
            y1 = f(x)
            x = x + f(x + s.k * y1)
        return x


class DecorrWrite(Base):
    """each write loses c * its component along the previous write's direction; c = rk."""
    def forward(s, x, subs):
        u = None
        for f in subs:
            y = f(x)
            if u is not None:
                y = y - s.k * (y * u).sum(-1, keepdim=True) * u
            u = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
            x = x + y
        return x


IDEAS.update(diffstream=DiffStream, readmax=ReadMax, iter2=Iter2, decorrwrite=DecorrWrite)


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
