"""Search 1 (compute): new per-token compute blocks that replace the FFN / MoE layer.

Use: --method inv --set inv=c_<name> ca=<int knob> cb=<float knob> cc=<str knob>
Budget (XS, d 64): dsv4's FFN layer has ~105k parameters and 128 active SwiGLU units per token.
Every block stays at or under ~105k parameters per layer. Masked blocks compute all units and
zero the unused ones (same function as a gathered kernel; the active unit count is what is matched).
Parameters named k_* keep their own init (GPT re-inits other 2-D weights to N(0, 0.02)).
ctx["ids"] = token ids [B, T]; x arrives as [B, T, d] and is flattened batch-major, as ids.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def P(*shape, std=0.02):
    return nn.Parameter(torch.randn(*shape) * std)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def idtab(V, R, seed):                       # token id -> row in [0, R)
    return torch.randint(0, R, (V,), generator=gen(seed))


def ksets(V, E, k, seed):                    # token id -> k distinct members of E
    return torch.rand(V, E, generator=gen(seed)).argsort(1)[:, :k]


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class SG(nn.Module):
    def __init__(self, d, h, d_out=None):
        super().__init__()
        self.w1, self.w3, self.w2 = P(d, h), P(d, h), P(h, d_out or d)

    def forward(self, x):
        return (F.silu(x @ self.w1) * (x @ self.w3)) @ self.w2


class HMoE(nn.Module):
    """All-hash MoE: token id -> k of E experts (fixed random table); gates = learned sigmoid scores
    of the picked experts renormalized x 2.5 (as the dsv4 hash layer). fn: swiglu | relu2 | cubic | bilin."""
    def __init__(self, d, V, E, h, k, seed, fn="swiglu", nl=3, d_in=None):
        super().__init__()
        d_in = d_in or d
        self.E, self.k, self.fn = E, k, fn
        self.W1 = P(E, d_in, h)
        if fn not in ("relu2", "tied"):  # swiglu_n: swiglu with rms-normed hidden
            self.W3 = P(E, d_in, h)
        self.W2 = P(E, h, d, std=0.02 / math.sqrt(2 * nl))
        if fn == "cubic":
            self.W4 = P(E, d_in, h)
        self.gate = P(E, d_in)
        self.register_buffer("sets", ksets(V, E, k, seed), persistent=False)

    def expert(self, e, xe):
        if self.fn == "swiglu":
            hh = F.silu(xe @ self.W1[e]) * (xe @ self.W3[e])
        elif self.fn == "relu2":
            hh = F.relu(xe @ self.W1[e]).square()
        elif self.fn == "cubic":
            hh = F.silu(xe @ self.W1[e]) * (xe @ self.W3[e]) * (1 + xe @ self.W4[e])
        elif self.fn == "bilin":
            hh = (xe @ self.W1[e]) * (xe @ self.W3[e])
        elif self.fn == "taylor":
            z = xe @ self.W1[e]
            hh = z + 0.5 * z * (xe @ self.W3[e])
        elif self.fn == "maxout":
            hh = torch.maximum(xe @ self.W1[e], xe @ self.W3[e])
        elif self.fn == "tied":
            z = xe @ self.W1[e]
            hh = F.silu(z) * z
        elif self.fn == "cos":
            hh = torch.cos(xe @ self.W1[e]) * (xe @ self.W3[e])
        elif self.fn == "swiglu_n":
            hh = rms(F.silu(xe @ self.W1[e]) * (xe @ self.W3[e]))
        return hh @ self.W2[e]

    def forward(self, x, ids, sets=None, xin=None):
        xin = x if xin is None else xin
        sets = self.sets[ids] if sets is None else sets
        g = torch.sigmoid(x @ self.gate.t()).gather(1, sets)
        g = g / (g.sum(-1, keepdim=True) + 1e-20) * 2.5
        N, k = sets.shape
        y = x.new_zeros(N, self.W2.shape[-1])
        fi, fg = sets.reshape(-1), g.reshape(-1, 1)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        for e in range(self.E):
            sel = (fi == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            t = tok[sel]
            y.index_add_(0, t, self.expert(e, xin.index_select(0, t)) * fg[sel])
        return y


class C(nn.Module):
    def __init__(self, d, cfg, layer, ctx):
        super().__init__()
        self.d, self.cfg, self.layer, self.ctx, self.V = d, cfg, layer, ctx, cfg.vocab
        self.ca, self.cb, self.cc = cfg.ca, cfg.cb, cfg.cc
        self.seed = 1000 * (layer + 1)

    def forward(self, x):
        B, T, d = x.shape
        ids = self.ctx["ids"]
        return self.f(x.reshape(-1, d), ids.reshape(-1), ids).view(B, T, d)

    @staticmethod
    def prev(ids, n=1):
        return torch.cat([torch.zeros_like(ids[:, :n]), ids[:, :-n]], 1).reshape(-1)


# ---------------- family A: hash MoE frame (control + expert-function variants) ----------------
class HashFrame(C):
    """all-hash MoE (control) and variants that change the expert function or its input."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        V, s = self.V, self.seed
        E, h, k = 16, cfg.expert_hidden, 3
        fn = {"sqrelu": "relu2", "cubicexp": "cubic", "bilin": "bilin"}.get(mode, "swiglu")
        if mode == "sqrelu":
            h = int(h * 1.5)
        if mode == "cubicexp":
            h = int(h * 0.75)
        if mode == "finehash":                      # E experts of width 512/E, k = 96 units / width
            E = self.ca or 64
            h = 512 // E
            k = 96 // h
        if mode == "iter":
            k = self.ca or 2
        self.moe = HMoE(d, V, E, h, k, s, fn, cfg.n_layers)
        self.sh = SG(d, cfg.shared_hidden)
        if mode == "dualhash":                      # 2 experts by id, 1 by an earlier token's id
            self.register_buffer("t2", ksets(V, E, 2, s + 1), persistent=False)
            self.register_buffer("t1", ksets(V, E, 1, s + 2), persistent=False)
        if mode == "expemb":
            self.k_e = P(128, d, std=0.02)
            self.register_buffer("tab", idtab(V, 128, s + 3), persistent=False)
        if mode == "idscale":
            self.k_S = nn.Parameter(torch.zeros(128, d))
            self.register_buffer("tab", idtab(V, 128, s + 3), persistent=False)
        if mode == "modemb":
            self.k_P = nn.Parameter(torch.zeros(d, d))
        if mode == "depth2":
            self.moe = HMoE(d, V, 16, cfg.expert_hidden, 2, s, "swiglu", cfg.n_layers)
            self.st2 = SG(d, 64)
            self.k_ws = nn.Parameter(torch.zeros(d))

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "dualhash":
            p = self.prev(ids2, self.ca or 1)
            sets = torch.cat([self.t2[ids], self.t1[p]], 1)
            return self.moe(x, ids, sets) + self.sh(x)
        if m == "expemb":
            return self.moe(x, ids, xin=x + self.k_e[self.tab[ids]]) + self.sh(x)
        if m == "idscale":
            return self.moe(x, ids) * (1 + self.k_S[self.tab[ids]]) + self.sh(x)
        if m == "modemb":
            emb = self.ctx["emb"].reshape(-1, self.d)
            return (self.moe(x, ids) + self.sh(x)) * (1 + torch.tanh(emb @ self.k_P))
        if m == "iter":
            y1 = self.moe(x, ids) + self.sh(x)
            x2 = rms(x + y1)
            return y1 + self.moe(x2, ids) + self.sh(x2)
        if m == "depth2":
            y1 = self.moe(x, ids) + self.sh(x)
            sc = x @ self.k_ws
            c = max(1, int(x.shape[0] * (self.ca or 50) / 100))
            top = torch.topk(sc.detach(), c).indices
            x2 = rms(x + y1).index_select(0, top)
            z = self.st2(x2) * torch.sigmoid(sc.index_select(0, top))[:, None] * 2
            return y1.index_add(0, top, z)
        return self.moe(x, ids) + self.sh(x)


class SharedW1(C):
    """hash MoE with 24 experts that share one input projection W1 and own W3 and W2."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        E, h = 24, cfg.expert_hidden
        self.E = E
        self.w1 = P(d, h)
        self.W3, self.W2 = P(E, d, h), P(E, h, d, std=0.02 / math.sqrt(2 * cfg.n_layers))
        self.register_buffer("sets", ksets(self.V, E, 3, self.seed), persistent=False)
        self.sh = SG(d, cfg.shared_hidden)

    def f(self, x, ids, ids2):
        a = F.silu(x @ self.w1)
        sets = self.sets[ids]
        z3 = torch.einsum("nd,nkdh->nkh", x, self.W3[sets])
        y = torch.einsum("nkh,nkhd->nd", a[:, None] * z3, self.W2[sets]) * (2.5 / 3)
        return y + self.sh(x)


class Compose(C):
    """compositional hash experts: banks A and B of 8 experts; a token uses k merged experts
    W_a + W_b (merged inside the nonlinearity); 64 possible merged experts per bank pair."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        E, h = 8, cfg.expert_hidden
        self.k = self.ca or 3
        L = cfg.n_layers
        self.A1, self.A3, self.A2 = P(E, d, h), P(E, d, h), P(E, h, d, std=0.02 / math.sqrt(2 * L))
        self.B1, self.B3, self.B2 = P(E, d, h), P(E, d, h), P(E, h, d, std=0.02 / math.sqrt(2 * L))
        self.register_buffer("sa", ksets(self.V, E, self.k, self.seed), persistent=False)
        self.register_buffer("sb", ksets(self.V, E, self.k, self.seed + 7), persistent=False)
        self.sh = SG(d, cfg.shared_hidden)

    def f(self, x, ids, ids2):
        a, b = self.sa[ids], self.sb[ids]                                  # [N,k]
        def gat(Z, s):
            return Z.gather(1, s[:, :, None].expand(-1, -1, Z.shape[-1]))
        z1 = gat(torch.einsum("nd,edh->neh", x, self.A1), a) + gat(torch.einsum("nd,edh->neh", x, self.B1), b)
        z3 = gat(torch.einsum("nd,edh->neh", x, self.A3), a) + gat(torch.einsum("nd,edh->neh", x, self.B3), b)
        hh = F.silu(z1) * z3                                               # [N,k,h]
        oa = torch.einsum("nkh,ehd->nked", hh, self.A2)
        ob = torch.einsum("nkh,ehd->nked", hh, self.B2)
        D = x.shape[1]
        oa = oa.gather(2, a[:, :, None, None].expand(-1, -1, 1, D)).squeeze(2)
        ob = ob.gather(2, b[:, :, None, None].expand(-1, -1, 1, D)).squeeze(2)
        return (oa + ob).sum(1) * (2.5 / self.k / 2) + self.sh(x)


# ---------------- family B: hashed neuron subsets (token id picks units, not experts) ----------------
class HMask(C):
    """one SwiGLU of 512 units + 32 always-on units; each token activates K units:
    hmask: K by id; hmask_prev: K/2 by id + K/2 by previous id; hmask_freq: frequent ids ca units,
    rare ids 192-ca; hmask_x: K by a fixed sign hash of x (64 subsets); overlap: 3 hashed windows."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, U = mode, 512
        self.U = U
        self.K = self.ca or 96
        self.w1, self.w3, self.w2 = P(d, U + 32), P(d, U + 32), P(U + 32, d)
        V, s = self.V, self.seed
        if mode in ("hmask", "hmask_freq", "hmask_bilin"):
            self.register_buffer("units", ksets(V, U, max(self.K, 192 - self.K), s), persistent=False)
        if mode == "hmask_prev":
            self.register_buffer("units", ksets(V, U, self.K // 2, s), persistent=False)
            self.register_buffer("units2", ksets(V, U, self.K // 2, s + 1), persistent=False)
        if mode == "hmask_x":
            self.register_buffer("proj", torch.randn(6, d, generator=gen(s)), persistent=False)
            self.register_buffer("units", ksets(64, U, self.K, s + 1), persistent=False)
        if mode == "overlap":
            W = self.ca or 48
            self.register_buffer("sets", ksets(V, 16, 3, s), persistent=False)
            win = (torch.arange(16)[:, None] * 32 + torch.arange(W)[None]) % U
            self.register_buffer("win", win, persistent=False)

    def mask(self, x, ids, ids2):
        N = x.shape[0]
        m = x.new_zeros(N, self.U + 32)
        m[:, self.U:] = 1
        mo = self.mode
        if mo in ("hmask", "hmask_bilin"):
            m.scatter_(1, self.units[ids][:, :self.K], 1.0)
        elif mo == "hmask_freq":
            top = self.ctx["freq_rank_top"][ids]
            u = self.units[ids]
            kf, kr = self.K, 192 - self.K
            ar = torch.arange(u.shape[1], device=x.device)[None]
            keep = torch.where(top[:, None], ar < kf, ar < kr).float()
            m.scatter_(1, u, keep)
        elif mo == "hmask_prev":
            m.scatter_(1, self.units[ids], 1.0)
            m.scatter_(1, self.units2[self.prev(ids2)], 1.0)
        elif mo == "hmask_x":
            code = (((x.detach() @ self.proj.t()) > 0).long() * (2 ** torch.arange(6, device=x.device))).sum(-1)
            m.scatter_(1, self.units[code], 1.0)
        elif mo == "overlap":
            u = self.win[self.sets[ids]].reshape(N, -1)
            m.scatter_(1, u, 1.0)
        return m

    def f(self, x, ids, ids2):
        z1 = x @ self.w1
        a = z1 if self.mode == "hmask_bilin" else F.silu(z1)
        hh = a * (x @ self.w3) * self.mask(x, ids, ids2)
        return hh @ self.w2


class NRoute(C):
    """learned neuron-level routing: 512 units, a router picks the top-K units per token
    (sigmoid score + sign-updated load bias); picked units are scaled by their score; +32 shared units."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        U = 384
        self.U, self.K = U, self.ca or 96
        self.w1, self.w3, self.w2 = P(d, U + 32), P(d, U + 32), P(U + 32, d)
        self.gate = P(U, d)
        self.register_buffer("bias", torch.zeros(U))
        self.register_buffer("count", torch.zeros(U))

    def f(self, x, ids, ids2):
        s = torch.sigmoid(x @ self.gate.t())
        idx = torch.topk(s.detach() + self.bias, self.K, dim=1).indices
        g = torch.zeros_like(s).scatter(1, idx, s.gather(1, idx) * 2)
        if self.training:
            with torch.no_grad():
                self.count += torch.bincount(idx.reshape(-1), minlength=self.U).float()
        m = torch.cat([g, torch.ones_like(g[:, :32])], 1)
        return (F.silu(x @ self.w1) * (x @ self.w3) * m) @ self.w2

    def step_update(self):
        if self.count.sum() > 0:
            self.bias += 1e-3 * torch.sign(self.count.mean() - self.count)
            self.count.zero_()


# ---------------- family C: id-conditioned dense FFN (identity modulates units) ----------------
class IdDense(C):
    """dense SwiGLU-128 with a hashed token-id table (R rows) that changes it:
    hgate: hidden x (1 + G[id]); hbias: G[id] added to the W1 pre-activation; idglu: the W3 gate is
    replaced by G[id] (width ca); idglu2: W3x + G[id]; rank1id: + u[id](v[id].x); rank4id: rank-4;
    hyperid: W1 gets a per-id rank-1 change a[id] b[id]^T; outmod: output x (1 + M[id]);
    stackmem: FFN(x + M[id])."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        h = cfg.dense_hidden
        R = self.ca or 512
        if mode == "idglu":
            h = self.ca or 128
            R = 65536 // h
        if mode == "rank4id":
            R = self.ca or 128
        self.h, self.R = h, R
        self.w1, self.w2 = P(d, h), P(h, d)
        if mode != "idglu":
            self.w3 = P(d, h)
        self.register_buffer("tab", idtab(self.V, R, self.seed), persistent=False)
        if mode in ("hgate", "hbias", "idglu2"):
            self.k_G = nn.Parameter(torch.zeros(R, h))
        if mode == "idglu":
            self.k_G = P(R, h, std=0.1)
        if mode == "rank1id":
            self.k_u, self.k_v = nn.Parameter(torch.zeros(R, d)), P(R, d, std=0.1)
        if mode == "rank4id":
            self.k_U, self.k_W = nn.Parameter(torch.zeros(R, 4, d)), P(R, d, 4, std=0.1)
        if mode == "hyperid":
            self.k_a, self.k_b = nn.Parameter(torch.zeros(R, h)), P(R, d, std=0.1)
        if mode in ("outmod",):
            self.k_M = nn.Parameter(torch.zeros(R, d))
        if mode == "stackmem":
            self.k_M = P(R, d, std=0.02)

    def f(self, x, ids, ids2):
        t, m = self.tab[ids], self.mode
        if m == "stackmem":
            x = x + self.k_M[t]
        z1 = x @ self.w1
        if m == "hbias":
            z1 = z1 + self.k_G[t]
        if m == "hyperid":
            z1 = z1 + self.k_a[t] * (self.k_b[t] * x).sum(-1, keepdim=True)
        if m == "idglu":
            hh = F.silu(z1) * self.k_G[t]
        elif m == "idglu2":
            hh = F.silu(z1) * (x @ self.w3 + self.k_G[t])
        else:
            hh = F.silu(z1) * (x @ self.w3)
        if m == "hgate":
            hh = hh * (1 + self.k_G[t])
        y = hh @ self.w2
        if m == "rank1id":
            y = y + self.k_u[t] * (self.k_v[t] * x).sum(-1, keepdim=True)
        if m == "rank4id":
            y = y + torch.einsum("nr,nrd->nd", torch.einsum("nd,ndr->nr", x, self.k_W[t]), self.k_U[t])
        if m == "outmod":
            y = y * (1 + self.k_M[t])
        return y


class LinExp(C):
    """dense SwiGLU-128 + 64 low-rank linear experts (rank 8); each token id picks 2 by hash."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        E = int(self.cc[1:]) if str(self.cc).startswith("E") else 64
        r, k = self.ca or 8, int(self.cb) or 2
        self.ff = SG(d, cfg.dense_hidden)
        self.k_U, self.k_W = nn.Parameter(torch.zeros(E, r, d)), P(E, d, r, std=0.1)
        self.register_buffer("sets", ksets(self.V, E, k, self.seed), persistent=False)

    def f(self, x, ids, ids2):
        s = self.sets[ids]
        z = torch.einsum("nd,nkdr->nkr", x, self.k_W[s])
        return self.ff(x) + torch.einsum("nkr,nkrd->nd", z, self.k_U[s])


# ---------------- family D: lookup memories ----------------
class Mem(C):
    """dense SwiGLU (width 128, or 96 for pkm/memx/codebook) + a value memory added to the output.
    memid: row = hash(id); membi: hash(prev, id); memtri: hash(prev2, prev, id); memgate: sigmoid(Wx) * M[id];
    twotable: M1[h1(id)] + M2[h2(id)]; memx: sign-LSH tables on x; pkm: product keys on x (Lample 2019);
    codebook: nearest learned centroid of x (straight-through softmax); softmaxffn: softmax over top-ca of 512 keys."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        w = 96 if mode in ("pkm", "memx", "codebook") else 128
        if mode == "softmaxffn":
            w = cfg.shared_hidden
        self.ff = SG(d, w)
        R = self.ca or 1024
        s = self.seed
        if mode in ("memid", "membi", "memtri", "memgate"):
            self.M = P(R, d)
            self.R = R
            self.register_buffer("tab", idtab(self.V, R, s), persistent=False)
        if mode == "memgate":
            self.wg = P(d, d)
        if mode == "twotable":
            R = self.ca or 512
            self.M1, self.M2 = P(R, d), P(R, d)
            self.register_buffer("t1", idtab(self.V, R, s), persistent=False)
            self.register_buffer("t2", idtab(self.V, R, s + 1), persistent=False)
        if mode == "memx":
            T = self.ca or 4
            bits = {2: 9, 4: 8, 8: 7}[T]
            self.T, self.bits = T, bits
            self.k_M = P(T, 2 ** bits, d, std=0.02)
            self.register_buffer("proj", torch.randn(T, bits, d, generator=gen(s)), persistent=False)
        if mode == "pkm":
            self.n, self.kk = 32, self.ca or 8
            self.wq = P(d, 64)
            self.k_K1, self.k_K2 = P(32, 32, std=1.0), P(32, 32, std=1.0)
            self.k_vals = P(1024, d, std=0.02)
        if mode == "codebook":
            K = self.ca or 256
            self.k_Cb = P(K, d, std=1.0)
            self.k_vals = P(K, d, std=0.02)
        if mode == "softmaxffn":
            self.k_keys, self.k_vals = P(512, d, std=1.0 / math.sqrt(d)), P(512, d, std=0.02)
            self.kk = self.ca or 32

    def f(self, x, ids, ids2):
        m, y = self.mode, self.ff(x)
        if m in ("memid", "memgate"):
            v = self.M[self.tab[ids]]
            return y + (v * torch.sigmoid(x @ self.wg) * 2 if m == "memgate" else v)
        if m == "membi":
            p = self.prev(ids2)
            return y + self.M[(p * 1000003 + ids * 7919) % self.R]
        if m == "memtri":
            p, p2 = self.prev(ids2), self.prev(ids2, 2)
            return y + self.M[(p2 * 999983 + p * 1000003 + ids * 7919) % self.R]
        if m == "twotable":
            return y + self.M1[self.t1[ids]] + self.M2[self.t2[ids]]
        if m == "memx":
            pw = 2 ** torch.arange(self.bits, device=x.device)
            code = ((torch.einsum("nd,tbd->ntb", x.detach(), self.proj) > 0).long() * pw).sum(-1)   # [N,T]
            out = 0
            for t in range(self.T):
                out = out + self.k_M[t][code[:, t]]
            return y + out
        if m == "pkm":
            q = x @ self.wq
            s1, s2 = q[:, :32] @ self.k_K1.t(), q[:, 32:] @ self.k_K2.t()
            kk = self.kk
            v1, i1 = s1.topk(kk, 1)
            v2, i2 = s2.topk(kk, 1)
            sc = (v1[:, :, None] + v2[:, None, :]).reshape(-1, kk * kk)
            ii = (i1[:, :, None] * self.n + i2[:, None, :]).reshape(-1, kk * kk)
            vs, j = sc.topk(kk, 1)
            w = F.softmax(vs, -1)
            return y + torch.einsum("nk,nkd->nd", w, self.k_vals[ii.gather(1, j)])
        if m == "codebook":
            sim = rms(x) @ F.normalize(self.k_Cb, dim=-1).t() * 4
            w = F.softmax(sim, -1)
            hard = F.one_hot(w.argmax(-1), w.shape[-1]).to(w.dtype)
            return y + (hard + w - w.detach()) @ self.k_vals
        if m == "softmaxffn":
            sc = x @ self.k_keys.t()
            v, i = sc.topk(self.kk, 1)
            return y + torch.einsum("nk,nkd->nd", F.softmax(v, -1), self.k_vals[i])
        raise ValueError(m)


class Peer(C):
    """PEER (He 2024): 576 single-neuron experts with product keys (24 x 24), 4 heads, top-kk per head; + shared 32."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.n, self.H, self.kk = 24, 4, self.ca or 8
        self.wq = P(d, self.H * 64)
        self.k_K1, self.k_K2 = P(self.H, 24, 32, std=1.0), P(self.H, 24, 32, std=1.0)
        self.u, self.v = P(576, d), P(576, d)
        self.sh = SG(d, cfg.shared_hidden)

    def f(self, x, ids, ids2):
        N, kk = x.shape[0], self.kk
        q = (x @ self.wq).view(N, self.H, 64)
        s1 = torch.einsum("nhc,hkc->nhk", q[..., :32], self.k_K1)
        s2 = torch.einsum("nhc,hkc->nhk", q[..., 32:], self.k_K2)
        v1, i1 = s1.topk(kk, -1)
        v2, i2 = s2.topk(kk, -1)
        sc = (v1[..., :, None] + v2[..., None, :]).reshape(N, self.H, -1)
        ii = (i1[..., :, None] * self.n + i2[..., None, :]).reshape(N, self.H, -1)
        vs, j = sc.topk(kk, -1)
        idx = ii.gather(-1, j)                                             # [N,H,kk]
        w = F.softmax(vs, -1)
        a = F.gelu(torch.einsum("nd,nhkd->nhk", x, self.u[idx]))
        return torch.einsum("nhk,nhkd->nd", w * a, self.v[idx]) + self.sh(x)


# ---------------- family E: multiplicative / tensor-product hidden ----------------
class Kron(C):
    """tensor-product hidden: h = vec(silu(Px) (x) b) with b = Qx (kron) or b = E[id] (kron_id);
    width r*r; + SwiGLU-96."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, r = mode, self.ca or 16
        self.r = r
        self.p, self.w2 = P(d, r), P(r * r, d)
        if mode == "kron":
            self.q = P(d, r)
        else:
            self.k_E = P(1024, r, std=1.0)
            self.register_buffer("tab", idtab(self.V, 1024, self.seed), persistent=False)
        self.ff = SG(d, 96)

    def f(self, x, ids, ids2):
        a = F.silu(x @ self.p)
        b = x @ self.q if self.mode == "kron" else self.k_E[self.tab[ids]]
        hh = (a[:, :, None] * b[:, None, :]).reshape(x.shape[0], -1)
        return hh @ self.w2 + self.ff(x)


# ---------------- round 2, exploit: ID-linear variants (SwiGLU + linear map looked up by token id) ----------------
class IdLin(C):
    """y = SwiGLU(x) + U[t]^T (W[t]^T x), t = row of a token-id table. ca = rows R, cb = rank r.
    Modes change the key (prev, pos, freqrows, two, bi, shared, emb), where the map acts (in, z3, gate),
    its form (act, diag), training (frozen), or the dense part (only, hmoe, dense96, l23)."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        R, r = self.ca or 128, int(self.cb) or 4
        h = cfg.dense_hidden
        if mode == "rank2id":
            R, r = self.ca or 256, 2
        if mode == "idlin_dense96":
            h = cfg.dense_hidden * 3 // 4
        if mode == "idlin_l23":
            R = 0 if layer == 0 else int(R * 1.5)
        self.R, self.r, self.h = R, r, h
        if mode == "idlin_hmoe":
            self.moe = HMoE(d, self.V, 8, cfg.expert_hidden, 2, self.seed + 5, "swiglu", cfg.n_layers)
            self.sh = SG(d, cfg.shared_hidden)
        elif mode != "idlin_only":
            self.ff = SG(d, h)
        if mode == "idlin_emb":                     # map weights generated from the token embedding
            self.k_gw, self.k_gu = P(d, d * r, std=0.1 / math.sqrt(d)), nn.Parameter(torch.zeros(d, r * d))
            return
        if R == 0:
            return
        V, sd = self.V, self.seed
        self.register_buffer("tab", idtab(V, R, sd), persistent=False)
        if mode in ("idlin_two", "idlin_bi"):
            self.register_buffer("tab2", idtab(V, R, sd + 1), persistent=False)
        if mode == "idlin_freqrows":                # frequent ids own rows; the rest share 64 hashed rows
            self.register_buffer("tab3", idtab(V, 64, sd + 2), persistent=False)
        if mode == "idlin_diag":
            self.k_s = nn.Parameter(torch.zeros(self.ca or 1024, d))
            self.register_buffer("tab", idtab(V, self.ca or 1024, sd), persistent=False)
            return
        dout = h if mode == "idlin_z3" else d
        if mode == "idlin_shared":
            key = "_idlin_shared"
            if key not in ctx:
                ctx[key] = (nn.Parameter(torch.zeros(R, r, d)), P(R, d, r, std=0.1))
            self.k_U, self.k_W = ctx[key]
            return
        self.k_U, self.k_W = nn.Parameter(torch.zeros(R, r, dout)), P(R, d, r, std=0.1)
        if mode == "idlin_frozen":
            self.k_U = nn.Parameter(torch.randn(R, r, dout) * 0.02, requires_grad=False)
            self.k_W.requires_grad_(False)
        if mode == "idlin_gate":
            self.k_g = nn.Parameter(torch.zeros(d))

    def rows(self, x, ids, ids2):
        m = self.mode
        if m == "idlin_prev":
            return self.tab[self.prev(ids2)]
        if m == "idlin_pos":
            B, T = ids2.shape
            return torch.arange(T, device=x.device).repeat(B) % self.R
        if m == "idlin_freqrows":
            fr = self.ctx["freq_rank"][ids]
            own = self.R - 64
            return torch.where(fr < own, fr, own + self.tab3[ids])
        return self.tab[ids]

    def lin(self, x, t, U=None, W=None):
        U = self.k_U if U is None else U
        W = self.k_W if W is None else W
        z = torch.einsum("nd,ndr->nr", x, W[t])
        if self.mode == "idlin_act":
            z = F.silu(z) * 2
        return torch.einsum("nr,nrd->nd", z, U[t])

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "idlin_hmoe":
            return self.moe(x, ids) + self.sh(x) + self.lin(x, self.tab[ids])
        if m == "idlin_emb":
            e = self.ctx["emb"].reshape(-1, self.d)
            W = (e @ self.k_gw).view(-1, self.d, self.r)
            U = (e @ self.k_gu).view(-1, self.r, self.d)
            return self.ff(x) + torch.einsum("nr,nrd->nd", torch.einsum("nd,ndr->nr", x, W), U)
        if self.R == 0:
            return self.ff(x)
        if m == "idlin_diag":
            return self.ff(x) + x * self.k_s[self.tab[ids]]
        if m == "idlin_z3":
            t = self.tab[ids]
            ff = self.ff
            z3 = x @ ff.w3 + self.lin(x, t)
            return (F.silu(x @ ff.w1) * z3) @ ff.w2
        t = self.rows(x, ids, ids2)
        if m == "idlin_only":
            return self.lin(x, t)
        if m == "idlin_in":
            return self.ff(x + self.lin(x, t))
        y = self.lin(x, t)
        if m == "idlin_two":
            y = y + self.lin(x, self.tab2[ids])
        if m == "idlin_bi":
            y = y + self.lin(x, self.tab2[self.prev(ids2)])
        if m == "idlin_gate":
            y = y * 2 * torch.sigmoid(x @ self.k_g)[:, None]
        return self.ff(x) + y


# ---------------- round 2, explore: new dense shapes, hash-MoE variants, merging ----------------
class Wide(C):
    """wide hidden at small parameter cost. lowrank_wide: W1, W3 factorized rank 32, width 512;
    sparse_wide: fixed random 25% masks on W1, W3, W2, width 512; monarch: block-diagonal W1, W3
    (4 blocks of 16 -> 128) after a d x d mix; chanmix: 4 group SwiGLUs (16 -> 256 -> 16) + d x d mix;
    randfeat: frozen random W1 width 1024 (relu), trained W2; embbilin: Wo((Wa emb) * (Wb x)) + SwiGLU."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        s = self.seed
        if mode == "lowrank_wide":
            self.a1, self.b1, self.a3, self.b3 = P(d, 32), P(32, 512), P(d, 32), P(32, 512)
            self.w2 = P(512, d)
        if mode == "sparse_wide":
            self.w1, self.w3, self.w2 = P(d, 512, std=0.04), P(d, 512, std=0.04), P(512, d, std=0.04)
            g = gen(s)
            for nm in ("m1", "m3"):
                self.register_buffer(nm, (torch.rand(d, 512, generator=g) < 0.25).float(), persistent=False)
            self.register_buffer("m2", (torch.rand(512, d, generator=g) < 0.25).float(), persistent=False)
        if mode == "monarch":
            self.mix = P(d, d)
            self.B1, self.B3 = P(4, d // 4, 128, std=0.04), P(4, d // 4, 128, std=0.04)
            self.w2 = P(512, d)
        if mode == "chanmix":
            g = d // 4
            self.mix = P(d, d)
            self.G1, self.G3, self.G2 = P(4, g, 256, std=0.04), P(4, g, 256, std=0.04), P(4, 256, g)
        if mode == "randfeat":
            self.register_buffer("w1", torch.randn(d, 1024, generator=gen(s)) / math.sqrt(d), persistent=False)
            self.register_buffer("b1", torch.randn(1024, generator=gen(s + 1)) * 0.5, persistent=False)
            self.w2 = P(1024, d)
            self.ff = SG(d, cfg.shared_hidden)
        if mode == "embbilin":
            self.wa, self.wb, self.wo = P(d, 256), P(d, 256), P(256, d)
            self.ff = SG(d, cfg.dense_hidden)

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "lowrank_wide":
            return (F.silu(x @ self.a1 @ self.b1) * (x @ self.a3 @ self.b3)) @ self.w2
        if m == "sparse_wide":
            return (F.silu(x @ (self.w1 * self.m1)) * (x @ (self.w3 * self.m3))) @ (self.w2 * self.m2)
        if m in ("monarch", "chanmix"):
            N = x.shape[0]
            xm = (x + x @ self.mix).view(N, 4, -1)
            if m == "monarch":
                hh = F.silu(torch.einsum("ngc,gch->ngh", xm, self.B1)) * torch.einsum("ngc,gch->ngh", xm, self.B3)
                return hh.reshape(N, -1) @ self.w2
            hh = F.silu(torch.einsum("ngc,gch->ngh", xm, self.G1)) * torch.einsum("ngc,gch->ngh", xm, self.G3)
            return torch.einsum("ngh,ghc->ngc", hh, self.G2).reshape(N, -1)
        if m == "randfeat":
            return F.relu(x @ self.w1 + self.b1) @ self.w2 + self.ff(x)
        if m == "embbilin":
            e = self.ctx["emb"].reshape(-1, self.d)
            return ((e @ self.wa) * (x @ self.wb)) @ self.wo + self.ff(x)
        raise ValueError(m)


class HashX(C):
    """hash-MoE variants (E16 w32 k3 + shared 32 unless noted). taylor / maxout / tiedglu / cosffn: expert
    function; freqbucket: experts keyed by frequency bucket (64 buckets by rank); embkey: by 6 sign bits of
    the token embedding; prodexp: shared * (1 + tanh(moe)) + moe; gatedres: (moe + shared) * 2 sigmoid(Wg x);
    resffn2: two half-width hash MoEs in sequence; skipwide: top ca% tokens by a learned score get a
    SwiGLU-256, all get a SwiGLU-32; statemap: SwiGLU-128 + rank-4 map keyed by a 7-bit LSH of x (control)."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        V, s, L = self.V, self.seed, cfg.n_layers
        eh = cfg.expert_hidden
        fn = {"taylor": "taylor", "maxout": "maxout", "tiedglu": "tied", "cosffn": "cos"}.get(mode, "swiglu")
        if mode == "tiedglu":
            eh = int(eh * 1.5)
        if mode == "resffn2":
            self.moe2 = HMoE(d, V, 16, eh // 2, 3, s + 9, "swiglu", L)
            eh = eh // 2
        if mode == "skipwide":
            self.big = SG(d, 256)
            self.k_ws = nn.Parameter(torch.zeros(d))
        elif mode == "statemap":
            self.ff = SG(d, cfg.dense_hidden)
            self.register_buffer("proj", torch.randn(7, d, generator=gen(s)), persistent=False)
            self.k_U, self.k_W = nn.Parameter(torch.zeros(128, 4, d)), P(128, d, 4, std=0.1)
        else:
            self.moe = HMoE(d, V, 16, eh, 3, s, fn, L)
        self.sh = SG(d, cfg.shared_hidden)
        if mode == "gatedres":
            self.k_g = nn.Parameter(torch.zeros(d, d))
        if mode == "embkey":
            self.register_buffer("proj", torch.randn(6, d, generator=gen(s + 3)), persistent=False)
            self.register_buffer("bsets", ksets(64, 16, 3, s + 4), persistent=False)
        if mode == "freqbucket":
            self.register_buffer("bsets", ksets(64, 16, 3, s + 4), persistent=False)

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "skipwide":
            sc = x @ self.k_ws
            c = max(1, int(x.shape[0] * (self.ca or 50) / 100))
            top = torch.topk(sc.detach(), c).indices
            z = self.big(x.index_select(0, top)) * (2 * torch.sigmoid(sc.index_select(0, top)))[:, None]
            return self.sh(x).index_add(0, top, z)
        if m == "statemap":
            code = (((x.detach() @ self.proj.t()) > 0).long() * (2 ** torch.arange(7, device=x.device))).sum(-1)
            z = torch.einsum("nd,ndr->nr", x, self.k_W[code])
            return self.ff(x) + torch.einsum("nr,nrd->nd", z, self.k_U[code])
        if m == "embkey":
            e = self.ctx["emb"].reshape(-1, self.d).detach()
            code = (((e @ self.proj.t()) > 0).long() * (2 ** torch.arange(6, device=x.device))).sum(-1)
            return self.moe(x, ids, self.bsets[code]) + self.sh(x)
        if m == "freqbucket":
            b = (self.ctx["freq_rank"][ids] * 64) // self.V
            return self.moe(x, ids, self.bsets[b]) + self.sh(x)
        y = self.moe(x, ids)
        if m == "prodexp":
            return self.sh(x) * (1 + torch.tanh(y)) + y
        if m == "gatedres":
            return (y + self.sh(x)) * 2 * torch.sigmoid(x @ self.k_g)
        if m == "resffn2":
            y1 = y + self.sh(x)
            return y1 + self.moe2(rms(x + y1), ids)
        return y + self.sh(x)


class IdMerge(C):
    """token-conditioned expert merging: 8 SwiGLU experts of width ca (default 32); a hashed id row gives
    softmax mixing weights; the experts' pre-activations are merged before the nonlinearity
    (z = sum a_e x W_e), so each token runs one merged expert. + shared 32."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        E, h = 8, self.ca or 32
        self.W1, self.W3, self.W2 = P(E, d, h), P(E, d, h), P(E, h, d)
        self.k_L = nn.Parameter(torch.zeros(512, E))
        self.register_buffer("tab", idtab(self.V, 512, self.seed), persistent=False)
        self.sh = SG(d, cfg.shared_hidden)

    def f(self, x, ids, ids2):
        a = F.softmax(self.k_L[self.tab[ids]], -1)                         # [N,E]
        z1 = torch.einsum("ne,nd,edh->nh", a, x, self.W1)
        z3 = torch.einsum("ne,nd,edh->nh", a, x, self.W3)
        hh = F.silu(z1) * z3
        return torch.einsum("ne,nh,ehd->nd", a, hh, self.W2) * 4 + self.sh(x)


class StemFull(C):
    """STEM-style reference (over budget): SwiGLU-128 whose W3 branch is replaced by a full per-id vector table."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.w1, self.w2 = P(d, 128), P(128, d)
        self.k_G = P(self.V, 128, std=0.1)

    def f(self, x, ids, ids2):
        return (F.silu(x @ self.w1) * self.k_G[ids]) @ self.w2


IDLIN = ("rank2id", "idlin_only", "idlin_hmoe", "idlin_in", "idlin_gate", "idlin_two", "idlin_freqrows",
         "idlin_prev", "idlin_bi", "idlin_l23", "idlin_shared", "idlin_emb", "idlin_act", "idlin_diag",
         "idlin_z3", "idlin_frozen", "idlin_pos", "idlin_dense96")
WIDE = ("lowrank_wide", "sparse_wide", "monarch", "chanmix", "randfeat", "embbilin")
HASHX = ("taylor", "maxout", "tiedglu", "cosffn", "freqbucket", "embkey", "prodexp", "gatedres", "resffn2",
         "skipwide", "statemap")


class Engram(nn.Module):
    """Engram-style memory (DeepSeek, arXiv 2601.07372), lab baseline for order O32: a hashed table keyed by the
    bigram (previous id, id), 2 hash heads of K rows x d/2 concatenated, context-aware gate
    a = sigmoid(<rms(h), rms(e Wk)> / sqrt(d)), value e Wv, then y = v~ + SiLU(causal depthwise conv4(rms(v~))),
    v~ = a * e Wv. Added to the block output. Parameters: K*d + 2d^2 + 4d. Shared by compute and mixing."""
    def __init__(self, d, K, seed, heads=2):
        super().__init__()
        self.d, self.K, self.H = d, K, heads
        self.k_tab = nn.Parameter(torch.randn(heads, K, d // heads, generator=gen(seed)) * 0.02)
        self.k_wk, self.k_wv = P(d, d), P(d, d)
        self.k_conv = nn.Parameter(torch.zeros(4, d))
        self.mul = [(1000003 + 7919 * h, 2654435761 % 1000033 + 104729 * h, 12345 + 31 * h) for h in range(heads)]

    def forward(self, x, ids2):
        B, T = ids2.shape
        cur = ids2.reshape(-1)
        pv = torch.cat([torch.zeros_like(ids2[:, :1]), ids2[:, :-1]], 1).reshape(-1)
        e = torch.cat([self.k_tab[h][(pv * a + cur * b + c) % self.K] for h, (a, b, c) in enumerate(self.mul)], -1)
        g = torch.sigmoid((rms(x) * rms(e @ self.k_wk)).sum(-1, keepdim=True) / math.sqrt(self.d))
        v = (g * (e @ self.k_wv)).view(B, T, -1)
        r = rms(v)
        c = sum(self.k_conv[j] * torch.cat([torch.zeros_like(r[:, :j]), r[:, :T - j]], 1) for j in range(4))
        return (v + F.silu(c)).reshape(B * T, -1)


class X7(C):
    """round 7. name = x7_<map>; body relu^2 width cb (default 192); map on sw form (z = x Ws, rank 4, y += z U[row]),
    one table of 2ca rows unless noted. k2b: 2 id rows + 2 previous-id rows; pair: id row + row by hash(prev, id);
    k2pair: 2 id rows + pair row; k2fbp: frequent-id row + hashed row + previous-id row + pair row; shk2: one table of
    n_layers*2ca rows shared by all layers, 2 rows per id, own Ws per layer; tri: id row + previous-id row + row by
    hash(prev2, prev, id); k2r8: 2 id rows at rank 8 (ca rows); prod: y += z (U1[h1] * (1 + U2[h2])), two tables of
    ca rows; mul: y += (x * v[id]) Wm, 8ca rows of d scales + shared d x d; hb: per-id hidden bias A[id] B (8ca rows,
    rank 16) inside relu^2, no output map; hbsw: hb + sw; hshiftsw: hidden h + a*h_prev (a per unit, init 0) + sw;
    xshiftsw: body input x + m*(x_prev - x) (m per channel, init 0) + sw; zprev: sw with mixing weights from the
    previous token's x; k2: control (swk2)."""
    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx)
        mp = name.split("_", 1)[1]
        self.mp = mp
        V, s = self.V, self.seed
        w = int(self.cb) or 192
        self.w1, self.w2 = P(d, w), P(w, d)
        ca = self.ca or 128
        r = 8 if mp == "k2r8" else 4
        R = ca if mp in ("k2r8", "prod") else 2 * ca
        self.R, self.r = R, r
        for i in range(1, 5):
            self.register_buffer(f"t{i}", idtab(V, R, s + 30 + i), persistent=False)
        if mp in ("hb", "hbsw"):
            Rh = 8 * ca
            self.register_buffer("th", idtab(V, Rh, s + 39), persistent=False)
            self.k_A, self.k_B = nn.Parameter(torch.zeros(Rh, 16)), P(16, w, std=0.1)
            if mp == "hb":
                return
        if mp == "hshiftsw":
            self.k_a = nn.Parameter(torch.zeros(w))
        if mp == "xshiftsw":
            self.k_m = nn.Parameter(torch.zeros(d))
        if mp == "mul":
            Rm = 8 * ca
            self.register_buffer("tm", idtab(V, Rm, s + 38), persistent=False)
            self.k_v, self.k_Wm = nn.Parameter(torch.zeros(Rm, d)), P(d, d)
            return
        self.k_Ws = P(d, r, std=0.1)
        if mp in ("shk2", "zbs"):
            key = "_x7_" + mp
            RS = cfg.n_layers * R
            if key not in ctx:
                ctx[key] = nn.Parameter(torch.zeros(RS, r, d))
            self.k_U = ctx[key]
            self.register_buffer("t1", idtab(V, RS, 7001), persistent=False)
            self.register_buffer("t2", idtab(V, RS, 7002), persistent=False)
            self.register_buffer("t3", idtab(V, RS, 7003), persistent=False)
            if mp == "zbs":
                self.k_Wp = P(d, r, std=0.1)
            return
        self.k_U = nn.Parameter(torch.zeros(R, r, d))
        if mp == "prod":
            self.k_U2 = nn.Parameter(torch.zeros(R, r, d))
        if mp in ("k2pairg", "k2gate"):
            self.k_g = P(d, 1, std=0.02)
        if mp in ("k2zp", "k2zprev2", "k2zpg", "k2zpb", "k2zpc", "k2zp2", "k3zp", "zponly"):
            self.k_Wp = P(d, r, std=0.1)
        if mp in ("k2eng", "k2engh"):   # O32: current-id maps + Engram in place of the previous-id map
            extra = (2 if mp == "k2eng" else 1) * ca * r * d          # = the U table (k2eng) or half of it (k2engh)
            self.eng = Engram(d, extra // d - 2 * d - 4, s + 50)
        if mp in ("k2zpbcopy", "copyonly", "k2zpbcnt", "zpcur", "k2zpbema", "emaonly", "k2zpbquad"):   # round 19
            self.k_Wp = P(d, r, std=0.1)
            if mp == "k2zpbcnt":
                self.k_c = nn.Parameter(torch.zeros(1))
            if mp in ("k2zpbema", "emaonly"):
                self.k_a4 = nn.Parameter(torch.zeros(4, w))
            if mp == "k2zpbquad":
                self.k_Uq = nn.Parameter(torch.zeros(R, r * (r + 1) // 2, d))
                self.register_buffer("iu", torch.triu_indices(r, r), persistent=False)
        if mp == "k2zpg":
            self.k_g = P(d, 1, std=0.02)
        if mp == "k2zph":
            self.k_Wp = P(w, r, std=0.02)
        R10 = ("zb2", "zbb", "zbf", "zbs", "zbpair", "zbt", "shiftffn", "zb2sf", "sf64", "sf2", "bodycat")
        if mp in R10:
            self.k_Wp = P(d, r, std=0.1)
        if mp in ("zbpair", "zbt"):
            self.k_g = P(d, 1, std=0.02)
        if mp in ("zb2", "zb2sf"):
            self.k_Wp2 = P(d, r, std=0.1)
        if mp in ("shiftffn", "zb2sf", "k2sf", "sf2"):
            self.k_s1, self.k_s2 = P(d, 32), P(32, d)
        if mp == "sf64":
            self.k_s1, self.k_s2 = P(d, 64), P(64, d)
        if mp == "sf2":
            self.k_s3, self.k_s4 = P(d, 32), P(32, d)
        if mp == "bodycat":
            self.w1, self.w2, self.k_w1p = P(d, int(self.cb) or 128), P(int(self.cb) or 128, d), P(d, int(self.cb) or 128)
        if mp == "k2zcat":
            self.k_Wp = P(d, r, std=0.1)
        if mp == "zh":
            self.k_Wh = P(w, r, std=0.02)
        if mp == "k2pairsep":
            self.k_Up = nn.Parameter(torch.zeros(R // 2, r, d))

    def shift(self, v, ids2):
        B, T = ids2.shape
        v = v.view(B, T, -1)
        return torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], 1).reshape(B * T, -1)

    def nextafter(self, ids2):
        """id that followed the last earlier occurrence of the current id in the sequence, and whether one exists"""
        B, T = ids2.shape
        m = (ids2[:, :, None] == ids2[:, None, :]) & torch.ones(T, T, dtype=torch.bool, device=ids2.device).tril(-1)
        pos = torch.arange(T, device=ids2.device)
        last = torch.where(m, pos.view(1, 1, T), torch.full_like(pos.view(1, 1, T), -1)).max(-1).values
        nxt = torch.gather(ids2, 1, (last + 1).clamp(0, T - 1))
        return nxt.reshape(-1), (last >= 0).reshape(-1)

    def ph(self, a, b):
        return (a * 1000003 + b * 7919 + 12345) % self.R

    def f(self, x, ids, ids2):
        mp = self.mp
        xb = x
        if mp == "xshiftsw":
            xb = x + self.k_m * (self.shift(x, ids2) - x)
        z1 = xb @ self.w1
        if mp == "bodycat":
            z1 = z1 + self.shift(x, ids2) @ self.k_w1p
        if mp in ("hb", "hbsw"):
            z1 = z1 + self.k_A[self.th[ids]] @ self.k_B
        h = F.relu(z1).square()
        if mp == "hshiftsw":
            h = h + self.k_a * self.shift(h, ids2)
        y = h @ self.w2
        if mp == "hb":
            return y
        if mp == "mul":
            return y + (x * self.k_v[self.tm[ids]]) @ self.k_Wm
        z = (self.shift(x, ids2) if mp == "zprev" else x) @ self.k_Ws
        if mp == "k2zcat":
            z = z + self.shift(x, ids2) @ self.k_Wp
        if mp == "zh":
            z = z + h @ self.k_Wh
        U = self.k_U
        E = lambda rows: torch.einsum("nr,nrd->nd", z, U[rows])
        pv = self.prev(ids2)
        t1, t2 = self.t1[ids], self.t2[ids]
        if mp in ("k2eng", "k2engh"):
            return y + E(t1) + E(t2) + self.eng(x, ids2)
        if mp == "zponly":   # round 18: previous-id map of the previous x only (k2zpb minus its two current-id maps)
            return y + torch.einsum("nr,nrd->nd", self.shift(x, ids2) @ self.k_Wp, U[self.t3[pv]])
        if mp in ("k2zpbcopy", "copyonly", "k2zpbcnt", "zpcur", "k2zpbema", "emaonly", "k2zpbquad"):   # round 19: new terms on k2zpb (or on the body alone for *only)
            B, T = ids2.shape
            if mp not in ("copyonly", "emaonly"):
                m = E(t1) + E(t2)
                if mp == "k2zpbcnt":   # current-id maps scaled by log(1 + earlier occurrences of the id)
                    eq = (ids2[:, :, None] == ids2[:, None, :]) & torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
                    m = m * (1 + self.k_c * torch.log1p(eq.sum(-1).float()).reshape(-1, 1))
                y = y + m
                xz = x if mp == "zpcur" else self.shift(x, ids2)   # zpcur: previous-id map of the current x
                y = y + torch.einsum("nr,nrd->nd", xz @ self.k_Wp, U[self.t3[pv]])
            if mp in ("k2zpbcopy", "copyonly"):   # map keyed by the id that followed this id last time
                nxt, has = self.nextafter(ids2)
                y = y + E(self.t4[nxt]) * has.reshape(-1, 1).to(y.dtype)
            if mp in ("k2zpbema", "emaonly"):   # hidden units read 4 causal EMAs of their own past values
                hv = h.view(B, T, -1)
                pos = torch.arange(T, device=x.device)
                lag = (pos[:, None] - 1 - pos[None, :]).clamp(min=0).float()
                mask = (pos[None, :] < pos[:, None]).float()
                e = 0
                for k, lam in enumerate((0.5, 0.8, 0.9, 0.97)):
                    e = e + self.k_a4[k] * torch.einsum("ts,bsw->btw", (1 - lam) * lam ** lag * mask, hv)
                y = y + e.reshape(B * T, -1) @ self.w2
            if mp == "k2zpbquad":   # per-id map of the 10 pairwise products of the rank-4 code
                q = (z[:, :, None] * z[:, None, :])[:, self.iu[0], self.iu[1]]
                y = y + torch.einsum("nq,nqd->nd", q, self.k_Uq[self.t4[ids]])
            return y
        if mp == "prod":
            return y + torch.einsum("nr,nrd->nd", z, self.k_U[t1] * (1 + self.k_U2[t2]))
        if mp in ("k2fbp", "zbf"):
            fr = self.ctx["freq_rank"][ids]
            own = self.R - 64
            t1 = torch.where(fr < own, fr, own + t2 % 64)
        y = y + E(t1)
        if mp in ("k2", "k2b", "k2pair", "shk2", "k2r8", "k2fbp", "k2pairg", "k2gate", "k2zp", "k2zprev2", "k2zcat",
                  "zh", "k2pairsep", "k2zpg", "k2zpb", "k2zpc", "k2zp2", "k3zp", "k2zph",
                  "zb2", "zbb", "zbf", "zbs", "zbpair", "zbt", "shiftffn", "zb2sf", "k2sf", "sf64", "sf2", "bodycat"):
            y = y + E(t2)
        if mp == "k2sf":
            return y + F.relu(self.shift(x, ids2) @ self.k_s1).square() @ self.k_s2
        if mp in ("zb2", "zbb", "zbf", "zbs", "zbpair", "zbt", "shiftffn", "zb2sf", "sf64", "sf2", "bodycat"):
            xp = self.shift(x, ids2)
            pb = torch.einsum("nr,nrd->nd", xp @ self.k_Wp, U[self.t3[pv]])
            if mp == "zbt":
                pb = torch.sigmoid(x @ self.k_g) * pb
            y = y + pb
            if mp in ("zb2", "zb2sf"):
                x2 = self.shift(xp, ids2)
                y = y + torch.einsum("nr,nrd->nd", x2 @ self.k_Wp2, U[self.t4[self.prev(ids2, 2)]])
            if mp == "zbb":
                y = y + torch.einsum("nr,nrd->nd", xp @ self.k_Wp, U[self.t4[pv]])
            if mp == "zbpair":
                y = y + torch.sigmoid(x @ self.k_g) * E(self.ph(pv, ids))
            if mp in ("shiftffn", "zb2sf", "sf64", "sf2"):
                y = y + F.relu(xp @ self.k_s1).square() @ self.k_s2
            if mp == "sf2":
                y = y + F.relu(self.shift(xp, ids2) @ self.k_s3).square() @ self.k_s4
            return y
        if mp == "k3zp":
            y = y + E(self.t4[ids])
        if mp in ("k2zpg", "k2zpb", "k2zpc", "k2zp2", "k3zp", "k2zph"):
            zp = self.shift(h, ids2) @ self.k_Wp if mp == "k2zph" else self.shift(x, ids2) @ self.k_Wp
            if mp == "k2zpc":
                zp = zp + z
            rows = self.t3[pv] if mp == "k2zpb" else self.ph(pv, ids)
            add = torch.einsum("nr,nrd->nd", zp, U[rows])
            if mp == "k2zp2":
                add = add + torch.einsum("nr,nrd->nd", zp, U[(pv * 7919 + ids * 1000003 + 777) % self.R])
            if mp == "k2zpg":
                add = torch.sigmoid(x @ self.k_g) * add
            return y + add
        if mp == "k2gate":
            m = E(t1) + E(t2)
            return (y - m) + torch.sigmoid(x @ self.k_g) * m
        if mp == "k2pairg":
            return y + torch.sigmoid(x @ self.k_g) * E(self.ph(pv, ids))
        if mp in ("k2zp", "k2zprev2"):
            zp = self.shift(x, ids2) @ self.k_Wp
            rows = self.ph(pv, ids) if mp == "k2zp" else t1
            return y + torch.einsum("nr,nrd->nd", zp, U[rows])
        if mp == "k2pairsep":
            return y + torch.einsum("nr,nrd->nd", z, self.k_Up[self.ph(pv, ids) % (self.R // 2)])
        if mp in ("k2b", "tri", "k2fbp"):
            y = y + E(self.t3[pv])
        if mp == "k2b":
            y = y + E(self.t4[pv])
        if mp in ("pair", "k2pair", "k2fbp"):
            y = y + E(self.ph(pv, ids))
        if mp == "tri":
            p2 = self.prev(ids2, 2)
            y = y + E((p2 * 2654435 + pv * 1000003 + ids * 7919 + 99) % self.R)
        return y


# ---------------- round 26: Engram bed (x7_k2eng) + one new mechanism per mode; no mode reads earlier token ids ----------------
R26_NEW = ("br_norm3", "br_route", "br_scal", "freq_mem", "freq_body", "freq_map", "id_trust", "id_bodygain", "id_film",
          "id_film_out", "map_rms", "sg_memx", "mem_drop", "map_drop", "mem_warm", "map_noise", "pos_mem", "map_top2",
          "map_soft", "map_relu", "map_sq", "mem_cap", "body_cap", "comp_gate", "agree_gate", "map2body", "hid_norm",
          "map_lr3", "map_pos", "id_halves", "mem_sq", "h_noise", "map_rinit", "id_readout", "hgate_mem", "hgate_map",
          "x_center", "h_center")
R26_VAR = ("base", "peri", "peri_nomem", "tok_tab", "tok_tab2", "kv_out32", "xh_bil", "map_lr03")
R26_MODES = R26_VAR + R26_NEW


R27_NEW = ("z_temp", "zm2body", "pos_hid", "bn_hid", "x_drop_ch", "h_l2_cap", "h_bcenter", "h_dead_revive", "w2_lr2", "w1_lr2",
           "body_lr05", "body2map", "width_ramp", "width_rev", "mem_alloc_early", "maps_ramp", "body_sign_split", "hid_hash_mix",
           "w2_id_sign", "hid_rot_pos", "x_groupnorm", "peri_group", "unit_lr_spread", "h_osc", "freq_width", "w_noise",
           "zm_hgate", "norm_skip", "pos_width", "id_thresh", "h_pow")
R27_VAR = ("w2_zero", "layer_drop", "iter2", "body_res2", "z_rope", "w1_binary", "peri_gain_id", "w2_wnorm")
R27_SOLO = ("peri_group", "peri_gain_id")          # replace Peri-LN, run without the P_ prefix
R28_NEW = ("g_freq_up", "g_freq_down", "g_tok_norm", "g_tok_clip", "g_pos_ramp", "g_mem_x2", "g_body_x2", "g_maps_x2",
           "g_hid_noise", "g_sign_mix", "r_l1h", "r_decor", "r_out_l2", "r_mem_l2", "r_usage", "r_noise_cons", "r_hsparse",
           "r_memorth", "r_w_orth", "t_ema_h", "t_ema_y", "t_cummean_x", "t_seq_gain", "t_fastw", "t_seq_norm",
           "v_width_grow", "v_drop_decay", "v_mem_freeze", "v_body_freeze", "g_ortho_mem")
R28_VAR = ("u_h_prev", "u_mem_prev", "u_act_prev", "mem_alloc_late", "h_osc_w1", "sign_q", "h_dead_revive10", "pos_hid_lin",
           "body_res64", "h_l2_cap_fix", "w_ema_eval")
PIVA_MODES = ("al05", "al25", "al50", "al70", "al95", "al90", "al98", "al83", "al00")   # pivot A allocation sweep; x9_peri is the
# f=0.83 point; al83 = wrapper null (default split rebuilt through this path); al00 = ablation (no memory, whole budget to width)


PIVA_KEY = ("key1", "key3")
PIVA_SK = ("al95k1", "al95k3", "al50k1", "al50k3")   # pA9: split x memory key   # pA8: the default Engram memory keyed on the current id alone (key1) or on 3 ids (key3)


class EngramK(Engram):                             # pA8: same table, gate, value and conv as Engram; only the hash key changes
    def __init__(self, d, K, seed, n):
        super().__init__(d, K, seed)
        self.n = n

    def forward(self, x, ids2):
        B, T = ids2.shape
        sh = lambda k: torch.cat([torch.zeros_like(ids2[:, :k]), ids2[:, :T - k]], 1).reshape(-1)
        cur = ids2.reshape(-1)
        pv = sh(1) if self.n >= 2 else torch.zeros_like(cur)
        p2 = sh(2) if self.n >= 3 else torch.zeros_like(cur)
        e = torch.cat([self.k_tab[h][(pv * a + cur * b + p2 * (7 + 2 * h) * 999983 + c) % self.K]
                       for h, (a, b, c) in enumerate(self.mul)], -1)
        g = torch.sigmoid((rms(x) * rms(e @ self.k_wk)).sum(-1, keepdim=True) / math.sqrt(self.d))
        v = (g * (e @ self.k_wv)).view(B, T, -1)
        r = rms(v)
        c = sum(self.k_conv[j] * torch.cat([torch.zeros_like(r[:, :j]), r[:, :T - j]], 1) for j in range(4))
        return (v + F.silu(c)).reshape(B * T, -1)


class NoMem(nn.Module):                            # pivot A ablation: memory branch returns zeros
    def forward(self, x, ids2):
        return torch.zeros_like(x)


DEEP_MODES = ("zp", "zpcur", "zpnull", "meki", "l3", "s_zp", "s_zpcur", "s_meki", "s_l3")
# deep arm (11-23): zp = x7_k2zpb's previous-id map (shared U table) on Engram; zpcur = same map keyed by the current id
# (ablation); zpnull = same parameters, map output x 0 (wrapper null); meki / l3 = MeKi-style (2602.03359) / L3-style
# (2601.21461) memory in place of Engram, same parameters; s_* = Engram at half its rows + the add-on at the other half.


class MeKiMem(nn.Module):   # MeKi-style: per-current-id vector (dm) + sigmoid gate of the hidden state, out = gain*rms(v Wout)
    def __init__(self, d, P_budget, seed, dm=32):
        super().__init__()
        self.R = max(16, int((P_budget - 2 * dm * d - d - 1) // dm))
        self.register_buffer("tid", idtab(8192, self.R, seed), persistent=False)
        self.k_M = nn.Parameter(torch.randn(self.R, dm, generator=gen(seed + 1)) * 0.02)
        self.k_a = nn.Parameter(torch.ones(1))
        self.k_wg, self.k_wo = P(d, dm), P(dm, d)
        self.k_g = nn.Parameter(torch.full((d,), 0.1))

    def forward(self, x, ids2):
        cur = ids2.reshape(-1)
        v = self.k_a * rms(self.k_M[self.tid[cur % self.tid.shape[0]]]) + torch.sigmoid(rms(x) @ self.k_wg)
        return self.k_g * rms(v @ self.k_wo)


class L3Mem(nn.Module):     # L3-style: k embeddings per current-id bucket (keys tied to values), softmax by the hidden
    def __init__(self, d, P_budget, seed, k=4):   # state, out = W_mix [LN(u); x]
        super().__init__()
        self.k, self.d = k, d
        self.R = max(16, int((P_budget - 2 * d * d - d) // (k * d)))
        self.register_buffer("tid", idtab(8192, self.R, seed), persistent=False)
        self.k_E = nn.Parameter(torch.randn(self.R, k, d, generator=gen(seed + 1)) * 0.02)
        self.k_mix = P(2 * d, d)
        self.k_g = nn.Parameter(torch.ones(d))

    def forward(self, x, ids2):
        E = self.k_E[self.tid[ids2.reshape(-1) % self.tid.shape[0]]]                 # n, k, d
        a = torch.softmax(torch.einsum("nkd,nd->nk", E, rms(x)) / math.sqrt(self.d), -1)
        u = self.k_g * rms(torch.einsum("nk,nkd->nd", a, E))
        return torch.cat([u, x], -1) @ self.k_mix


class StackMem(nn.Module):  # Engram + add-on, both read the same x, outputs summed
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b, self.K = a, b, a.K

    def forward(self, x, ids2):
        return self.a(x, ids2) + self.b(x, ids2)


class AddAux(torch.autograd.Function):              # returns y unchanged; backward adds lam * d(pen) as an auxiliary loss
    @staticmethod
    def forward(c, y, pen, lam):
        c.lam, c.sh = lam, pen.shape
        return y.view_as(y)

    @staticmethod
    def backward(c, g):
        return g, g.new_full(c.sh, c.lam), None


class TokGrad(torch.autograd.Function):             # forward identity; backward rescales each token's gradient
    @staticmethod
    def forward(c, y, mode):
        c.mode = mode
        return y.view_as(y)

    @staticmethod
    def backward(c, g):
        nr = g.norm(dim=-1, keepdim=True)
        mu = nr.mean()
        if c.mode == "norm": return g * mu / (nr + 1e-12), None
        return g * torch.clamp(2 * mu / (nr + 1e-12), max=1.0), None


def gscale(v, sc):                                    # value v, gradient scaled by sc (tensor or float)
    return v * sc + (v * (1 - sc)).detach()


def cmean(v, B, T):                                   # causal running mean over positions
    v3 = v.view(B, T, -1)
    return (v3.cumsum(1) / torch.arange(1, T + 1, device=v.device, dtype=v.dtype)[None, :, None]).view(B * T, -1)


def cema(v, a, B, T):                                 # causal EMA with per-channel decay a in (0, 1)
    v3, out, st = v.view(B, T, -1), [], torch.zeros_like(v.view(B, T, -1)[:, 0])
    for t in range(T):
        st = a * st + (1 - a) * v3[:, t]
        out.append(st)
    return torch.stack(out, 1).view(B * T, -1)


def rope(v, pos):                                     # rotate channel pairs (i, i + n/2) by angle pos * 10000^(-i/(n/2))
    hn = v.shape[1] // 2
    th = pos.float()[:, None] * torch.pow(10000.0, -torch.arange(hn, device=v.device).float() / hn)[None]
    a, b, c, s = v[:, :hn], v[:, hn:2 * hn], torch.cos(th).to(v.dtype), torch.sin(th).to(v.dtype)
    return torch.cat([a * c - b * s, a * s + b * c, v[:, 2 * hn:]], 1)


class X9(X7):
    """rounds 26-27. name = x9_<mode> or x9_P_<mode>: the Engram baseline x7_k2eng (relu^2 body yb + 2 current-id maps ym +
    Engram memory ye, output yb + ym + ye) with one change; P_ puts Peri-LN (rms + gain) on the output. x9_base = x7_k2eng,
    x9_peri = Engram + Peri-LN. No mode reads earlier token ids."""
    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx, "x7_k2eng")
        m = name.split("_", 1)[1]
        self.pp = m.startswith("P_")
        m = self.m9 = m[2:] if self.pp else m
        assert m in R26_MODES + R27_NEW + R27_VAR + R28_NEW + R28_VAR + PIVA_MODES + PIVA_KEY + PIVA_SK + DEEP_MODES, m
        L, s = cfg.n_layers, self.seed
        fl = layer / max(L - 1, 1)                     # 0 at the first layer, 1 at the last
        w, R, r, V = self.w1.shape[1], self.R, self.r, self.V
        z = lambda *sh: nn.Parameter(torch.zeros(*sh))
        if m in ("width_ramp", "width_rev"):          # same total width, 2/3x .. 4/3x across layers
            w = int(round(w * ((2 + 2 * (fl if m == "width_ramp" else 1 - fl)) / 3)))
            self.w1, self.w2 = P(d, w), P(w, d)
        if m in ("body_res2", "body_res64"):
            rw = 32 if m == "body_res2" else 64
            self.w1, self.w2, self.k_r1, self.k_r2 = P(d, w - rw), P(w - rw, d), P(d, rw), P(rw, d)
        if m == "body_sign_split": self.w1, self.k_w1b = P(d, w // 2), P(d, w - w // 2)
        if m == "sign_q": self.w1, self.k_w1b = P(d, w // 4), P(d, w - w // 4)
        if m == "mem_alloc_early":                    # memory rows 1.5x .. 0.5x across layers
            K = self.eng.K
            self.eng = Engram(d, int(K * (1.5 - fl)), s + 50)
        if m == "mem_alloc_late": self.eng = Engram(d, int(self.eng.K * (0.5 + fl)), s + 50)
        if m in PIVA_MODES:                            # pivot A: fixed body + memory budget 2w + K (units of d), memory share f
            f = int(m[2:]) / 100
            tot = 2 * w + self.eng.K
            K = int(round(f * tot))
            w = int(round((tot - K) / 2)) if K else int(round((tot + 2 * d + 4) / 2))   # al00: Wk, Wv, conv params go to width too
            self.w1, self.w2 = P(d, w), P(w, d)
            self.eng = Engram(d, K, s + 50) if K else NoMem()
        if m in PIVA_KEY: self.eng = EngramK(d, self.eng.K, s + 50, 1 if m == "key1" else 3)
        if m in PIVA_SK:                               # pA9: split f = al<NN> with the key1 / key3 memory
            f = int(m[2:4]) / 100
            tot = 2 * w + self.eng.K
            K = int(round(f * tot))
            w = int(round((tot - K) / 2))
            self.w1, self.w2 = P(d, w), P(w, d)
            self.eng = EngramK(d, K, s + 50, int(m[5]))
        if m in DEEP_MODES:                            # deep arm: same total parameters as x9_peri
            K0, sd = self.eng.K, s + 50
            if m in ("zp", "zpcur", "zpnull"):
                self.eng = Engram(d, K0 - (d * r + d - 1) // d, sd)
                self.k_Wp = P(d, r, std=0.1)
            elif m == "meki": self.eng = MeKiMem(d, K0 * d + 2 * d * d + 4 * d, sd)
            elif m == "l3": self.eng = L3Mem(d, K0 * d + 2 * d * d + 4 * d, sd)
            else:
                Ka = K0 // 2
                A = (K0 - Ka) * d
                if m in ("s_zp", "s_zpcur"):
                    R3 = (A - d * r) // (r * d)
                    self.k_Wp, self.k_U3 = P(d, r, std=0.1), z(R3, r, d)
                    self.register_buffer("t5", idtab(V, R3, s + 35), persistent=False)
                    self.eng = Engram(d, Ka, sd)
                else:
                    self.eng = StackMem(Engram(d, Ka, sd), (MeKiMem if m == "s_meki" else L3Mem)(d, A, sd + 7))
        if m == "maps_ramp":                          # id-map rows 0.5x .. 1.5x across layers
            R = self.R = int(R * (0.5 + fl))
            for i in (1, 2):
                self.register_buffer(f"t{i}", idtab(V, R, s + 30 + i), persistent=False)
            self.k_U = z(R, r, self.d)
        w = self.w1.shape[1] + (self.k_w1b.shape[1] if m in ("body_sign_split", "sign_q") else 0)
        if self.pp or m in ("peri", "peri_nomem", "peri_group", "peri_gain_id"): self.k_pg = nn.Parameter(torch.full((d,), 0.1))
        if m == "peri_gain_id": self.k_tt = z(509, d)
        # ---- round 26 parameters
        if m == "br_norm3": self.k_bg = nn.Parameter(torch.full((3, d), 0.1))
        if m == "br_route": self.k_br = z(d, 3)
        if m == "br_scal": self.k_bs = nn.Parameter(torch.ones(3))
        if m.startswith("freq_") and m != "freq_width" or m == "pos_mem": self.k_fa = z(2)
        if m in ("id_trust", "id_bodygain", "id_halves"): self.k_it = z(R)
        if m == "id_film": self.k_fA, self.k_fB = z(R, 4), P(4, w, std=0.1)
        if m == "id_film_out": self.k_fA, self.k_fB = z(R, 4), P(4, d, std=0.1)
        if m == "map_rms": self.k_mg = nn.Parameter(torch.full((d,), 0.05))
        if m == "mem_warm": self.register_buffer("n_step", torch.zeros(()), persistent=False)
        if m in ("mem_cap", "body_cap"): self.k_cap = nn.Parameter(torch.full((1,), 2.0))
        if m == "comp_gate": self.k_cg = z(d, 1)
        if m == "agree_gate": self.k_ag = z(1)
        if m == "map2body": self.k_mb = z(1)
        if m == "map_pos": self.k_mp = z(16, r)
        if m == "mem_sq": self.k_ms = z(d)
        if m == "hid_norm": self.k_hn = nn.Parameter(torch.full((1,), 0.03))
        if m == "map_rinit": self.k_U.data.normal_(0, 0.02)
        if m == "id_readout": self.k_rA, self.k_rB, self.k_rc = z(R, 4), P(4, w, std=0.1), P(d)
        if m in ("hgate_mem", "hgate_map"): self.k_hg = z(w)
        if m in ("tok_tab", "tok_tab2"): self.k_tt = z(509, d)
        if m == "tok_tab2": self.k_tt2 = z(1021, d)
        if m == "kv_out32": self.k_kok, self.k_kov = P(32, d, std=1 / math.sqrt(d)), z(32, d)
        if m == "xh_bil": self.k_hU, self.k_hV, self.k_hR = P(w, 8, std=1 / math.sqrt(w)), P(d, 8, std=1 / math.sqrt(d)), z(8, d)
        # ---- round 27 parameters
        if m == "z_temp": self.k_zt = z(d)
        if m == "zm2body": self.k_zb = z(r, d)
        if m == "bn_hid":
            self.k_bnw, self.k_bnb = nn.Parameter(torch.full((w,), 0.15)), z(w)
            self.register_buffer("bn_m", torch.zeros(w), persistent=False)
            self.register_buffer("bn_v", torch.ones(w), persistent=False)
        if m == "h_l2_cap": self.k_lc = z(1)
        self.lay9 = layer
        if m in ("h_dead_revive", "h_dead_revive10"): self.register_buffer("dr_f", torch.full((w,), 0.5), persistent=False)
        if m in ("h_osc", "h_osc_w1"): self.k_om = z(w) if m == "h_osc" else nn.Parameter(torch.ones(w))
        if m in ("pos_hid", "pos_hid_lin"): self.k_ph = z(16, w)
        if m in ("t_ema_h", "t_ema_y"):
            nc = w if m == "t_ema_h" else d
            self.k_ea, self.k_eg = nn.Parameter(torch.full((nc,), 2.0)), z(nc)
        if m in ("t_cummean_x", "t_seq_gain", "t_seq_norm", "u_mem_prev"): self.k_cg = z(d)
        if m in ("u_h_prev", "u_act_prev"): self.k_up = z(w)
        if m == "t_fastw": self.k_fq, self.k_fk, self.k_fv, self.k_fo = P(d, 16, std=d ** -0.5), P(d, 16, std=d ** -0.5), P(d, 16, std=d ** -0.5), z(16, d)
        if m.startswith("v_") or m == "w_ema_eval": self.register_buffer("n_step9", torch.zeros(()), persistent=False)
        if m == "w_ema_eval":
            self.register_buffer("e_w1", self.w1.data.clone(), persistent=False)
            self.register_buffer("e_w2", self.w2.data.clone(), persistent=False)
        if m == "h_bcenter": self.register_buffer("hb_m", torch.zeros(w), persistent=False)
        c1 = {"w1_lr2": 2.0, "body_lr05": 0.5}.get(m, 1.0)
        c2 = {"w2_lr2": 2.0, "body_lr05": 0.5}.get(m, 1.0)
        if m == "unit_lr_spread":
            c = torch.exp(torch.linspace(math.log(0.3), math.log(3.0), w))[torch.randperm(w, generator=gen(s + 61))]
            self.register_buffer("ulr", c, persistent=False)
            self.w1.data /= c[None]
            self.w2.data /= c[:, None]
        self.c1, self.c2 = c1, c2
        self.w1.data /= c1
        self.w2.data /= c2
        if m == "hid_hash_mix":
            self.register_buffer("pm", torch.stack([torch.randperm(w, generator=gen(s + 70 + i)) for i in range(4)]), persistent=False)
            self.register_buffer("pid", idtab(V, 4, s + 75), persistent=False)
        if m == "w2_id_sign":
            self.register_buffer("sg", (torch.randint(0, 2, (8, w), generator=gen(s + 80)) * 2 - 1).float(), persistent=False)
            self.register_buffer("sgid", idtab(V, 8, s + 81), persistent=False)
        if m == "id_thresh": self.k_ith = z(509)
        if m == "h_pow": self.k_hp = z(w)
        if m == "zm_hgate": self.k_zg = z(r, w)
        if m == "norm_skip": self.k_ns = z(d)
        if m == "w2_zero": self.w2.data.zero_()
        if m == "w2_wnorm": self.k_wn = nn.Parameter(self.w2.data.norm(dim=1).clone())

    def s2(self, u):                                  # learned gain 2*sigmoid(a + b*(4u-2)), u in [0,1], 1 at init
        return 2 * torch.sigmoid(self.k_fa[0] + self.k_fa[1] * (4 * u - 2))[:, None]

    def body(self, xb, x, ids, t1, zm, pos):
        m, tr = self.m9, self.training
        w1, w2 = self.w1 * self.c1, self.w2 * self.c2
        if m == "unit_lr_spread": w1, w2 = w1 * self.ulr[None], w2 * self.ulr[:, None]
        if m == "w_noise" and tr: w1 = w1 * (1 + 0.05 * torch.randn_like(w1))
        if m == "w1_binary": w1 = w1 + (torch.sign(w1) * w1.abs().mean() - w1).detach()
        if m == "w2_wnorm": w2 = F.normalize(w2, dim=1) * self.k_wn[:, None]
        if m == "w_ema_eval":                         # eval with an EMA (0.95 per step) of the body weights
            if tr:
                with torch.no_grad():
                    self.e_w1.lerp_(self.w1.data, 0.05)
                    self.e_w2.lerp_(self.w2.data, 0.05)
            else:
                w1, w2 = self.e_w1 * self.c1, self.e_w2 * self.c2
        if m == "v_body_freeze" and float(self.n_step9) > 200: w1, w2 = w1.detach(), w2.detach()
        if m == "r_w_orth" and tr and not self._inner:
            wn = F.normalize(w1, dim=0)
            G = wn.t() @ wn
            self._pen.append((((G - torch.eye(G.shape[0], device=G.device)) ** 2).mean(), 0.1))
        if m == "t_cummean_x": xb = xb + self.k_cg * cmean(x, *self._bt)
        elif m == "t_seq_norm": xb = xb * (1 + self.k_cg * (torch.rsqrt(cmean(x * x, *self._bt) + 1e-6) - 1))
        if m == "x_groupnorm": xb = rms(xb.view(xb.shape[0], 8, -1)).view(xb.shape[0], -1)
        if m == "z_rope": xb = rope(xb, pos)
        if m == "x_drop_ch": xb = F.dropout(xb, 0.1, tr)
        z1 = torch.cat([xb.abs() @ w1, xb @ self.k_w1b], 1) if m in ("body_sign_split", "sign_q") else xb @ w1
        if m == "z_temp": z1 = z1 * torch.exp(x @ self.k_zt)[:, None]
        elif m == "bn_hid":
            if tr:
                mu, va = z1.mean(0), z1.var(0, unbiased=False)
                with torch.no_grad():
                    self.bn_m.lerp_(mu.detach(), 0.05)
                    self.bn_v.lerp_(va.detach(), 0.05)
            else:
                mu, va = self.bn_m, self.bn_v
            z1 = (z1 - mu) * torch.rsqrt(va + 1e-5) * self.k_bnw + self.k_bnb
        elif m in ("h_dead_revive", "h_dead_revive10"):
            z1 = z1 + (0.5 * ((0.2 if m == "h_dead_revive" else 0.1) - self.dr_f).clamp(min=0)).to(z1.dtype)
        elif m == "u_h_prev" and self.lay9 and not self._inner: z1 = z1 + self.k_up * self.ctx["x9h"]
        elif m == "id_thresh": z1 = z1 - self.k_ith[ids % 509][:, None]
        h = F.relu(z1).square()
        if m == "h_pow": h = h * torch.exp(self.k_hp.clamp(-1, 1) * torch.log(F.relu(z1) + 1e-6))
        if m in ("h_dead_revive", "h_dead_revive10") and tr:
            with torch.no_grad(): self.dr_f.lerp_((h > 0).float().mean(0), 0.05)
        B, T = self._bt
        ns = float(self.n_step9) if hasattr(self, "n_step9") else 0.0
        if m == "pos_hid_lin": h = h * (1 + self.k_ph[(pos // 4).clamp(max=15)])
        elif m == "h_l2_cap_fix":
            nh = h.norm(dim=-1, keepdim=True)
            h = h * 2 / (1 + nh / (nh.mean().detach() + 1e-6))
        elif m == "t_ema_h": h = h + self.k_eg * cema(h, torch.sigmoid(self.k_ea), B, T)
        elif m == "u_act_prev" and self.lay9 and not self._inner: h = h * 2 * torch.sigmoid(self.k_up * (self.ctx["x9a"] - 0.5))
        elif m == "v_width_grow":
            keep = torch.arange(h.shape[1], device=h.device) < h.shape[1] * min(1.0, 0.5 + 0.5 * ns / 150)
            h = h * keep.to(h.dtype)[None]
        elif m == "v_drop_decay" and tr: h = F.dropout(h, max(0.0, 0.2 * (1 - ns / 300)), True)
        if tr and h.requires_grad and not self._inner:
            if m == "g_hid_noise": h.register_hook(lambda g: g * (1 + 0.1 * torch.randn_like(g)))
            elif m == "g_sign_mix": h.register_hook(lambda g: 0.5 * g + 0.5 * torch.sign(g) * g.abs().mean())
            elif m == "r_l1h": self._pen.append((h.abs().mean() / (h.detach().abs().mean() + 1e-6), 0.01))
            elif m == "r_decor":
                hn = (h - h.mean(0)) / (h.std(0) + 1e-4)
                C = hn.t() @ hn / hn.shape[0]
                self._pen.append((((C - torch.diag(torch.diag(C))) ** 2).mean(), 0.1))
            elif m == "r_usage":
                pu = h.mean(0) / (h.mean().detach() + 1e-6)
                self._pen.append((((pu - 1) ** 2).mean(), 0.01))
            elif m == "r_hsparse": self._pen.append((((torch.sigmoid(10 * z1).mean(0) - 0.1) ** 2).mean(), 0.1))
        if m == "u_h_prev" and not self._inner: self.ctx["x9h"] = h
        if m == "u_act_prev" and not self._inner: self.ctx["x9a"] = (h > 0).float().mean(0)
        if m == "id_film": h = h * (1 + self.k_fA[t1] @ self.k_fB)
        elif m == "id_halves":
            g = 2 * torch.sigmoid(self.k_it[t1])[:, None]
            hw = h.shape[1] // 2
            h = torch.cat([h[:, :hw] * g, h[:, hw:] * (2 - g)], 1)
        elif m == "hid_norm": h = rms(h) * self.k_hn
        elif m == "h_center": h = h - h.mean(-1, keepdim=True)
        elif m == "h_noise" and tr: h = h * (1 + 0.1 * torch.randn_like(h))
        if m == "pos_hid": h = h * (1 + self.k_ph[(torch.log2(pos.float() + 1) * 2).long().clamp(max=15)])
        elif m == "h_l2_cap":
            nh = h.norm(dim=-1, keepdim=True)
            a = F.softplus(self.k_lc)
            h = h * (1 + a) / (1 + a * nh / (nh.mean().detach() + 1e-6))
        elif m == "h_bcenter":                        # subtract a detached running per-unit mean (same in train and eval)
            if tr:
                with torch.no_grad(): self.hb_m.lerp_(h.detach().mean(0), 0.05)
            h = h - self.hb_m
        elif m == "hid_hash_mix": h = h.gather(1, self.pm[self.pid[ids]])
        elif m == "w2_id_sign": h = h * self.sg[self.sgid[ids]]
        elif m == "hid_rot_pos": h = rope(h, pos)
        elif m in ("h_osc", "h_osc_w1"): h = h * torch.cos(self.k_om * z1)
        elif m == "freq_width":                       # the 512 most frequent ids use only the first half of the units
            keep = (self.ctx["freq_rank"][ids] >= 512)[:, None] | (torch.arange(h.shape[1], device=h.device) < h.shape[1] // 2)[None]
            h = h * keep.to(h.dtype)
        elif m == "pos_width":                        # active units grow from 1/2 at position 0 to all at the last position
            T = int(pos.max()) + 1
            keep = torch.arange(h.shape[1], device=h.device)[None] < (h.shape[1] * (0.5 + 0.5 * pos.float() / max(T - 1, 1)))[:, None]
            h = h * keep.to(h.dtype)
        elif m == "zm_hgate": h = h * (1 + torch.tanh(zm @ self.k_zg))
        yb = h @ w2
        if m == "xh_bil": yb = yb + ((h @ self.k_hU) * (x @ self.k_hV)) @ self.k_hR
        elif m == "id_readout": yb = yb + (h * (self.k_rA[t1] @ self.k_rB)).sum(-1, keepdim=True) / math.sqrt(h.shape[1]) * self.k_rc
        elif m == "kv_out32": yb = yb + torch.softmax(rms(yb) @ self.k_kok.t(), -1) @ self.k_kov
        elif m in ("body_res2", "body_res64"): yb = yb + F.relu((x + yb) @ self.k_r1).square() @ self.k_r2
        return yb, h

    def f(self, x, ids, ids2):
        m = self.m9
        B, T = ids2.shape
        n = x.shape[0]
        tr = self.training
        pos = torch.arange(T, device=x.device).repeat(B)
        drop = lambda v: v * (torch.rand(n, 1, device=x.device) > 0.1).to(v.dtype) / 0.9 if tr else v
        self._bt, self._pen, self._inner = (B, T), [], False
        if hasattr(self, "n_step9") and tr: self.n_step9 += 1
        U = self.k_U * (3.0 if m == "map_lr3" else 0.3 if m == "map_lr03" else 1.0)
        t1, t2 = self.t1[ids], self.t2[ids]

        def maps(xin):
            zm = xin @ self.k_Ws
            if m == "map_soft": zm = 2 * torch.softmax(zm, -1)
            elif m == "map_relu": zm = 1.4 * F.relu(zm)
            elif m == "map_sq": zm = zm * zm.abs()
            elif m == "map_top2": zm = zm * (zm.abs() >= zm.abs().topk(2, -1).values[:, -1:]).to(zm.dtype)
            elif m == "map_pos": zm = zm + self.k_mp[(torch.log2(pos.float() + 1) * 2).long().clamp(max=15)]
            return zm, torch.einsum("nr,nrd->nd", zm, U[t1]) + torch.einsum("nr,nrd->nd", zm, U[t2])

        zm, ym = maps(x)
        xb = x - x.mean(-1, keepdim=True) if m == "x_center" else x
        if m == "map2body": xb = xb + self.k_mb * ym
        elif m == "zm2body": xb = xb + zm @ self.k_zb
        yb, h = self.body(xb, x, ids, t1, zm, pos)
        if m == "iter2": yb, h = self.body(x + yb, x, ids, t1, zm, pos)
        if m == "body2map": zm, ym = maps(x + yb)
        if m == "v_mem_freeze" and tr and float(self.n_step9) > 200:
            with torch.no_grad(): ye = self.eng(x, ids2)
        else:
            ye = self.eng(x.detach() if m == "sg_memx" else x, ids2)
        if m in ("zp", "zpcur", "zpnull", "s_zp", "s_zpcur"):   # previous-id map of the previous x, added to the map branch
            pkey = self.prev(ids2) if m in ("zp", "zpnull", "s_zp") else ids
            Ut, tt = (U, self.t3) if m in ("zp", "zpcur", "zpnull") else (self.k_U3, self.t5)
            yz = torch.einsum("nr,nrd->nd", self.shift(x, ids2) @ self.k_Wp, Ut[tt[pkey]])
            ym = ym + (0.0 * yz if m == "zpnull" else yz)
        if m == "u_mem_prev":
            ye0 = ye
            if self.lay9: ye = ye + self.k_cg * self.ctx["x9m"]
            self.ctx["x9m"] = ye0
        if tr:
            if m == "g_mem_x2": ye = gscale(ye, 2.0)
            elif m == "g_body_x2": yb = gscale(yb, 2.0)
            elif m == "g_maps_x2": ym = gscale(ym, 2.0)
            elif m == "g_ortho_mem" and ye.requires_grad:   # memory gradient minus its part along the body output
                ybn = F.normalize(yb.detach(), dim=-1)
                ye.register_hook(lambda g: g - (g * ybn).sum(-1, keepdim=True) * ybn)
            elif m == "r_mem_l2": self._pen.append((ye.pow(2).mean() / (ye.detach().pow(2).mean() + 1e-8), 0.01))
            elif m == "r_memorth": self._pen.append((F.cosine_similarity(yb, ye, dim=-1).pow(2).mean(), 0.05))
            elif m == "r_noise_cons":
                self._inner = True
                yb2, _ = self.body(xb + 0.1 * xb.detach().std() * torch.randn_like(xb), x, ids, t1, zm, pos)
                self._inner = False
                self._pen.append((((yb2 - yb.detach()) ** 2).mean() / (yb.detach().pow(2).mean() + 1e-8), 0.05))
        if m == "map_rms": ym = rms(ym) * self.k_mg
        elif m == "map_noise" and tr: ym = ym + 0.1 * ym.detach().pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(ym)
        elif m == "mem_drop": ye = drop(ye)
        elif m == "map_drop": ym = drop(ym)
        elif m == "mem_warm":
            if tr: self.n_step += 1
            ye = ye * (self.n_step / 100).clamp(max=1)
        elif m in ("freq_mem", "freq_body", "freq_map"):
            s = self.s2(torch.log1p(self.ctx["freq_rank"][ids].float()) / math.log(self.V))
            if m == "freq_mem": ye = ye * s
            elif m == "freq_body": yb = yb * s
            else: ym = ym * s
        elif m == "pos_mem": ye = ye * self.s2(torch.log1p(pos.float()) / math.log(T))
        elif m == "id_trust": ye = ye * 2 * torch.sigmoid(self.k_it[t1])[:, None]
        elif m == "id_bodygain": yb = yb * 2 * torch.sigmoid(self.k_it[t1])[:, None]
        elif m in ("mem_cap", "body_cap"):
            c = self.k_cap.abs() + 0.1
            if m == "mem_cap": ye = c * torch.tanh(ye / c)
            else: yb = c * torch.tanh(yb / c)
        elif m == "mem_sq": ye = ye + self.k_ms * ye * ye.abs()
        elif m in ("hgate_mem", "hgate_map"):
            gh = 1 + torch.tanh(rms(h) @ self.k_hg / math.sqrt(h.shape[1]))[:, None]
            if m == "hgate_mem": ye = ye * gh
            else: ym = ym * gh
        elif m == "comp_gate":
            g = torch.sigmoid(x @ self.k_cg)
            yb, ye = 2 * g * yb, 2 * (1 - g) * ye
        elif m == "agree_gate": ye = ye * (1 + torch.tanh(self.k_ag * F.cosine_similarity(yb, ye, dim=-1)[:, None]))
        elif m == "br_route":
            wt = 3 * torch.softmax(x @ self.k_br, -1)
            yb, ym, ye = yb * wt[:, :1], ym * wt[:, 1:2], ye * wt[:, 2:]
        elif m == "br_scal": yb, ym, ye = yb * self.k_bs[0], ym * self.k_bs[1], ye * self.k_bs[2]
        if m == "br_norm3": y = rms(yb) * self.k_bg[0] + rms(ym) * self.k_bg[1] + rms(ye) * self.k_bg[2]
        elif m == "peri_nomem": y = rms(yb + ym) * self.k_pg + ye
        else: y = yb + ym + ye
        if m in ("tok_tab", "tok_tab2"): y = y + self.k_tt[ids % 509]
        if m == "tok_tab2": y = y + self.k_tt2[ids % 1021]
        if m == "norm_skip": y = y + self.k_ns * x
        if m == "t_ema_y": y = y + self.k_eg * cema(y, torch.sigmoid(self.k_ea), B, T)
        elif m == "t_fastw":                          # causal linear attention (elu+1 features, rank 16) in the FFN slot
            q, k = F.elu(x @ self.k_fq) + 1, F.elu(x @ self.k_fk) + 1
            v = x @ self.k_fv
            kv = torch.einsum("btr,bts->btrs", k.view(B, T, -1), v.view(B, T, -1)).cumsum(1)
            num = torch.einsum("btr,btrs->bts", q.view(B, T, -1), kv)
            den = (q.view(B, T, -1) * k.view(B, T, -1).cumsum(1)).sum(-1, keepdim=True)
            y = y + (num / (den + 1e-6)).reshape(n, -1) @ self.k_fo
        elif m == "r_out_l2" and tr: self._pen.append((y.pow(2).mean() / (y.detach().pow(2).mean() + 1e-8), 0.01))
        if m == "peri" or self.pp: y = rms(y) * self.k_pg
        elif m == "peri_group": y = rms(y.view(n, 8, -1)).view(n, -1) * self.k_pg
        elif m == "peri_gain_id": y = rms(y) * (self.k_pg + self.k_tt[ids % 509])
        elif m == "id_film_out": y = y * (1 + self.k_fA[t1] @ self.k_fB)
        if m == "id_film_out" and self.pp: y = y * (1 + self.k_fA[t1] @ self.k_fB)
        if m == "layer_drop" and tr: y = y * (torch.rand(B, 1, 1, device=x.device) > 0.1).to(y.dtype).expand(B, T, 1).reshape(n, 1) / 0.9
        if m == "t_seq_gain": y = y * 2 * torch.sigmoid(self.k_cg * cmean(x, B, T))
        if tr:
            if m in ("g_tok_norm", "g_tok_clip"): y = TokGrad.apply(y, "norm" if m == "g_tok_norm" else "clip")
            elif m in ("g_freq_up", "g_freq_down"):
                u = (torch.log1p(self.ctx["freq_rank"][ids].float()) / math.log(self.V))[:, None]
                y = gscale(y, 0.5 + u if m == "g_freq_up" else 1.5 - u)
            elif m == "g_pos_ramp": y = gscale(y, (0.5 + pos.float() / max(T - 1, 1))[:, None])
            for pen, lam in self._pen: y = AddAux.apply(y, pen, lam)
        return y


# ---------------- round 20 (O33): how hash-MoE experts compute, 40 mechanisms, no token-history tables ----------------
X8_MODES = ("base", "a_wta", "a_wta25", "a_gsort", "a_lat", "a_div", "a_pair", "a_cos", "a_rbf", "a_l1",
            "x_sq", "x_abs", "x_sign", "x_sh", "x_sq_sh", "o_norm", "o_gate", "o_rotx", "o_rotx_full", "o_proj_e",
            "o_clip", "c_temp", "c_soft", "c_lin", "w_tie", "w_fac", "it2", "it2_sh", "it_h", "hc_sh", "hc_all",
            "hc_route", "d_drop", "d_hnoise", "m_hid", "m_mix", "rf", "rf512", "rf_sin", "p_pos", "p_pos_out")
X9_MODES = ("n_prod", "n_pow", "n_rat", "n_beta", "n_quad", "n_singate", "n_bump", "h_sort", "h_softmax",
            "s_xtopk", "s_shrink", "s_lsh", "hy_film", "kv_e", "f_learn", "dp_sh", "dp_skip", "o_tokgate",
            "q_tern", "c_max", "cx", "kan", "trop", "w_basis", "w_shv", "w_shg", "i_bias", "g_self", "g_agree",
            "g_out", "g_rank", "g_equal", "c_attn", "c_temp_tok", "c_prod", "o_norm_v", "a_l1_pm", "a_rbf_pm",
            "rf_pm", "o_clip_sum", "w_wide", "o_peri")
X8_MODES = X8_MODES + X9_MODES
X10_MODES = ("nest_ov", "nest_ov3", "ch_half", "ch_quarter", "rot_in", "rot_out", "pc_tanh", "pc_glu", "kron",
             "kron_s", "ta_act", "in_top25", "in_top50", "out_top25", "out_top50", "lsh_tab", "lsh_tab2", "dend",
             "ig", "smax", "prod_u", "ad_width", "norm_h", "norm_a", "hr_blk", "hr_blk3", "xl_w1", "xl_w2",
             "q_energy", "cat4", "w_nm", "kan_16", "kan_k8", "kan_sh", "kan_pm", "kan_in", "kan_l1", "o_norm_kan",
             "kan_kl", "kan_peri")
X8_MODES = X8_MODES + X10_MODES
X11_NEW = ("e_skip", "delta_in", "attn_in", "norm_feat", "norm_gate", "x0_in", "mh_out", "e_bias", "sgn_w2",
           "in_spline", "out_spline", "hh_refl", "o_sq", "gate_sh", "sh_gate_r", "ksum", "rot_y", "rot_e", "cmul_out",
           "perm_mix", "outer_out", "yx_glu")
PERI_ALT = ("o_dyt", "o_ln", "peri_e", "peri_sh", "peri_both", "peri_tok")
X8_MODES = X8_MODES + X11_NEW + PERI_ALT
# round 23: "x8_P_<mode>" = <mode> with Peri-LN (rms + gain on the FFN output, as o_peri) on top
R23_MODES = tuple("P_" + m for m in X11_NEW + ("rot_out", "q_energy", "norm_h", "kan_pm", "lsh_tab2", "kan_16", "kan_kl",
                                              "o_norm_v", "kv_e", "rf_pm", "kan_l1", "o_norm_kan")) + PERI_ALT
X12_NEW = ("y_carry", "odd_act", "conf_out", "ent_in", "e_chain", "tok_tab", "tok_scale", "tok_h", "sat_glu", "dual_w2",
           "dual_w1", "abs_glu", "sq_glu", "xr_prod", "g_inside", "sharp_tok", "kv_out", "acorr_h", "lg_in", "rh_to_sh",
           "xh_glu", "align_gate", "attn_mul", "eh_carry", "attn_h", "sig_out", "tok_gate")
X8_MODES = X8_MODES + X12_NEW + ("peri_tokh", "out_spline8", "outer16", "mh_out2")
# round 24: "_eb" suffix = per-expert output bias (e_bias) stacked on the named mode
R24_MODES = tuple("P_" + m for m in X12_NEW + ("kan_kl_eb", "kv_e_eb", "o_norm_kan_eb", "q_energy_eb", "kan_pm_eb", "sgn_w2_eb",
                                              "out_spline8", "outer16", "mh_out2", "n_rat", "m_hid", "x_sq")) + ("peri_tokh",)
X13_NEW = ("het_act", "het_act4", "het_lin", "cplx_glu", "quat_glu", "dual_glu", "g_tanh", "g_zsum", "g_sign", "o_gs",
           "o_orth_sh", "o_orth_e", "h_cmax", "h_cmax_a", "o_tanh_e", "o_glu_e", "hs_gate", "hpos_gate", "leak_b",
           "bilin_mix", "x_div", "x_div_o", "g_width", "g_from_a", "g_bias_e", "sr_balance", "x_quant", "x_signbits",
           "act_mix", "g_sq", "h_x_bil", "o_mlp", "x_comp", "tok_tab2", "tok_tabg", "tok_sig", "kv_out32", "tok_both",
           "sig_out_h", "acorr_o")
X8_MODES = X8_MODES + X13_NEW
R25_MODES = tuple("P_" + m for m in X13_NEW)
SIGNED_G = ("g_tanh", "g_zsum", "g_sign")
KAN_E = ("kan", "kan_16", "kan_k8", "kan_pm", "kan_in", "kan_l1", "o_norm_kan", "kan_kl", "kan_peri")


class X8(C):
    """name = x8_<mode>. An all-hash MoE written here (E experts, top-k by a fixed hash of the current token id,
    sigmoid scores of the picked experts renormalized x 2.5, one shared SwiGLU), as the allhash lab baseline,
    plus one change to how experts compute or combine. x8_base = the replica (control). Added parameters are
    k_*; additive ones start at zero. a_wta*, a_gsort, a_cos, o_norm, o_rotx_full, c_soft, w_tie, w_fac, d_* change step 0."""

    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx)
        pm = name[3:]
        self.peri = pm.startswith("P_")
        m = self.mode = pm[2:] if self.peri else pm
        self.eb = m.endswith("_eb")
        if self.eb:
            m = self.mode = m[:-3]
        assert m in X8_MODES, m
        E, k, nl = cfg.n_experts, cfg.topk, cfg.n_layers
        h = int(self.cb) or 48
        if m == "w_tie":
            h = h * 3 // 2                      # tied W2 = W1^T frees one matrix: 1.5x width, same params
        if m == "w_fac":
            h = h * 4 // 3                      # W2_e = A_e B (rank 16): width 64 at the same params
        h = {"n_prod": h * 3 // 4, "w_shv": 3 * cfg.n_experts * h // (2 * cfg.n_experts + 1),
             "w_shg": 3 * cfg.n_experts * h // (2 * cfg.n_experts + 1),
             "a_l1_pm": h - 5, "a_rbf_pm": h - 11, "rf_pm": h - 5, "w_wide": h + 5, "ch_half": h * 3 // 2, "ch_quarter": h * 2, "dend": h * 3 // 4,
             "smax": h * 3 // 4, "kan_pm": h - 5, "q_energy": h - 8, "sgn_w2": h * 3 // 4, "dual_w1": h * 3 // 4, "dual_w2": h * 3 // 4}.get(m, h)
        hs = cfg.shared_hidden
        self.E, self.k, self.h, self.hs = E, k, h, hs
        z = lambda *sh: nn.Parameter(torch.zeros(*sh))
        rn3 = lambda *sh, std: nn.Parameter(torch.randn(*sh) * std)
        self.k_gate = rn3(E, d, std=0.02)
        if m == "w_shg":
            self.k_W1s, self.k_W3 = rn3(d, h, std=0.02), rn3(E, d, h, std=0.02)
        elif m == "w_shv":
            self.k_W1, self.k_W3s = rn3(E, d, h, std=0.02), rn3(d, h, std=0.02)
        elif m in ("nest_ov", "nest_ov3"):            # expert e = window of w units in a circular pool of E*h units
            w = h * (2 if m == "nest_ov" else 3)
            self.k_P1, self.k_P3 = rn3(d, E * h, std=0.02), rn3(d, E * h, std=0.02)
            self.register_buffer("nidx", (torch.arange(E)[:, None] * h + torch.arange(w)[None]) % (E * h), persistent=False)
        elif m in ("kron", "kron_s"):                 # hidden = outer product of a 10-unit and a 12-unit factor
            self.k_W1, self.k_W3 = rn3(E, d, 10, std=0.02), rn3(E, d, 12, std=0.02)
        else:
            self.k_W1, self.k_W3 = rn3(E, d, h, std=0.02), rn3(E, d, h, std=0.02)
        if m == "w_tie":
            self.k_ts = nn.Parameter(torch.tensor(1 / math.sqrt(2 * nl)))
        elif m == "w_fac":
            self.k_A, self.k_B = rn3(E, h, 16, std=0.045), rn3(16, d, std=0.045 / math.sqrt(2 * nl))
        elif m in ("nest_ov", "nest_ov3"):
            self.k_P2 = rn3(E * h, d, std=0.02 / math.sqrt(2 * nl))
        else:
            self.k_W2 = rn3(E, 120 if m in ("kron", "kron_s") else h, d, std=0.02 / math.sqrt(2 * nl))
        if m in ("nest_ov", "nest_ov3"):
            self.h = h = self.nidx.shape[1]
        if m in ("kron", "kron_s"):
            self.h = 120
        self.k_s1, self.k_s3 = rn3(d, hs, std=0.02), rn3(d, hs, std=0.02)
        self.k_s2 = rn3(hs, d, std=0.02 / math.sqrt(2 * nl))
        self.register_buffer("sets", ksets(self.V, E, k, self.seed + 7), persistent=False)
        if m in ("a_lat", "it_h"): self.k_L = z(h, h)
        if m == "a_div": self.k_dv = z(E, 1, 1)
        if m == "a_pair": self.k_pc = z(E, 1, 1)
        if m == "a_cos": self.k_ct = nn.Parameter(torch.full((E, 1, 1), 0.02 * d))
        if m in ("a_rbf", "a_rbf_pm"): self.k_c, self.k_Ro = rn3(E, 16, d, std=1.0), z(E, 16, d)
        if m in ("a_l1", "a_l1_pm", "kan_l1"): self.k_c, self.k_Ro = rn3(E, 8, d, std=1.0), z(E, 8, d)
        if m in ("x_sq", "x_abs", "x_sign"): self.k_Wq = z(d, h)
        if m == "x_sh": self.k_Wq = z(hs, h)
        if m == "x_sq_sh": self.k_Wq = z(d, hs)
        if m in ("o_norm",): self.k_on = nn.Parameter(torch.full((E, 1, 1), 0.05))
        if m in ("o_norm_v", "o_norm_kan"): self.k_on = nn.Parameter(torch.full((E, 1, d), 0.05))
        if m in ("o_peri", "kan_peri"): self.k_pg = nn.Parameter(torch.full((d,), 0.1))
        if m == "n_prod": self.k_W4 = rn3(E, d, h, std=0.02)
        if m in ("n_pow", "n_beta", "n_quad", "n_singate", "n_bump"): self.k_np = z(E, 1, 1)
        if m == "n_rat": self.k_r = z(3, E, 1, 1)
        if m in ("h_softmax", "c_max", "g_self", "g_agree", "c_prod", "o_clip_sum"): self.k_s = z(1)
        if m == "s_shrink": self.k_th = nn.Parameter(torch.full((E, 1, h), 1e-3))
        if m == "s_lsh":
            self.register_buffer("lshR", torch.randn(E, d, h, generator=gen(self.seed + 5)), persistent=False)
        if m == "hy_film": self.k_fu, self.k_fv = rn3(d, 4, std=1 / math.sqrt(d)), z(E, 4, h)
        if m == "kv_e": self.k_kk, self.k_kv = rn3(E, 8, d, std=1 / math.sqrt(d)), z(E, 8, d)
        if m == "f_learn": self.k_fw, self.k_fo = rn3(d, 16, std=3 / math.sqrt(d)), z(16, h)
        if m == "dp_sh": self.k_D = z(h, h)
        if m == "dp_skip": self.k_P, self.k_Q = rn3(E, d, 4, std=0.02), z(E, 4, d)
        if m in ("o_tokgate", "c_temp_tok", "g_out"): self.k_u = z(d)
        if m == "cx": self.k_cb = z(E, 1, h // 2)
        if m in KAN_E or m == "kan_sh":
            nP = 16 if m == "kan_16" else 8
            kn = torch.linspace(-1.5, 1.5, 8) if m == "kan_k8" else torch.linspace(-1.0, 1.0, 4)
            self.register_buffer("knots", kn, persistent=False)
            if m == "kan_sh":
                self.k_kP, self.k_kc, self.k_kR = rn3(d, 32, std=1 / math.sqrt(d)), nn.Parameter(torch.ones(32, 4)), z(32, d)
            else:
                self.k_kP, self.k_kc = rn3(E, d, nP, std=1 / math.sqrt(d)), nn.Parameter(torch.ones(E, nP, len(kn)))
                if m == "kan_in":
                    self.k_kA = z(E, nP, h)
                else:
                    self.k_kR = z(E, nP, d)
            if m == "kan_kl": self.k_kt = nn.Parameter(kn.clone())
        if m in ("rot_in", "rot_out", "rot_y", "cmul_out"): self.k_rw = z(d, d // 2)
        if (self.peri and m not in ("o_peri", "kan_peri")) or m in ("o_dyt", "o_ln", "peri_e", "peri_sh", "peri_both", "peri_tok", "peri_tokh"):
            self.k_pg = nn.Parameter(torch.full((d,), 0.1))
        if m == "peri_both": self.k_pg2 = nn.Parameter(torch.full((d,), 0.1))
        if m == "peri_tok": self.k_pt = z(d, d)
        if m == "o_dyt": self.k_al = nn.Parameter(torch.tensor(math.log(20.0)))
        if m in ("e_skip", "e_bias"): self.k_es = z(E, 1, d)
        if m in ("delta_in", "attn_in", "x0_in", "ksum"): self.k_di = z(d)
        if m == "norm_feat": self.k_nf = z(E, 1, h)
        if m == "norm_gate": self.k_ng = z(d)
        if m in ("mh_out", "mh_out2"):
            js = (1, 2, 3) if m == "mh_out" else (1,)
            self.register_buffer("sets2", torch.stack([ksets(self.V, E, k, self.seed + 7 + 100 * j) for j in js]), persistent=False)
        if m == "sgn_w2": self.k_W2b = rn3(E, h, d, std=0.02 / math.sqrt(2 * nl))
        if m in ("in_spline", "out_spline", "out_spline8"):
            nk = 8 if m == "out_spline8" else 4
            self.k_isp = z(d, nk)
            self.register_buffer("sknots", torch.linspace(-1.0, 1.0, nk), persistent=False)
        if m == "hh_refl": self.k_hv, self.k_s = rn3(d, d, std=1 / math.sqrt(d)), z(1)
        if m == "o_sq": self.k_osq = z(d)
        if m == "gate_sh": self.k_gs = z(hs, d)
        if m in ("sh_gate_r", "perm_mix", "yx_glu"): self.k_sg = z(d, d)
        if m == "perm_mix": self.register_buffer("perm", torch.randperm(d, generator=gen(self.seed + 17)), persistent=False)
        if m == "rot_e": self.k_rwe = z(E, d, d // 2)
        if m == "cmul_out": self.k_rm = z(d, d // 2)
        if m in ("outer_out", "outer16"):
            ro = 16 if m == "outer16" else 4
            self.k_ou, self.k_ov, self.k_ow = rn3(d, ro, std=1 / math.sqrt(d)), rn3(d, ro, std=1 / math.sqrt(d)), z(ro, d)
        if self.eb: self.k_eb = z(E, 1, d)
        if m == "odd_act": self.k_od = z(E, 1, h)
        if m == "conf_out": self.k_cf = z(d)
        if m == "ent_in": self.k_en = z(E, 1, h)
        if m == "e_chain": self.k_ch = z(E, 1, h)
        if m in ("tok_tab", "tok_scale", "peri_tokh"): self.k_tt = z(509, d)
        if m == "tok_h": self.k_tt = z(509, h)
        if m == "tok_gate": self.k_tt = z(509, E)
        if m == "sat_glu": self.k_sat = z(E, 1, h)
        if m == "dual_w1": self.k_dsv, self.k_dW = rn3(E, d, std=1 / math.sqrt(d)), z(E, d, h)
        if m == "dual_w2": self.k_dsv, self.k_dW = rn3(E, d, std=1 / math.sqrt(d)), z(E, h, d)
        if m == "xr_prod": self.k_xp = z(d)
        if m == "g_inside": self.k_gi = z(E, 1, h)
        if m == "sharp_tok": self.k_st = z(d, E)
        if m == "kv_out": self.k_kok, self.k_kov = rn3(16, d, std=1 / math.sqrt(d)), z(16, d)
        if m == "acorr_h": self.k_ac = z(E, 1, h)
        if m == "lg_in": self.k_lgW = z(E, E, h)
        if m == "rh_to_sh": self.k_rs = z(h, hs)
        if m == "xh_glu": self.k_xg = z(E, 1, h)
        if m == "align_gate": self.k_ag = z(d)
        if m == "attn_mul": self.k_am = z(d, d)
        if m == "eh_carry": self.k_ehc = z(E, 1, h)
        if m == "attn_h": self.k_ah = z(d, h)
        if m == "sig_out": self.k_so = z(E, d)
        if m == "y_carry": self.k_yc = z(d)
        # round 25
        if m in ("het_act", "het_act4", "het_lin"):
            self.register_buffer("eidx", torch.arange(E).view(E, 1, 1), persistent=False)
        if m == "g_zsum": self.k_gz = z(1)
        if m == "g_sign": self.k_gsn = nn.Parameter(torch.full((E,), 3.0))
        if m in ("o_gs", "o_orth_e"): self.k_og = z(E, 1, 1)
        if m == "o_orth_sh": self.k_og = z(1)
        if m in ("h_cmax", "h_cmax_a", "leak_b", "bilin_mix"): self.k_hx = z(E, 1, h)
        if m in ("o_tanh_e",): self.k_ot = z(E, 1, d)
        if m == "o_glu_e": self.k_ou = z(E, d)
        if m in ("hs_gate", "hpos_gate"): self.k_hg = z(d)
        if m in ("x_div", "x_div_o"): self.k_xd = z(E, 1, 1)
        if m == "g_width": self.k_gw = z(E, 1, 1)
        if m == "g_from_a": self.k_ga = z(E, 1)
        if m == "g_bias_e": self.k_gb = z(E)
        if m == "sr_balance": self.k_sb = z(d)
        if m == "x_quant": self.k_xq = z(d)
        if m == "x_signbits":
            self.register_buffer("pw", 2 ** torch.arange(8), persistent=False)
            self.k_lt = z(256, d)
        if m == "act_mix": self.k_amx = nn.Parameter(torch.tensor([4.0, 0.0, 0.0]).repeat(E, 1, h, 1))
        if m == "g_sq": self.k_gq = z(d, E)
        if m == "h_x_bil": self.k_hU, self.k_hV, self.k_hR = rn3(E, h, 8, std=1 / math.sqrt(h)), rn3(d, 8, std=1 / math.sqrt(d)), z(E, 8, d)
        if m == "o_mlp": self.k_oA, self.k_oB = rn3(d, 16, std=1 / math.sqrt(d)), z(16, d)
        if m == "x_comp": self.k_xc = z(1, 1, h)
        if m in ("tok_tab2", "tok_tabg", "tok_sig", "tok_both"): self.k_tt = z(509, d)
        if m == "tok_tab2": self.k_tt2 = z(1021, d)
        if m == "tok_tabg": self.k_tg = z(d)
        if m == "tok_both": self.k_tt3 = z(509, E)
        if m == "tok_sig": self.k_so = z(E, d)
        if m == "kv_out32": self.k_kok, self.k_kov = rn3(32, d, std=1 / math.sqrt(d)), z(32, d)
        if m == "sig_out_h": self.k_soh = z(E, h)
        if m == "acorr_o": self.k_ac = z(d)
        if m == "pc_tanh": self.k_pc = z(d)
        if m == "pc_glu": self.k_pW = z(d, d)
        if m in ("ch_half", "ch_quarter"):
            g_, nc = gen(self.seed + 11), (d // 2 if m == "ch_half" else d // 4)
            msk = torch.zeros(E, d, 1)
            for e in range(E):
                msk[e, torch.randperm(d, generator=g_)[:nc]] = 1
            self.register_buffer("chm", msk, persistent=False)
        if m == "ta_act": self.k_ta = z(d, E)
        if m in ("lsh_tab", "lsh_tab2"):
            nt = 1 if m == "lsh_tab" else 2
            self.register_buffer("lR", torch.randn(nt, d, 8, generator=gen(self.seed + 13)), persistent=False)
            self.register_buffer("pw", 2 ** torch.arange(8), persistent=False)
            self.k_lt = z(nt, 256, d)
        if m in ("dend", "smax"): self.k_W1b = rn3(E, d, h, std=0.02)
        if m == "ig": self.k_iu, self.k_iv = rn3(d, 4, std=1 / math.sqrt(d)), z(E, 4, d)
        if m == "prod_u": self.k_pu, self.k_pR = rn3(E, d, 8, std=0.1), z(E, 8, d)
        if m == "ad_width": self.k_aw = z(d, E)
        if m == "norm_h": self.k_hn = nn.Parameter(torch.full((E, 1, 1), 0.02))
        if m == "norm_a": self.k_an = nn.Parameter(torch.full((E, 1, 1), 0.2))
        if m in ("hr_blk", "hr_blk3"): self.k_hb = rn3(d, E * 4, std=0.02)
        if m in ("xl_w1", "xl_w2"): self.k_xs = z(E, 1, 1)
        if m == "q_energy": self.k_qP, self.k_qR = rn3(E, d, 16, std=1 / math.sqrt(d)), z(E, 8, d)
        if m == "trop": self.k_tW, self.k_tR = rn3(E, d, 8, std=1.0), z(E, 8, d)
        if m == "w_basis": self.k_Bs, self.k_al = rn3(4, d, h, std=0.02), z(E, 4)
        if m == "i_bias": self.k_b1 = z(E, 1, h)
        if m == "c_attn": self.k_aq, self.k_ak, self.k_s = rn3(d, 8, std=1 / math.sqrt(d)), rn3(d, 8, std=1 / math.sqrt(d)), z(1)
        if m == "o_gate": self.k_Wg = z(d, d)
        if m == "o_rotx": self.k_oc = z(1)
        if m == "o_proj_e": self.k_oc = z(E, 1, 1)
        if m == "o_clip": self.k_t = z(E, 1, 1)
        if m in ("c_temp", "c_soft"): self.k_tau = z(1)
        if m == "c_lin": self.k_sv = z(E, d)
        if m in ("it2", "it2_sh"): self.k_ic = z(1)
        if m in ("hc_sh", "hc_all"): self.k_Hc = z(hs, hs)
        if m == "hc_route": self.k_hr = z(1)
        if m == "m_hid": self.k_mh = z(E, 1, 1)
        if m == "m_mix": self.k_mA = z(E, E)
        if m in ("rf", "rf512", "rf_sin", "rf_pm"):
            n = 512 if m == "rf512" else 256
            self.register_buffer("rfW", torch.randn(d, n, generator=gen(self.seed + 3)) / math.sqrt(d), persistent=False)
            self.register_buffer("rfb", torch.rand(n, generator=gen(self.seed + 4)) * 2 * math.pi, persistent=False)
            self.k_rfO = z(n, d)
        if m == "p_pos": self.k_Wp = z(16, h)
        if m == "p_pos_out": self.k_Wp = z(16, d)

    def pe(self, ids2):
        B, T = ids2.shape
        t = torch.arange(T, device=ids2.device, dtype=torch.float32).repeat(B)[:, None]
        fr = torch.exp(-torch.arange(8, device=ids2.device, dtype=torch.float32) * math.log(T) / 8)
        return torch.cat([torch.sin(t * fr), torch.cos(t * fr)], 1)

    def shared_h(self, x, extra=0):
        a = x @ self.k_s1 + extra
        if self.mode == "x_sq_sh":
            a = a + (x * x) @ self.k_Wq
        return F.silu(a) * (x @ self.k_s3)

    def kanphi(self, xe):
        u = torch.einsum("end,edu->enu", xe, self.k_kP)
        t = self.k_kt if self.mode == "kan_kl" else self.knots
        return (F.relu(u[..., None] - t) * self.k_kc[:, None]).sum(-1)

    def rot(self, v, th):
        v2 = v.unflatten(-1, (v.shape[-1] // 2, 2))
        c, s_ = torch.cos(th), torch.sin(th)
        return torch.stack([c * v2[..., 0] - s_ * v2[..., 1], s_ * v2[..., 0] + c * v2[..., 1]], -1).flatten(-2)

    def xin(self, x):
        m = self.mode
        if m == "x_quant":
            return x + self.k_xq * ((x * 4).round() / 4 - x).detach()
        if m == "rot_in":
            return self.rot(x, x @ self.k_rw)
        if m in ("in_top25", "in_top50"):
            kk = x.shape[-1] // (4 if m == "in_top25" else 2)
            return x * (x.abs() >= x.abs().topk(kk, -1).values[:, -1:]).to(x.dtype)
        if m == "ig":
            return x[None] * 2 * torch.sigmoid(torch.einsum("nr,erd->end", x @ self.k_iu, self.k_iv))
        if m in ("delta_in", "attn_in", "x0_in"):
            src = {"delta_in": "delta", "attn_in": "attn", "x0_in": "x8r0"}[m]
            return x + self.k_di * rms(self.ctx[src].reshape(x.shape))
        if m == "in_spline":
            return x + (F.relu(x[..., None] - self.sknots) * self.k_isp).sum(-1)
        return x

    def W1(self):
        m = self.mode
        if m in ("nest_ov", "nest_ov3"):
            return self.k_P1[:, self.nidx].permute(1, 0, 2)
        if m in ("ch_half", "ch_quarter"):
            return self.k_W1 * self.chm
        if m == "xl_w1" and self.layer:
            return self.k_W1 + self.k_xs * self.ctx[("x8xl", m)]
        if m == "w_nm":
            W = self.k_W1
            keep = W.unflatten(1, (W.shape[1] // 4, 4)).abs().argsort(2).argsort(2) >= 2
            return W + (W * keep.flatten(1, 2).to(W.dtype) - W).detach()
        if self.mode == "w_shg":
            return self.k_W1s[None].expand(self.E, -1, -1)
        if self.mode == "w_basis":
            return self.k_W1 + torch.einsum("eb,bdh->edh", self.k_al, self.k_Bs)
        return self.k_W1

    def W3(self):
        if self.mode in ("nest_ov", "nest_ov3"):
            return self.k_P3[:, self.nidx].permute(1, 0, 2)
        if self.mode in ("ch_half", "ch_quarter"):
            return self.k_W3 * self.chm
        if self.mode == "w_shv":
            return self.k_W3s[None].expand(self.E, -1, -1)
        return self.k_W3

    def W2(self):
        m = self.mode
        if m == "w_tie":
            return self.k_W1.transpose(1, 2) * self.k_ts
        if m == "w_fac":
            return self.k_A @ self.k_B
        if m in ("nest_ov", "nest_ov3"):
            return self.k_P2[self.nidx]
        if m == "xl_w2" and self.layer:
            return self.k_W2 + self.k_xs * self.ctx[("x8xl", m)]
        return self.k_W2

    def experts(self, xin, x, M, hsh, ids2):
        """all E experts on all tokens: xin [N,d] or [E,N,d] -> O [E,N,d] (unrouted tokens get weight 0 later)."""
        m, h = self.mode, self.h
        eq = "nd,edh->enh" if xin.dim() == 2 else "end,edh->enh"
        if m == "a_cos":
            xn = F.normalize(xin, dim=-1)
            a = torch.einsum(eq, xn, F.normalize(self.k_W1, dim=1)) * self.k_ct
        else:
            a = torch.einsum(eq, xin, self.W1())
        b = torch.einsum(eq, xin, self.W3())
        if m == "i_bias":
            a = a + self.k_b1
        elif m == "f_learn":
            a = a + (torch.sin(x @ self.k_fw) @ self.k_fo)[None]
        elif m == "norm_feat":
            a = a + self._lrv[None, :, None] * self.k_nf
        elif m == "ent_in":
            a = a + self._ent[None, :, None] * self.k_en
        elif m == "lg_in":
            a = a + torch.einsum("nf,efh->enh", self._lg, self.k_lgW)
        elif m == "eh_carry":
            pH = self.ctx.get(("x8eh", self.layer - 1)) if self.layer else None
            if pH is not None and pH.shape == a.shape:
                a = a + self.k_ehc * pH
        elif m == "dual_w1":
            xe = (xin if xin.dim() == 3 else xin[None]).expand(self.E, -1, -1)
            a = a + torch.sigmoid(torch.einsum("end,ed->en", xe, self.k_dsv))[..., None] * torch.einsum(eq, xin, self.k_dW)
        elif m == "sharp_tok":
            a = a * (1 + torch.tanh(x @ self.k_st)).t()[:, :, None]
        elif m == "g_inside":
            a = a * (1 + self.k_gi * self._G.t()[:, :, None])
        elif m == "kan_in":
            xe = xin if xin.dim() == 3 else xin[None].expand(self.E, -1, -1)
            a = a + torch.einsum("enu,euh->enh", self.kanphi(xe), self.k_kA)
        if m in ("x_sq", "x_abs", "x_sign"):
            f = {"x_sq": x * x, "x_abs": x.abs(), "x_sign": torch.sign(x)}[m]
            a = a + (f @ self.k_Wq)[None]
        elif m == "x_sh":
            a = a + (hsh @ self.k_Wq)[None]
        elif m == "p_pos":
            a = a + (self.pe(ids2) @ self.k_Wp)[None]
        if m == "a_gsort":
            s = a.unflatten(-1, (h // 2, 2)).sort(-1, descending=True).values
            g = F.silu(s.flatten(-2))
        elif m == "n_pow":
            g = F.silu(a) * (1 + a.abs()).pow(self.k_np)
        elif m == "n_rat":
            g = F.silu(a) + (self.k_r[0] * a + self.k_r[1] * a * a) / (1 + (self.k_r[2] * a).abs())
        elif m == "n_beta":
            g = a * torch.sigmoid(self.k_np.exp() * a)
        elif m == "n_singate":
            g = F.silu(a) + self.k_np * torch.sin(4 * a)
        elif m == "n_bump":
            g = F.silu(a) * torch.exp(-self.k_np * a * a)
        elif m == "cx":
            p = a.unflatten(-1, (h // 2, 2))
            r = (p * p).sum(-1, keepdim=True).add(1e-8).sqrt()
            g = (p * F.relu(r + self.k_cb[..., None]) / r).flatten(-2)
        elif m == "ta_act":
            g = F.silu(a) + torch.tanh(x @ self.k_ta).t()[:, :, None] * F.relu(a) ** 2
        elif m == "smax":
            g = torch.logaddexp(a, torch.einsum(eq, xin, self.k_W1b))
        elif m == "dend":
            g = F.silu(F.relu(a) - F.relu(torch.einsum(eq, xin, self.k_W1b)))
        elif m == "norm_a":
            g = F.silu(rms(a) * self.k_an)
        elif m == "odd_act":
            g = F.silu(a) - self.k_od * F.silu(-a)
        elif m == "abs_glu":
            g = a.abs()
        elif m == "sq_glu":
            g = a * a
        elif m == "het_act":
            g = torch.where(self.eidx < self.E // 2, F.silu(a), F.relu(a) ** 2)
        elif m == "het_act4":
            r = self.eidx % 4
            g = torch.where(r == 0, F.silu(a), torch.where(r == 1, F.relu(a) ** 2,
                            torch.where(r == 2, F.gelu(a, approximate="tanh"), F.softplus(a) - math.log(2))))
        elif m == "het_lin":
            g = torch.where(self.eidx % 2 == 0, F.silu(a), a)
        elif m == "act_mix":
            w = torch.softmax(self.k_amx, -1)
            g = w[..., 0] * F.silu(a) + w[..., 1] * F.relu(a) ** 2 + w[..., 2] * torch.tanh(a)
        elif m == "h_cmax_a":
            g = F.silu(a) + self.k_hx * F.silu(a.cummax(-1).values)
        else:
            g = F.silu(a)
        if m == "g_from_a":
            self._amean = a.mean(-1)
        elif m == "hpos_gate":
            self._apos = torch.sigmoid(4 * a).mean(-1)
        if m in ("kron", "kron_s"):
            H = (g[..., :, None] * (F.silu(b) if m == "kron_s" else b)[..., None, :]).flatten(-2)
        elif m in ("cplx_glu", "dual_glu"):
            gp, bp = g.unflatten(-1, (h // 2, 2)), b.unflatten(-1, (h // 2, 2))
            re = gp[..., 0] * bp[..., 0] - (gp[..., 1] * bp[..., 1] if m == "cplx_glu" else 0)
            H = torch.stack([re, gp[..., 0] * bp[..., 1] + gp[..., 1] * bp[..., 0]], -1).flatten(-2)
        elif m == "quat_glu":
            p, q = g.unflatten(-1, (h // 4, 4)), b.unflatten(-1, (h // 4, 4))
            a1, b1, c1, d1 = p.unbind(-1)
            a2, b2, c2, d2 = q.unbind(-1)
            H = torch.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                             a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2], -1).flatten(-2)
        else:
            H = g * b
        if m == "leak_b":
            H = H + self.k_hx * b
        elif m == "bilin_mix":
            H = H + self.k_hx * a * b
        elif m == "h_cmax":
            H = H + self.k_hx * H.cummax(-1).values
        elif m == "x_div":
            s = torch.einsum("en,ne->n", H.abs().mean(-1), M) / self.k
            H = H / (1 + F.softplus(self.k_xd - 4) * s[None, :, None])
        elif m == "x_comp":
            w = a.masked_fill(M.t()[:, :, None] == 0, -1e4).softmax(0)
            H = H * (1 + self.k_xc * (self.k * w - 1))
        elif m == "g_width":
            wd = h * torch.sigmoid(4 + self.k_gw * (self.k * self._G.t()[:, :, None] / 2.5 - 1))
            H = H * torch.sigmoid(wd - torch.arange(h, device=x.device, dtype=H.dtype) - 0.5)
        elif m == "sig_out_h":
            H = H + (torch.sigmoid(self._lg) @ self.k_soh)[None]
        if m == "n_prod":
            H = H * (1 + torch.einsum(eq, xin, self.k_W4))
        elif m == "n_quad":
            H = H * (1 + self.k_np * b)
        elif m == "i_bias" or m == "f_learn":
            pass
        if m == "sat_glu":
            H = H / (1 + F.softplus(self.k_sat - 4) * H.abs())
        elif m == "tok_h":
            H = H * (1 + self.k_tt[self._ids % 509])[None]
        elif m == "attn_h":
            H = H * (1 + torch.tanh(rms(self.ctx["attn"].reshape(x.shape)) @ self.k_ah))[None]
        elif m == "acorr_h":
            fa = torch.fft.rfft(a.float(), dim=-1)
            H = H + self.k_ac * (torch.fft.irfft(fa.real ** 2 + fa.imag ** 2, n=h, dim=-1) / h).to(H.dtype)
        elif m == "xh_glu":
            H = H * (1 + self.k_xg * torch.tanh(torch.einsum("enh,ne->nh", H, M) / self.k)[None])
        elif m == "e_chain":
            H = F.silu(a + self.k_ch * H.roll(1, 0)) * b
        if m == "eh_carry":
            self.ctx[("x8eh", self.layer)] = H
        self._H = H
        if m == "norm_h":
            H = rms(H) * self.k_hn
        elif m == "cat4":
            H = H * 4 * torch.softmax(a.unflatten(-1, (h // 4, 4)), -1).flatten(-2)
        elif m == "ad_width":
            wd = h * torch.sigmoid(4 + x @ self.k_aw).t()[:, :, None]
            H = H * torch.sigmoid(wd - torch.arange(h, device=x.device, dtype=H.dtype) - 0.5)
        elif m in ("hr_blk", "hr_blk3"):
            sc = (x @ self.k_hb).view(-1, self.E, 4).transpose(0, 1)          # [E,N,4]
            kk = 2 if m == "hr_blk" else 3
            msk = (sc >= sc.topk(kk, -1).values[..., -1:]).to(H.dtype)
            wt = torch.softmax(sc, -1) * msk
            wt = wt / wt.sum(-1, keepdim=True) * kk
            H = (H.unflatten(-1, (4, h // 4)) * wt[..., None]).flatten(-2) * 4 / kk
        elif m == "h_sort":
            H = H.sort(-1, descending=True).values
        elif m == "h_softmax":
            H = H * (1 + self.k_s * (h * torch.softmax(a, -1) - 1))
        elif m == "s_shrink":
            H = torch.sign(H) * F.relu(H.abs() - F.relu(self.k_th))
        elif m == "s_lsh":
            xe = xin if xin.dim() == 3 else xin[None]
            H = H * 2 * (torch.einsum("end,edh->enh", xe.expand(self.E, -1, -1), self.lshR) > 0).to(H.dtype)
        elif m == "q_tern":
            sc = H.abs().mean(-1, keepdim=True) + 1e-8
            H = H + ((H / sc).round().clamp(-1, 1) * sc - H).detach()
        elif m == "hy_film":
            H = H * (1 + torch.einsum("nr,erh->enh", x @ self.k_fu, self.k_fv))
        elif m == "dp_sh":
            H = H + F.silu(H @ self.k_D)
        elif m == "s_xtopk":
            Hr = H.permute(1, 0, 2).gather(1, self._sets[:, :, None].expand(-1, -1, h)).flatten(1)
            kk = Hr.shape[1] // 4
            thr = Hr.abs().kthvalue(kk, dim=-1).values
            H = H * (H.abs() > thr[None, :, None]).to(H.dtype)
        if m in ("a_wta", "a_wta25"):
            kk = h // 2 if m == "a_wta" else h - h // 4
            thr = H.kthvalue(kk, dim=-1, keepdim=True).values
            H = H * (H > thr).to(H.dtype)
        elif m == "a_lat":
            H = H * (1 + torch.tanh(H @ self.k_L))
        elif m == "a_div":
            H = H / (1 + torch.exp(self.k_dv - 4) * H.abs().mean(-1, keepdim=True))
        elif m == "a_pair":
            H = H + self.k_pc * H * H.roll(1, -1)
        elif m == "it_h":
            H = F.silu(a + H @ self.k_L) * b
        elif m == "m_hid":
            Hm = torch.einsum("enh,ne->nh", H, M) / self.k
            H = H + self.k_mh * Hm[None]
        elif m == "d_hnoise" and self.training:
            H = H * (1 + 0.1 * torch.randn_like(H))
        O = torch.einsum("enh,ehd->end", H, self.W2())
        if m == "sgn_w2":
            O = O + torch.einsum("enh,ehd->end", -F.relu(-H), self.k_W2b - self.k_W2)
        elif m == "e_skip":
            O = O + self.k_es * (xin if xin.dim() == 3 else xin[None])
        elif m == "e_bias":
            O = O + self.k_es
        elif m == "dual_w2":
            xe = (xin if xin.dim() == 3 else xin[None]).expand(self.E, -1, -1)
            O = O + torch.sigmoid(torch.einsum("end,ed->en", xe, self.k_dsv))[..., None] * torch.einsum("enh,ehd->end", H, self.k_dW)
        if self.eb:
            O = O + self.k_eb
        if m == "o_tanh_e":
            O = O + self.k_ot * torch.tanh(rms(O))
        elif m == "o_glu_e":
            O = O * (1 + torch.tanh(torch.einsum("end,ed->en", rms(O), self.k_ou)))[..., None]
        elif m == "h_x_bil":
            O = O + torch.einsum("enr,erd->end", torch.einsum("enh,ehr->enr", H, self.k_hU) * (x @ self.k_hV)[None], self.k_hR)
        elif m == "x_div_o":
            s = torch.einsum("en,ne->n", O.abs().mean(-1), M) / self.k
            O = O / (1 + F.softplus(self.k_xd - 4) * s[None, :, None])
        if m in ("kv_e", "trop", "dp_skip", "q_energy", "prod_u") or (m in KAN_E and m != "kan_in"):
            xe = xin if xin.dim() == 3 else xin[None].expand(self.E, -1, -1)
            if m == "kv_e":
                O = O + torch.einsum("enu,eud->end", torch.softmax(torch.einsum("end,eud->enu", xe, self.k_kk), -1), self.k_kv)
            elif m in KAN_E:
                O = O + torch.einsum("enu,eud->end", self.kanphi(xe), self.k_kR)
            elif m == "q_energy":
                qz = torch.einsum("end,edu->enu", xe, self.k_qP).unflatten(-1, (8, 2))
                O = O + torch.einsum("enu,eud->end", qz.pow(2).sum(-1), self.k_qR)
            elif m == "prod_u":
                pz = torch.einsum("end,edu->enu", torch.log(xe.abs() + 1e-3), self.k_pu).clamp(-5, 5).exp()
                O = O + torch.einsum("enu,eud->end", pz, self.k_pR)
            elif m == "trop":
                tr = (xe[..., None] + self.k_tW[:, None]).amax(2)
                O = O + torch.einsum("enu,eud->end", tr, self.k_tR)
            else:
                O = O + torch.einsum("enr,erd->end", torch.einsum("end,edr->enr", xe, self.k_P), self.k_Q)
        if m in ("a_rbf", "a_l1", "a_rbf_pm", "a_l1_pm", "kan_l1"):
            xe = xin if xin.dim() == 3 else xin[None].expand(self.E, -1, -1)
            if m in ("a_rbf", "a_rbf_pm"):
                d2 = (xe * xe).sum(-1, keepdim=True) + (self.k_c * self.k_c).sum(-1)[:, None] \
                    - 2 * torch.einsum("end,eud->enu", xe, self.k_c)
                phi = torch.exp(-d2 / (2 * x.shape[-1]))
            else:
                phi = torch.exp(-(xe[:, :, None] - self.k_c[:, None]).abs().mean(-1))
            O = O + torch.einsum("enu,eud->end", phi, self.k_Ro)
        return O

    def f(self, x, ids, ids2):
        m, E, k = self.mode, self.E, self.k
        N, d = x.shape
        sets = self.sets[ids]
        lg = x @ self.k_gate.t()
        if m == "g_sq":
            lg = lg + (x * x) @ self.k_gq
        elif m == "g_bias_e":
            lg = lg + self.k_gb
        key = ("x8lg", self.layer)
        if m == "hc_route":
            pl = self.ctx.get(("x8lg", self.layer - 1)) if self.layer else None
            self.ctx[key] = lg
            if pl is not None and pl.shape == lg.shape:
                lg = lg + self.k_hr * pl
        if m in ("c_temp", "c_soft"):
            lg = lg * self.k_tau.exp()
        elif m == "c_temp_tok":
            lg = lg * (x @ self.k_u).exp()[:, None]
        self._sets = sets
        if m == "x0_in" and self.layer == 0:
            self.ctx["x8r0"] = self.ctx["resid"]
        if m in ("norm_feat", "norm_gate"):
            self._lrv = torch.log(self.ctx["resid"].reshape(N, d).pow(2).mean(-1).add(1e-12).sqrt())
        if m in ("xl_w1", "xl_w2") and self.layer == 0:
            self.ctx[("x8xl", m)] = self.k_W1 if m == "xl_w1" else self.k_W2
        if m == "c_soft":
            g = torch.softmax(lg.gather(1, sets), -1)
        else:
            g = torch.sigmoid(lg).gather(1, sets)
        if m == "c_lin":
            g = g * (1 + (x @ self.k_sv.t()).gather(1, sets))
        if m == "d_drop" and self.training:
            drop = (torch.rand(N, 1, device=x.device) < 0.25) & \
                   (torch.arange(k, device=x.device)[None] == torch.randint(0, k, (N, 1), device=x.device))
            g = g.masked_fill(drop, 0.0)
        if m == "g_equal":
            g = torch.ones_like(g)
        elif m == "g_rank":
            rk = g.argsort(-1, descending=True).argsort(-1)
            g = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.25, 0.125], device=x.device, dtype=g.dtype)[rk]
        if m == "tok_gate":
            g = g * (1 + torch.tanh(self.k_tt[ids % 509].gather(1, sets)))
        elif m == "tok_both":
            g = g * (1 + torch.tanh(self.k_tt3[ids % 509].gather(1, sets)))
        elif m == "g_tanh":
            g = torch.tanh(lg).gather(1, sets)
        elif m == "g_zsum":
            g = g - self.k_gz * g.mean(-1, keepdim=True)
        elif m == "g_sign":
            g = g * torch.tanh(self.k_gsn)[sets]
        if m in SIGNED_G:
            g = g / (g.abs().sum(-1, keepdim=True) + 1e-6) * 2.5
        else:
            g = g / (g.sum(-1, keepdim=True) + 1e-20) * 2.5
        G = torch.zeros(N, E, device=x.device, dtype=x.dtype).scatter(1, sets, g.to(x.dtype))
        M = torch.zeros(N, E, device=x.device, dtype=x.dtype).scatter(1, sets, 1.0)
        self._G, self._lg, self._ids = G, lg, ids
        if m == "ent_in":
            pr = g / 2.5
            self._ent = -(pr * pr.clamp_min(1e-9).log()).sum(-1)
        extra = 0
        if m in ("hc_sh", "hc_all"):
            prev = [self.ctx.get(("x8hs", j)) for j in range(self.layer)]
            prev = [p for p in prev if p is not None and p.shape[0] == N]
            if m == "hc_sh":
                prev = prev[-1:]
            if prev:
                extra = sum(prev) @ self.k_Hc
        hsh = self.shared_h(x, extra)
        if m in ("hc_sh", "hc_all"):
            self.ctx[("x8hs", self.layer)] = hsh
        sh = hsh @ self.k_s2
        if m == "it2_sh":
            sh = self.shared_h(x + self.k_ic * sh) @ self.k_s2
        xin = x + self.k_di * rms(sh) if m == "ksum" else self.xin(x)
        O = self.experts(xin, x, M, hsh, ids2)
        if m == "o_gs":
            mo = F.normalize(torch.einsum("end,ne->nd", O, M), dim=-1)[None]
            O = O - self.k_og * (O * mo).sum(-1, keepdim=True) * mo
        elif m == "o_orth_e":
            sn = F.normalize(sh, dim=-1)[None]
            O = O - self.k_og * (O * sn).sum(-1, keepdim=True) * sn
        elif m == "g_from_a":
            G = G * (1 + torch.tanh(self.k_ga * self._amean)).t()
            G = G / (G.sum(-1, keepdim=True) + 1e-20) * 2.5
        if m == "rot_e":
            O = self.rot(O, torch.einsum("nd,edr->enr", x, self.k_rwe))
        elif m == "rh_to_sh":
            sh = self.shared_h(x, (torch.einsum("enh,ne->nh", self._H, M) / k) @ self.k_rs) @ self.k_s2
        if m == "it2":
            O = self.experts(x[None] + self.k_ic * O, x, M, hsh, ids2)
        if m in ("o_rotx", "o_rotx_full", "o_proj_e"):
            xh = F.normalize(x, dim=-1)[None]
            c = 1.0 if m == "o_rotx_full" else self.k_oc
            O = O - c * (O * xh).sum(-1, keepdim=True) * xh
        elif m in ("o_norm", "o_norm_v", "o_norm_kan"):
            O = rms(O) * self.k_on
        elif m == "o_gate":
            O = O * 2 * torch.sigmoid(O @ self.k_Wg)
        elif m == "o_clip":
            t = self.k_t.exp()
            O = t * torch.tanh(O / t)
        elif m == "m_mix":
            O = O + torch.einsum("ef,fnd->end", self.k_mA, O * M.t()[:, :, None])
        if m == "g_out":
            g2 = torch.sigmoid(lg.gather(1, sets) + (O @ self.k_u).t().gather(1, sets))
            g2 = g2 / (g2.sum(-1, keepdim=True) + 1e-20) * 2.5
            G = torch.zeros(N, E, device=x.device, dtype=x.dtype).scatter(1, sets, g2.to(x.dtype))
        elif m == "g_self":
            G = G * (1 + torch.tanh(self.k_s * torch.log(O.pow(2).mean(-1).add(1e-12).sqrt()))).t()
        elif m == "g_agree":
            mo = torch.einsum("end,ne->nd", O, M) / k
            G = G * (1 + self.k_s * F.cosine_similarity(O, mo[None], dim=-1)).t()
        yr = torch.einsum("end,ne->nd", O, G)
        if m in ("mh_out", "mh_out2"):                # output channel group j combines its own hash set of experts
            ng = 4 if m == "mh_out" else 2
            sig, q = torch.sigmoid(lg), d // ng
            parts = [yr[:, :q]]
            for j in range(ng - 1):
                sj = self.sets2[j][ids]
                gj = sig.gather(1, sj)
                gj = gj / (gj.sum(-1, keepdim=True) + 1e-20) * 2.5
                Gj = torch.zeros(N, E, device=x.device, dtype=x.dtype).scatter(1, sj, gj.to(x.dtype))
                parts.append(torch.einsum("end,ne->nd", O[..., (j + 1) * q:(j + 2) * q], Gj))
            yr = torch.cat(parts, -1)
        if m in ("c_max", "c_attn", "c_prod"):
            Or = (O * G.t()[:, :, None]).permute(1, 0, 2).gather(1, sets[:, :, None].expand(-1, -1, d))  # [N,k,d]
            if m == "c_max":
                yr = yr + self.k_s * (Or * k).logsumexp(1) / k
            elif m == "c_attn":
                at = torch.softmax((Or @ self.k_aq) @ (Or @ self.k_ak).transpose(1, 2) / math.sqrt(8), -1)
                yr = yr + self.k_s * (at @ Or).sum(1)
            else:
                pr = Or[:, 0] * Or[:, 1] + Or[:, 0] * Or[:, 2] + Or[:, 1] * Or[:, 2] if k == 3 else Or[:, 0] * Or[:, 1]
                yr = yr + self.k_s * pr
        elif m == "o_tokgate":
            yr = yr * 2 * torch.sigmoid(x @ self.k_u)[:, None]
        elif m == "o_clip_sum":
            t = self.k_s.exp()
            yr = t * torch.tanh(yr / t)
        if m == "rot_out":
            yr = self.rot(yr, x @ self.k_rw)
        elif m in ("out_spline", "out_spline8"):
            yr = yr + (F.relu(rms(yr)[..., None] - self.sknots) * self.k_isp).sum(-1)
        elif m == "hh_refl":
            v = F.normalize(x @ self.k_hv, dim=-1)
            yr = yr - self.k_s * 2 * (yr * v).sum(-1, keepdim=True) * v
        elif m == "o_sq":
            yr = yr + self.k_osq * rms(yr) * yr
        elif m == "gate_sh":
            yr = yr * (1 + torch.tanh(hsh @ self.k_gs))
        elif m == "perm_mix":
            yr = yr + (x @ self.k_sg) * yr[:, self.perm]
        elif m in ("outer_out", "outer16"):
            yr = yr + ((x @ self.k_ou) * (yr @ self.k_ov)) @ self.k_ow
        elif m == "yx_glu":
            yr = yr * (1 + torch.tanh(x @ self.k_sg))
        elif m == "cmul_out":
            yr = (1 + (x @ self.k_rm)).repeat_interleave(2, -1) * self.rot(yr, x @ self.k_rw)
        elif m in ("peri_e", "peri_both"):
            yr = rms(yr) * self.k_pg
        elif m == "conf_out":
            yr = yr * (1 + torch.tanh(self.k_cf * (4 * torch.sigmoid(lg).gather(1, sets).amax(-1, keepdim=True) - 2)))
        elif m in ("kv_out", "kv_out32"):
            yr = yr + torch.softmax(rms(yr) @ self.k_kok.t(), -1) @ self.k_kov
        elif m == "o_orth_sh":
            sn = F.normalize(sh, dim=-1)
            yr = yr - self.k_og * (yr * sn).sum(-1, keepdim=True) * sn
        elif m == "acorr_o":
            fy = torch.fft.rfft(yr.float(), dim=-1)
            yr = yr + self.k_ac * rms(torch.fft.irfft(fy.real ** 2 + fy.imag ** 2, n=d, dim=-1)).to(yr.dtype)
        elif m == "o_mlp":
            yr = yr + F.silu(rms(yr) @ self.k_oA) @ self.k_oB
        if m == "sh_gate_r":
            sh = sh * (1 + torch.tanh(rms(yr) @ self.k_sg))
        elif m == "peri_sh":
            sh = rms(sh) * self.k_pg
        elif m == "peri_both":
            sh = rms(sh) * self.k_pg2
        elif m == "pc_tanh":
            yr = yr + self.k_pc * torch.tanh(rms(yr))
        elif m == "pc_glu":
            yr = yr * (1 + torch.tanh(rms(yr) @ self.k_pW))
        elif m in ("out_top25", "out_top50"):
            kk = d // (4 if m == "out_top25" else 2)
            yr = yr * (yr.abs() >= yr.abs().topk(kk, -1).values[:, -1:]).to(yr.dtype)
        if m == "sr_balance":
            t = torch.tanh(x @ self.k_sb)[:, None]
            y = yr * (1 + t) + sh * (1 - t)
        else:
            y = yr + sh
        if m in ("tok_tab2", "tok_sig", "tok_both"):
            y = y + self.k_tt[ids % 509]
        if m == "tok_tab2":
            y = y + self.k_tt2[ids % 1021]
        elif m == "tok_tabg":
            y = y + self.k_tt[ids % 509] * 2 * torch.sigmoid(x @ self.k_tg)[:, None]
        elif m == "tok_sig":
            y = y + torch.sigmoid(lg) @ self.k_so
        elif m == "x_signbits":
            y = y + self.k_lt[((x[:, :8] > 0).long() * self.pw).sum(-1)]
        elif m == "hs_gate":
            hn = torch.einsum("en,ne->n", torch.log(self._H.pow(2).mean(-1).add(1e-12).sqrt()), M) / k
            y = y * (1 + torch.tanh(self.k_hg * hn[:, None]))
        elif m == "hpos_gate":
            p = torch.einsum("en,ne->n", self._apos, M) / k
            y = y * (1 + torch.tanh(self.k_hg * (4 * p[:, None] - 2)))
        if m == "xr_prod":
            y = y + self.k_xp * rms(yr) * rms(sh)
        elif m == "tok_tab":
            y = y + self.k_tt[ids % 509]
        elif m == "tok_scale":
            y = y * (1 + self.k_tt[ids % 509])
        elif m == "sig_out":
            y = y + torch.sigmoid(lg) @ self.k_so
        elif m == "y_carry":
            pv = self.ctx.get(("x8y", self.layer - 1)) if self.layer else None
            self.ctx[("x8y", self.layer)] = y
            if pv is not None and pv.shape == y.shape:
                y = y + self.k_yc * pv
        if m in ("lsh_tab", "lsh_tab2"):
            for i in range(self.k_lt.shape[0]):
                y = y + self.k_lt[i][((x @ self.lR[i] > 0).long() * self.pw).sum(-1)]
        elif m == "kan_sh":
            u = x @ self.k_kP
            y = y + (F.relu(u[..., None] - self.knots) * self.k_kc).sum(-1) @ self.k_kR
        if m == "rot_y":
            y = self.rot(y, x @ self.k_rw)
        elif m == "norm_gate":
            y = y * (1 + torch.tanh(self.k_ng * self._lrv[:, None]))
        elif m == "align_gate":
            y = y * (1 + torch.tanh(self.k_ag * F.cosine_similarity(y, x, dim=-1)[:, None]))
        elif m == "attn_mul":
            y = y * (1 + torch.tanh(rms(self.ctx["attn"].reshape(x.shape)) @ self.k_am))
        if m in ("o_peri", "kan_peri") or self.peri:
            y = rms(y) * self.k_pg
        elif m == "o_dyt":
            y = torch.tanh(self.k_al.exp() * y) * self.k_pg
        elif m == "o_ln":
            y = F.layer_norm(y, (d,)) * self.k_pg
        elif m == "peri_tok":
            y = rms(y) * self.k_pg * (1 + torch.tanh(x @ self.k_pt))
        elif m == "peri_tokh":
            y = rms(y) * (self.k_pg + self.k_tt[ids % 509])
        if m in ("rf", "rf512", "rf_pm"):
            y = y + F.relu(x @ self.rfW) @ self.k_rfO
        elif m == "rf_sin":
            y = y + torch.sin(x @ self.rfW + self.rfb) @ self.k_rfO
        elif m == "p_pos_out":
            y = y * (1 + self.pe(ids2) @ self.k_Wp)
        return y


def make(name, d, cfg, layer, ctx):
    if name.startswith("x8_"):
        return X8(d, cfg, layer, ctx, name)
    if name.startswith("x9_"):
        return X9(d, cfg, layer, ctx, name)
    if name.startswith("x7_"):
        return X7(d, cfg, layer, ctx, name)
    if name.startswith("x5_"):
        return X5(d, cfg, layer, ctx, name)
    if name.startswith("hm4_"):
        return HM4(d, cfg, layer, ctx, name)
    if name.startswith("x4_"):
        return X4(d, cfg, layer, ctx, name)
    if name in IDLIN3:
        return IdLin3(d, cfg, layer, ctx, name)
    if name in DENSE3:
        return Dense3(d, cfg, layer, ctx, name)
    if name in COMBO3:
        return Combo3(d, cfg, layer, ctx, name)
    if name == "linexp_bi":
        return LinExpBi(d, cfg, layer, ctx)
    if name in IDLIN:
        return IdLin(d, cfg, layer, ctx, name)
    if name in WIDE:
        return Wide(d, cfg, layer, ctx, name)
    if name in HASHX:
        return HashX(d, cfg, layer, ctx, name)
    if name.startswith("linexp"):
        return LinExp(d, cfg, layer, ctx)
    if name in ("hashmoe", "sqrelu", "cubicexp", "bilin", "finehash", "iter", "dualhash", "expemb",
                "idscale", "modemb", "depth2"):
        return HashFrame(d, cfg, layer, ctx, name)
    if name in ("hmask", "hmask_prev", "hmask_freq", "hmask_x", "overlap", "hmask_bilin"):
        return HMask(d, cfg, layer, ctx, name)
    if name in ("hgate", "hbias", "idglu", "idglu2", "rank1id", "rank4id", "hyperid", "outmod", "stackmem"):
        return IdDense(d, cfg, layer, ctx, name)
    if name in ("memid", "membi", "memtri", "memgate", "twotable", "memx", "pkm", "codebook", "softmaxffn"):
        return Mem(d, cfg, layer, ctx, name)
    if name in ("kron", "kron_id"):
        return Kron(d, cfg, layer, ctx, name)
    return dict(sharedw1=SharedW1, compose=Compose, nroute=NRoute, linexp=LinExp, peer=Peer, idmerge=IdMerge, stem_full=StemFull)[name](d, cfg, layer, ctx)


# ---------------- round 3, exploit: what the ID-linear gain depends on ----------------
class IdLin3(C):
    """SwiGLU(x) + per-id low-rank map, round 3 variants. ca = rows R, cb = rank r (or lr mult for idlin_lr)."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, m = mode, mode
        V, sd, L = self.V, self.seed, cfg.n_layers
        R, r = self.ca or 128, int(self.cb) or 4
        h = cfg.dense_hidden
        self.c = 1.0
        if m == "idlin_lr":
            self.c, r = float(self.cb) or 3.0, 4
        if m == "idlin_dense64":
            h = cfg.dense_hidden // 2
            R = self.ca or 176
        if m == "idlin_l0only":
            R = (self.ca or 128) * 3 if layer == 0 else 0
        if m == "idlin_lastonly":
            R = (self.ca or 128) * 3 if layer == L - 1 else 0
        if m in ("idlin_bisep", "idlin_twosep"):
            R = (self.ca or 128) // 2
        self.R, self.r, self.h = R, r, h
        if m == "idlin_hmoebi":
            self.moe = HMoE(d, V, 8, cfg.expert_hidden, 2, sd + 5, "swiglu", L)
            self.sh = SG(d, cfg.shared_hidden)
        else:
            self.ff = SG(d, h)
        for i in range(1, 4):
            self.register_buffer(f"t{i}", idtab(V, max(R, 1), sd + i), persistent=False)
        din = h if m == "idlin_hid" else d
        if m == "lin_shared":
            k = self.ca or 64
            self.k_A, self.k_B = P(d, k, std=0.1), nn.Parameter(torch.zeros(k, d))
            return
        if m == "idlin_core":
            self.k_A, self.k_B = P(d, 16, std=0.1), nn.Parameter(torch.zeros(16, d))
            self.k_C = nn.Parameter(torch.eye(16).repeat(R, 1, 1) + torch.randn(R, 16, 16) * 0.1)
            return
        if m == "idlin_vecgate":
            R = self.ca or 512
            self.register_buffer("t1", idtab(V, R, sd + 1), persistent=False)
            self.k_A, self.k_B = P(d, 64, std=0.1), nn.Parameter(torch.zeros(64, d))
            self.k_g = nn.Parameter(torch.ones(R, 64))
            return
        if R == 0:
            return
        if m == "idlin_shareW":
            self.k_Ws = P(d, r, std=0.1)
            self.k_U = nn.Parameter(torch.zeros(R, r, d))
            return
        if m == "idlin_shareU":
            self.k_W, self.k_Us = P(R, d, r, std=0.1), nn.Parameter(torch.zeros(r, d))
            return
        if m == "idlin_sharedbi":
            key = "_idlin3_shared"
            if key not in ctx:
                ctx[key] = (nn.Parameter(torch.zeros(R, r, d)), P(R, d, r, std=0.1))
            self.k_U, self.k_W = ctx[key]
            return
        self.k_U, self.k_W = nn.Parameter(torch.zeros(R, r, d)), P(R, din, r, std=0.1 / self.c)
        if m == "idlin_initU":
            self.k_U = P(R, r, d, std=0.02)
        if m in ("idlin_bisep", "idlin_twosep"):
            self.k_U2, self.k_W2 = nn.Parameter(torch.zeros(R, r, d)), P(R, d, r, std=0.1)
        if m == "idquad":
            self.k_W2 = P(R, d, r, std=0.3)
        if m == "idlin_bias":
            self.k_b = nn.Parameter(torch.zeros(R, d))
        if m == "idlin_bigate":
            self.k_g = nn.Parameter(torch.zeros(d))
        if m == "idlin_posbi":
            self.k_Up, self.k_Wp = nn.Parameter(torch.zeros(64, r, d)), P(64, d, r, std=0.1)
        if m == "idlin_soft2":
            self.k_s = nn.Parameter(torch.zeros(V))

    def lin(self, x, t, U=None, W=None):
        U = self.k_U if U is None else U
        W = self.k_W if W is None else W
        if self.mode == "idlin_ortho":
            W = F.normalize(W, dim=1)
        z = torch.einsum("nd,ndr->nr", x, W[t]) * self.c
        return torch.einsum("nr,nrd->nd", z, U[t]) * self.c

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "idlin_hmoebi":
            p = self.prev(ids2)
            return self.moe(x, ids) + self.sh(x) + self.lin(x, self.t1[ids]) + self.lin(x, self.t2[p])
        y0 = self.ff(x)
        if m == "lin_shared":
            return y0 + (x @ self.k_A) @ self.k_B
        if m == "idlin_core":
            z = torch.einsum("nk,nkj->nj", x @ self.k_A, self.k_C[self.t1[ids]])
            return y0 + z @ self.k_B
        if m == "idlin_vecgate":
            return y0 + ((x @ self.k_A) * self.k_g[self.t1[ids]]) @ self.k_B
        if self.R == 0:
            return y0
        t, p = self.t1[ids], None
        if m in ("idlin_tri", "idlin_bigate", "idlin_sharedbi", "idlin_bisep", "idlin_freqbi", "idlin_bipair"):
            p = self.prev(ids2)
        if m == "idlin_freqbi":
            fr = self.ctx["freq_rank"][ids]
            own = self.R - 64
            t = torch.where(fr < own, fr, own + self.t3[ids] % 64)
        if m == "idlin_shareW":
            return y0 + torch.einsum("nr,nrd->nd", x @ self.k_Ws, self.k_U[t])
        if m == "idlin_shareU":
            return y0 + torch.einsum("nd,ndr->nr", x, self.k_W[t]) @ self.k_Us
        if m == "idlin_embin":
            return y0 + self.lin(self.ctx["emb"].reshape(-1, self.d), t)
        if m == "idlin_siluin":
            return y0 + self.lin(F.silu(x), t)
        if m == "idlin_onff":
            return y0 + self.lin(y0, t)
        if m == "idlin_hid":
            ff = self.ff
            hh = F.silu(x @ ff.w1) * (x @ ff.w3)
            return hh @ ff.w2 + self.lin(hh, t)
        if m == "idquad":
            z = torch.einsum("nd,ndr->nr", x, self.k_W[t]) * torch.einsum("nd,ndr->nr", x, self.k_W2[t])
            return y0 + torch.einsum("nr,nrd->nd", z, self.k_U[t])
        y = self.lin(x, t)
        if m == "idlin_tri":
            y = y + self.lin(x, self.t2[p]) + self.lin(x, self.t3[self.prev(ids2, 2)])
        elif m in ("idlin_sharedbi", "idlin_freqbi"):
            y = y + self.lin(x, self.t2[p])
        elif m == "idlin_bipair":
            y = y + self.lin(x, (p * 1000003 + ids * 7919) % self.R)
        elif m == "idlin_bigate":
            y = y + self.lin(x, self.t2[p]) * 2 * torch.sigmoid(x @ self.k_g)[:, None]
        elif m == "idlin_three":
            y = y + self.lin(x, self.t2[ids]) + self.lin(x, self.t3[ids])
        elif m == "idlin_bisep":
            y = y + self.lin(x, self.t2[p], self.k_U2, self.k_W2)
        elif m == "idlin_twosep":
            y = y + self.lin(x, self.t2[ids], self.k_U2, self.k_W2)
        elif m == "idlin_soft2":
            a = torch.sigmoid(self.k_s[ids])[:, None] * 2
            y = a * y + (2 - a) * self.lin(x, self.t2[ids])
        elif m == "idlin_posbi":
            B, T = ids2.shape
            pos = torch.arange(T, device=x.device).repeat(B) % 64
            y = y + self.lin(x, pos, self.k_Up, self.k_Wp)
        elif m == "idlin_bias":
            y = y + self.k_b[t]
        elif m == "idlin_tanh":
            y = torch.tanh(y)
        elif m == "idlin_drop" and self.training:
            y = y * (torch.rand(y.shape[0], 1, device=y.device) > 0.1).float() / 0.9
        if m == "idlin_ffnmul":
            return y0 * (1 + y)
        return y0 + y


class LinExpBi(C):
    """linexp + one more linear expert picked by the previous token id."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        E, r = 64, self.ca or 8
        self.ff = SG(d, cfg.dense_hidden)
        self.k_U, self.k_W = nn.Parameter(torch.zeros(E, r, d)), P(E, d, r, std=0.1)
        self.register_buffer("sets", ksets(self.V, E, 2, self.seed), persistent=False)
        self.register_buffer("sets2", ksets(self.V, E, 1, self.seed + 1), persistent=False)

    def f(self, x, ids, ids2):
        s = torch.cat([self.sets[ids], self.sets2[self.prev(ids2)]], 1)
        z = torch.einsum("nd,nkdr->nkr", x, self.k_W[s])
        return self.ff(x) + torch.einsum("nkr,nkrd->nd", z, self.k_U[s])


# ---------------- round 3, explore: dense function shapes and zero-parameter id tricks ----------------
class Dense3(C):
    """dense blocks at ~the SwiGLU-128 + map budget. geglu / reglu / relu2_dense / bilin_dense / maxout_dense:
    activation shape at width ca; topk_dense: SwiGLU-512 keeping the top ca units; normffn: hidden RMS-normed;
    mixact: half silu-gated half relu^2 units; sinffn: sin(W1 x) * W3 x; pattention: TokenFormer-style
    gelu(norm(x K^T)) V over 512 keys; kanffn: 8 RBF bases per channel -> linear, + SwiGLU-96; hashnet: SwiGLU-512
    whose W1, W3 entries are hashed into 32k shared weights; xlshare: one SwiGLU-384 shared by all layers + own
    SwiGLU-32; ffn2x: the same SwiGLU applied twice; tiedw13: W3 = W1 with columns rolled by 1, width 192;
    ratffn: learned rational activation (deg 3/2) gated; softhid: softmax over hidden units x width;
    complexffn: 32 complex channels, complex linear -> modReLU -> complex linear; lowW2_wide: SwiGLU-256 with W2
    rank 32. idsign / idperm / idnorm: SwiGLU-128 whose input is x * fixed random signs per id / x rolled by
    hash(id) / x * (1 + g[id]) learned."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, m = mode, mode
        s, V = self.seed, self.V
        w = self.ca or 0
        if m in ("geglu", "reglu", "bilin_dense"):
            self.ff = SG(d, w or 160)
        if m == "relu2_dense":
            self.w1, self.w2 = P(d, w or 256), P(w or 256, d)
        if m == "maxout_dense":
            self.w1, self.w3, self.w4, self.w2 = P(d, w or 128), P(d, w or 128), P(d, w or 128), P(w or 128, d)
        if m == "topk_dense":
            self.ff = SG(d, 512)
        if m in ("normffn", "softhid", "sinffn", "idsign", "idperm", "ratffn"):
            self.ff = SG(d, w or 160)
        if m == "mixact":
            self.ff = SG(d, (w or 160) // 2)
            self.r1, self.r2 = P(d, (w or 160) // 2 * 3 // 2), P((w or 160) // 2 * 3 // 2, d)
        if m == "pattention":
            n = w or 512
            self.k_K, self.k_V = P(n, d, std=1.0 / math.sqrt(d)), P(n, d)
            self.ff = SG(d, 64)
        if m == "kanffn":
            self.register_buffer("cent", torch.linspace(-2, 2, 8), persistent=False)
            self.k_Wk = nn.Parameter(torch.zeros(d * 8, d))
            self.ff = SG(d, 96)
        if m == "hashnet":
            self.k_pool = P(32768, std=0.04)
            g = gen(s)
            self.register_buffer("h1", torch.randint(0, 32768, (d, 512), generator=g), persistent=False)
            self.register_buffer("h3", torch.randint(0, 32768, (d, 512), generator=g), persistent=False)
            self.register_buffer("sg1", torch.randint(0, 2, (d, 512), generator=g).float() * 2 - 1, persistent=False)
            self.w2 = P(512, d)
        if m == "xlshare":
            key = "_xlshare"
            if key not in ctx:
                ctx[key] = SG(d, w or 384)
            self.shared = ctx[key]
            self.own = SG(d, 32)
        if m == "ffn2x":
            self.ff = SG(d, w or 160)
        if m == "tiedw13":
            self.w1, self.w2 = P(d, w or 192), P(w or 192, d)
        if m == "ratffn":
            self.k_p = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 0.0]))
            self.k_q = nn.Parameter(torch.tensor([0.0, 0.0]))
        if m == "complexffn":
            hc = w or 96
            self.ar, self.ai = P(d // 2, hc, std=0.1), P(d // 2, hc, std=0.1)
            self.br, self.bi = P(hc, d // 2), P(hc, d // 2)
            self.k_bias = nn.Parameter(torch.zeros(hc))
        if m == "lowW2_wide":
            hw = w or 256
            self.w1, self.w3 = P(d, hw), P(d, hw)
            self.a2, self.b2 = P(hw, 32), P(32, d)
        if m == "idsign":
            self.register_buffer("sgn", (torch.randint(0, 2, (V, d), generator=gen(s)) * 2 - 1).float(), persistent=False)
        if m == "idperm":
            self.register_buffer("shift", torch.randint(0, d, (V,), generator=gen(s)), persistent=False)
        if m == "idnorm":
            self.ff = SG(d, cfg.dense_hidden)
            self.k_g = nn.Parameter(torch.zeros(1024, d))
            self.register_buffer("tab", idtab(V, 1024, s), persistent=False)

    def f(self, x, ids, ids2):
        m = self.mode
        if m in ("geglu", "reglu", "bilin_dense", "normffn", "softhid", "sinffn", "ratffn", "idsign", "idperm", "ffn2x"):
            ff = self.ff
            if m == "idsign":
                x = x * self.sgn[ids]
            if m == "idperm":
                N, d = x.shape
                idx = (torch.arange(d, device=x.device)[None] + self.shift[ids][:, None]) % d
                x = x.gather(1, idx)
            z1, z3 = x @ ff.w1, x @ ff.w3
            if m == "ffn2x":
                y1 = (F.silu(z1) * z3) @ ff.w2
                x2 = rms(x + y1)
                return y1 + (F.silu(x2 @ ff.w1) * (x2 @ ff.w3)) @ ff.w2
            a = {"geglu": F.gelu, "reglu": F.relu, "bilin_dense": lambda z: z, "sinffn": torch.sin}.get(m, F.silu)(z1)
            if m == "ratffn":
                p, q = self.k_p, self.k_q
                a = (p[0] + p[1] * z1 + p[2] * z1 ** 2 + p[3] * z1 ** 3) / (1 + (q[0] * z1).abs() + (q[1] * z1 ** 2).abs())
            hh = a * z3
            if m == "normffn":
                hh = rms(hh)
            if m == "softhid":
                hh = F.softmax(hh, -1) * hh.shape[-1] * hh.detach().abs().mean(-1, keepdim=True)
            return hh @ ff.w2
        if m == "idnorm":
            return self.ff(x * (1 + self.k_g[self.tab[ids]]))
        if m == "relu2_dense":
            return F.relu(x @ self.w1).square() @ self.w2
        if m == "maxout_dense":
            return torch.maximum(torch.maximum(x @ self.w1, x @ self.w3), x @ self.w4) @ self.w2
        if m == "topk_dense":
            ff = self.ff
            hh = F.silu(x @ ff.w1) * (x @ ff.w3)
            k = self.ca or 128
            v, i = hh.abs().topk(k, -1)
            return torch.zeros_like(hh).scatter(1, i, hh.gather(1, i)) @ ff.w2
        if m == "mixact":
            return self.ff(x) + F.relu(x @ self.r1).square() @ self.r2
        if m == "pattention":
            sc = x @ self.k_K.t()
            sc = F.gelu(sc / (sc.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(sc.shape[-1]))
            return sc @ self.k_V + self.ff(x)
        if m == "kanffn":
            b = torch.exp(-((x[:, :, None] - self.cent) * 2) ** 2).reshape(x.shape[0], -1)
            return b @ self.k_Wk + self.ff(x)
        if m == "hashnet":
            w1, w3 = self.k_pool[self.h1] * self.sg1, self.k_pool[self.h3]
            return (F.silu(x @ w1) * (x @ w3)) @ self.w2
        if m == "xlshare":
            return self.shared(x) + self.own(x)
        if m == "tiedw13":
            z = x @ self.w1
            return (F.silu(z) * z.roll(1, -1)) @ self.w2
        if m == "complexffn":
            xr, xi = x[:, ::2], x[:, 1::2]
            hr, hi = xr @ self.ar - xi @ self.ai, xr @ self.ai + xi @ self.ar
            mag = torch.sqrt(hr ** 2 + hi ** 2 + 1e-6)
            sc = F.relu(mag + self.k_bias) / mag
            hr, hi = hr * sc, hi * sc
            yr, yi = hr @ self.br - hi @ self.bi, hr @ self.bi + hi @ self.br
            return torch.stack([yr, yi], -1).reshape(x.shape[0], -1)
        if m == "lowW2_wide":
            return ((F.silu(x @ self.w1) * (x @ self.w3)) @ self.a2) @ self.b2
        raise ValueError(m)


class Combo3(C):
    """ID-linear (rank 4, ca rows) added to another body: maxout_idlin: hash MoE with maxout experts (E16 w32 k3,
    ca default 64) + shared; idsign_hmoe: hash MoE whose experts see x * random signs of the id (no map);
    geglu_idlin / relu2_idlin: GeGLU-128 / relu^2-192 + map."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, m = mode, mode
        V, s, L = self.V, self.seed, cfg.n_layers
        if m in ("maxout_idlin", "idsign_hmoe"):
            fn = "maxout" if m == "maxout_idlin" else "swiglu"
            eh = cfg.expert_hidden if m == "idsign_hmoe" else 16
            self.moe = HMoE(d, V, 16, eh, 3, s, fn, L)
            self.sh = SG(d, cfg.shared_hidden)
        if m == "idsign_hmoe":
            self.register_buffer("sgn", (torch.randint(0, 2, (V, d), generator=gen(s + 1)) * 2 - 1).float(), persistent=False)
            return
        if m == "geglu_idlin":
            self.ff = SG(d, cfg.dense_hidden)
        if m == "relu2_idlin":
            self.w1, self.w2 = P(d, 192), P(192, d)
        R = self.ca or 128
        self.register_buffer("tab", idtab(V, R, s + 2), persistent=False)
        self.k_U, self.k_W = nn.Parameter(torch.zeros(R, 4, d)), P(R, d, 4, std=0.1)

    def f(self, x, ids, ids2):
        m = self.mode
        if m == "idsign_hmoe":
            return self.moe(x, ids, xin=x * self.sgn[ids]) + self.sh(x)
        t = self.tab[ids]
        y = torch.einsum("nr,nrd->nd", torch.einsum("nd,ndr->nr", x, self.k_W[t]), self.k_U[t])
        if m == "maxout_idlin":
            return self.moe(x, ids) + self.sh(x) + y
        if m == "geglu_idlin":
            ff = self.ff
            return (F.gelu(x @ ff.w1) * (x @ ff.w3)) @ ff.w2 + y
        return F.relu(x @ self.w1).square() @ self.w2 + y


IDLIN3 = ("idlin_tri", "idlin_bipair", "idlin_bigate", "idlin_three", "idlin_lr", "idlin_shareW", "idlin_shareU",
          "idlin_core", "idlin_vecgate", "idlin_dense64", "idlin_hirank", "idlin_freqbi", "idlin_hmoebi",
          "idlin_initU", "idlin_tanh", "idlin_l0only", "idlin_lastonly", "idlin_sharedbi", "idlin_embin",
          "idlin_bisep", "idlin_bias", "idlin_siluin", "idlin_ffnmul", "idlin_onff", "idlin_hid", "lin_shared",
          "idlin_ortho", "idlin_drop", "idlin_posbi", "idquad", "idlin_twosep", "idlin_soft2")
DENSE3 = ("geglu", "reglu", "relu2_dense", "bilin_dense", "maxout_dense", "topk_dense", "normffn", "mixact",
          "sinffn", "pattention", "kanffn", "hashnet", "xlshare", "ffn2x", "tiedw13", "ratffn", "softhid",
          "complexffn", "lowW2_wide", "idsign", "idperm", "idnorm")
COMBO3 = ("maxout_idlin", "idsign_hmoe", "geglu_idlin", "relu2_idlin")


# ---------------- round 4: body x map grid + normed-hidden family ----------------
def _hid_norm(h, kind, g=None):
    if kind == "n":
        return rms(h)
    if kind == "ln":
        h = h - h.mean(-1, keepdim=True)
        return rms(h)
    if kind == "gn":
        N, H = h.shape
        return rms(h.reshape(N, 8, H // 8)).reshape(N, H)
    if kind == "nd":
        return h * torch.rsqrt(h.detach().pow(2).mean(-1, keepdim=True) + 1e-6)
    if kind == "gain":
        return h * g
    if kind == "ng":
        return rms(h) * g
    return h


class X4(C):
    """name = x4_<body>_<map>. body: sg / relu2 / geglu / kan / pat SwiGLU-family hidden, optionally with a hidden
    norm suffix (sgn = rms-normed hidden, sgln = centered+normed, sggn = 8-group norm, sgnd = scale by detached rms,
    sggain = learned scalar gain init 20 (control), sgng = norm + learned per-unit gain, sgzn = rms on W1x and W3x
    before gating, tri / trin = silu(W1x) W3x W4x cubic gate, kanh = learned 8-RBF activation per unit).
    map: none / id (per-id rank-4 U W, ca rows) / sw (shared W, per-id rank-4 U, 2ca rows) / bi (id + previous id) /
    three (3 hashes of id) / freq (own rows for frequent ids) / l0 (all rows in layer 0). cb = body width."""
    BODY_W = {"sg": 128, "relu2": 192, "geglu": 128, "tri": 96, "kan": 96, "pat": 64, "kanh": 128}

    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx)
        _, body, mp = name.split("_", 2)
        self.body, self.mp = body, mp
        V, s = self.V, self.seed
        base = next(b for b in ("relu2", "geglu", "tri", "kanh", "kan", "pat", "sg") if body.startswith(b))
        self.base, self.norm = base, body[len(base):]
        w = int(self.cb) or self.BODY_W[base]
        if base == "relu2":
            self.w1, self.w2 = P(d, w), P(w, d)
        elif base == "tri":
            self.w1, self.w3, self.w4, self.w2 = P(d, w), P(d, w), P(d, w), P(w, d)
        else:
            self.w1, self.w3, self.w2 = P(d, w), P(d, w), P(w, d)
        if base == "kan":
            self.register_buffer("cent", torch.linspace(-2, 2, 8), persistent=False)
            self.k_Wk = nn.Parameter(torch.zeros(d * 8, d))
        if base == "kanh":
            self.register_buffer("cent", torch.linspace(-1, 1, 8), persistent=False)
            a0 = torch.zeros(w, 8)
            self.k_a = nn.Parameter(a0)
        if base == "pat":
            n = 512
            self.k_K, self.k_V = P(n, d, std=1.0 / math.sqrt(d)), P(n, d)
        self.g = None
        if self.norm == "gain":
            self.k_gs = nn.Parameter(torch.tensor(20.0))
        if self.norm == "ng":
            self.k_gu = nn.Parameter(torch.ones(w))
        R = self.ca or 128
        if mp == "l0":
            R = R * 3 if layer == 0 else 0
        if mp == "sw":
            R = R * 2
        self.R = R
        if mp == "none" or R == 0:
            return
        r = 4
        for i in range(1, 4):
            self.register_buffer(f"t{i}", idtab(V, R, s + 10 + i), persistent=False)
        if mp == "sw":
            self.k_Ws, self.k_U = P(d, r, std=0.1), nn.Parameter(torch.zeros(R, r, d))
            return
        nt = {"bi": 2, "three": 3}.get(mp, 1)
        R1 = R // nt if mp in ("bi", "three") else R
        self.R1 = R1
        for i in range(1, 4):
            self.register_buffer(f"t{i}", idtab(V, R1, s + 10 + i), persistent=False)
        self.k_U, self.k_W = nn.Parameter(torch.zeros(nt * R1, r, d)), P(nt * R1, d, r, std=0.1)

    def hidden(self, x):
        b, nk = self.base, self.norm
        if b == "relu2":
            h = F.relu(x @ self.w1).square()
        elif b == "tri":
            h = F.silu(x @ self.w1) * (x @ self.w3) * (x @ self.w4)
        elif b == "kanh":
            z = x @ self.w1
            phi = torch.exp(-((z[:, :, None] - self.cent) * 4) ** 2)
            h = (F.silu(z) + (phi * self.k_a).sum(-1)) * (x @ self.w3)
        else:
            z1, z3 = x @ self.w1, x @ self.w3
            if nk == "zn":
                z1, z3 = rms(z1), rms(z3)
            h = (F.gelu(z1) if b == "geglu" else F.silu(z1)) * z3
        if nk and nk != "zn":
            g = self.k_gs if nk == "gain" else (self.k_gu if nk == "ng" else None)
            h = _hid_norm(h, nk, g)
        return h

    def lin(self, x, t):
        z = torch.einsum("nd,ndr->nr", x, self.k_W[t])
        return torch.einsum("nr,nrd->nd", z, self.k_U[t])

    def f(self, x, ids, ids2):
        y = self.hidden(x) @ self.w2
        if self.base == "kan":
            y = y + torch.exp(-((x[:, :, None] - self.cent) * 2) ** 2).reshape(x.shape[0], -1) @ self.k_Wk
        if self.base == "pat":
            sc = x @ self.k_K.t()
            y = y + F.gelu(sc / (sc.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(sc.shape[-1])) @ self.k_V
        mp = self.mp
        if mp == "none" or self.R == 0:
            return y
        if mp == "sw":
            return y + torch.einsum("nr,nrd->nd", x @ self.k_Ws, self.k_U[self.t1[ids]])
        if mp == "freq":
            fr = self.ctx["freq_rank"][ids]
            own = self.R - 64
            t = torch.where(fr < own, fr, own + self.t2[ids] % 64)
            return y + self.lin(x, t)
        y = y + self.lin(x, self.t1[ids])
        if mp == "bi":
            y = y + self.lin(x, self.R1 + self.t2[self.prev(ids2)])
        if mp == "three":
            y = y + self.lin(x, self.R1 + self.t2[ids]) + self.lin(x, 2 * self.R1 + self.t3[ids])
        return y


class HM4(C):
    """hm4_plain: hash MoE (8 experts, k 2, expert width cfg.expert_hidden) + shared SwiGLU, as in idlin_hmoebi without
    maps (control for this code path); hm4_n: the same with rms-normed hidden in experts and shared expert;
    hm4_nid: hm4_n + rank-4 id map (ca rows)."""
    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx)
        self.mode = name
        n = name != "hm4_plain"
        self.moe = HMoE(d, self.V, 8, cfg.expert_hidden, 2, self.seed + 5, "swiglu_n" if n else "swiglu", cfg.n_layers)
        self.sh = SG(d, cfg.shared_hidden)
        self.n = n
        if name == "hm4_nid":
            R = self.ca or 128
            self.register_buffer("tab", idtab(self.V, R, self.seed + 7), persistent=False)
            self.k_U, self.k_W = nn.Parameter(torch.zeros(R, 4, d)), P(R, d, 4, std=0.1)

    def f(self, x, ids, ids2):
        sh = self.sh
        hh = F.silu(x @ sh.w1) * (x @ sh.w3)
        y = self.moe(x, ids) + (rms(hh) if self.n else hh) @ sh.w2
        if self.mode == "hm4_nid":
            t = self.tab[ids]
            y = y + torch.einsum("nr,nrd->nd", torch.einsum("nd,ndr->nr", x, self.k_W[t]), self.k_U[t])
        return y


# ---------------- round 5: relu-power bodies x shared-input per-id output maps ----------------
class X5(C):
    """name = x5_<body>_<map>. body (width cb, default 192): relu2, relu3, relu15 (relu^1.5), srelu2 (silu^2),
    relu2b (relu^2 with learned per-unit threshold), deep2 (relu^2 d->128->96->d), relu2k (relu^2 keeping the top half
    of units), relu2sg (relu^2 gate times W3x, width 128), none (no body). map (sw = sum_k (w_k.x) U[row][k], rank 4,
    2ca rows): sw, swbi (half rows by id, half by previous id), swfreq (own rows for frequent ids), swk2 (2 hashed rows
    per id summed), swsh (one table shared by all layers, 6ca rows), swt (tanh on the 4 mixing weights), swn (rms on
    the mixing weights), swh (mixing weights = first 4 body hidden units), swr1 / swr2 / swr8 / swr16 (rank r at equal
    parameters), swx (sw plus per-id rank-4 input map, i.e. rank4id), mem (per-id vector, 8ca rows), none."""
    def __init__(self, d, cfg, layer, ctx, name):
        super().__init__(d, cfg, layer, ctx)
        _, body, mp = name.split("_", 2)
        self.body, self.mp = body, mp
        V, s = self.V, self.seed
        w = int(self.cb) or (128 if body == "relu2sg" else 192)
        if body == "deep2":
            self.w1, self.wm, self.w2 = P(d, 128), P(128, 96, std=0.02), P(96, d)
        elif body == "relu2sg":
            self.w1, self.w3, self.w2 = P(d, w), P(d, w), P(w, d)
        elif body != "none":
            self.w1, self.w2 = P(d, w), P(w, d)
        if body == "relu2b":
            self.k_b = nn.Parameter(torch.zeros(w))
        ca = self.ca or 128
        r = {"swr1": 1, "swr2": 2, "swr8": 8, "swr16": 16}.get(mp, 4)
        R = 2 * ca * 4 // r
        if mp == "mem":
            R = 8 * ca
        self.R, self.r = R, r
        if mp == "none":
            return
        for i in range(1, 3):
            self.register_buffer(f"t{i}", idtab(V, R if mp != "swbi" else R // 2, s + 20 + i), persistent=False)
        if mp == "mem":
            self.k_M = nn.Parameter(torch.zeros(R, d))
            return
        if mp == "swsh":
            key = "_x5_swsh"
            if key not in ctx:
                ctx[key] = nn.Parameter(torch.zeros(3 * R, r, d))
            self.k_U = ctx[key]
            self.register_buffer("t1", idtab(V, 3 * R, 7777), persistent=False)
            self.k_Ws = P(d, r, std=0.1)
            return
        self.k_Ws = P(d, r, std=0.1)
        self.k_U = nn.Parameter(torch.zeros(R, r, d))
        if mp == "swx":
            self.R = R // 2
            self.k_U = nn.Parameter(torch.zeros(R // 2, r, d))
            self.k_W = P(R // 2, d, r, std=0.1)
            self.register_buffer("t1", idtab(V, R // 2, s + 21), persistent=False)

    def hid(self, x):
        b = self.body
        if b == "deep2":
            return F.relu(F.relu(x @ self.w1).square() @ self.wm).square()
        z = x @ self.w1
        if b == "relu2":
            return F.relu(z).square()
        if b == "relu3":
            return F.relu(z) ** 3
        if b == "relu15":
            return F.relu(z) ** 1.5
        if b == "srelu2":
            return F.silu(z).square()
        if b == "relu2b":
            return F.relu(z - self.k_b).square()
        if b == "relu2sg":
            return F.relu(z).square() * (x @ self.w3)
        if b == "relu2k":
            h = F.relu(z).square()
            v, i = h.topk(h.shape[-1] // 2, -1)
            return torch.zeros_like(h).scatter(1, i, v)
        raise ValueError(b)

    def f(self, x, ids, ids2):
        y = 0
        h = None
        if self.body != "none":
            h = self.hid(x)
            y = h @ self.w2
        mp = self.mp
        if mp == "none":
            return y
        t = self.t1[ids]
        if mp == "mem":
            return y + self.k_M[t]
        if mp == "swx":
            z = torch.einsum("nd,ndr->nr", x, self.k_W[t])
        elif mp == "swh":
            z = h[:, :4]
        else:
            z = x @ self.k_Ws
        if mp == "swt":
            z = torch.tanh(z)
        if mp == "swn":
            z = rms(z)
        if mp in ("swfreq", "swk2freq", "swfreqbi", "swk2fb"):
            fr = self.ctx["freq_rank"][ids]
            own = self.R - 64
            t = torch.where(fr < own, fr, own + self.t2[ids] % 64)
        y = y + torch.einsum("nr,nrd->nd", z, self.k_U[t])
        if mp == "swbi":
            y = y + torch.einsum("nr,nrd->nd", z, self.k_U[self.R // 2 + self.t2[self.prev(ids2)]])
        if mp == "swk2":
            y = y + torch.einsum("nr,nrd->nd", z, self.k_U[self.t2[ids]])
        if mp in ("swk3", "swk4", "swk2bi", "swk2freq", "swfreqbi", "swk2fb"):
            k = {"swk3": 3, "swk4": 4, "swfreqbi": 1}.get(mp, 2)
            for i in range(2, k + 1):
                y = y + torch.einsum("nr,nrd->nd", z, self.k_U[(self.t1[ids] * (2 * i + 1) + i * 7919) % self.R])
            if mp in ("swk2bi", "swfreqbi", "swk2fb"):
                y = y + torch.einsum("nr,nrd->nd", z, self.k_U[self.t2[self.prev(ids2)]])
        return y
