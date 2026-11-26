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
        if mp in ("k2zp", "k2zprev2", "k2zpg", "k2zpb", "k2zpc", "k2zp2", "k3zp"):
            self.k_Wp = P(d, r, std=0.1)
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


def make(name, d, cfg, layer, ctx):
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
