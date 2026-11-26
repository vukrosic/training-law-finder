"""Invention search (IDEAS.md): FFN replacements that change how a token picks its computation.

Every invention keeps the dsv4 XS budget unless its IDEAS.md line says otherwise: experts of width
cfg.expert_hidden, 3 per token (cfg.topk), plus one shared expert of width cfg.shared_hidden.
Selection uses sigmoid scores plus a load bias updated by sign at rate 1e-3 (the DeepSeek
aux-free rule) so that inventions are compared on routing, not on balancing.
A shared dict `ctx` (set by GPT.forward and Block) carries: ids [B,T], attn [B,T,d] (what attention
added in this block), delta [B,T,d] (block input minus previous block input), prev_logits (router
logits of the previous layer, for ECHO).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(x, w1, w3, w2):
    return (F.silu(x @ w1) * (x @ w3)) @ w2


class Bank(nn.Module):
    """E SwiGLU experts [E,d,h]; forward(x [N,d_in], idx [N,k], g [N,k]) -> [N,d_out]."""
    def __init__(self, E, d_in, h, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        self.W1 = nn.Parameter(torch.randn(E, d_in, h) * 0.02)
        self.W3 = nn.Parameter(torch.randn(E, d_in, h) * 0.02)
        self.W2 = nn.Parameter(torch.randn(E, h, d_out) * 0.02)
        self.E = E

    def hidden(self, x, idx):                       # [N,k,h]
        return F.silu(torch.einsum("nd,nkdh->nkh", x, self.W1[idx])) * torch.einsum("nd,nkdh->nkh", x, self.W3[idx])

    def down(self, hdn, idx):                       # [N,k,d]
        return torch.einsum("nkh,nkhd->nkd", hdn, self.W2[idx])

    def forward(self, x, idx, g, pre=None):
        # dispatch loop over experts (no per-token weight gather)
        N, k = idx.shape
        y = x.new_zeros(N, self.W2.shape[-1])
        fi, fg = idx.reshape(-1), g.reshape(-1, 1).to(x.dtype)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        for e in range(self.E):
            sel = (fi == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            t = tok[sel]
            xe = x.index_select(0, t)
            if pre is not None:
                xe = pre(e, xe)
            oe = (F.silu(xe @ self.W1[e]) * (xe @ self.W3[e])) @ self.W2[e]
            y.index_add_(0, t, oe * fg[sel])
        return y


class Balancer(nn.Module):
    """sigmoid score + sign-updated bias for selection; gates = renormalized scores x 2.5."""
    def __init__(self, E, rate=1e-3, scale=2.5):
        super().__init__()
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("count", torch.zeros(E))
        self.rate, self.scale, self.E = rate, scale, E

    def pick(self, logits, k, mask=None):
        s = torch.sigmoid(logits.float())
        sel = s.detach() + self.bias
        if mask is not None:
            sel = sel.masked_fill(mask, float("-inf"))
        idx = torch.topk(sel, k, dim=-1).indices
        g = torch.gather(s, -1, idx)
        g = g / (g.sum(-1, keepdim=True) + 1e-20) * self.scale
        if self.training:
            with torch.no_grad():
                self.count += torch.bincount(idx.reshape(-1), minlength=self.E).float()
        return idx, g

    def step_update(self):
        if self.count.sum() > 0:
            self.bias += self.rate * torch.sign(self.count.mean() - self.count)
            self.count.zero_()


def gate_param(d, E):
    return nn.Parameter(torch.randn(E, d) * 0.02)


class Inv(nn.Module):
    """Base: shared expert + one routed bank; subclasses override route features or the whole flow."""
    def __init__(self, d, cfg, layer, ctx, E=None, h=None):
        super().__init__()
        self.cfg, self.layer, self.ctx = cfg, layer, ctx
        self.E = E or cfg.n_experts
        self.k = cfg.topk
        h = h or cfg.expert_hidden
        self.sh = nn.ModuleList()   # shared expert as plain weights
        self.sw1 = nn.Parameter(torch.randn(d, cfg.shared_hidden) * 0.02)
        self.sw3 = nn.Parameter(torch.randn(d, cfg.shared_hidden) * 0.02)
        self.sw2 = nn.Parameter(torch.randn(cfg.shared_hidden, d) * 0.02)
        self.d, self.h = d, h
        self.inv_aux = {}

    def shared(self, x):
        return swiglu(x, self.sw1, self.sw3, self.sw2)

    def forward(self, x):
        B, T, d = x.shape
        y = self.routed(x.reshape(-1, d), B, T)
        return y.view(B, T, d) + self.shared(x)


class Feature(Inv):
    """Plain routed bank whose router reads a feature other than x (ATTN, DELTA, LOOKAHEAD, COUNT, ECHO)."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        self.bank = Bank(self.E, d, self.h)
        self.bal = Balancer(self.E)
        self.gate = gate_param(d, self.E)
        if mode == "count":
            self.count_emb = nn.Parameter(torch.zeros(8, self.E))
        if mode == "echo":
            self.echo = nn.Parameter(torch.zeros(()))
        if mode == "grid":
            self.gr, self.gcol = gate_param(d, 4), gate_param(d, 4)
        if mode == "embin":
            self.bank = Bank(self.E, 2 * d, self.h, d_out=d)
        if mode == "product":
            self.mod = Bank(self.E, d, self.h)
            self.gmod = gate_param(d, self.E)
            self.bmod = Balancer(self.E)
        if mode == "rotated":                                 # each expert sees x through a fixed random rotation
            self.register_buffer("R", torch.linalg.qr(torch.randn(self.E, d, d, generator=torch.Generator().manual_seed(77 + layer)))[0])
        if mode == "recurse":                                 # apply the chosen experts again on x + output
            self.wr = nn.Parameter(torch.zeros(d))
            self.br = nn.Parameter(torch.tensor(-2.0))
        if mode == "pos":
            self.pos_emb = nn.Parameter(torch.zeros(8, self.E))
        if mode == "doc":
            self.gate2 = gate_param(d, self.E)
        if mode in ("copy", "inhibit"):
            self.mix = nn.Parameter(torch.ones(()))
        if mode == "hebb":
            self.keys = nn.Parameter(F.normalize(torch.randn(self.E, d), dim=-1), requires_grad=False)
            self.register_buffer("ksum", torch.zeros(self.E, d))
        if mode == "skipk":
            self.wq = nn.Parameter(torch.zeros(d))
        if mode == "cubic":
            self.Pu = nn.Parameter(torch.randn(self.E, d, d) * 0.02)
            self.Pv = nn.Parameter(torch.randn(self.E, d, self.h) * 0.02)

    def routed(self, xf, B, T):
        m = self.mode
        if m == "attn":
            f = F.rms_norm(self.ctx["attn"], (self.d,)).reshape(-1, self.d)
        elif m == "delta":
            f = F.rms_norm(self.ctx["delta"], (self.d,)).reshape(-1, self.d)
        elif m == "lookahead":
            xs = xf.view(B, T, -1)
            f = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, :-1]], 1).reshape(-1, self.d)
        else:
            f = xf
        if m == "stale":                                      # route on the previous layer's FFN input
            f = self.ctx["stale"] if self.ctx.get("stale") is not None else xf
            self.ctx["stale"] = xf
        if m == "chunk":                                      # 4-token chunks routed from the previous chunk's mean
            xs = xf.view(B, T // 4, 4, -1).mean(2)
            xs = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, :-1]], 1)
            f = xs.repeat_interleave(4, 1).reshape(-1, self.d)
        logits = f @ self.gate.t()
        if m == "grid":                                       # expert (i, j): i from x_t, j from x_{t-1}
            pv = prev_tokens(xf, B, T, 1)
            logits = ((xf @ self.gr.t())[:, :, None] + (pv @ self.gcol.t())[:, None, :]).reshape(-1, 16)
        if m == "count":
            ids = self.ctx["ids"]
            same = (ids[:, :, None] == ids[:, None, :]).tril(-1).sum(-1)          # earlier copies
            bucket = torch.clamp(torch.log2(same.float() + 1).floor().long(), max=7)
            logits = logits + self.count_emb[bucket.reshape(-1)]
        if m == "echo":
            if self.ctx.get("prev_logits") is not None:
                logits = logits + self.echo * self.ctx["prev_logits"]
            self.ctx["prev_logits"] = logits
        E = self.E
        if m == "neighbor":                                   # average with the previous token's logits
            L = logits.view(B, T, E)
            logits = (0.5 * (L + torch.cat([L[:, :1], L[:, :-1]], 1))).reshape(-1, E)
        if m == "pos":                                        # position bucket (8 per sequence) adds logits
            pb = (torch.arange(T, device=xf.device) * 8 // T).repeat(B)
            logits = logits + self.pos_emb[pb]
        if m == "doc":                                        # running mean of the sequence so far
            xs = xf.view(B, T, -1)
            cm = xs.cumsum(1) / torch.arange(1, T + 1, device=xf.device).view(1, T, 1)
            logits = logits + cm.reshape(-1, self.d) @ self.gate2.t()
        if m == "copy":                                       # reuse logits of the last earlier copy of this id
            ids = self.ctx["ids"]
            same = (ids[:, :, None] == ids[:, None, :]).tril(-1)
            pos = (same * torch.arange(1, T + 1, device=xf.device)).argmax(-1)    # 0 if none
            has = same.any(-1)
            L = logits.view(B, T, E)
            prevL = torch.gather(L, 1, pos.unsqueeze(-1).expand(B, T, E))
            logits = (L + self.mix * prevL * has.unsqueeze(-1)).reshape(-1, E)
        if m == "inhibit":                                    # penalize experts the previous layer chose
            pm = self.ctx.get("prev_map")
            if pm is not None and pm.shape[1] == E:
                logits = logits - self.mix * pm
        if m == "hebb":                                       # cosine to keys moved by a Hebbian rule
            logits = 8 * F.normalize(xf, dim=-1) @ self.keys.t()
        if m == "anti":                                       # 2 best + the lowest-scoring expert
            s_ = torch.sigmoid(logits.float())
            sel = s_.detach() + self.bal.bias
            idx = torch.cat([torch.topk(sel, self.k - 1, -1).indices, sel.argmin(-1, keepdim=True)], 1)
            g = torch.gather(s_, -1, idx)
            g = g / (g.sum(-1, keepdim=True) + 1e-20) * self.bal.scale
            if self.training:
                with torch.no_grad():
                    self.bal.count += torch.bincount(idx.reshape(-1), minlength=E).float()
        elif m in ("normk", "skipk", "attnk"):                         # variable experts per token, mean k
            idx, g = self.bal.pick(logits, self.k + 1)
            if m == "normk":                                  # residual norm tercile -> 2, 3 or 4 experts
                q = self.ctx["resid"].reshape(-1, self.d).norm(dim=-1)
            elif m == "attnk":                                # attention output norm tercile -> 2, 3 or 4
                q = self.ctx["attn"].reshape(-1, self.d).norm(dim=-1)
            else:                                             # learned score; bottom quarter skips experts
                q = xf @ self.wq
            r = q.argsort().argsort().float() / q.numel()
            if m in ("normk", "attnk"):
                nk = 2 + (r > 1 / 3).long() + (r > 2 / 3).long()
            else:
                nk = torch.where(r < 0.25, 0, 4)
                g = g * (2 * torch.sigmoid(xf @ self.wq)).unsqueeze(-1)
            g = g * (torch.arange(self.k + 1, device=xf.device) < nk.unsqueeze(-1))
        else:
            idx, g = self.bal.pick(logits, self.k)
        if m == "inhibit":
            self.ctx["prev_map"] = torch.zeros_like(logits).scatter(1, idx, 1.0)
        if m == "hebb" and self.training:
            with torch.no_grad():
                self.ksum.index_add_(0, idx.reshape(-1), F.normalize(xf, dim=-1).repeat_interleave(self.k, 0))
        if m == "cubic":
            return self.cubic(xf, idx, g)
        if m == "rotated":
            return self.bank(xf, idx, g, pre=lambda e, xe: xe @ self.R[e])
        if m == "recurse":
            out = self.bank(xf, idx, g)
            return out + torch.sigmoid(xf @ self.wr + self.br).unsqueeze(-1) * self.bank(xf + out, idx, g)
        if m == "diff":                                       # experts write the change f(x_t) - f(x_{t-1})
            return self.bank(xf, idx, g) - self.bank(prev_tokens(xf, B, T, 1), idx, g)
        if m == "embin":
            return self.bank(torch.cat([xf, self.ctx["emb"].reshape(-1, self.d)], -1), idx, g)
        if m == "product":                                    # 2 value experts times 1 modulating expert
            i2, g2 = self.bmod.pick(xf @ self.gmod.t(), 1)
            v = self.bank(xf, idx[:, :2], g[:, :2])
            return v * 2 * torch.sigmoid(self.mod(xf, i2, g2 / self.bmod.scale))
        return self.bank(xf, idx, g)

    def cubic(self, x, idx, g):
        """CUBIC-EXPERT: every routed expert carries its own HYPER rank-1 change of W1."""
        N, k = idx.shape
        y = x.new_zeros(N, self.d)
        fi, fg = idx.reshape(-1), g.reshape(-1, 1).to(x.dtype)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        bk = self.bank
        for e in range(self.E):
            sel = (fi == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            t = tok[sel]
            xe = x.index_select(0, t)
            a = xe @ bk.W1[e] + (xe * (xe @ self.Pu[e])).sum(-1, keepdim=True) * (xe @ self.Pv[e])
            y.index_add_(0, t, ((F.silu(a) * (xe @ bk.W3[e])) @ bk.W2[e]) * fg[sel])
        return y

    def step_update(self):
        if self.mode == "hebb":
            n = self.ksum.norm(dim=-1, keepdim=True)
            upd = torch.where(n > 0, self.ksum / n.clamp_min(1e-9), self.keys.data)
            self.keys.data = F.normalize(0.9 * self.keys.data + 0.1 * upd, dim=-1)
            self.ksum.zero_()


class Tree(Inv):
    """TREE: pick 1 of 4 parents; parent output picks 2 of 16 children reading x + parent output."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.par, self.chi = Bank(4, d, self.h), Bank(self.E, d, self.h)
        self.gp, self.gc = gate_param(d, 4), gate_param(d, self.E)
        self.bp, self.bc = Balancer(4), Balancer(self.E)

    def routed(self, xf, B, T):
        i1, g1 = self.bp.pick(xf @ self.gp.t(), 1)
        p = self.par(xf, i1, g1 / self.bp.scale)                  # parent output, gate in (0,1]
        z = xf + p
        i2, g2 = self.bc.pick(F.rms_norm(z, (self.d,)) @ self.gc.t(), self.k - 1)
        return p + self.chi(z, i2, g2)


class Chain(Inv):
    """CHAIN: k experts applied in sequence; each step routes on the current running state."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        self.gates = nn.ParameterList([gate_param(d, self.E) for _ in range(self.k)])
        self.bal = Balancer(self.E)

    def routed(self, xf, B, T):
        z, out, used = xf, 0, None
        for j in range(self.k):
            idx, g = self.bal.pick(F.rms_norm(z, (self.d,)) @ self.gates[j].t(), 1, mask=used)
            y = self.bank(z, idx, g / self.bal.scale)
            used = torch.zeros(xf.shape[0], self.E, dtype=torch.bool, device=xf.device) if used is None else used
            used = used.scatter(1, idx, True)
            out = out + y
            z = z + y
        return out


class DepthPool(Inv):
    """DEPTH-POOL: one bank of E x n_layers experts shared by all layers; per-layer router."""
    pools = {}

    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        key = id(ctx)
        if layer == 0:
            self.bank = Bank(self.E * cfg.n_layers, d, self.h)
            ctx["pool"] = self.bank
        else:
            object.__setattr__(self, "bank_ref", ctx["pool"])    # parameters registered once, in layer 0
        self.gate = gate_param(d, self.E * cfg.n_layers)
        self.bal = Balancer(self.E * cfg.n_layers)

    def routed(self, xf, B, T):
        bank = self.bank if self.layer == 0 else self.bank_ref
        idx, g = self.bal.pick(xf @ self.gate.t(), self.k)
        return bank(xf, idx, g)


class Assemble(Inv):
    """ASSEMBLE: 6 pieces of a width-96 SwiGLU; token sums 3 pieces' weights, then one nonlinearity."""
    def __init__(self, d, cfg, layer, ctx, P=6):
        super().__init__(d, cfg, layer, ctx)
        H = self.h * self.k
        self.bank = Bank(P, d, H)
        self.gate = gate_param(d, P)
        self.bal = Balancer(P)

    def routed(self, xf, B, T):
        idx, g = self.bal.pick(xf @ self.gate.t(), self.k)
        g = g / self.bal.scale                                         # convex weights
        w1 = torch.einsum("nk,nkdh->ndh", g, self.bank.W1[idx])
        w3 = torch.einsum("nk,nkdh->ndh", g, self.bank.W3[idx])
        w2 = torch.einsum("nk,nkhd->nhd", g, self.bank.W2[idx])
        hdn = F.silu(torch.einsum("nd,ndh->nh", xf, w1)) * torch.einsum("nd,ndh->nh", xf, w3)
        return torch.einsum("nh,nhd->nd", hdn, w2) * self.bal.scale


class Hyper(Inv):
    """HYPER: SwiGLU whose input weights get a per-token low-rank change made from x:
    W1(x) = W1 + sum_r u_r(x) v_r(x)^T, u_r = x Pu_r (d), v_r = x Pv_r (H).
    knobs: hyp_rank r; hyp_where w1 | w13 (also W3) | w3; hyp_gen lin | mlp (u, v from a silu layer);
    hyp_base dense (dense SwiGLU of width hyp_width or k*h, plus the shared expert) | moe (plain
    routed MoE; the change goes into the shared expert); hyp_off = control without the change."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        c = cfg
        self.r, self.where, self.gen, self.base = c.hyp_rank, c.hyp_where, c.hyp_gen, c.hyp_base
        self.off = c.hyp_off
        self.squash, self.use_gain = c.hyp_squash, c.hyp_gain
        if self.base == "moe":
            self.bank = Bank(self.E, d, self.h)
            self.gate = gate_param(d, self.E)
            self.bal = Balancer(self.E)
            H = c.shared_hidden
        else:
            H = c.hyp_width or self.h * self.k
            self.w1 = nn.Parameter(torch.randn(d, H) * 0.02)
            self.w3 = nn.Parameter(torch.randn(d, H) * 0.02)
            self.w2 = nn.Parameter(torch.randn(H, d) * 0.02)
        self.H = H
        self.hprev = c.hyp_prev
        if self.hprev and self.base != "moe":
            self.w1p = nn.Parameter(torch.randn(d, H) * 0.02)
            self.w3p = nn.Parameter(torch.randn(d, H) * 0.02)
        if self.gen == "mlp":
            self.g1 = nn.Parameter(torch.randn(d, d) * 0.02)
        nw = 2 if self.where == "w13" else 1
        self.pu = nn.Parameter(torch.randn(nw, self.r, d, d) * c.hyp_pinit)
        self.pv = nn.Parameter(torch.randn(nw, self.r, d, H) * c.hyp_pinit)
        self.gain = nn.Parameter(torch.zeros(nw, H))                # used if hyp_gain: starts as no change

    def delta(self, xf, j):
        z = F.silu(xf @ self.g1) if self.gen == "mlp" else xf
        u = torch.einsum("nd,rde->nre", z, self.pu[j])           # [N,r,d]
        v = torch.einsum("nd,rdh->nrh", z, self.pv[j])           # [N,r,H]
        sc = torch.einsum("nre,ne->nr", u, xf)                   # x . u_r (quadratic form of x)
        if self.squash == "tanh":
            sc = torch.tanh(sc)
        elif self.squash.startswith("clip"):
            c_ = float(self.squash[4:])
            sc = sc.clamp(-c_, c_)
        out = torch.einsum("nr,nrh->nh", sc, v)
        return out * self.gain[j] if self.use_gain else out

    def ffn(self, xf, w1, w3, w2, pv=None):
        a, b = xf @ w1, xf @ w3
        if pv is not None:
            a, b = a + pv @ self.w1p, b + pv @ self.w3p
        if not self.off:
            if self.where in ("w1", "w13"):
                a = a + self.delta(xf, 0)
            if self.where == "w3":
                b = b + self.delta(xf, 0)
            if self.where == "w13":
                b = b + self.delta(xf, 1)
        return (F.silu(a) * b) @ w2

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        if self.base == "moe":
            idx, g = self.bal.pick(xf @ self.gate.t(), self.k)
            y = self.bank(xf, idx, g) + self.ffn(xf, self.sw1, self.sw3, self.sw2)
        else:
            pv = prev_tokens(xf, B, T, 1) if self.hprev else None
            y = self.ffn(xf, self.w1, self.w3, self.w2, pv) + self.shared(xf)
        return y.view(B, T, d)


class Pairs(Inv):
    """PAIRS: 3 up-halves and 3 down-halves picked by separate routers, paired by rank."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        self.gu, self.gd = gate_param(d, self.E), gate_param(d, self.E)
        self.bu, self.bd = Balancer(self.E), Balancer(self.E)

    def routed(self, xf, B, T):
        iu, gu = self.bu.pick(xf @ self.gu.t(), self.k)
        idn, gd = self.bd.pick(xf @ self.gd.t(), self.k)
        hdn = self.bank.hidden(xf, iu)
        y = self.bank.down(hdn, idn)
        return (y * (gu * gd / self.bu.scale).unsqueeze(-1)).sum(1)


class Continuous(Inv):
    """CONTINUOUS: 3 heads each pick a real position on a line of 32 experts (width h/2), blend 2 nearest."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.E2 = 2 * self.E
        self.bank = Bank(self.E2, d, self.h // 2)
        self.wz = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.offs = nn.Parameter(torch.linspace(-2, 2, self.k))

    def routed(self, xf, B, T):
        z = torch.sigmoid(xf @ self.wz.t() + self.offs) * (self.E2 - 1.001)   # [N,k]
        lo = z.detach().floor().long()
        fr = z - lo
        idx = torch.cat([lo, lo + 1], 1)
        g = torch.cat([1 - fr, fr], 1)
        return self.bank(xf, idx, g)


class Preview(Inv):
    """PREVIEW: expert e scores a token by ||x P_e||^2 (rank-4 preview), no router weights."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        self.P = nn.Parameter(torch.randn(self.E, d, 4) * (1 / math.sqrt(d)))
        self.bal = Balancer(self.E)

    def routed(self, xf, B, T):
        xn = F.rms_norm(xf, (self.d,))
        e = torch.einsum("nd,edr->ner", xn, self.P).pow(2).sum(-1)
        idx, g = self.bal.pick(e - 4.0, self.k)
        return self.bank(xf, idx, g)


class LSH(Inv):
    """LSH-CODE: 3 heads read 4 learned sign bits each as an expert number; gate = mean bit margin."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        nb = int(math.log2(self.E))
        self.nb = nb
        self.proj = nn.Parameter(torch.randn(self.k, nb, d) * 0.02)
        self.register_buffer("pw", 2 ** torch.arange(nb))

    def routed(self, xf, B, T):
        p = torch.einsum("nd,kbd->nkb", F.rms_norm(xf, (self.d,)), self.proj)
        idx = ((p > 0).long() * self.pw).sum(-1)                        # [N,k]
        g = torch.sigmoid(p.abs() * 4).mean(-1) * 2.5 / self.k * 2
        return self.bank(xf, idx, g)


def prev_tokens(xf, B, T, n):
    """[N, n*d]: states of the n previous tokens (zeros before the start)."""
    xs = xf.view(B, T, -1)
    outs = [torch.cat([torch.zeros_like(xs[:, :j]), xs[:, :-j]], 1) for j in range(1, n + 1)]
    return torch.cat(outs, -1).reshape(B * T, -1)


class BigramExpert(Inv):
    """BIGRAM-EXPERT: experts read [x_t, x_{t-1}, ..., x_{t-bg_win}]; router reads x_t (bg_route=cat: all).
    bg_mode=conv: experts read x_t + a learned causal mix of the bg_win previous states (input stays d)."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.win, self.rmode, self.mode = cfg.bg_win, cfg.bg_route, cfg.bg_mode
        din = d if self.mode == "conv" else d * (1 + self.win)
        self.emb_in = cfg.bg_emb                              # none | cur | prev | both: raw token embeddings
        din += d * {"none": 0, "cur": 1, "prev": 1, "both": 2}[self.emb_in]
        if self.emb_in != "none" and cfg.bg_win == 0:
            din -= 0
        self.bank = Bank(self.E, din, self.h, d_out=d)
        self.gate = gate_param(d * (1 + self.win) if self.rmode == "cat" else d, self.E)
        self.bal = Balancer(self.E)
        if self.mode == "conv":
            self.cw = nn.Parameter(torch.zeros(self.win, d))

    def routed(self, xf, B, T):
        pv = prev_tokens(xf, B, T, self.win)
        cat = torch.cat([xf, pv], -1)
        idx, g = self.bal.pick((cat if self.rmode == "cat" else xf) @ self.gate.t(), self.k)
        if self.mode == "conv":
            inp = xf + (pv.view(-1, self.win, self.d) * self.cw).sum(1)
        else:
            inp = cat
        if self.emb_in != "none":
            e = self.ctx["emb"].reshape(-1, self.d)
            ep = prev_tokens(e, B, T, 1)
            inp = torch.cat([inp] + ([e] if self.emb_in in ("cur", "both") else []) + ([ep] if self.emb_in in ("prev", "both") else []), -1)
        return self.bank(inp, idx, g)


class MultiWidth(Inv):
    """MULTI-WIDTH: 6 experts of width h/2, 6 of width h, 4 of width 2h; router picks 3 of the 16."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.banks = nn.ModuleList([Bank(6, d, self.h // 2), Bank(6, d, self.h), Bank(4, d, 2 * self.h)])
        self.off = [0, 6, 12, 16]
        self.gate = gate_param(d, 16)
        self.bal = Balancer(16)

    def routed(self, xf, B, T):
        idx, g = self.bal.pick(xf @ self.gate.t(), self.k)
        y = 0
        for b, (lo, hi) in enumerate(zip(self.off[:-1], self.off[1:])):
            m = (idx >= lo) & (idx < hi)
            y = y + self.banks[b](xf, (idx - lo).clamp(0, hi - lo - 1), g * m)
        return y


class SliceOut(Inv):
    """SLICE-OUT: expert e writes only quarter e % 4 of the residual stream; tokens pick 3 experts."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h, d_out=d // 4)
        self.gate = gate_param(d, self.E)
        self.bal = Balancer(self.E)

    def routed(self, xf, B, T):
        idx, g = self.bal.pick(xf @ self.gate.t(), self.k)
        y = xf.new_zeros(xf.shape[0], 4, self.d // 4)
        for q in range(4):
            m = (idx % 4 == q)
            y[:, q] = self.bank(xf, idx, g * m)
        return y.reshape(-1, self.d)


class Tiny(Inv):
    """TINY: 512 one-neuron experts, 96 per token (= 3 experts of width 32), router on x."""
    def __init__(self, d, cfg, layer, ctx, E=512):
        super().__init__(d, cfg, layer, ctx, E=E, h=1)
        self.bank = Bank(E, d, 1)
        self.kk = self.k * cfg.expert_hidden
        self.gate = gate_param(d, E)
        self.bal = Balancer(E, scale=float(self.kk) / 3 * 2.5 / 2.5)

    def routed(self, xf, B, T):
        idx, g = self.bal.pick(xf @ self.gate.t(), self.kk)
        return (self.bank.down(self.bank.hidden(xf, idx), idx) * g.unsqueeze(-1)).sum(1)


class Market(Inv):
    """MARKET: buy every expert whose score - price beats a global threshold (1..6 experts per token);
    prices = load bias; the threshold moves so the mean number bought stays k."""
    def __init__(self, d, cfg, layer, ctx, kmax=6):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        self.gate = gate_param(d, self.E)
        self.bal = Balancer(self.E)
        self.kmax = kmax
        self.register_buffer("thr", torch.tensor(0.5))
        self.register_buffer("bought", torch.zeros(()))

    def routed(self, xf, B, T):
        s = torch.sigmoid((xf @ self.gate.t()).float())
        sel = s.detach() + self.bal.bias
        v, idx = torch.topk(sel, self.kmax, dim=-1)
        keep = (v > self.thr)
        keep[:, 0] = True
        g = torch.gather(s, -1, idx) * keep
        g = g / (g.sum(-1, keepdim=True) + 1e-20) * 2.5
        if self.training:
            with torch.no_grad():
                self.bal.count += torch.bincount(idx[keep], minlength=self.E).float()
                self.bought = keep.float().sum(-1).mean()
        return self.bank(xf, idx, g)

    def step_update(self):
        self.thr += 0.01 * torch.sign(self.bought - self.k)


class FreqK(Inv):
    """FREQ-K: token ids in the most frequent half of training tokens get k-1 experts, the rest k+1."""
    def __init__(self, d, cfg, layer, ctx):
        super().__init__(d, cfg, layer, ctx)
        self.bank = Bank(self.E, d, self.h)
        self.gate = gate_param(d, self.E)
        self.bal = Balancer(self.E)

    def routed(self, xf, B, T):
        ids = self.ctx["ids"].reshape(-1)
        freq = self.ctx["freq_rank_top"][ids]                           # True = frequent id
        idx, g = self.bal.pick(xf @ self.gate.t(), self.k + 1)
        mask = torch.ones_like(g)
        mask[freq, -2:] = 0                                             # frequent: keep k-1
        g = torch.gather(torch.sigmoid((xf @ self.gate.t()).float()), -1, idx) * mask
        g = g / (g.sum(-1, keepdim=True) + 1e-20) * 2.5
        return self.bank(xf, idx, g)


class Wild(Inv):
    """Round w1: ATTN-ENTROPY, ATTN-TARGET, QUERY-ROUTE, HEAD-ROUTE, RESIDUAL-EXPERT, SKIP-TO-PREV, TWO-SPEED."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode = mode
        self.bank = Bank(self.E, d, self.h)
        nh = cfg.n_heads
        if mode in ("attnent", "attntarget", "queryroute", "headroute"):
            ctx["need_att"] = True
        if mode == "skipprev":                                # 16 own experts + the previous layer's 16
            self.bal = Balancer(2 * self.E)
            self.gate = gate_param(d, 2 * self.E)
            self.prev = ctx.get("last_bank")
            ctx["last_bank"] = self.bank
        else:
            self.bal = Balancer(self.E)
            self.gate = gate_param(d, self.E)
        if mode == "attnent":                                 # 4 entropy buckets per head add logits
            self.ent_emb = nn.Parameter(torch.zeros(nh, 4, self.E))
        if mode == "headroute":                               # head h scores only its own group of E/nh experts
            self.hgate = nn.Parameter(torch.randn(nh, d // nh, self.E // nh) * 0.02)
        if mode == "twospeed":
            self.step = 0
            self.acc = {}
            for name in ("W1", "W3", "W2"):
                prm = getattr(self.bank, name)
                prm.register_hook(lambda gr, name=name: self.slow_hook(gr, name))

    def slow_hook(self, gr, name):
        """experts E/2..E-1 receive their summed gradient only every 4th step"""
        if not self.training:
            return gr
        h = self.E // 2
        a = self.acc.get(name)
        a = gr[h:].clone() if a is None else a + gr[h:]
        out = gr.clone()
        if self.step % 4 == 3:
            out[h:] = a / 4
            self.acc[name] = None
        else:
            out[h:] = 0
            self.acc[name] = a
        return out

    def step_update(self):
        if self.mode == "twospeed":
            self.step += 1

    def routed(self, xf, B, T):
        m, c, d = self.mode, self.ctx, self.d
        if m == "attntarget":                                 # hidden state at the most-attended position
            tgt = c["att"].mean(1).argmax(-1)                 # [B,T]
            xs = xf.view(B, T, d)
            f = torch.gather(xs, 1, tgt.unsqueeze(-1).expand(B, T, d)).reshape(-1, d)
        elif m == "queryroute":
            f = F.rms_norm(c["q"], (d,)).reshape(-1, d)
        else:
            f = xf
        logits = f @ self.gate.t()
        if m == "attnent":
            att = c["att"]                                    # [B,h,T,T]
            ent = -(att * (att + 1e-9).log()).sum(-1)         # [B,h,T]
            ent = ent / torch.log(torch.arange(1, T + 1, device=xf.device).float() + 1)
            bk = torch.clamp((ent * 4).long(), 0, 3)          # [B,h,T]
            nh = att.shape[1]
            add = sum(self.ent_emb[hh][bk[:, hh]] for hh in range(nh))   # [B,T,E]
            logits = logits + add.reshape(-1, self.E)
        if m == "headroute":
            hd = c["heads"]                                   # [B,h,T,dh]
            nh = hd.shape[1]
            logits = torch.cat([F.rms_norm(hd[:, hh], (hd.shape[-1],)) @ self.hgate[hh] for hh in range(nh)], -1)
            logits = logits.reshape(-1, self.E) + xf @ self.gate.t()
        if m == "skipprev":
            mask = None
            if self.prev is None:
                mask = torch.zeros(2 * self.E, dtype=torch.bool, device=xf.device)
                mask[self.E:] = True
                mask = mask.expand(xf.shape[0], -1)
            idx, g = self.bal.pick(logits, self.k, mask)
            own = idx < self.E
            y = self.bank(xf, torch.where(own, idx, 0), g * own)
            if self.prev is not None:
                y = y + self.prev(xf, torch.where(own, 0, idx - self.E), g * (~own))
            return y
        idx, g = self.bal.pick(logits, self.k)
        if m == "residual":                                   # each expert fits what the earlier ones left
            y = xf.new_zeros(xf.shape)
            for j in range(self.k):
                y = y + self.bank(xf - y, idx[:, j:j + 1], g[:, j:j + 1])
            return y
        return self.bank(xf, idx, g)


def rand_table(R, E, k, seed):
    return torch.rand(R, E, generator=torch.Generator().manual_seed(seed)).argsort(1)[:, :k]


class HashInv(Inv):
    """Round h: inventions built on token-id hash routing (the all-hash MoE in this frame).
    Modes: hash (base), hashpos, hashcount, hashbg3, hashtwo, hashgroup, hashemb, hashdepth."""
    def __init__(self, d, cfg, layer, ctx, mode):
        super().__init__(d, cfg, layer, ctx)
        self.mode, V, E, k = mode, cfg.vocab, self.E, self.k
        self.V = V
        self.gate = gate_param(d, E)
        self.bank = Bank(E, 2 * d if mode == "hashemb" else d, self.h, d_out=d)
        sd = 1234 + 17 * layer
        if mode == "hashpos":
            self.register_buffer("tab", rand_table(4 * V, E, k, sd), persistent=False)
        elif mode == "hashcount":
            self.register_buffer("tab", rand_table(2 * V, E, k, sd), persistent=False)
        elif mode == "hashbg3" and layer == cfg.n_layers - 1:
            self.register_buffer("tab", rand_table(1 << 16, E, k, sd), persistent=False)
        elif mode == "hashtwo":
            self.register_buffer("tab", rand_table(V, E, 1, sd), persistent=False)
            self.register_buffer("tabp", rand_table(V, E, 1, sd + 1), persistent=False)
            self.register_buffer("tabb", rand_table(1 << 16, E, 1, sd + 2), persistent=False)
        elif mode == "hashgroup":                             # id picks a group of G experts, x picks k inside
            G = getattr(cfg, "hg_size", 8)
            self.register_buffer("tab", rand_table(V, E, G, sd), persistent=False)
            self.bal = Balancer(E)
        elif mode == "hashdepth":                             # one pool of 3E experts shared by all layers
            if "pool" not in ctx:
                ctx["pool"] = Bank(E * cfg.n_layers, d, self.h)
                self.pool = ctx["pool"]
            else:
                object.__setattr__(self, "pool", ctx["pool"])
            del self.bank
            self.gate = gate_param(d, E * cfg.n_layers)
            self.register_buffer("tab", rand_table(V, E * cfg.n_layers, k, sd), persistent=False)
        elif mode == "hashk":                                 # per-layer key letter from cfg.hk
            self.key = cfg.hk[layer]
            R = V if self.key in "up" else 1 << 16
            self.register_buffer("tab", rand_table(R, E, k, sd), persistent=False)
        else:
            self.register_buffer("tab", rand_table(V, E, k, sd), persistent=False)
        self.he = cfg.he                                      # expert input: none / add / cat raw embedding
        if self.he == "cat" and mode != "hashemb":
            self.bank = Bank(E, 2 * d, self.h, d_out=d)
        if self.he == "addg":
            self.eg = nn.Parameter(torch.zeros(d))

    def key_of(self, flat, ids):
        c = self.key
        pv = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1).reshape(-1)
        if c == "u":
            return flat
        if c == "p":
            return pv
        if c == "b":
            return (pv * 8191 + flat * 131) % (1 << 16)
        p2 = torch.cat([torch.zeros_like(ids[:, :2]), ids[:, :-2]], 1).reshape(-1)
        if c == "t":                                          # trigram
            return (p2 * 524287 + pv * 8191 + flat * 131) % (1 << 16)
        if c == "s":                                          # skip-bigram (t-2, t)
            return (p2 * 8191 + flat * 131) % (1 << 16)
        raise ValueError(c)

    def routed(self, xf, B, T):
        m, ids = self.mode, self.ctx["ids"]
        flat = ids.reshape(-1)
        pv = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1).reshape(-1)
        if m == "hashpos":
            pb = (torch.arange(T, device=xf.device) * 4 // T).repeat(B)
            idx = self.tab[flat + self.V * pb]
        elif m == "hashcount":                                # first vs repeated occurrence in the sequence
            seen = (ids[:, :, None] == ids[:, None, :]).tril(-1).any(-1).reshape(-1).long()
            idx = self.tab[flat + self.V * seen]
        elif m == "hashbg3" and self.layer == self.cfg.n_layers - 1:
            idx = self.tab[(pv * 8191 + flat * 131) % (1 << 16)]
        elif m == "hashtwo":
            idx = torch.cat([self.tab[flat], self.tabp[pv], self.tabb[(pv * 8191 + flat * 131) % (1 << 16)]], 1)
        elif m == "hashgroup":
            grp = self.tab[flat]
            mask = torch.ones(flat.numel(), self.E, dtype=torch.bool, device=xf.device).scatter(1, grp, False)
            idx, g = self.bal.pick(xf @ self.gate.t(), self.k, mask)
            return self.bank(xf, idx, g)
        elif m == "hashk":
            idx = self.tab[self.key_of(flat, ids)]
        else:
            idx = self.tab[flat]
        g = torch.gather(torch.sigmoid((xf @ self.gate.t()).float()), 1, idx)
        g = g / (g.sum(-1, keepdim=True) + 1e-20) * 2.5
        if m == "hashemb" or self.he == "cat":
            return self.bank(torch.cat([xf, self.ctx["emb"].reshape(-1, self.d)], -1), idx, g)
        if self.he == "add":
            return self.bank(xf + self.ctx["emb"].reshape(-1, self.d), idx, g)
        if self.he == "addg":
            return self.bank(xf + self.eg * self.ctx["emb"].reshape(-1, self.d), idx, g)
        if m == "hashdepth":
            return self.pool(xf, idx, g)
        return self.bank(xf, idx, g)


def context_vectors(train, V, n=4_000_000, dim=64, seed=0, side="both", gap=1):
    """random-indexing co-occurrence vectors: token a = sum of random codes of its left and right neighbours"""
    import numpy as np
    t = np.asarray(train[:n]).astype(np.int64)
    rng = np.random.default_rng(seed)
    RL, RR = rng.standard_normal((V, dim)).astype(np.float32), rng.standard_normal((V, dim)).astype(np.float32)
    C = np.zeros((V, 2 * dim), np.float32)
    for c0 in range(0, len(t) - gap, 500_000):        # chunks keep the temp array small
        c1 = min(c0 + 500_000, len(t) - gap)
        np.add.at(C[:, :dim], t[c0 + gap:c1 + gap], RL[t[c0:c1]])
        np.add.at(C[:, dim:], t[c0:c1], RR[t[c0 + gap:c1 + gap]])
    freq = np.bincount(t, minlength=V).astype(np.float32)
    if side == "left":
        C[:, dim:] = 0
    elif side == "right":
        C[:, :dim] = 0
    C /= np.maximum(freq, 1)[:, None]
    C -= C[freq > 0].mean(0)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-6
    return torch.from_numpy(C), torch.from_numpy(freq)


def kmeans(X, w, K, seed, iters=25):
    g = torch.Generator().manual_seed(seed)
    live = (w > 0).nonzero().squeeze(1)
    cen = X[live[torch.randperm(live.numel(), generator=g)[:K]]].clone()
    for _ in range(iters):
        a = (X @ cen.t()).argmax(1)
        for c in range(K):
            m = (a == c) & (w > 0)
            if m.any():
                cen[c] = F.normalize((X[m] * w[m, None]).sum(0), dim=0)
    return cen


def build_tables(train, cfg, model):
    """fixed token -> expert tables from training statistics (cfg.ht)"""
    V, E, k, L = cfg.vocab, cfg.n_experts, cfg.topk, cfg.n_layers
    C, freq = context_vectors(train, V, n=cfg.ht_n)
    side = {"clusterL": "left", "clusterR": "right"}.get(cfg.ht)
    Ct = context_vectors(train, V, n=cfg.ht_n, side=side)[0] if side else C
    if cfg.emb_init:                                 # control: same statistics into the embedding instead
        ev = getattr(cfg, "emb_var", "base")
        Ce = C
        if ev in ("left", "right"):
            Ce = context_vectors(train, V, n=cfg.ht_n, side=ev)[0]
        if ev == "win2":                             # neighbours at distance 1 and 2
            Ce = torch.cat([C, context_vectors(train, V, n=cfg.ht_n, seed=1, gap=2)[0]], 1)
        P = torch.randn(Ce.shape[1], cfg.d_model, generator=torch.Generator().manual_seed(5))
        W = Ce @ P
        W = W / W.std() * 0.02 * getattr(cfg, "emb_scale", 1.0)
        if ev == "mix":                              # half random, half co-occurrence
            W = (W + torch.randn(W.shape, generator=torch.Generator().manual_seed(6)) * 0.02) / 2 ** 0.5
        Wh = W
        if ev in ("bigram", "bigramL"):              # emb(a) = mean code of tokens after a, head(b) = code of b
            import numpy as np                       # so emb(a) . head(b) ~ count(a, b) / count(a): a bigram model at step 0
            t = np.asarray(train[:cfg.ht_n]).astype(np.int64)
            R = np.random.default_rng(11).standard_normal((V, cfg.d_model)).astype(np.float32)
            A = np.zeros((V, cfg.d_model), np.float32)
            for c0 in range(0, len(t) - 1, 500_000):   # chunks keep the temp array under 300 MB
                c1 = min(c0 + 500_000, len(t) - 1)
                np.add.at(A, t[c0:c1], R[t[c0 + 1:c1 + 1]])
            fr = np.bincount(t, minlength=V).astype(np.float32)
            A /= np.maximum(fr, 1)[:, None]
            if ev == "bigramL":                      # also the left context on the input side
                B = np.zeros((V, cfg.d_model), np.float32)
                R2 = np.random.default_rng(12).standard_normal((V, cfg.d_model)).astype(np.float32)
                for c0 in range(0, len(t) - 1, 500_000):
                    c1 = min(c0 + 500_000, len(t) - 1)
                    np.add.at(B, t[c0 + 1:c1 + 1], R2[t[c0:c1]])
                A = A + B / np.maximum(fr, 1)[:, None]
            A -= A[fr > 0].mean(0)
            sc = 0.02 * getattr(cfg, "emb_scale", 1.0)
            W = torch.from_numpy(A / A.std() * sc)
            Wh = torch.from_numpy(R / R.std() * sc)
        if ev == "bigsvd" and getattr(cfg, "svd_mode", "logp") == "sparse":
            # ported from videos/03 (scale125 infrastructure): same matrix as svd_mode=logp without a dense
            # V x V array: after row centring, log(c + a) - log(rowsum + aV) = log1p(c / a) - rowmean
            import numpy as np
            t = np.asarray(train[:cfg.ht_n]).astype(np.int64)
            pid, cnt = np.unique(t[:-1] * V + t[1:], return_counts=True)
            al = getattr(cfg, "svd_alpha", 0.1)
            L = torch.sparse_coo_tensor(torch.from_numpy(np.stack([pid // V, pid % V])),
                                        torch.from_numpy(np.log1p(cnt / al).astype(np.float32)), (V, V)).coalesce().to_sparse_csr()
            Lt = L.to_sparse_coo().t().coalesce().to_sparse_csr()
            rm = (L @ torch.ones(V, 1)) / V                       # row means, V x 1
            A = lambda X: L @ X - rm @ X.sum(0, keepdim=True)      # (L - rm 1^T) X
            At = lambda Y: Lt @ Y - (rm.t() @ Y).expand(V, -1)     # (L - rm 1^T)^T Y
            r = getattr(cfg, "svd_rank", 0) or cfg.d_model
            g = torch.Generator().manual_seed(0)
            Q = torch.linalg.qr(A(torch.randn(V, r + 10, generator=g)))[0]
            for _ in range(3):                                     # power iterations, re-orthonormalized
                Q = torch.linalg.qr(A(torch.linalg.qr(At(Q))[0]))[0]
            Ub, S_, Vt = torch.linalg.svd(At(Q).t(), full_matrices=False)   # B = Q^T A = (A^T Q)^T
            U, S_, Vv = (Q @ Ub)[:, :r], S_[:r], Vt[:r].t()
            sc = getattr(cfg, "init_std", 0.02) * getattr(cfg, "emb_scale", 1.0)
            W, Wh = U * S_.sqrt(), Vv * S_.sqrt()
            W, Wh = W / W.std() * sc, Wh / Wh.std() * sc
        elif ev == "bigsvd":                         # rank-d SVD of the smoothed log P(b | a) matrix
            import numpy as np
            t = torch.from_numpy(np.asarray(train[:cfg.ht_n]).astype(np.int64))
            Mf = np.zeros(V * V, np.float32)          # float32 counts, no int64 V x V temp
            np.add.at(Mf, (t[:-1] * V + t[1:]).numpy(), 1.0)
            M = torch.from_numpy(Mf).view(V, V)
            al, sm = getattr(cfg, "svd_alpha", 0.1), getattr(cfg, "svd_mode", "logp")
            rs = M.sum(1, keepdim=True)
            if sm == "ppmi":                         # positive PMI (Levy & Goldberg)
                cs = M.sum(0, keepdim=True)
                M.mul_(rs.sum()).div_(rs + 1e-9).div_(cs + 1e-9).add_(1e-9).log_().clamp_(min=0)   # in place, no V x V outer product
            else:
                M.add_(al).log_().sub_(torch.log(rs + al * V))   # in place: one V x V float32 matrix
                if sm != "nocenter":
                    M -= M.mean(1, keepdim=True)     # softmax ignores a per-row constant
            r = getattr(cfg, "svd_rank", 0) or cfg.d_model
            U, S_, Vh = torch.svd_lowrank(M, q=r, niter=3)
            del M
            sc = getattr(cfg, "init_std", 0.02) * getattr(cfg, "emb_scale", 1.0)
            W = U * S_.sqrt()
            Wh = Vh * S_.sqrt()
            if r < cfg.d_model:                      # pad the free dimensions with small random values
                gq = torch.Generator().manual_seed(9)
                W = torch.cat([W / W.std(), torch.randn(V, cfg.d_model - r, generator=gq)], 1)
                Wh = torch.cat([Wh / Wh.std(), torch.randn(V, cfg.d_model - r, generator=gq)], 1)
            W, Wh = W / W.std() * sc, Wh / Wh.std() * sc
        with torch.no_grad():
            if ev in ("bigram", "bigramL", "bigsvd"):
                side_ = getattr(cfg, "svd_side", "both")   # ablation: which matrix gets the factor
                if side_ in ("both", "emb"):
                    model.emb.weight.copy_(W.to(model.emb.weight.device))
                if side_ in ("both", "head"):
                    model.head.weight.copy_(Wh.to(model.head.weight.device))
                if side_ == "swap":                  # factors on the wrong sides: emb gets head's, head gets emb's
                    model.emb.weight.copy_(Wh.to(model.emb.weight.device))
                    model.head.weight.copy_(W.to(model.head.weight.device))
            elif ev != "headonly":
                model.emb.weight.copy_(W.to(model.emb.weight.device))
            if ev in ("head", "headonly"):           # untied output head gets the same vectors
                model.head.weight.copy_(W.to(model.head.weight.device))
    if cfg.ht == "rand_emb":
        return
    order = torch.argsort(-freq)
    rank = torch.empty(V, dtype=torch.long)
    rank[order] = torch.arange(V)
    for li, b in enumerate(model.blocks):
        f = b.ffn
        ht = cfg.ht
        if ht == "same":
            tab = rand_table(V, E, k, 1234)
        elif ht in ("clusterL", "clusterR"):         # table from one side of the context only
            cen = kmeans(Ct, (freq > 0).float(), E, 99 + li)
            tab = torch.topk(Ct @ cen.t(), k, 1).indices
        elif ht == "cluster1":                       # cluster table in layer 1 only
            if li == 0:
                cen = kmeans(C, (freq > 0).float(), E, 99)
                tab = torch.topk(C @ cen.t(), k, 1).indices
            else:
                tab = rand_table(V, E, k, 1234 + 17 * li)
        elif ht == "clusterhalf":                    # nearest centroid's expert + 2 random others
            cen = kmeans(C, (freq > 0).float(), E, 99 + li)
            first = (C @ cen.t()).argmax(1, keepdim=True)
            r = torch.rand(V, E, generator=torch.Generator().manual_seed(1234 + li)).scatter(1, first, -1.0)
            tab = torch.cat([first, torch.topk(r, k - 1, 1).indices], 1)
        elif ht in ("cluster", "clusterw"):          # similar-context tokens share experts
            cen = kmeans(C, freq.sqrt() if ht == "clusterw" else (freq > 0).float(), E, 99 + li)
            tab = torch.topk(C @ cen.t(), k, 1).indices
        elif ht == "anti":                           # similar-context tokens get different experts
            cen = kmeans(C, (freq > 0).float(), 6, 99 + li)
            cl = (C @ cen.t()).argmax(1)
            off = torch.randint(0, E, (6,), generator=torch.Generator().manual_seed(7 + li))
            r = torch.zeros(V, dtype=torch.long)
            for c in range(6):                       # rank by frequency inside each cluster
                m = (cl == c).nonzero().squeeze(1)
                r[m[torch.argsort(-freq[m])]] = torch.arange(m.numel())
            base = (3 * r + off[cl]) % E
            tab = torch.stack([(base + j) % E for j in range(k)], 1)
        elif ht == "freq":                           # frequent tokens get disjoint expert sets
            base = (3 * rank + 5 * li) % E
            tab = torch.stack([(base + j) % E for j in range(k)], 1)
        elif ht == "rand":
            tab = rand_table(V, E, k, 1234 + 17 * li)
        elif ht == "bucket":                         # tokens share G random buckets; one random expert set per bucket
            G = cfg.ht_g
            bk = torch.randint(0, G, (V,), generator=torch.Generator().manual_seed(4321 + li))
            tab = rand_table(G, E, k, 1234 + 17 * li)[bk]
        elif ht == "freqbucket":                     # the G-1 most frequent tokens own a bucket, the rest share one
            G = cfg.ht_g
            bk = torch.clamp(rank, max=G - 1)
            tab = rand_table(G, E, k, 1234 + 17 * li)[bk]
        elif ht == "rareskip":                       # any router: a set of ids skips the routed experts in every layer
            sk = rank >= cfg.ht_top                  # skip_sel=rare: ids below the top ht_top
            if cfg.skip_sel == "freq":               # control: the most frequent ids skip, same token mass
                mass = freq[sk].sum()
                cs = torch.cumsum(freq[order], 0)
                sk = torch.zeros(V, dtype=torch.bool); sk[order[:int((cs < mass).sum()) + 1]] = True
            elif cfg.skip_sel == "rand":             # control: random ids skip, same token mass
                mass = freq[sk].sum()
                pr = torch.randperm(V, generator=torch.Generator().manual_seed(31))
                cs = torch.cumsum(freq[pr], 0)
                sk = torch.zeros(V, dtype=torch.bool); sk[pr[:int((cs < mass).sum()) + 1]] = True
            if li == 0:
                print(f"skip set: {int(sk.sum())} ids, token mass {float(freq[sk].sum() / freq.sum()):.3f}", flush=True)
            f.rare_mask = sk.to(next(model.parameters()).device)
            cfg.rare_mode = cfg.rare_mode if cfg.rare_mode != "none" else "skip"
            continue
        elif ht in ("freqlearn", "freqskip"):        # frequent ids hashed; rare ids learned / shared expert only
            tab = rand_table(V, E, k, 1234 + 17 * li)
            f.rare_mask = (rank >= cfg.ht_top).to(f.hash_table.device)
            cfg.rare_mode = "learned" if ht == "freqlearn" else "skip"
        elif ht in ("freqrare", "freqcluster"):      # frequent ids own sets; rare ids in R random / context-cluster groups
            tab = rand_table(V, E, k, 1234 + 17 * li)
            R = cfg.ht_g
            rare = rank >= cfg.ht_top
            if ht == "freqrare":
                grp = torch.randint(0, R, (V,), generator=torch.Generator().manual_seed(4321 + li))
            else:
                cen = kmeans(C, rare.float(), R, 99 + li)
                grp = (C @ cen.t()).argmax(1)
            rt = rand_table(R, E, k, 777 + 17 * li)
            tab[rare] = rt[grp[rare]]
        elif ht == "rarebucket":                     # G-1 random buckets for frequent tokens hashed, rare share bucket 0
            G = cfg.ht_g
            bk = torch.randint(1, max(G, 2), (V,), generator=torch.Generator().manual_seed(4321 + li))
            bk[rank >= cfg.ht_top] = 0
            tab = rand_table(G, E, k, 1234 + 17 * li)[bk]
        else:
            raise ValueError(ht)
        if hasattr(f, "hash_table"):                 # native all-hash MoE (model.py)
            f.hash_table = tab.to(f.hash_table.device)
        else:
            f.tab = tab.to(f.tab.device)


def make(name, d, cfg, layer, ctx):
    if name.startswith("c_"):
        import blocks
        return blocks.make(name[2:], d, cfg, layer, ctx)
    if name in ("attn", "delta", "lookahead", "count", "echo", "plain", "neighbor", "doc", "copy", "inhibit",
                "hebb", "anti", "normk", "skipk", "cubic", "pos", "grid", "stale", "chunk", "diff",
                "attnk", "product", "embin", "rotated", "recurse"):
        return Feature(d, cfg, layer, ctx, name)
    if name in ("attnent", "attntarget", "queryroute", "headroute", "residual", "skipprev", "twospeed"):
        return Wild(d, cfg, layer, ctx, name)
    if name.startswith("hash"):
        return HashInv(d, cfg, layer, ctx, name)
    return dict(tree=Tree, chain=Chain, depthpool=DepthPool, assemble=Assemble, hyper=Hyper, pairs=Pairs,
                continuous=Continuous, preview=Preview, lsh=LSH, bigram=BigramExpert, market=Market,
                multiwidth=MultiWidth, sliceout=SliceOut, tiny=Tiny,
                freqk=FreqK)[name](d, cfg, layer, ctx)
