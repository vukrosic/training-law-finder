"""Idea mixers for the mixing search (registered into mixers.REG)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mixers import register, RMS, rope, rope_tables, ShortConv, l2n, delta_rule, dt_bias_init, causal_mask, MLA, GatedDeltaNet, KimiDeltaAttention, Mamba2


def logsig(x):
    return F.logsigmoid(x)


class GDNBranch(nn.Module):
    """gated delta rule on given (pre-normalised) per-head q,k,v; own beta / decay from x."""
    def __init__(self, d, H, dk, chan=False):
        super().__init__()
        self.H, self.dk, self.chan = H, dk, chan
        self.b = nn.Linear(d, H, bias=False)
        self.a = nn.Linear(d, H * dk if chan else H, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(dt_bias_init(H * dk if chan else H))
        self.norm = RMS(dk)

    def forward(self, x, q, k, v):
        B, T, _ = x.shape
        beta = self.b(x).sigmoid().transpose(1, 2)
        a = F.softplus(self.a(x) + self.dt_bias)
        if self.chan:
            g = (-self.A_log.exp()[:, None] * a.view(B, T, self.H, self.dk)).transpose(1, 2)
        else:
            g = (-self.A_log.exp() * a).transpose(1, 2)
        y = delta_par(l2n(q) * self.dk ** -0.5, l2n(k), v, beta, g, chan=self.chan)
        return self.norm(y)


from mixers import delta_par  # noqa: E402
from mixers import CTX  # noqa: E402  (model.forward puts the token ids here)


@register("xattn")
class XAttn(nn.Module):
    """softmax attention with switchable parts (round-1 search). Options (mxo):
    conv:K  short conv (identity init, no activation) on the streams in cw (default qkv)
    xconv:K conv on the block input before the projections (Canon-style)
    rec:    alibi (learned slopes) | fox (data-dependent forget gate, Forgetting Transformer)
    og:     sig | silu  elementwise output gate from x
    qkn:1   QK RMSNorm;  temp:1 learned per-head temperature
    erase:L delta-erase attention (ours): logit_ij += sum_{j<m<=i} log(1 - lam*beta_m*relu(cos(k_m,k_j)))
    innov:1 values = delta-rule innovations u (DeltaFormer-style, dot kernel)
    dbr:1   add a gated-delta-rule branch on the same q,k,v (chan:1 = channel decay)
    mem:1   GDN readout as one extra softmax slot (memory slot)
    clock:1 data-dependent RoPE clock (positions = cumsum softplus(w.x))
    lse:1   confidence gate from (logsumexp - log(i+1))
    hop2:1  second pass with query + first output
    stick:1 stick-breaking attention;  sink:1 learned sink logit
    win:W   sliding window;  rope:0 NoPE;  heads:H"""
    def __init__(self, cfg, layer, o):
        super().__init__()
        d = cfg.d_model
        self.o_ = o
        self.H = o.get("heads", cfg.n_heads)
        self.dh = d // self.H
        H, dh = self.H, self.dh
        self.ev = int(o.get("ev", 1))                 # value expansion (GDN-style expand_v)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        if self.ev > 1:
            self.vx = nn.Linear(d, d * self.ev, bias=False)
        self.out = nn.Linear(d * self.ev, d, bias=False)
        if o.get("gout", 0):                          # GDN-style output: per-head RMSNorm * silu(gate)
            self.gon = RMS(dh * self.ev)
            self.gog = nn.Linear(d, d * self.ev, bias=False)
        if o.get("wk", 0):                            # separate write keys for the innovation solve
            self.wkp = nn.Linear(d, d, bias=False)
        if o.get("imix", 0):                          # values = u + m * v
            self.imx = nn.Parameter(torch.full((H, 1, 1), float(o["imix"])))
        if "ibb" in o:
            self.ibbp = nn.Parameter(torch.full((H,), float(o["ibb"])))
        self.conv = o.get("conv", 0)
        self.cw = str(o.get("cw", "qkv"))
        if self.conv:
            self.convs = nn.ModuleDict({c: ShortConv(d, self.conv, act=bool(o.get("cact", 0)), init="ident") for c in self.cw})
        if o.get("xconv", 0):
            self.xc = ShortConv(d, o["xconv"], act=False, init="ident")
        self.rec = o.get("rec", "none")
        if self.rec == "alibi":
            sl = torch.tensor([2.0 ** (-8.0 / H * (h + 1)) for h in range(H)]) * o.get("asc", 1.0)
            self.slope = nn.Parameter(inv_softplus_t(sl))
        if self.rec == "fox":
            self.fg = nn.Linear(d, H, bias=True)
        self.og = o.get("og", "none")
        if self.og != "none":
            self.gate = nn.Linear(d, d * self.ev, bias=False)
        if o.get("qkn", 0):
            self.qn, self.kn = RMS(dh), RMS(dh)
        if o.get("temp", 0):
            self.lt = nn.Parameter(torch.zeros(H))
        if o.get("erase", 0):
            lam = float(o["erase"])
            self.elam = nn.Parameter(torch.full((H,), math.log(lam / (1 - lam))))
            self.eb = nn.Linear(d, H, bias=True)
        if o.get("innov", 0):
            self.ib = nn.Linear(d, H, bias=False)
            if o.get("idec", 0):
                self.ia = nn.Linear(d, H, bias=False)
                self.iA_log = nn.Parameter(torch.log(torch.empty(H).uniform_(0, 16)))
                self.idt = nn.Parameter(dt_bias_init(H))
            if o.get("ipr", 0):
                self.ipw = nn.Parameter(torch.zeros(H))
            if o.get("itemp", 0):
                self.itw = nn.Parameter(torch.zeros(H))
            if o.get("ib2", 0):
                self.ib2 = nn.Linear(d, H, bias=False)
            if o.get("iidb", 0):                          # per-token-id innovation beta (ours)
                self.iidb = nn.Embedding(cfg.vocab, H); nn.init.zeros_(self.iidb.weight)
            if o.get("inrm", 0):
                self.inw = nn.Parameter(torch.ones(H, 1, 1))
        if o.get("dbr", 0) or o.get("mem", 0):
            self.br = GDNBranch(d, H, dh, chan=bool(o.get("chan", 0)))
            self.brs = nn.Parameter(torch.full((H, 1, 1), float(o.get("brs", 0.2))))
        if o.get("mem", 0):
            self.mw = nn.Linear(d, H, bias=True)
        if o.get("clock", 0):
            self.ck = nn.Linear(d, H, bias=True)
        if o.get("lse", 0):
            self.la = nn.Parameter(torch.ones(H))
            self.lb = nn.Parameter(torch.full((H,), 2.0))
        if o.get("sink", 0):
            self.sk = nn.Parameter(torch.zeros(H))
        self.rope_on = o.get("rope", 1)
        c, s_ = rope_tables(dh, cfg.seq_len)
        self.register_buffer("rc", c, persistent=False)
        self.register_buffer("rs", s_, persistent=False)
        self.register_buffer("inv", 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh)), persistent=False)

    def post_init(self):
        with torch.no_grad():
            if hasattr(self, "fg"):
                self.fg.bias.fill_(3.0)
            if hasattr(self, "ck"):
                self.ck.bias.fill_(0.5413)            # softplus = 1: starts as plain RoPE
                self.ck.weight.zero_()
            if hasattr(self, "eb"):
                self.eb.bias.zero_()
            if hasattr(self, "mw"):
                self.mw.bias.fill_(-2.0)

    def forward(self, x, cos, sin):
        o = self.o_
        B, T, D = x.shape
        H, dh = self.H, self.dh
        dev = x.device
        xin = self.xc(x) if hasattr(self, "xc") else x
        q, k, v = self.qkv(xin).split(D, -1)
        if self.ev > 1:
            v = self.vx(xin)
        if self.conv:
            q = self.convs["q"](q) if "q" in self.convs else q
            k = self.convs["k"](k) if "k" in self.convs else k
            v = self.convs["v"](v) if "v" in self.convs else v
        q, k = (t.view(B, T, H, dh).transpose(1, 2) for t in (q, k))
        v = v.view(B, T, H, dh * self.ev).transpose(1, 2)
        q0, k0 = q, k
        if hasattr(self, "qn"):
            q, k = self.qn(q), self.kn(k)
        if hasattr(self, "ck"):
            tau = F.softplus(self.ck(x)).cumsum(1).transpose(1, 2) - 1.0          # [B,H,T], starts at 0
            ang = tau[..., None] * self.inv
            q, k = rope(q, ang.cos(), ang.sin()), rope(k, ang.cos(), ang.sin())
        elif self.rope_on:
            q, k = rope(q, self.rc[:T], self.rs[:T]), rope(k, self.rc[:T], self.rs[:T])
        s = (q @ k.transpose(-1, -2)) * dh ** -0.5
        if hasattr(self, "lt"):
            s = s * self.lt.exp()[:, None, None]
        ii = torch.arange(T, device=dev)
        dist = (ii[:, None] - ii[None, :]).float()
        mask = dist < 0
        if o.get("win", 0):
            mask = mask | (dist >= o["win"])
        if self.rec == "alibi":
            s = s - F.softplus(self.slope)[:, None, None] * dist.clamp(min=0)
        elif self.rec == "fox":
            Fc = logsig(self.fg(x)).cumsum(1).transpose(1, 2)                     # [B,H,T]
            s = s + (Fc[..., :, None] - Fc[..., None, :])
        if hasattr(self, "elam"):
            kn = l2n(k0)
            c = F.relu(kn @ kn.transpose(-1, -2))                                 # [B,H,j,m]
            bm = self.eb(x).sigmoid().transpose(1, 2)                             # [B,H,m]
            E = torch.log1p(-torch.sigmoid(self.elam)[:, None, None] * bm[..., None, :] * c * 0.999)
            E = E.masked_fill(dist.T <= 0, 0.0)                                   # keep m > j  ([j,m]: m-j>0)
            P = E.cumsum(-1)                                                      # [j, i] = sum_{j<m<=i}
            s = s + P.transpose(-1, -2)
        s = s.masked_fill(mask, float("-inf"))
        if o.get("stick", 0):
            ls_neg = logsig(-s).masked_fill(mask, 0.0)
            rev = ls_neg.flip(-1).cumsum(-1).flip(-1) - ls_neg                    # sum_{m>j, m<=i}
            a = (logsig(s) + rev).masked_fill(mask, float("-inf")).exp()
        elif hasattr(self, "mw") or hasattr(self, "sk"):
            extra = (self.mw(x).transpose(1, 2) if hasattr(self, "mw") else self.sk[:, None].expand(H, T).unsqueeze(0).expand(B, H, T))
            full = torch.cat([s, extra[..., None]], -1).softmax(-1)
            a, a_x = full[..., :T], full[..., T]
        else:
            a = s.softmax(-1)
        if o.get("innov", 0):
            kn = l2n(self.wkp(x).view(B, T, H, dh).transpose(1, 2)) if hasattr(self, "wkp") else l2n(k0)
            bl = self.ib(x)
            if hasattr(self, "iidb"):
                bl = bl + self.iidb(CTX["ids"])
            if hasattr(self, "ibbp"):
                bl = bl + self.ibbp
            beta = bl.sigmoid().transpose(1, 2)
            KK = kn @ kn.transpose(-1, -2)
            if o.get("iwin", 0):
                KK = KK.masked_fill(dist > o["iwin"], 0.0)
            v_raw = v
            if hasattr(self, "ia"):
                Gc = (-self.iA_log.exp() * F.softplus(self.ia(x) + self.idt)).transpose(1, 2).cumsum(2)
                KK = KK * (Gc[..., :, None] - Gc[..., None, :]).masked_fill(dist < 0, float("-inf")).exp()
            A = (KK * beta[..., :, None]).masked_fill(dist <= 0, 0.0)
            I = torch.eye(T, device=dev).expand_as(A)
            v = torch.linalg.solve_triangular(I + A, beta[..., None] * v, upper=False, unitriangular=True)
            if hasattr(self, "ib2"):                      # second solve with its own beta (ours)
                b2 = self.ib2(x).sigmoid().transpose(1, 2)
                A2 = (KK * b2[..., :, None]).masked_fill(dist <= 0, 0.0)
                v = torch.linalg.solve_triangular(I + A2, b2[..., None] * v, upper=False, unitriangular=True)
            for _ in range(int(o.get("innov", 0)) - 1):    # order-n: innovations of the innovations (ours)
                v = torch.linalg.solve_triangular(I + A, beta[..., None] * v, upper=False, unitriangular=True)
            if o.get("iq", 0):                            # queries = innovations of the queries (ours)
                qq = torch.linalg.solve_triangular(I + A, beta[..., None] * l2n(q0), upper=False, unitriangular=True)
                if self.rope_on:
                    qq = rope(qq, self.rc[:T], self.rs[:T])
                s = (qq @ k.transpose(-1, -2)) * dh ** -0.5 * 4.0
                a = s.masked_fill(mask, float("-inf")).softmax(-1)
            if hasattr(self, "inw"):                      # RMS-normalise each innovation vector (per token)
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * self.inw * 0.1
            if o.get("ihalf", 0):                         # innovations only in head 0, raw values elsewhere
                hm = (torch.arange(H, device=dev) == 0).view(1, H, 1, 1)
                v = torch.where(hm, v, v_raw)
            if o.get("itemp", 0):                         # sharper attention for surprising tokens (ours)
                if not hasattr(self, "itw"):
                    raise RuntimeError("itemp needs init")
                tq = torch.exp(self.itw[:, None] * 0.5 * torch.log(v.pow(2).sum(-1) + 1e-6))    # [B,H,T]
                st = (s.masked_fill(mask, 0.0) * tq[..., None]).masked_fill(mask, float("-inf"))
                if hasattr(self, "sk"):
                    ex = (self.sk[:, None] * tq)[..., None]
                    a = torch.cat([st, ex], -1).softmax(-1)[..., :T]
                else:
                    a = st.softmax(-1)
            if o.get("ikey", 0):                          # keys = innovations of the keys (ours)
                kq = torch.linalg.solve_triangular(I + A, beta[..., None] * k0, upper=False, unitriangular=True)
                if self.rope_on:
                    kq = rope(kq, self.rc[:T], self.rs[:T])
                s = (q @ kq.transpose(-1, -2)) * dh ** -0.5
                a = s.masked_fill(mask, float("-inf")).softmax(-1)
            if hasattr(self, "imx"):
                v = v + self.imx * v_raw
            if hasattr(self, "ipw"):
                pr = self.ipw[:, None, None] * torch.log(v.norm(dim=-1) + 1e-4)[..., None, :]
                a = (s + pr).masked_fill(mask, float("-inf")).softmax(-1)
        y = a @ v
        if o.get("hop2", 0):
            q2 = q + y
            s2 = ((q2 @ k.transpose(-1, -2)) * dh ** -0.5).masked_fill(mask, float("-inf"))
            y = s2.softmax(-1) @ v
        if hasattr(self, "br"):
            g = self.br(x, q0, k0, v)
            if hasattr(self, "mw"):
                y = y + a_x[..., None] * g * self.brs
            else:
                y = y + g * self.brs
        if hasattr(self, "la"):
            lse = torch.logsumexp(s, -1) - torch.log(ii.float() + 1)
            y = y * torch.sigmoid(self.la[:, None] * lse + self.lb[:, None])[..., None]
        if hasattr(self, "gon"):
            y = self.gon(y.transpose(1, 2)) * F.silu(self.gog(x).view(B, T, H, -1))
            return self.out(y.reshape(B, T, -1))
        y = y.transpose(1, 2).reshape(B, T, -1)
        if self.og == "sig":
            y = y * torch.sigmoid(self.gate(x))
        elif self.og == "silu":
            y = y * F.silu(self.gate(x))
        return self.out(y)


def inv_softplus_t(x):
    return x + torch.log(-torch.expm1(-x))


@register("hyb")
class Hybrid(nn.Module):
    """per-layer pattern, pat:<letters> one per layer: a attn(xattn), m mla, g gdn, k kda, s kda-silu, 2 mamba2."""
    def __init__(self, cfg, layer, o):
        super().__init__()
        pat = str(o.get("pat", "ggm"))
        c = pat[layer % len(pat)]
        if c == "a":
            self.m = XAttn(cfg, layer, o)
        elif c == "m":
            self.m = MLA(cfg, layer, o)
        elif c == "g":
            self.m = GatedDeltaNet(cfg, layer, o)
        elif c == "k":
            self.m = KimiDeltaAttention(cfg, layer, o)
        elif c == "s":
            self.m = KimiDeltaAttention(cfg, layer, dict(o, og="silu"))
        elif c == "2":
            self.m = Mamba2(cfg, layer, o)
        elif c == "i":                                 # innovation attention with sink (round-2 leader)
            self.m = XAttn(cfg, layer, dict(innov=1, sink=1))
        elif c == "j":                                 # order-2 innovation attention with sink (round-2 FW leader)
            self.m = XAttn(cfg, layer, dict(innov=2, sink=1))
        elif c == "S":                                 # GDN 1 head + D skip (round-2 code leader)
            self.m = GDNY(cfg, layer, dict(skip=1, heads=1))
        elif c == "P":                                 # DeltaProduct-2, 1 head
            self.m = GDNZ(cfg, layer, dict(dprod=2, heads=1))
        elif c == "e":
            self.m = GatedDeltaNet(cfg, layer, dict(ev=2))
        elif c == "h":
            self.m = GatedDeltaNet(cfg, layer, dict(ev=2, heads=1))
        elif c == "1":
            self.m = GatedDeltaNet(cfg, layer, dict(heads=1))

    def forward(self, x, cos, sin):
        return self.m(x, cos, sin)


@register("gdnx")
class GDNX(GatedDeltaNet):
    """Gated DeltaNet + switches: rope:1 (RoPE on normed q,k), negeig:1 (beta in (0,2), Grazzi 2024),
    loc:W (add sliding-window softmax attention on the same q,k,v, RoPE, scale locs),
    ts:1 (two decay timescales: head 0 fast A in [8,16], head 1 slow A in [0.1,1])."""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        self.ox = o
        if o.get("ts", 0) and self.H >= 2:
            with torch.no_grad():
                A = torch.cat([torch.empty(self.H // 2).uniform_(8, 16), torch.empty(self.H - self.H // 2).uniform_(0.1, 1)])
                self.A_log.copy_(A.log())
        c, s_ = rope_tables(self.dk, cfg.seq_len)
        self.register_buffer("rc", c, persistent=False)
        self.register_buffer("rs", s_, persistent=False)
        if o.get("loc", 0):
            self.ls = nn.Parameter(torch.full((self.H, 1, 1), float(o.get("locs", 0.5))))

    def forward(self, x, cos, sin):
        o = self.ox
        B, T, _ = x.shape
        q, k, v, beta = self.qkvb(x)
        if o.get("negeig", 0):
            beta = beta * 2
        if o.get("rope", 0):
            q, k = rope(q, self.rc[:T], self.rs[:T]), rope(k, self.rc[:T], self.rs[:T])
        g = (-self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)).transpose(1, 2)
        y = delta_par(q * self.dk ** -0.5, k, v, beta, g)
        if o.get("loc", 0):
            qr, kr = rope(q, self.rc[:T], self.rs[:T]), rope(k, self.rc[:T], self.rs[:T])
            ii = torch.arange(T, device=x.device)
            dist = ii[:, None] - ii[None, :]
            s = (qr @ kr.transpose(-1, -2)) * float(o.get("lt", 8.0))             # normed q,k: fixed temperature
            s = s.masked_fill((dist < 0) | (dist >= o["loc"]), float("-inf"))
            y = y + (s.softmax(-1) @ v) * self.ls
        y = self.onorm(y.transpose(1, 2)) * F.silu(self.g(x).view(B, T, self.H, self.dv))
        return self.o(y.reshape(B, T, -1))


@register("mlax")
class MLAX(MLA):
    """MLA + lconv:K identity-init causal conv on the KV latent before its RMSNorm (ours)."""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        self.lc = ShortConv(self.r + self.rd, o.get("lconv", 4), act=False, init="ident")

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H = self.H
        q = self.wq(x).view(B, T, H, self.nope + self.rd).transpose(1, 2)
        qn, qr = q.split([self.nope, self.rd], -1)
        c, kr = self.lc(self.wkv_a(x)).split([self.r, self.rd], -1)
        qr = rope(qr, self.rc[:T], self.rs[:T])
        kr = rope(kr.unsqueeze(1), self.rc[:T], self.rs[:T]).expand(B, H, T, self.rd)
        kv = self.wkv_b(self.kv_norm(c)).view(B, T, H, self.nope + self.vd).transpose(1, 2)
        kn, v = kv.split([self.nope, self.vd], -1)
        y = F.scaled_dot_product_attention(torch.cat([qn, qr], -1), torch.cat([kn, kr], -1), v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, H * self.vd))


def delta_par_u(q, k, v, beta, logdecay):
    """delta_par (scalar decay) that also returns the innovations u."""
    B, H, T, dk = q.shape
    dev = q.device
    strict = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
    incl = torch.ones(T, T, dtype=torch.bool, device=dev).tril(0)
    G = logdecay.cumsum(2)
    dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
    A = ((k @ k.transpose(-1, -2)) * dec * beta[..., :, None]).masked_fill(~strict, 0.0)
    I = torch.eye(T, device=dev).expand_as(A)
    u = torch.linalg.solve_triangular(I + A, beta[..., None] * v, upper=False, unitriangular=True)
    return ((q @ k.transpose(-1, -2)) * dec).masked_fill(~incl, 0.0) @ u, u


def delta_vdec(q, k, v, beta, g):
    """delta rule with one decay per VALUE channel (g [B,H,T,dv]): each value column is its own
    scalar-decay delta rule on shared keys.  S_t = S_{t-1} diag(a_t);  S_t += beta k (v - S_t^T k)^T."""
    B, H, T, dk = q.shape
    dev = q.device
    strict = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
    incl = torch.ones(T, T, dtype=torch.bool, device=dev).tril(0)
    G = g.cumsum(2).transpose(2, 3)                                         # [B,H,dv,T]
    dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()   # [B,H,dv,T,T]
    kk = (k @ k.transpose(-1, -2))[:, :, None]
    A = (kk * dec * beta[:, :, None, :, None]).masked_fill(~strict, 0.0)
    I = torch.eye(T, device=dev).expand_as(A)
    rhs = (beta[..., None] * v).transpose(2, 3)[..., None]                  # [B,H,dv,T,1]
    u = torch.linalg.solve_triangular(I + A, rhs, upper=False, unitriangular=True)[..., 0]   # [B,H,dv,T]
    W = ((q @ k.transpose(-1, -2))[:, :, None] * dec).masked_fill(~incl, 0.0)
    return (W @ u[..., None])[..., 0].transpose(2, 3)                       # [B,H,T,dv]


@register("gdny")
class GDNY(GatedDeltaNet):
    """Gated DeltaNet + round-2 switches:
    skip:1   learned per-channel D skip on v (Mamba-style)
    inno:G   add G * innovation u_t to the readout (ours: token reports what was new)
    vdec:1   decay per value channel instead of one per head (ours)
    oconv:K  identity-init causal conv on the readout before the gate
    iattn:1  add softmax attention over the innovations u (RoPE on normed q,k)
    rope:1   RoPE on normed q,k"""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        self.oy = o
        d = cfg.d_model
        if o.get("skip", 0):
            self.D = nn.Parameter(torch.zeros(self.H, 1, self.dv))
        if o.get("inno", 0):
            self.ig = nn.Parameter(torch.full((self.H, 1, 1), float(o["inno"])))
        if o.get("vdec", 0):
            self.a = nn.Linear(d, self.H * self.dv, bias=False)
            self.dt_bias = nn.Parameter(dt_bias_init(self.H * self.dv))
        if o.get("oconv", 0):
            self.oc = ShortConv(self.H * self.dv, o["oconv"], act=False, init="ident")
        if o.get("iattn", 0):
            self.ias = nn.Parameter(torch.full((self.H, 1, 1), 0.5))
        if o.get("sq", 0):                             # softmax-over-slots query (GSA-like probe)
            self.sqt = nn.Parameter(torch.full((self.H, 1, 1), float(o["sq"])))
        if o.get("idb", 0):                            # per-token-id write strength (ours)
            self.idb = nn.Embedding(cfg.vocab, self.H); nn.init.zeros_(self.idb.weight)
        if o.get("idg", 0):                            # per-token-id forget rate (ours)
            self.idg = nn.Embedding(cfg.vocab, self.H); nn.init.zeros_(self.idg.weight)
        if o.get("idk", 0):                            # per-token-id key offset (ours)
            self.idk = nn.Embedding(cfg.vocab, self.H * self.dk); nn.init.zeros_(self.idk.weight)
        if o.get("ve", 0):                             # per-token-id value table added to v (value embedding)
            self.ve = nn.Embedding(cfg.vocab, self.H * self.dv); nn.init.zeros_(self.ve.weight)
        if o.get("vo", 0):                             # per-token-id table added to the readout (ours)
            self.vo = nn.Embedding(cfg.vocab, self.H * self.dv); nn.init.zeros_(self.vo.weight)
        if o.get("idgain", 0):                         # per-token-id channel gain on the readout (ours)
            self.idgn = nn.Embedding(cfg.vocab, self.H * self.dv); nn.init.zeros_(self.idgn.weight)
        if o.get("pskip", 0):                          # skip from the previous token's value (ours)
            self.D2 = nn.Parameter(torch.zeros(self.H, 1, self.dv))
        c, s_ = rope_tables(self.dk, cfg.seq_len)
        self.register_buffer("rc", c, persistent=False)
        self.register_buffer("rs", s_, persistent=False)

    def forward(self, x, cos, sin):
        o = self.oy
        B, T, _ = x.shape
        H = self.H
        q, k, v, beta = self.qkvb(x)
        ids = CTX.get("ids")
        vt = lambda e: e(ids).view(B, T, H, self.dv).transpose(1, 2)
        if hasattr(self, "ve"):
            v = v + vt(self.ve)
        if o.get("kprev", 0):                          # write key from the previous token: S learns prev -> current
            k = torch.cat([k[:, :, :1], k[:, :, :-1]], 2)
        if hasattr(self, "idb"):
            beta = (self.b(x) + self.idb(ids)).sigmoid().transpose(1, 2)
        if hasattr(self, "idk"):
            k = l2n(k + self.idk(ids).view(B, T, H, self.dk).transpose(1, 2))
        if o.get("rope", 0):
            q, k = rope(q, self.rc[:T], self.rs[:T]), rope(k, self.rc[:T], self.rs[:T])
        if hasattr(self, "sqt"):
            q = (q * self.sqt).softmax(-1) * self.dk ** 0.5
        a = F.softplus(self.a(x) + self.dt_bias + (self.idg(ids) if hasattr(self, "idg") else 0.0))
        if o.get("vdec", 0):
            g = (-self.A_log.exp()[:, None] * a.view(B, T, H, self.dv)).transpose(1, 2)
            y = delta_vdec(q * self.dk ** -0.5, k, v, beta, g)
            u = None
        else:
            g = (-self.A_log.exp() * a).transpose(1, 2)
            y, u = delta_par_u(q * self.dk ** -0.5, k, v, beta, g)
        if hasattr(self, "ig"):
            y = y + self.ig * u
        if hasattr(self, "D"):
            y = y + self.D * v
        if hasattr(self, "D2"):
            y = y + self.D2 * F.pad(v[:, :, :-1], (0, 0, 1, 0))
        if hasattr(self, "vo"):
            y = y + vt(self.vo)
        if hasattr(self, "idgn"):
            y = y * (1 + vt(self.idgn))
        if hasattr(self, "ias"):
            qr, kr = rope(q, self.rc[:T], self.rs[:T]), rope(k, self.rc[:T], self.rs[:T])
            s = ((qr @ kr.transpose(-1, -2)) * 8.0).masked_fill(causal_mask(T, x.device), float("-inf"))
            y = y + (s.softmax(-1) @ u) * self.ias
        if hasattr(self, "oc"):
            y = self.oc(y.transpose(1, 2).reshape(B, T, -1)).view(B, T, H, self.dv).transpose(1, 2)
        y = self.onorm(y.transpose(1, 2)) * F.silu(self.g(x).view(B, T, H, self.dv))
        return self.o(y.reshape(B, T, -1))


def delta_vb(q, k, v, beta, g):
    """delta rule with decay g and write strength beta both per VALUE channel ([B,H,T,dv]).
    each value column c: S_c += beta_c k (v_c - S_c^T k)."""
    B, H, T, dk = q.shape
    dev = q.device
    strict = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
    incl = torch.ones(T, T, dtype=torch.bool, device=dev).tril(0)
    G = g.cumsum(2).transpose(2, 3)                                         # [B,H,dv,T]
    dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
    bt = beta.transpose(2, 3)                                               # [B,H,dv,T]
    A = ((k @ k.transpose(-1, -2))[:, :, None] * dec * bt[..., :, None]).masked_fill(~strict, 0.0)
    I = torch.eye(T, device=dev).expand_as(A)
    rhs = (bt * v.transpose(2, 3))[..., None]
    u = torch.linalg.solve_triangular(I + A, rhs, upper=False, unitriangular=True)[..., 0]
    W = ((q @ k.transpose(-1, -2))[:, :, None] * dec).masked_fill(~incl, 0.0)
    return (W @ u[..., None])[..., 0].transpose(2, 3)


@register("gdnz")
class GDNZ(GatedDeltaNet):
    """Gated DeltaNet + round-2 switches (second batch):
    rd2:G     second read of the same state with query l2n(first read) (ours), mixed with gate init G
    vbeta:1   write strength per value channel (ours)
    dprod:N   DeltaProduct (Siems et al. 2025): N delta writes per token (known probe)"""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        self.oz = o
        d = cfg.d_model
        if o.get("rd2", 0):
            assert self.dv == self.dk
            self.r2 = nn.Parameter(torch.full((self.H, 1, 1), float(o["rd2"])))
        if o.get("vbeta", 0):
            self.b = nn.Linear(d, self.H * self.dv, bias=False)
        if o.get("skip", 0):
            self.D = nn.Parameter(torch.zeros(self.H, 1, self.dv))
        n = o.get("dprod", 1)
        if n > 1:
            self.kx = nn.Linear(d, self.H * self.dk * (n - 1), bias=False)
            self.vx = nn.Linear(d, self.H * self.dv * (n - 1), bias=False)
            self.bx = nn.Linear(d, self.H * (n - 1), bias=False)
            self.ckx = ShortConv(self.H * self.dk * (n - 1))
            self.cvx = ShortConv(self.H * self.dv * (n - 1))

    def forward(self, x, cos, sin):
        o = self.oz
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        q = l2n(self.cq(self.q(x)).view(B, T, H, dk)).transpose(1, 2)
        k = l2n(self.ck(self.k(x)).view(B, T, H, dk)).transpose(1, 2)
        v = self.cv(self.v(x)).view(B, T, H, dv).transpose(1, 2)
        g = (-self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)).transpose(1, 2)
        n = o.get("dprod", 1)
        if o.get("vbeta", 0):
            beta = self.b(x).sigmoid().view(B, T, H, dv).transpose(1, 2)
            y = delta_vb(q * dk ** -0.5, k, v, beta, g[..., None].expand(B, H, T, dv))
        elif n > 1:
            beta = self.b(x).sigmoid().transpose(1, 2)
            kx = l2n(self.ckx(self.kx(x)).view(B, T, n - 1, H, dk)).permute(0, 3, 1, 2, 4)
            vx = self.cvx(self.vx(x)).view(B, T, n - 1, H, dv).permute(0, 3, 1, 2, 4)
            bx = self.bx(x).sigmoid().view(B, T, n - 1, H).permute(0, 3, 1, 2)
            K = torch.cat([k[:, :, :, None], kx], 3).reshape(B, H, T * n, dk)
            V = torch.cat([v[:, :, :, None], vx], 3).reshape(B, H, T * n, dv)
            Bt = torch.cat([beta[..., None], bx], 3).reshape(B, H, T * n)
            Gd = torch.cat([g[..., None], torch.zeros(B, H, T, n - 1, device=x.device)], 3).reshape(B, H, T * n)
            Q = torch.zeros(B, H, T, n, dk, device=x.device)
            Q[:, :, :, -1] = q * dk ** -0.5
            y = delta_par(Q.reshape(B, H, T * n, dk), K, V, Bt, Gd).view(B, H, T, n, dv)[:, :, :, -1]
        else:
            beta = self.b(x).sigmoid().transpose(1, 2)
            y, u = delta_par_u(q * dk ** -0.5, k, v, beta, g)
            if hasattr(self, "r2"):
                q2 = l2n(y)
                Gc = g.cumsum(2)
                incl = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(0)
                dec = (Gc[..., :, None] - Gc[..., None, :]).masked_fill(~incl, float("-inf")).exp()
                y = y + self.r2 * (((q2 * dk ** -0.5) @ k.transpose(-1, -2)) * dec).masked_fill(~incl, 0.0) @ u
        if hasattr(self, "D"):
            y = y + self.D * v
        y = self.onorm(y.transpose(1, 2)) * F.silu(self.g(x).view(B, T, H, dv))
        return self.o(y.reshape(B, T, -1))


# ------------------------------------------------------------------ round 7 (gdnw)
def delta_gen(q, k, v, beta, G, lam=None, rbw=False, mom=0.0):
    """delta rule, scalar decay, parallel form. G = cumulative log decay [B,H,T].
    lam: [H,1,1] in (0,1) scales the correction (0 = linear attention, 1 = delta rule).
    rbw: read S_{t-1} (before the current write). mom: momentum on the writes, m_t = mom*m_{t-1} + k_t u_t^T
    (approximate: u ignores the momentum)."""
    B, H, T, dk = q.shape
    dev = q.device
    strict = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
    incl = torch.ones(T, T, dtype=torch.bool, device=dev).tril(0)
    dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
    A = (k @ k.transpose(-1, -2)) * dec * beta[..., :, None]
    if lam is not None:
        A = A * lam
    A = A.masked_fill(~strict, 0.0)
    I = torch.eye(T, device=dev).expand_as(A)
    u = torch.linalg.solve_triangular(I + A, beta[..., None] * v, upper=False, unitriangular=True)
    M = dec
    if mom:
        ii = torch.arange(T, device=dev)
        L = (float(mom) ** (ii[:, None] - ii[None, :]).clamp(min=0).float()).masked_fill(~incl, 0.0)
        M = dec @ L
    W = (q @ k.transpose(-1, -2)) * M
    return W.masked_fill(~(strict if rbw else incl), 0.0) @ u, u


class IndHead(nn.Module):
    """extra delta-rule memory keyed by token ids (ours): write key = E[previous id] (or a hash of the
    previous two / three ids), value = W x_t (or Ev[id_t]); read query = E[current id] (or hash of the last two / three).
    The state maps 'token seen' -> 'what came after it last time': a learned, soft induction head.
    Round-9 knobs: R buckets, ad decay init, dv value dim, nh heads, og output gate, skip key (id t-2 -> query id t-1)."""
    def __init__(self, cfg, mode="x", key="uni", dk=32, dv=32, hk=0, R=None, ad=-4.0, nh=1, og=0, beta1=0):
        super().__init__()
        d = cfg.d_model
        self.mode, self.key, self.hk, self.nh, self.dk, self.dv, self.beta1 = mode, key, hk, nh, dk, dv, beta1
        R = int(R) if R else (65536 if key in ("bi", "tri", "skip") else cfg.vocab)
        if key in ("skip", "uni"): R = cfg.vocab
        self.R = R
        if key == "none":                              # control: value branch with no memory read
            pass
        elif hk:                                       # keys from hidden states instead of ids
            self.wq, self.wk = nn.Linear(d, nh * dk, bias=False), nn.Linear(d, nh * dk, bias=False)
        else:
            self.E = nn.Embedding(R, nh * dk)
            nn.init.normal_(self.E.weight, 0, 1.0)
        if mode == "id":
            self.Ev = nn.Embedding(cfg.vocab, nh * dv); nn.init.normal_(self.Ev.weight, 0, 0.02)
        else:
            self.wv = nn.Linear(d, nh * dv, bias=False)
        self.wb = nn.Linear(d, nh, bias=True)
        self.ad = nn.Parameter(torch.full((nh,), float(ad)))   # softplus(-4) ~ 0.018 per step
        self.wo = nn.Linear(nh * dv, d, bias=False); nn.init.zeros_(self.wo.weight)
        if og:
            self.gate_o = nn.Linear(d, nh * dv, bias=False)

    def forward(self, x, ids):
        B, T, _ = x.shape
        nh, dk, dv = self.nh, self.dk, self.dv
        if self.key == "none":
            y = self.wv(x)
            if hasattr(self, "gate_o"):
                y = y * F.silu(self.gate_o(x))
            return self.wo(y)
        if self.hk:
            xp = F.pad(x[:, :-1], (0, 0, 1, 0))
            kk, qq = self.wk(xp), self.wq(x)
        else:
            ip = F.pad(ids[:, :-1], (1, 0))
            if self.key == "bi":
                ipp = F.pad(ids[:, :-2], (2, 0))
                kk = self.E((ipp * 1000003 + ip) % self.R)
                qq = self.E((ip * 1000003 + ids) % self.R)
            elif self.key == "tri":
                ipp = F.pad(ids[:, :-2], (2, 0)); i3 = F.pad(ids[:, :-3], (3, 0))
                kk = self.E(((i3 * 1000003 + ipp) * 999983 + ip) % self.R)
                qq = self.E(((ipp * 1000003 + ip) * 999983 + ids) % self.R)
            elif self.key == "skip":                   # key id t-2 (write) / query id t-1 (read): x _ y patterns
                ipp = F.pad(ids[:, :-2], (2, 0))
                kk, qq = self.E(ipp), self.E(ip)
            else:
                kk, qq = self.E(ip), self.E(ids)
        kk = l2n(kk.view(B, T, nh, dk)).transpose(1, 2)
        qq = l2n(qq.view(B, T, nh, dk)).transpose(1, 2)
        v = (self.Ev(ids) if self.mode == "id" else self.wv(x)).view(B, T, nh, dv).transpose(1, 2)
        beta = torch.ones(B, nh, T, device=x.device) if self.beta1 else self.wb(x).sigmoid().transpose(1, 2)
        G = (-F.softplus(self.ad)[None, :, None] * torch.ones(B, nh, T, device=x.device)).cumsum(-1)
        y, _ = delta_gen(qq, kk, v, beta, G)
        y = y.transpose(1, 2).reshape(B, T, nh * dv)
        if hasattr(self, "gate_o"):
            y = y * F.silu(self.gate_o(x))
        return self.wo(y)


class NGramHead(nn.Module):
    """exact-match n-gram head (Known probe: Akyurek et al. 2024, arXiv 2401.12973): attend to positions j whose
    previous n ids equal the current last n ids, with a learned recency slope; copy W v_j."""
    def __init__(self, cfg, n=1, dv=32):
        super().__init__()
        d = cfg.d_model
        self.n = n
        self.wv = nn.Linear(d, dv, bias=False)
        self.sl = nn.Parameter(torch.tensor(0.1))
        self.wo = nn.Linear(dv, d, bias=False); nn.init.zeros_(self.wo.weight)

    def forward(self, x, ids):
        B, T, _ = x.shape
        ii = torch.arange(T, device=x.device)
        m = torch.ones(B, T, T, dtype=torch.bool, device=x.device)
        for s in range(self.n):                        # id_{j-1-s} == id_{t-s}
            cur = F.pad(ids[:, :T - s], (s, 0), value=-1) if s else ids
            prv = F.pad(ids[:, :T - 1 - s], (1 + s, 0), value=-2)
            m = m & (cur[:, :, None] == prv[:, None, :])
        m = m & (ii[None, :, None] >= ii[None, None, :])
        sc = (-self.sl.abs() * (ii[:, None] - ii[None, :]).float())[None].expand(B, T, T)
        sc = sc.masked_fill(~m, float("-inf"))
        has = m.any(-1, keepdim=True)
        w = torch.where(has, sc, torch.zeros_like(sc)).softmax(-1) * has
        return self.wo(w @ self.wv(x))


@register("gdnw")
class GDNW(GDNY):
    """GDN with round-7 switches (each default off), on top of GDNY's skip/ve/vo/idgain/pskip/kprev:
    qs:S learned per-head query scale (init S; 0 -> dk^-0.5) | negeig:1 beta in (0,2) | skipinit:1 D starts at 1
    ck:K conv kernel | nocqk:1 no conv on q,k | nocv:1 no conv on v | bb:B learned beta bias init B
    og:sig|none output gate | ai:1 A~U(1,16), ai:2 A~U(0,4) | nodec:1 no forgetting (nodec:2 head 0 only)
    vres:L value residual from layer 0 | xread:1 read layer 0's state (xread:2 previous layer's)
    mom:E momentum on writes | qktie:1 q = k | nq:2 second query on the same state | rbw:1 read before write
    noqn:1 no L2 norm on q | idog:1 per-id output-gate bias | lam:P delta/linear blend init P
    ind:x|id induction memory by ids (indk:bi bigram keys, indh:1 hidden keys) | ngh:N exact-match n-gram head
    kid:1 keys = E[id] | qid:1 queries = E[id] | pmem:P persistent learned key/value rows"""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        self.ow, self.layer = o, layer
        d, H, dk, dv = cfg.d_model, self.H, self.dk, self.dv
        ck = int(o.get("ck", 4))
        if ck != 4:
            self.cq, self.ck, self.cv = ShortConv(H * dk, ck), ShortConv(H * dk, ck), ShortConv(H * dv, ck)
        if o.get("qs", 0) != 0:
            init = float(o["qs"]) if float(o["qs"]) > 0 else dk ** -0.5
            self.qsl = nn.Parameter(torch.full((H, 1, 1), math.log(init)))
        if o.get("skipinit", 0) and hasattr(self, "D"):
            self.D.data.fill_(1.0)
        if "bb" in o:
            self.bbias = nn.Parameter(torch.full((H,), float(o["bb"])))
        if o.get("ai", 0):
            lo, hi = (1, 16) if int(o["ai"]) == 1 else (0, 4)
            with torch.no_grad():
                self.A_log.copy_(torch.log(torch.empty(H).uniform_(lo, hi).clamp(min=1e-3)))
        if o.get("vres", 0) and layer > 0:
            self.vrl = nn.Parameter(torch.full((H, 1, 1), float(o["vres"])))
        if o.get("xread", 0) and layer > 0:
            self.xrg = nn.Parameter(torch.full((H, 1, 1), 0.3))
        if o.get("nq", 0):
            self.q2 = nn.Linear(d, H * dk, bias=False)
            self.on2 = RMS(dv)
            self.o2 = nn.Linear(H * dv, d, bias=False); nn.init.zeros_(self.o2.weight)
        if o.get("idog", 0):
            self.idgt = nn.Embedding(cfg.vocab, H * dv); nn.init.zeros_(self.idgt.weight)
        if o.get("lam", 0):
            p = float(o["lam"])
            self.laml = nn.Parameter(torch.full((H, 1, 1), math.log(p / (1 - p))))
        indl = str(o.get("indl", ""))                  # layers that get the induction memory, e.g. indl:02
        if (o.get("ind", 0) or o.get("indh", 0)) and (not indl or str(layer) in indl):
            kw = dict(dk=int(o.get("indd", 32)), dv=int(o.get("indv", 32)), R=o.get("indR"), ad=float(o.get("indad", -4.0)),
                      nh=int(o.get("indnh", 1)), og=int(o.get("indog", 0)), beta1=int(o.get("indb1", 0)))
            self.indm = IndHead(cfg, mode=str(o.get("ind", "x")) if o.get("ind", 0) != 1 else "x",
                                key=str(o.get("indk", "uni")), hk=int(o.get("indh", 0)), **kw)
            if o.get("ind2", 0):
                self.indm2 = IndHead(cfg, mode="x", key=str(o.get("ind2k", "bi")), **kw)
            if o.get("ind3", 0):
                self.indm3 = IndHead(cfg, mode="x", key=str(o.get("ind3k", "tri")), **kw)
        if o.get("ngh", 0):
            self.nghm = NGramHead(cfg, n=int(o["ngh"]))
        if o.get("kid", 0) or o.get("qid", 0):
            self.Eid = nn.Embedding(cfg.vocab, H * dk); nn.init.normal_(self.Eid.weight, 0, 1.0)
        if o.get("pmem", 0):
            P = int(o["pmem"])
            self.pk = nn.Parameter(torch.randn(H, P, dk))
            self.pv = nn.Parameter(torch.randn(H, P, dv) * 0.02)
            self.pb = nn.Parameter(torch.zeros(H, P))

    def forward(self, x, cos, sin):
        o = self.ow
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        ids = CTX.get("ids")
        qr = self.q(x) if not o.get("qktie", 0) else self.k(x)
        qr = qr if o.get("nocqk", 0) else self.cq(qr)
        kr = self.k(x) if o.get("nocqk", 0) else self.ck(self.k(x))
        q = qr.view(B, T, H, dk).transpose(1, 2)
        q = q if o.get("noqn", 0) else l2n(q)
        k = l2n(kr.view(B, T, H, dk)).transpose(1, 2)
        vr = self.v(x) if o.get("nocv", 0) else self.cv(self.v(x))
        v = vr.view(B, T, H, dv).transpose(1, 2)
        bl = self.b(x) + (self.bbias if hasattr(self, "bbias") else 0.0)
        beta = bl.sigmoid().transpose(1, 2)
        if o.get("negeig", 0):
            beta = beta * 2
        if hasattr(self, "Eid"):
            e = l2n(self.Eid(ids).view(B, T, H, dk)).transpose(1, 2)
            if o.get("kid", 0):
                k = e
            if o.get("qid", 0):
                q = e
        vt = lambda emb: emb(ids).view(B, T, H, dv).transpose(1, 2)
        if hasattr(self, "ve"):
            v = v + vt(self.ve)
        if self.layer == 0 and o.get("vres", 0):
            CTX["v0"] = v
        if hasattr(self, "vrl"):
            v = v + self.vrl * CTX["v0"]
        if o.get("kprev", 0):
            k = torch.cat([k[:, :, :1], k[:, :, :-1]], 2)
        q, k = self.pre_qk(q, k, x)
        a = F.softplus(self.a(x) + self.dt_bias)
        g = (-self.A_log.exp() * a).transpose(1, 2)
        nd = int(o.get("nodec", 0))
        if nd == 1:
            g = g * 0
        elif nd == 2:
            g = torch.cat([g[:, :1] * 0, g[:, 1:]], 1)
        G = g.cumsum(2)
        sc = self.qsl.exp() if hasattr(self, "qsl") else dk ** -0.5
        lam = self.laml.sigmoid() if hasattr(self, "laml") else None
        if hasattr(self, "pk"):                        # persistent rows prepended at the start of the sequence
            P = self.pk.shape[1]
            kq = torch.cat([l2n(self.pk)[None].expand(B, -1, -1, -1), k], 2)
            vq = torch.cat([self.pv[None].expand(B, -1, -1, -1), v], 2)
            bq = torch.cat([self.pb.sigmoid()[None].expand(B, -1, -1), beta], 2)
            qq = torch.cat([torch.zeros(B, H, P, dk, device=x.device), q * sc], 2)
            Gq = torch.cat([torch.zeros(B, H, P, device=x.device), G], 2)
            y, u = delta_gen(qq, kq, vq, bq, Gq, lam, bool(o.get("rbw", 0)), float(o.get("mom", 0)))
            y, u = y[:, :, P:], u[:, :, P:]
        else:
            y, u = delta_gen(q * sc, k, v, beta, G, lam, bool(o.get("rbw", 0)), float(o.get("mom", 0)))
        y = self.extra_read(y, q * sc, k, v, beta, u, G, x)
        if hasattr(self, "xrg"):
            k0, u0, G0 = CTX["xr"]
            incl = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(0)
            dec0 = (G0[..., :, None] - G0[..., None, :]).masked_fill(~incl, float("-inf")).exp()
            y = y + self.xrg * (((q * sc) @ k0.transpose(-1, -2)) * dec0).masked_fill(~incl, 0.0) @ u0
        if o.get("xread", 0) and (self.layer == 0 or int(o["xread"]) == 2):
            CTX["xr"] = (k, u, G)
        if hasattr(self, "D"):
            y = y + self.D * v
        if hasattr(self, "D2"):
            y = y + self.D2 * F.pad(v[:, :, :-1], (0, 0, 1, 0))
        if hasattr(self, "vo"):
            y = y + vt(self.vo)
        if hasattr(self, "idgn"):
            y = y * (1 + vt(self.idgn))
        gl = (self.g(x) + (self.idgt(ids) if hasattr(self, "idgt") else 0.0)).view(B, T, H, dv)
        og = o.get("og", "silu")
        gate = F.silu(gl) if og == "silu" else (gl.sigmoid() if og == "sig" else 1.0)
        yo = self.onorm(y.transpose(1, 2))
        out = self.o((yo * gate if og != "none" else yo).reshape(B, T, -1))
        if hasattr(self, "q2"):
            q2 = l2n(self.q2(x).view(B, T, H, dk)).transpose(1, 2)
            incl = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(0)
            dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
            y2 = (((q2 * sc) @ k.transpose(-1, -2)) * dec).masked_fill(~incl, 0.0) @ u
            y2 = self.on2(y2.transpose(1, 2)) * (gate if og != "none" else 1.0)
            out = out + self.o2(y2.reshape(B, T, -1))
        if hasattr(self, "indm"):
            out = out + self.indm(x, ids)
        if hasattr(self, "indm2"):
            out = out + self.indm2(x, ids)
        if hasattr(self, "indm3"):
            out = out + self.indm3(x, ids)
        if hasattr(self, "nghm"):
            out = out + self.nghm(x, ids)
        return out

    def pre_qk(self, q, k, x):
        return q, k

    def extra_read(self, y, qs, k, v, beta, u, G, x):
        return y



def _incl(T, dev, shift=0):
    return torch.ones(T, T, dtype=torch.bool, device=dev).tril(-shift)


@register("gdnv")
class GDNV(GDNW):
    """GDN with round-8 read-side switches (ours unless marked), on top of every GDNW switch:
    hop2:1   second read with query q2 = l2n(q + W_h y1): two-hop retrieval from the same state in one layer
    qdec:B   query-controlled recency at read time: W_ij *= exp(-softplus(l(x_i)+B) (i-j))
    ts2:F    second delta state with decay G*F (F<1 slower forgetting), added per channel
    hier:C   chunk memory: delta rule over chunk-mean keys/values, read by tokens of later chunks
    lbk:D    extra read of old memory only (keys at j <= t-D)
    drope:B  data-dependent rotary angles on q,k (Known: Mamba-3 style), angle step softplus(l(x)+B)"""
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, layer, o)
        d, H, dk, dv = cfg.d_model, self.H, self.dk, self.dv
        if o.get("hop2", 0):
            self.Wh = nn.Parameter(torch.randn(H, dv, dk) * dv ** -0.5)
            self.hg = nn.Parameter(torch.full((H, 1, 1), 0.3))
        if "qdec" in o:
            self.lqd = nn.Linear(d, H, bias=False)
            self.bqd = nn.Parameter(torch.full((H,), float(o["qdec"])))
        if o.get("ts2", 0):
            self.tsm = nn.Parameter(torch.full((H, 1, dv), 0.5))
        if o.get("hier", 0):
            self.hcg = nn.Parameter(torch.full((H, 1, 1), 0.3))
        if o.get("lbk", 0):
            self.lbg = nn.Parameter(torch.full((H, 1, 1), 0.3))
        if o.get("bk", 0):                             # bigram-identity keys mixed into the main delta state (ours)
            self.bkR = int(o.get("bkR", 65536))
            self.Ebk = nn.Embedding(self.bkR, H * dk); nn.init.normal_(self.Ebk.weight, 0, 1.0)
            self.bkg = nn.Parameter(torch.full((H, 1, 1), float(o["bk"])))
        if "drope" in o:
            self.ldr = nn.Linear(d, H, bias=False)
            self.bdr = nn.Parameter(torch.full((H,), float(o["drope"])))
            self.register_buffer("dinv", 1.0 / (100.0 ** (torch.arange(dk // 2).float() / (dk // 2))), persistent=False)

    def pre_qk(self, q, k, x):
        if hasattr(self, "Ebk"):
            ids = CTX["ids"]; B, T = ids.shape
            ip, ipp = F.pad(ids[:, :-1], (1, 0)), F.pad(ids[:, :-2], (2, 0))
            ek = l2n(self.Ebk((ipp * 1000003 + ip) % self.bkR).view(B, T, self.H, self.dk)).transpose(1, 2)
            eq = l2n(self.Ebk((ip * 1000003 + ids) % self.bkR).view(B, T, self.H, self.dk)).transpose(1, 2)
            q, k = l2n(q + self.bkg * eq), l2n(k + self.bkg * ek)
        if not hasattr(self, "ldr"):
            return q, k
        th = F.softplus(self.ldr(x) + self.bdr).cumsum(1).transpose(1, 2)[..., None] * self.dinv   # [B,H,T,dk/2]
        c, s = th.cos(), th.sin()
        def rot(z):
            z1, z2 = z.chunk(2, -1)
            return torch.cat([z1 * c - z2 * s, z1 * s + z2 * c], -1)
        return rot(q), rot(k)

    def extra_read(self, y, qs, k, v, beta, u, G, x):
        o = self.ow
        B, H, T, dk = qs.shape
        dev = x.device
        incl = _incl(T, dev)
        dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
        if hasattr(self, "lqd"):
            r = F.softplus(self.lqd(x) + self.bqd).transpose(1, 2)[..., None]            # [B,H,T,1]
            ii = torch.arange(T, device=dev)
            age = (ii[:, None] - ii[None, :]).clamp(min=0).float()
            y = ((qs @ k.transpose(-1, -2)) * dec * torch.exp(-r * age)).masked_fill(~incl, 0.0) @ u
        if hasattr(self, "Wh"):
            q2 = l2n(qs / qs.norm(dim=-1, keepdim=True).clamp(min=1e-6) + torch.einsum("bhtv,hvk->bhtk", y, self.Wh))
            y2 = ((q2 * self.dk ** -0.5 if not hasattr(self, "qsl") else q2 * self.qsl.exp()) @ k.transpose(-1, -2) * dec).masked_fill(~incl, 0.0) @ u
            y = y + self.hg * y2
        if hasattr(self, "tsm"):
            y2, _ = delta_gen(qs, k, v, beta, G * float(o["ts2"]))
            y = y + self.tsm * y2
        if hasattr(self, "hcg"):
            C = int(o["hier"]); N = T // C
            kc = l2n(k[:, :, :N * C].reshape(B, H, N, C, dk).mean(3))
            vc = v[:, :, :N * C].reshape(B, H, N, C, -1).mean(3)
            bc = beta[:, :, :N * C].reshape(B, H, N, C).mean(3)
            A = ((kc @ kc.transpose(-1, -2)) * bc[..., :, None]).masked_fill(~_incl(N, dev, 1), 0.0)
            uc = torch.linalg.solve_triangular(torch.eye(N, device=dev).expand_as(A) + A, bc[..., None] * vc, upper=False, unitriangular=True)
            tc = torch.arange(T, device=dev) // C
            m = torch.arange(N, device=dev)[None, :] < tc[:, None]                          # chunk c finished before t's chunk
            y = y + self.hcg * ((qs @ kc.transpose(-1, -2)).masked_fill(~m, 0.0) @ uc)
        if hasattr(self, "lbg"):
            D = int(o["lbk"])
            y = y + self.lbg * ((qs @ k.transpose(-1, -2)) * dec).masked_fill(~_incl(T, dev, D), 0.0) @ u
        return y


@register("indw")
class IndWrap(nn.Module):
    """any base mixer (base:kda|gdn|mla|xattn|mamba2) plus the round-9 induction delta memories (ind/ind2/ind3 knobs as in gdnw)."""
    def __init__(self, cfg, layer, o):
        super().__init__()
        from mixers import REG
        self.m = REG[str(o.get("base", "kda"))](cfg, layer, o)
        indl = str(o.get("indl", ""))
        if (o.get("ind", 0) or o.get("indh", 0)) and (not indl or str(layer) in indl):
            kw = dict(dk=int(o.get("indd", 32)), dv=int(o.get("indv", 32)), R=o.get("indR"), ad=float(o.get("indad", -4.0)),
                      nh=int(o.get("indnh", 1)), og=int(o.get("indog", 0)), beta1=int(o.get("indb1", 0)))
            self.indm = IndHead(cfg, mode="x", key=str(o.get("indk", "uni")), hk=int(o.get("indh", 0)), **kw)
            if o.get("ind2", 0):
                self.indm2 = IndHead(cfg, mode="x", key=str(o.get("ind2k", "bi")), **kw)
            if o.get("ind3", 0):
                self.indm3 = IndHead(cfg, mode="x", key=str(o.get("ind3k", "tri")), **kw)
        if o.get("eng"):
            self.eng = Engram(cfg.d_model, int(o["eng"]))
        wbl = str(o.get("wbl", ""))                                     # round 17: layers that get the side branch
        if o.get("wb") and (not wbl or str(layer) in wbl):
            pick = lambda t: Wild7 if t in Wild7.NEW else Wild6 if t in Wild6.NEW else Wild5 if t in Wild5.NEW else Wild4 if t in Wild4.NEW else (Wild3 if t in Wild3.NEW else (Wild2 if t in Wild2.NEW else Wild))
            self.wbs = nn.ModuleList([pick(t)(cfg, t, db=int(o.get("wbd", 32))) for t in str(o["wb"]).split("+")])
            self.wbz = float(o.get("wbz", 0))                           # wbz:1 = wrapper null: branch runs, output x 0
        if o.get("br"):
            bw = dict(db=int(o.get("brd", 32)), gate=int(o.get("brg", 1)), w=int(o.get("brw", 16)), C=int(o.get("brC", 8)),
                      sc=str(o.get("brsc", "0.5:0.9:0.99")), nh=int(o.get("brnh", 1)))
            self.brs = nn.ModuleList([Branch(cfg, t, **bw) for t in str(o["br"]).split("+")])

    def forward(self, x, cos, sin):
        out = self.m(x, cos, sin)
        ids = CTX["ids"]
        for n in ("indm", "indm2", "indm3"):
            if hasattr(self, n):
                out = out + getattr(self, n)(x, ids)
        if hasattr(self, "eng"):
            out = out + self.eng(x, ids)
        if hasattr(self, "wbs"):
            for b in self.wbs:
                out = out + (b(x) * 0.0 if self.wbz else b(x))
        if hasattr(self, "brs"):
            for b in self.brs:
                out = out + b(x)
        return out


class Engram(nn.Module):
    """Engram-style memory (DeepSeek, arXiv 2601.07372), lab baseline for order O32, copied from
    compute/arena/env/blocks.py: a hashed table keyed by the bigram (previous id, id), 2 hash heads of K rows x d/2
    concatenated, context gate a = sigmoid(<rms(h), rms(e Wk)> / sqrt(d)), value e Wv, then y = v~ + SiLU(causal
    depthwise conv4(rms(v~))), v~ = a * e Wv. Added to the mixer output. Parameters: K*d + 2d^2 + 4d. Option eng:K."""
    def __init__(self, d, K, heads=2):
        super().__init__()
        self.d, self.K, self.H = d, K, heads
        self.k_tab = nn.Parameter(torch.randn(heads, K, d // heads) * 0.02)
        self.k_wk, self.k_wv = nn.Parameter(torch.randn(d, d) * 0.02), nn.Parameter(torch.randn(d, d) * 0.02)
        self.k_conv = nn.Parameter(torch.zeros(4, d))
        self.mul = [(1000003 + 7919 * h, 2654435761 % 1000033 + 104729 * h, 12345 + 31 * h) for h in range(heads)]

    def post_init(self):
        nn.init.zeros_(self.k_conv)                    # the model re-init draws every 2-D tensor; keep conv at 0

    def forward(self, x, ids2):
        B, T = ids2.shape
        rms = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = x.reshape(B * T, -1)
        cur = ids2.reshape(-1)
        pv = torch.cat([torch.zeros_like(ids2[:, :1]), ids2[:, :-1]], 1).reshape(-1)
        e = torch.cat([self.k_tab[h][(pv * a + cur * b + c) % self.K] for h, (a, b, c) in enumerate(self.mul)], -1)
        g = torch.sigmoid((rms(x) * rms(e @ self.k_wk)).sum(-1, keepdim=True) / math.sqrt(self.d))
        v = (g * (e @ self.k_wv)).view(B, T, -1)
        r = rms(v)
        c = sum(self.k_conv[j] * torch.cat([torch.zeros_like(r[:, :j]), r[:, :T - j]], 1) for j in range(4))
        return v + F.silu(c)


class Wild(nn.Module):
    """round-15 side branch on a base mixer (option wb:<typ>[+<typ>]), 28 cross-position ops no search had tried.
    u = W_i x (db channels) -> causal op over positions -> * silu(W_g x) -> W_o. Ops (Known = published mechanism):
    dlag: sum of u at lags 1,2,4..128, per-channel weights (Known: dilated conv / WaveNet) | dlagg: lag weights chosen
    per token by softmax(W x) | lconv: free causal kernel of length T per channel (Known: long conv) | plaw: power-law
    kernel (age+1)^-a per channel (ours) | cplx: damped oscillation kernel r^age cos(w age) (Known: LRU/S4D) | topk:
    softmax attention over the top 8 past keys (Known: Explicit Sparse Transformer) | lbpool: means over log-spaced
    windows 1..128, one scale per channel group (ours) | lbatt: query picks among the 8 window means (ours) | diff:
    differential attention A1 - l A2 (Known: Diff Transformer) | succ: attention on past keys returning the token after
    the match, u_{s+1} (Known: induction head as a block) | volt: u_t * lag mix of u (ours, Volterra) | routla: linear
    attention with soft same-route weights over 4 routes and decay (Known: Mixture-of-Memories) | sgu: u * (W_T z + b),
    causal T x T token mix (Known: gMLP) | tmix: softmax of a learned causal T x T matrix (Known: Synthesizer) | cov:
    decayed 8x8 outer products (ours, second-order pooling) | gsa: 8 decayed slots written by softmax(W x), read by
    softmax over slots (Known: GSA/ABC) | lsparse: attention over lags 0,1,2,4..128 only (Known: LogSparse) | rank: soft
    running rank of u_t among past u (ours) | ldelay: learned real-valued delay per channel (Known: DCLS delays) | lsh:
    attention inside random-projection buckets of k (Known: Reformer) | tsince: soft time since the channel last fired
    (ours) | gauss: 4 content-free Gaussian windows, learned centre and width (Known: hard-coded Gaussian attention) |
    gaussc: gauss prior + content scores | cvar: decayed variance of u (ours) | segmean: mean over the current segment,
    soft boundaries from W x (Known: dynamic chunking) | spmax: sparsemax attention (Known) | relu2: relu(qk)^2 / t
    (Known: ReLU attention) | sigm: sigmoid attention with bias -log T (Known: sigmoid attention)."""
    LAGS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d, T = cfg.d_model, cfg.seq_len
        self.typ, self.db = typ, db
        dout = {"cov": 64, "rank": 16, "tsince": 16}.get(typ, db)
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, dout, bias=False)
        self.wo = nn.Linear(dout, d, bias=False)
        if typ in ("topk", "lbatt", "succ", "routla", "gsa", "lsparse", "lsh", "gaussc", "spmax", "relu2", "sigm", "sgu"):
            self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "diff":
            self.wq, self.wk = nn.Linear(d, 2 * db, bias=False), nn.Linear(d, 2 * db, bias=False)
            self.lam = nn.Parameter(torch.tensor(0.0))                     # l = 0.8 * sigmoid(.) * 2 -> 0.8 at init
        if typ in ("dlag", "volt"):
            self.lw = nn.Parameter(torch.full((len(self.LAGS), db), 0.1))
        if typ == "dlagg":
            self.wa = nn.Linear(d, len(self.LAGS), bias=False)
        if typ == "lconv":
            self.K = nn.Parameter(torch.zeros(db, T))
        if typ == "plaw":
            self.pa = nn.Parameter(torch.linspace(0.3, 2.0, db))
        if typ == "cplx":
            self.rho = nn.Parameter(torch.linspace(2.0, 5.0, db))
            self.om = nn.Parameter(torch.linspace(0.0, 0.8, db))
        if typ == "routla":
            self.wr = nn.Linear(d, 4, bias=False)
        if typ in ("sgu", "tmix"):
            self.Wt = nn.Parameter(torch.zeros(T, T))
            self.bt = nn.Parameter(torch.ones(T))
        if typ == "gsa":
            self.wa = nn.Linear(d, 8, bias=False)
        if typ == "lsh":
            self.register_buffer("R", torch.randn(db, 4), persistent=False)
        if typ == "ldelay":
            self.dl = nn.Parameter(torch.linspace(-4.0, 2.0, db))
        if typ in ("gauss", "gaussc"):
            self.mu = nn.Parameter(torch.tensor([1.0, 4.0, 16.0, 64.0]))
            self.ls = nn.Parameter(torch.log(torch.tensor([1.0, 2.0, 8.0, 32.0])))
        if typ == "segmean":
            self.wb = nn.Linear(d, 1, bias=True)

    def post_init(self):                                                  # the model re-init draws every 2-D tensor
        if hasattr(self, "lw"): nn.init.constant_(self.lw, 0.1)
        if hasattr(self, "K"):
            with torch.no_grad():
                self.K.copy_(0.1 * 0.9 ** torch.arange(self.K.shape[1]).float()[None].expand_as(self.K))
        if hasattr(self, "Wt"):
            nn.init.zeros_(self.Wt)

    @staticmethod
    def _shift(u, l):
        return torch.cat([torch.zeros_like(u[:, :l]), u[:, :-l]], 1) if l < u.shape[1] else torch.zeros_like(u)

    @staticmethod
    def _msoft(s, m):                                                      # masked softmax, rows with no key -> 0
        has = m.any(-1, keepdim=True)
        w = s.masked_fill(~m, float("-inf"))
        return torch.where(has, w, torch.zeros_like(w)).softmax(-1) * has

    def forward(self, x):
        B, T, _ = x.shape
        u = self.wi(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        incl = age >= 0
        agef = age.clamp(min=0).float()
        t, db = self.typ, self.db
        sc = db ** -0.5
        if t in ("dlag", "volt"):
            lags = self.LAGS if t == "dlag" else self.LAGS[:4]
            y = sum(self.lw[n] * self._shift(u, l) for n, l in enumerate(lags))
            if t == "volt":
                y = u * y
        elif t == "dlagg":
            a = self.wa(x).softmax(-1)
            y = sum(a[..., n:n + 1] * self._shift(u, l) for n, l in enumerate(self.LAGS))
        elif t in ("lconv", "plaw", "cplx"):
            if t == "lconv":
                M = self.K[:, :T][:, age.clamp(min=0)]
            elif t == "plaw":
                M = (agef[None] + 1) ** (-self.pa.clamp(min=0.05)[:, None, None])
            else:
                r = torch.sigmoid(self.rho)[:, None, None]
                M = (1 - r) * r ** agef[None] * torch.cos(self.om[:, None, None] * agef[None])
            M = M.masked_fill(~incl[None], 0.0)
            if t == "plaw":
                M = M / M.sum(-1, keepdim=True)
            y = torch.einsum("ctj,bjc->btc", M, u)
        elif t in ("lbpool", "lbatt"):
            S = torch.cat([torch.zeros_like(u[:, :1]), u.cumsum(1)], 1)        # S[t+1] = sum u[0..t]
            means = []
            for w in self.LAGS:
                lo = (ii + 1 - w).clamp(min=0)
                means.append((S[:, ii + 1] - S[:, lo]) / (ii + 1 - lo).float()[None, :, None])
            if t == "lbpool":
                g = db // len(self.LAGS)
                y = torch.cat([m[..., n * g:(n + 1) * g] for n, m in enumerate(means)], -1)
            else:
                Mn = torch.stack(means, 2)                                     # [B,T,8,db]
                a = (torch.einsum("btc,btkc->btk", self.wq(x), Mn) * sc).softmax(-1)
                y = torch.einsum("btk,btkc->btc", a, Mn)
        elif t == "diff":
            q, k = self.wq(x), self.wk(x)
            A1 = (q[..., :db] @ k[..., :db].transpose(-1, -2) * sc).masked_fill(~incl, float("-inf")).softmax(-1)
            A2 = (q[..., db:] @ k[..., db:].transpose(-1, -2) * sc).masked_fill(~incl, float("-inf")).softmax(-1)
            y = (A1 - 0.8 * 2 * torch.sigmoid(self.lam) * A2) @ u
        elif t in ("routla", "gsa"):
            q, k = self.wq(x), self.wk(x)
            if t == "routla":
                p = self.wr(x).softmax(-1)
                A = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-1, -2) * (p @ p.transpose(-1, -2)) * 0.97 ** agef
                A = A.masked_fill(~incl, 0.0)
                y = (A @ u) / A.sum(-1, keepdim=True).clamp(min=1e-6)
            else:
                a = self.wa(x).softmax(-1)                                     # [B,T,8]
                M = (0.05 * 0.95 ** agef).masked_fill(~incl, 0.0)
                Ks = (M @ (a[..., :, None] * k[..., None, :]).reshape(B, T, -1)).view(B, T, 8, db)
                Vs = (M @ (a[..., :, None] * u[..., None, :]).reshape(B, T, -1)).view(B, T, 8, db)
                w = (torch.einsum("btc,btmc->btm", q, Ks) * sc).softmax(-1)
                y = torch.einsum("btm,btmc->btc", w, Vs)
        elif t in ("sgu", "tmix"):
            W = self.Wt[:T, :T]
            if t == "sgu":
                y = u * ((W.masked_fill(~incl, 0.0) @ self.wq(x)) + self.bt[:T, None])
            else:
                y = (W + torch.where(incl, 0.0, float("-inf"))).softmax(-1) @ u
        elif t == "cov":
            v = u[..., :8]
            o = (v[..., :, None] * v[..., None, :]).reshape(B, T, 64)
            y = (0.1 * 0.9 ** agef).masked_fill(~incl, 0.0) @ o
        elif t == "cvar":
            M = (0.1 * 0.9 ** agef).masked_fill(~incl, 0.0)
            M = M / M.sum(-1, keepdim=True)
            m = M @ u
            y = M @ (u * u) - m * m
        elif t == "rank":
            v = u[..., :16]
            strict = age > 0
            r = torch.sigmoid((v[:, :, None, :] - v[:, None, :, :]) * 2.0) * strict[None, :, :, None]
            y = r.sum(2) / ii.clamp(min=1).float()[None, :, None]
        elif t == "tsince":
            v = u[..., :16]
            L = torch.log1p(-torch.sigmoid(4.0 * (v - 1.0)) + 1e-6).cumsum(1)  # [B,T,c]
            Lp = torch.cat([torch.zeros_like(L[:, :1]), L[:, :-1]], 1)         # L_{j-1}
            E = torch.exp((L[:, :, None, :] - Lp[:, None, :, :]).clamp(max=0.0)) * incl[None, :, :, None]
            y = torch.log1p(E.sum(2))
        elif t == "ldelay":
            dly = 128.0 * torch.sigmoid(self.dl)                               # [db]
            lo = dly.floor().long(); fr = dly - lo.float()
            def g(sh):
                idx = ii[:, None] - sh[None, :]                                # [T,db]
                ok = (idx >= 0).float()
                return torch.gather(u, 1, idx.clamp(min=0)[None].expand(B, T, db)) * ok[None]
            y = (1 - fr) * g(lo) + fr * g(lo + 1)
        elif t == "segmean":
            b = torch.sigmoid(self.wb(x)).squeeze(-1)                          # [B,T]
            L = torch.log1p(-b + 1e-6).cumsum(1)
            W = torch.exp((L[:, :, None] - L[:, None, :]).clamp(max=0.0)).masked_fill(~incl[None], 0.0)
            y = (W @ u) / W.sum(-1, keepdim=True)
        elif t in ("gauss", "gaussc"):
            hs = db // 4
            sig = torch.exp(self.ls)[:, None, None]
            s = -((agef[None] - self.mu[:, None, None]) ** 2) / (2 * sig ** 2)          # [4,T,T]
            s = s[None].expand(B, 4, T, T)
            if t == "gaussc":
                q, k = self.wq(x).view(B, T, 4, hs).transpose(1, 2), self.wk(x).view(B, T, 4, hs).transpose(1, 2)
                s = s + q @ k.transpose(-1, -2) * hs ** -0.5
            A = s.masked_fill(~incl, float("-inf")).softmax(-1)
            y = (A @ u.view(B, T, 4, hs).transpose(1, 2)).transpose(1, 2).reshape(B, T, db)
        else:
            q, k = self.wq(x), self.wk(x)
            s = q @ k.transpose(-1, -2) * sc
            if t == "topk":
                kth = s.masked_fill(~incl, float("-inf")).topk(min(8, T), -1).values[..., -1:]
                y = (s.masked_fill(~incl | (s < kth), float("-inf"))).softmax(-1) @ u
            elif t == "lsparse":
                m = incl & ((age & (age - 1)) == 0)
                y = s.masked_fill(~m, float("-inf")).softmax(-1) @ u
            elif t == "succ":
                vs = torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1)    # value at s = u_{s+1}
                y = self._msoft(s, (age > 0)[None].expand(B, T, T)) @ vs
            elif t == "lsh":
                bk = (k @ self.R).argmax(-1)                                   # [B,T]
                m = incl[None] & (bk[:, :, None] == bk[:, None, :])
                y = s.masked_fill(~m, float("-inf")).softmax(-1) @ u
            elif t == "spmax":
                z = s.masked_fill(~incl, -1e4)
                zs, _ = z.sort(-1, descending=True)
                kk = torch.arange(1, T + 1, device=dev, dtype=z.dtype)
                cs = zs.cumsum(-1)
                sup = (1 + kk * zs) > cs
                kz = sup.sum(-1, keepdim=True)
                tau = (cs.gather(-1, kz - 1) - 1) / kz.to(z.dtype)
                y = (z - tau).clamp(min=0) @ u
            elif t == "relu2":
                y = (F.relu(s) ** 2).masked_fill(~incl, 0.0) @ u / (ii + 1).float()[None, :, None]
            else:                                                              # sigm
                y = torch.sigmoid(s - math.log(T)).masked_fill(~incl, 0.0) @ u
        return self.wo(y * F.silu(self.gate(x)))


class Wild2(nn.Module):
    """round-16 side branch (option wb:<typ>, same wiring as Wild), 28 more cross-position ops.
    u = W_i x (db) -> causal op -> * silu(W_g x) -> W_o. Known = published mechanism:
    hyena: order-2 long conv with data gates (Known: Hyena) | mingru: minGRU recurrence h=(1-z)h+z u (Known) | stack:
    differentiable stack, depth 8 (Known: Joulin-Mikolov) | ccount: log1p of running soft event count (ours) | parity:
    cos/sin of pi * running soft count (ours) | clock: cos/sin of a content-driven phase (ours) | cope: contextual position
    attention (Known: CoPE) | alibi: 4-head attention with linear age bias (Known) | fox: attention + forget-gate bias
    (Known: Forgetting Transformer) | stickb: stick-breaking attention (Known) | selatt: selective attention, earlier
    queries mask keys (Known) | dval: attention reading u_s - u_{s-1} (ours) | talk: talking-heads attention (Known) |
    mta: 3x3 conv over the score map (Known: Multi-Token Attention) | hop: two-step Hopfield read (Known) | perf:
    random-feature linear attention (Known: Performer) | cosatt: cosine attention with learned temperature (Known) |
    sinkwin: first token + window 8 (Known: StreamingLLM) | hard: straight-through argmax attention (ours) | landmk:
    local chunk + summaries of finished 16-token chunks (Known: landmark attention) | emabank: per-token softmax over 8
    fixed EMAs (ours) | haar: Haar differences at 4 scales (ours) | novw: decayed mean weighted by novelty (ours) | ret:
    4-head retention (Known: RetNet) | soft: plain softmax attention (control for sigm) | relpb: softmax + log-bucket
    age bias (Known: T5 bias) | based: 2nd-order Taylor linear attention (Known: Based) | softpick: rectified softmax
    (Known: softpick)."""
    NEW = set("hyena mingru stack ccount parity clock cope alibi fox stickb selatt dval talk mta hop perf cosatt sinkwin "
              "hard landmk emabank haar novw ret soft relpb based softpick".split())

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d, T = cfg.d_model, cfg.seq_len
        self.typ, self.db = typ, db
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, db, bias=False)
        self.wo = nn.Linear(db, d, bias=False)
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "hyena":
            self.K1, self.K2 = nn.Parameter(torch.zeros(db, T)), nn.Parameter(torch.zeros(db, T))
        if typ == "stack":
            self.wa = nn.Linear(d, 3, bias=False)
        if typ == "clock":
            self.rate = nn.Parameter(torch.full((16,), -2.0))
        if typ == "cope":
            self.E = nn.Parameter(torch.zeros(65, db))
        if typ == "fox":
            self.wf = nn.Linear(d, 1, bias=True)
        if typ == "talk":
            self.P1, self.P2 = nn.Parameter(torch.eye(4)), nn.Parameter(torch.eye(4))
        if typ == "mta":
            self.Kc = nn.Parameter(torch.zeros(3, 3))
        if typ == "cosatt":
            self.tau = nn.Parameter(torch.tensor(math.log(10.0)))
        if typ == "perf":
            self.register_buffer("R", torch.randn(db, 32), persistent=False)
        if typ == "emabank":
            self.wa = nn.Linear(d, 8, bias=False)
        if typ == "novw":
            self.ns = nn.Parameter(torch.tensor(1.0))
        if typ == "relpb":
            self.bb = nn.Parameter(torch.zeros(16))

    def post_init(self):                                                  # the model re-init draws every 2-D tensor
        T = getattr(self, "K1", None)
        with torch.no_grad():
            if hasattr(self, "K1"):
                k = 0.1 * 0.9 ** torch.arange(self.K1.shape[1]).float()
                self.K1.copy_(k[None].expand_as(self.K1)); self.K2.copy_(k[None].expand_as(self.K2))
            if hasattr(self, "P1"):
                self.P1.copy_(torch.eye(4)); self.P2.copy_(torch.eye(4))
            if hasattr(self, "Kc"):
                self.Kc.zero_(); self.Kc[2, 2] = 1.0
            if hasattr(self, "E"):
                self.E.zero_()
            if hasattr(self, "wf"):
                nn.init.zeros_(self.wf.weight); nn.init.constant_(self.wf.bias, 3.0)

    @staticmethod
    def _sm(s, m):
        return s.masked_fill(~m, float("-inf")).softmax(-1)

    @staticmethod
    def _scan(alog, b, C=32):                                             # h_t = exp(alog_t) h_{t-1} + b_t, alog in [-0.7, 0]
        B, T, c = b.shape
        h0 = torch.zeros(B, 1, c, device=b.device, dtype=b.dtype)
        outs = []
        for s in range(0, T, C):
            a, bb = alog[:, s:s + C], b[:, s:s + C]
            A = a.cumsum(1)
            h = torch.exp(A) * (h0 + (bb * torch.exp(-A)).cumsum(1))
            outs.append(h); h0 = h[:, -1:]
        return torch.cat(outs, 1)

    def forward(self, x):
        B, T, _ = x.shape
        t, db = self.typ, self.db
        u, q, k = self.wi(x), self.wq(x), self.wk(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        incl = age >= 0
        agef = age.clamp(min=0).float()
        sc = db ** -0.5
        if t == "hyena":
            def conv(K, v):
                M = K[:, :T][:, age.clamp(min=0)].masked_fill(~incl[None], 0.0)
                return torch.einsum("ctj,bjc->btc", M, v)
            y = k * conv(self.K2, q * conv(self.K1, u))
        elif t == "mingru":
            z = 0.5 * torch.sigmoid(q)
            y = self._scan(torch.log1p(-z), z * u)
        elif t == "stack":
            a = self.wa(x).softmax(-1)                                       # push, pop, keep
            S = torch.zeros(B, 8, db, device=dev, dtype=u.dtype)
            outs = []
            for s in range(T):
                p, o, n = a[:, s, 0, None, None], a[:, s, 1, None, None], a[:, s, 2, None, None]
                pushed = torch.cat([u[:, s, None], S[:, :-1]], 1)
                popped = torch.cat([S[:, 1:], torch.zeros_like(S[:, :1])], 1)
                S = p * pushed + o * popped + n * S
                outs.append(S[:, 0])
            y = torch.stack(outs, 1)
        elif t == "ccount":
            y = torch.log1p(torch.sigmoid(u).cumsum(1))
        elif t in ("parity", "clock"):
            v = u[..., :db // 2]
            ph = math.pi * torch.sigmoid(v).cumsum(1) if t == "parity" else (F.softplus(v) * torch.exp(self.rate[:db // 2])).cumsum(1)
            y = torch.cat([torch.cos(ph), torch.sin(ph)], -1)
        elif t == "emabank":
            lam = torch.tensor([0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995], device=dev)
            M = ((1 - lam)[:, None, None] * lam[:, None, None] ** agef[None]).masked_fill(~incl[None], 0.0)
            E = torch.einsum("kts,bsc->btkc", M, u)
            y = torch.einsum("btk,btkc->btc", self.wa(x).softmax(-1), E)
        elif t == "haar":
            S = torch.cat([torch.zeros_like(u[:, :1]), u.cumsum(1)], 1)
            def wmean(lo, hi):                                               # mean of u over [lo, hi) clamped at 0
                lo, hi = lo.clamp(min=0), hi.clamp(min=0)
                n = (hi - lo).clamp(min=1).float()[None, :, None]
                return (S[:, hi] - S[:, lo]) / n
            g = db // 4; parts = []
            for n_, w in enumerate((2, 8, 32, 128)):
                h = w // 2
                d_ = wmean(ii + 1 - h, ii + 1) - wmean(ii + 1 - w, ii + 1 - h)
                parts.append(d_[..., n_ * g:(n_ + 1) * g])
            y = torch.cat(parts, -1)
        elif t == "novw":
            prev = torch.cat([torch.zeros_like(u[:, :1]), u[:, :-1]], 1)
            w = torch.sigmoid(self.ns * ((u - prev) ** 2).mean(-1) - 1.0)    # [B,T]
            M = (0.97 ** agef).masked_fill(~incl, 0.0)
            y = (M @ (w[..., None] * u)) / (M @ w[..., None]).clamp(min=1e-4)
        elif t in ("alibi", "talk", "ret"):
            hs = db // 4
            qh, kh, uh = (z.view(B, T, 4, hs).transpose(1, 2) for z in (q, k, u))
            s = qh @ kh.transpose(-1, -2) * hs ** -0.5                      # [B,4,T,T]
            if t == "alibi":
                m = torch.tensor([0.5, 0.25, 0.0625, 0.0156], device=dev)[:, None, None]
                A = self._sm(s - m * agef, incl)
            elif t == "talk":
                s = torch.einsum("hg,bgts->bhts", self.P1, s)
                A = torch.einsum("hg,bgts->bhts", self.P2, self._sm(s, incl))
            else:
                gam = torch.tensor([0.9, 0.95, 0.98, 0.995], device=dev)[:, None, None]
                A = (s * gam ** agef).masked_fill(~incl, 0.0)
            yh = A @ uh
            if t == "ret":
                yh = yh * torch.rsqrt(yh.pow(2).mean(-1, keepdim=True) + 1e-5)
            y = yh.transpose(1, 2).reshape(B, T, db)
        elif t in ("perf", "based"):
            if t == "perf":
                def phi(z):
                    z = z * sc ** 0.5
                    return torch.exp(z @ self.R - z.pow(2).sum(-1, keepdim=True) / 2 - 4.0)
            else:
                def phi(z):
                    z = z * sc ** 0.5
                    z8 = z[..., :8]
                    return torch.cat([torch.ones_like(z[..., :1]), z, (z8[..., :, None] * z8[..., None, :]).flatten(-2) / math.sqrt(2)], -1)
            A = (phi(q) @ phi(k).transpose(-1, -2)).masked_fill(~incl, 0.0)
            y = (A @ u) / A.sum(-1, keepdim=True).clamp(min=1e-4) if t == "perf" else (A @ u) / A.sum(-1, keepdim=True).abs().clamp(min=1.0)
        elif t == "landmk":
            C = 16; nc = T // C
            ch = ii // C
            km = k.view(B, nc, C, db).mean(2); um = u.view(B, nc, C, db).mean(2)
            sl = q @ k.transpose(-1, -2) * sc
            sm_ = q @ km.transpose(-1, -2) * sc                              # [B,T,nc]
            ml = incl & (ch[:, None] == ch[None, :])
            mm = (torch.arange(nc, device=dev)[None, :] * C + C - 1) < ii[:, None]
            A = self._sm(torch.cat([sl, sm_], -1), torch.cat([ml, mm], -1))
            y = A[..., :T] @ u + A[..., T:] @ um
        else:
            s = q @ k.transpose(-1, -2) * sc
            if t == "cope":
                g = torch.sigmoid(s).masked_fill(~incl, 0.0)
                p = g.flip(-1).cumsum(-1).flip(-1).clamp(max=64.0)          # p_ts = sum_{r=s..t} g_tr
                qE = q @ self.E.t()                                          # [B,T,65]
                lo = p.floor().long().clamp(max=63); fr = p - lo.float()
                pe = qE.gather(-1, lo) * (1 - fr) + qE.gather(-1, lo + 1) * fr
                y = self._sm(s + pe, incl) @ u
            elif t == "fox":
                Fc = F.logsigmoid(self.wf(x)).squeeze(-1).cumsum(1)          # [B,T]
                y = self._sm(s + Fc[:, :, None] - Fc[:, None, :], incl) @ u
            elif t == "stickb":
                lsp = -F.softplus(s).masked_fill(~incl, 0.0)                 # log(1 - beta)
                Cr = lsp.flip(-1).cumsum(-1).flip(-1)                        # sum_{r>=s}
                A = torch.exp((F.logsigmoid(s) + Cr - lsp).masked_fill(~incl, -1e4)) * incl
                y = A @ u
            elif t == "selatt":
                Ss = F.relu(s).masked_fill(~(age > 0), 0.0)
                Ss = torch.cat([torch.zeros_like(Ss[..., :1]), Ss[..., 1:]], -1)  # token 0 is never masked
                Fm = Ss.cumsum(-2)
                Fm = torch.cat([torch.zeros_like(Fm[:, :1]), Fm[:, :-1]], 1)    # masks from earlier queries
                y = self._sm(s - Fm, incl) @ u
            elif t == "dval":
                prev = torch.cat([torch.zeros_like(u[:, :1]), u[:, :-1]], 1)
                y = self._sm(s, incl) @ (u - prev)
            elif t == "mta":
                z = s.masked_fill(~incl, 0.0)
                z = F.pad(z, (2, 0, 2, 0))[:, None]
                z = F.conv2d(z, self.Kc[None, None])[:, 0]
                y = self._sm(z, incl) @ u
            elif t == "hop":
                q2 = q + self._sm(s, incl) @ k
                y = self._sm(q2 @ k.transpose(-1, -2) * sc, incl) @ u
            elif t == "cosatt":
                qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
                y = self._sm(torch.exp(self.tau) * qn @ kn.transpose(-1, -2), incl) @ u
            elif t == "sinkwin":
                y = self._sm(s, incl & ((age < 8) | (ii[None, :] == 0))) @ u
            elif t == "hard":
                A = self._sm(s, incl)
                Hd = F.one_hot(A.argmax(-1), T).to(A.dtype)
                y = ((Hd - A).detach() + A) @ u
            elif t == "relpb":
                bk = (torch.log2(agef + 1) * 2).floor().long().clamp(max=15)
                y = self._sm(s + self.bb[bk], incl) @ u
            elif t == "softpick":
                e = torch.expm1(s.clamp(max=20.0)).masked_fill(~incl, 0.0)
                y = (F.relu(e) / (e.abs().sum(-1, keepdim=True) + 1e-6)) @ u
            else:                                                            # soft
                y = self._sm(s, incl) @ u
        return self.wo(y * F.silu(self.gate(x)))


class Wild3(nn.Module):
    """round-17 side branch (option wb:<typ>, same wiring as Wild), 30 more cross-position ops.
    u = W_i x (db) -> causal op -> * silu(W_g x) -> W_o. Known = published mechanism:
    kern: softmax of -|q-k|^2 (Known: L2 attention) | maxlog: running max over windows 1..128, one window per channel
    group (ours) | cmax: max-plus decayed max, max_s u_s - d (t-s) (ours) | osc: rotation recurrence with input-driven
    frequency (Known: oscillatory SSM, LinOSS) | vq: attention over 16 codebook keys, values = per-code running means
    (Known: Transformer-VQ) | wkv: RWKV-4 WKV with decay and current-token bonus (Known: RWKV) | moba: top-2 finished
    16-token blocks by q . mean k plus own block (Known: Kimi MoBA) | nsa: gated sum of compressed, selected and window
    branches (Known: DeepSeek NSA) | fire: learned MLP bias of log age / log t (Known: FIRE) | mlstm: matrix memory with
    exponential input gate and stabiliser (Known: xLSTM mLSTM) | ssmax: logits * s log t (Known: scalable softmax) |
    qtemp: per-query temperature from x (Known: selective attention temperature) | rnd: window 4 + fixed random 10% +
    first token (Known: BigBird) | strid: window 16 + every 16th token (Known: Sparse Transformer) | onlstm: EMA with
    cumax master forget gate over channels (Known: ON-LSTM) | latch: soft set/reset latch per channel (ours) | elman:
    tanh RNN (Known: Elman) | cosf: relu linear attention with cos age reweighting (Known: cosFormer) | entg: softmax head
    scaled by 1 - normalised entropy (ours) | pom: sigmoid(q) * running mean of u*k (Known: PoM) | tnorm: linear attention
    without denominator + RMSNorm (Known: TransNormer) | hrr: holographic bind/unbind memory (Known: Hrrformer) | fastw:
    decayed Hebbian self-similarity read (Known: fast weights) | loglin: Fenwick-level weighted linear attention (Known:
    log-linear attention) | kerple: -r1 log(1 + r2 age) bias (Known: KERPLE) | moh: 4 heads, router keeps top 2 (Known:
    MoH) | hist: decayed soft histogram of 8 channels in 4 bins (ours) | kwta: decayed mean of the top 4 of 32 channels
    (ours) | flash: relu^2 inside 16-token chunks + linear attention across chunks (Known: FLASH/GAU) | levy: decayed Levy
    area of an 8-channel path (Known: path signatures)."""
    NEW = set("kern maxlog cmax osc vq wkv moba nsa fire mlstm ssmax qtemp rnd strid onlstm latch elman cosf entg pom "
              "tnorm hrr fastw loglin kerple moh hist kwta flash levy".split())

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d, T = cfg.d_model, cfg.seq_len
        self.typ, self.db = typ, db
        dout = {"levy": 28}.get(typ, db)
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, dout, bias=False)
        self.wo = nn.Linear(dout, d, bias=False)
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "cmax":
            self.dec = nn.Parameter(torch.linspace(-4.0, 0.0, db))
        if typ in ("osc", "wkv"):
            self.rho = nn.Parameter(torch.linspace(1.0, 4.0, db))
        if typ == "wkv":
            self.bonus = nn.Parameter(torch.zeros(db))
        if typ == "vq":
            self.C = nn.Parameter(torch.randn(16, db))
        if typ == "nsa":
            self.wg = nn.Linear(d, 3, bias=True)
        if typ == "fire":
            self.f1, self.fb, self.f2 = nn.Parameter(torch.zeros(16)), nn.Parameter(torch.zeros(16)), nn.Parameter(torch.zeros(16))
        if typ == "mlstm":
            self.wf, self.wig = nn.Linear(d, 1, bias=True), nn.Linear(d, 1, bias=True)
        if typ == "ssmax":
            self.sp = nn.Parameter(torch.tensor(0.24))
        if typ == "qtemp":
            self.wt = nn.Linear(d, 1, bias=True)
        if typ == "rnd":
            g = torch.Generator().manual_seed(17)
            self.register_buffer("Rm", torch.rand(T, T, generator=g) < 0.1, persistent=False)
        if typ == "onlstm":
            self.wf = nn.Linear(d, db, bias=False)
        if typ == "elman":
            self.Wh = nn.Parameter(torch.zeros(db, db))
        if typ == "loglin":
            self.wl = nn.Linear(d, 9, bias=True)
        if typ == "kerple":
            self.r = nn.Parameter(torch.tensor([0.0, 0.0]))
        if typ == "moh":
            self.wr = nn.Linear(d, 4, bias=False)

    def post_init(self):                                                  # the model re-init draws every 2-D tensor
        with torch.no_grad():
            if hasattr(self, "Wh"):
                nn.init.orthogonal_(self.Wh); self.Wh.mul_(0.5)
            if hasattr(self, "wf") and self.typ == "mlstm":
                nn.init.zeros_(self.wf.weight); nn.init.constant_(self.wf.bias, 3.0)
                nn.init.zeros_(self.wig.weight); nn.init.zeros_(self.wig.bias)
            if hasattr(self, "wt"):
                nn.init.zeros_(self.wt.weight); nn.init.constant_(self.wt.bias, 0.5413)   # softplus = 1
            if hasattr(self, "wl"):
                nn.init.zeros_(self.wl.weight); nn.init.constant_(self.wl.bias, 0.5413)
            if hasattr(self, "wg"):
                nn.init.zeros_(self.wg.weight); nn.init.zeros_(self.wg.bias)
            if hasattr(self, "C"):
                self.C.normal_(0.0, 0.5)

    _sm = staticmethod(Wild2._sm)
    _scan = staticmethod(Wild2._scan)
    _msoft = staticmethod(Wild._msoft)
    _shift = staticmethod(Wild._shift)

    def forward(self, x):
        B, T, _ = x.shape
        t, db = self.typ, self.db
        u, q, k = self.wi(x), self.wq(x), self.wk(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        incl = age >= 0
        agef = age.clamp(min=0).float()
        sc = db ** -0.5
        if t in ("maxlog", "cmax"):
            m = u
            levels = [u]
            dl = F.softplus(self.dec) if t == "cmax" else None
            w = 1
            while w < T:
                sh = torch.cat([torch.full_like(m[:, :w], -1e4), m[:, :-w]], 1) if w < T else None
                m = torch.maximum(m, sh - (dl * w if t == "cmax" else 0.0))
                levels.append(m); w *= 2
            if t == "cmax":
                y = m
            else:
                g = db // 8
                y = torch.cat([levels[n][..., n * g:(n + 1) * g] for n in range(8)], -1)
        elif t == "osc":
            om = F.softplus(q) * 0.5
            ph = om.cumsum(1)
            alog = torch.log(0.5 + 0.5 * torch.sigmoid(self.rho)).expand_as(u)
            c, s_ = torch.cos(ph), torch.sin(ph)
            y = c * self._scan(alog, c * u) + s_ * self._scan(alog, s_ * u)
        elif t == "wkv":
            w = 0.01 + 0.69 * torch.sigmoid(-self.rho)                      # per-channel decay rate in (0.01, 0.7)
            ek = torch.exp(k.clamp(-5.0, 5.0))
            alog = (-w).expand_as(u)
            hn, hd = self._scan(alog, ek * u), self._scan(alog, ek)
            hn = torch.cat([torch.zeros_like(hn[:, :1]), hn[:, :-1]], 1); hd = torch.cat([torch.zeros_like(hd[:, :1]), hd[:, :-1]], 1)
            cur = torch.exp(self.bonus) * ek
            y = (hn + cur * u) / (hd + cur)
        elif t == "vq":
            dist = (k[..., None, :] - self.C).pow(2).sum(-1)                 # [B,T,16]
            Hs = (-dist).softmax(-1)
            H = F.one_hot(dist.argmin(-1), 16).to(u.dtype) + Hs - Hs.detach()
            N = H.cumsum(1)                                                  # [B,T,16]
            Su = (H[..., :, None] * u[..., None, :]).cumsum(1)               # [B,T,16,db]
            has = N.detach() > 0.5
            wgt = self._msoft(q @ self.C.t() * sc + torch.log(N.clamp(min=1e-3)), has)
            y = torch.einsum("btc,btcd->btd", wgt, Su / N.clamp(min=1.0)[..., None])
        elif t in ("moba", "nsa"):
            C = 16; nc = T // C
            ch = ii // C
            km = k.view(B, nc, C, db).mean(2); um = u.view(B, nc, C, db).mean(2)
            s = q @ k.transpose(-1, -2) * sc
            sm_ = q @ km.transpose(-1, -2) * sc                              # [B,T,nc]
            fin = ((torch.arange(nc, device=dev)[None, :] * C + C - 1) < ii[:, None])[None].expand(B, T, nc)
            top = sm_.detach().masked_fill(~fin, -1e9).topk(2, -1).indices
            sel = torch.zeros_like(fin).scatter(-1, top, True) & fin          # [B,T,nc]
            msel = sel[:, :, ch] | (incl & (ch[:, None] == ch[None, :]))[None]
            if t == "moba":
                y = self._sm(s, msel) @ u
            else:
                g = torch.sigmoid(self.wg(x))
                y_c = self._msoft(sm_, fin) @ um
                y_s = self._sm(s, msel) @ u
                y_w = self._sm(s, incl & (age < 8)) @ u
                y = g[..., :1] * y_c + g[..., 1:2] * y_s + g[..., 2:] * y_w
        elif t == "mlstm":
            Fc = F.logsigmoid(self.wf(x)).squeeze(-1).cumsum(1)             # [B,T]
            ig = self.wig(x).squeeze(-1)
            D = (Fc[:, :, None] - Fc[:, None, :] + ig[:, None, :]).masked_fill(~incl, -1e4)
            mx = D.max(-1, keepdim=True).values
            Cm = (q @ k.transpose(-1, -2) * sc) * torch.exp(D - mx)
            n = torch.maximum(Cm.sum(-1, keepdim=True).abs(), torch.exp(-mx))
            y = (Cm @ u) / n
        elif t == "onlstm":
            fm = 0.5 + 0.499 * self.wf(x).softmax(-1).cumsum(-1)
            y = self._scan(torch.log(fm), (1 - fm) * u)
        elif t == "latch":
            pon, poff = torch.sigmoid(4.0 * (u - 1.0)), torch.sigmoid(4.0 * (q - 1.0))
            a = 1 - 0.495 * (pon + poff)
            y = 2 * self._scan(torch.log(a), 0.495 * pon) - 1
        elif t == "elman":
            h = torch.zeros(B, db, device=dev, dtype=u.dtype); outs = []
            for s_ in range(T):
                h = torch.tanh(h @ self.Wh + u[:, s_]); outs.append(h)
            y = torch.stack(outs, 1)
        elif t == "pom":
            y = torch.sigmoid(q) * (u * k).cumsum(1) / (ii + 1).float()[None, :, None]
        elif t == "hrr":
            kn, qn = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
            Mf = (torch.fft.rfft(kn, dim=-1) * torch.fft.rfft(u, dim=-1)).cumsum(1) / (ii + 1).float()[None, :, None]
            y = torch.fft.irfft(Mf * torch.conj(torch.fft.rfft(qn, dim=-1)), n=db, dim=-1)
        elif t == "fastw":
            A = (u @ u.transpose(-1, -2) * sc * 0.95 ** agef).masked_fill(~incl, 0.0)
            y = A @ u / (ii + 1).float().sqrt()[None, :, None]
        elif t in ("hist", "kwta", "levy"):
            M = (0.05 * 0.95 ** agef).masked_fill(~incl, 0.0)
            if t == "hist":
                cen = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=dev)
                o = (-(u[..., :8, None] - cen) ** 2 / 0.5).softmax(-1).reshape(B, T, 32)
            elif t == "kwta":
                thr = u.abs().topk(4, -1).values[..., -1:].detach()
                o = u * (u.abs() >= thr)
            else:
                v = u[..., :8]
                X = (0.1 * 0.9 ** agef).masked_fill(~incl, 0.0) @ v
                Xp = torch.cat([torch.zeros_like(X[:, :1]), X[:, :-1]], 1)
                Ar = Xp[..., :, None] * v[..., None, :]
                iu = torch.triu_indices(8, 8, 1, device=dev)
                o = (Ar - Ar.transpose(-1, -2))[..., iu[0], iu[1]]           # [B,T,28]
            y = M @ o
        elif t in ("moh",):
            hs = db // 4
            qh, kh, uh = (z.view(B, T, 4, hs).transpose(1, 2) for z in (q, k, u))
            yh = self._sm(qh @ kh.transpose(-1, -2) * hs ** -0.5, incl) @ uh   # [B,4,T,hs]
            r = self.wr(x).softmax(-1)
            thr = r.topk(2, -1).values[..., -1:]
            r = r * (r >= thr); r = r / r.sum(-1, keepdim=True)
            y = (yh * r.transpose(1, 2)[..., None] * 2).transpose(1, 2).reshape(B, T, db)
        else:
            s = q @ k.transpose(-1, -2) * sc
            if t == "kern":
                s = s - 0.5 * sc * (q.pow(2).sum(-1, keepdim=True) + k.pow(2).sum(-1)[:, None, :])
                y = self._sm(s, incl) @ u
            elif t == "fire":
                z = torch.log1p(agef) / torch.log1p(ii.clamp(min=8).float())[:, None]
                f = F.relu(z[..., None] * self.f1 + self.fb) @ self.f2
                y = self._sm(s + f, incl) @ u
            elif t == "ssmax":
                y = self._sm(s * self.sp * torch.log(ii + 2.0)[:, None], incl) @ u
            elif t == "qtemp":
                y = self._sm(s * F.softplus(self.wt(x)), incl) @ u
            elif t == "rnd":
                y = self._sm(s, incl & ((age < 4) | self.Rm[:T, :T] | (ii[None, :] == 0))) @ u
            elif t == "strid":
                y = self._sm(s, incl & ((age < 16) | (age % 16 == 0))) @ u
            elif t == "cosf":
                A = (F.relu(q) @ F.relu(k).transpose(-1, -2)) * torch.cos(math.pi / 2 * agef / T)
                A = A.masked_fill(~incl, 0.0)
                y = (A @ u) / A.sum(-1, keepdim=True).clamp(min=1e-4)
            elif t == "entg":
                A = self._sm(s, incl)
                H = -(A * torch.log(A.clamp(min=1e-9))).sum(-1, keepdim=True)
                y = (A @ u) * (1 - H / torch.log(ii + 2.0)[:, None])
            elif t == "tnorm":
                y = s.masked_fill(~incl, 0.0) @ u
                y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
            elif t == "loglin":
                xo = ii[:, None] ^ ii[None, :]
                lev = torch.where(age > 0, torch.floor(torch.log2(xo.clamp(min=1).float())).long() + 1, 0).clamp(max=8)
                lam = F.softplus(self.wl(x))                                 # [B,T,9]
                Wl = lam.gather(-1, lev[None].expand(B, T, T))
                A = (s * Wl).masked_fill(~incl, 0.0)
                y = (A @ u) / A.abs().sum(-1, keepdim=True).clamp(min=1.0)
            elif t == "kerple":
                r1, r2 = F.softplus(self.r[0]), F.softplus(self.r[1])
                y = self._sm(s - r1 * torch.log1p(r2 * agef), incl) @ u
            else:                                                            # flash
                C = 16; ch = ii // C
                same = incl & (ch[:, None] == ch[None, :])
                A1 = (F.relu(s) ** 2).masked_fill(~same, 0.0) / C
                A2 = s.masked_fill(~(ch[:, None] > ch[None, :]), 0.0) / (ch * C).clamp(min=1).float()[:, None]
                y = (A1 + A2) @ u
        return self.wo(y * F.silu(self.gate(x)))


class Wild4(nn.Module):
    """round-18 side branch (option wb:<typ>, same wiring as Wild), 30 more ops. Most are 4-head softmax side heads
    with a new distance prior, since gaussc (Gaussian age windows + content) is the only side head that held at eval
    2048 and its FineWeb gap grew from -0.000 at seq 256 to -0.035 at seq 512. Known = published mechanism:
    lapc: Laplace age windows + content (ours) | cauchyc: Cauchy (heavy-tail) age windows + content (ours) | gaussl:
    Gaussian windows in log age (ours) | gausscq: window centre predicted per query (Known: Luong local-p) | gaussg:
    gaussc prior scaled per query by a gate (ours) | dog: difference-of-Gaussians, centre-surround prior (ours) | sgauss:
    window width proportional to position (ours) | relfull: free bias per age (Known: Shaw relative position) | xpos:
    logits * gamma^age (Known: xPos) | sandwich: sum of cosines of age as bias (Known: Sandwich) | dape: MLP over
    [logits, ALiBi bias] (Known: DAPE) | rope: RoPE side head (Known: RoPE; control for "any position info") | drope:
    RoPE angle from a learned per-token clock (Known: contextual / selective RoPE) | mixprior: two Gaussian windows per
    head, learned mix (ours) | distt: temperature grows with log age (ours) | sm1: softmax with a null logit (Known:
    softmax+1 / attention sink) | ent15: 1.5-entmax (Known: entmax) | capb: tanh logit soft-cap (Known: Gemma 2) |
    kbias: per-key salience bias from x_s (ours) | agev: attention also returns a learned age embedding (ours) | cent:
    centred attention A - 1/(t+1) (ours) | dualt: gate between temperature 2 and 0.5 heads (ours) | l1att: softmax of
    -|q-k|_1 (ours) | pconv: causal depthwise conv on q, k, v before attention (Known: Primer) | attvar: attention returns
    mean and variance of values (ours) | dkey: keys are k_s - k_{s-1} (ours) | rglru: Griffin RG-LRU recurrence (Known:
    Griffin) | sdft: sliding 32-token DFT magnitudes at 4 frequencies (ours) | trend: decayed least-squares slope (ours)
    | acf: decayed autocorrelation at lags 1, 2, 4, 8 (ours)."""
    NEW = set("lapc cauchyc gaussl gausscq gaussg dog sgauss relfull xpos sandwich dape rope drope mixprior distt "
              "sm1 ent15 capb kbias agev cent dualt l1att pconv attvar dkey rglru sdft trend acf cauchy0".split())

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d, T = cfg.d_model, cfg.seq_len
        self.typ, self.db, self.H = typ, db, 4
        dout = {"attvar": 2 * db, "sdft": 32, "acf": 32}.get(typ, db)
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, dout, bias=False)
        self.wo = nn.Linear(dout, d, bias=False)
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        P = lambda v: nn.Parameter(torch.tensor(v))
        L = lambda v: nn.Parameter(torch.log(torch.tensor(v)))
        if typ in ("lapc", "cauchyc", "gaussg", "dog", "mixprior"):
            self.mu, self.ls = P([1.0, 4.0, 16.0, 64.0]), L([1.0, 2.0, 8.0, 32.0])
        if typ == "mixprior":
            self.mu2, self.ls2, self.mw = P([8.0, 32.0, 96.0, 192.0]), L([4.0, 8.0, 16.0, 32.0]), P([0.0] * 4)
        if typ == "dog":
            self.c = P([4.0, 2.0])
        if typ == "gaussg":
            self.wgp = nn.Linear(d, 4, bias=True)
        if typ == "gaussl":
            self.m, self.ls = nn.Parameter(torch.log1p(torch.tensor([1.0, 4.0, 16.0, 64.0]))), L([0.5] * 4)
        if typ == "gausscq":
            self.wm, self.ls = nn.Linear(d, 4, bias=True), L([1.0, 2.0, 8.0, 32.0])
        if typ == "sgauss":
            self.lr_ = L([0.05, 0.1, 0.25, 0.5])
        if typ == "relfull":
            self.rb = nn.Parameter(torch.zeros(4, T))
        if typ == "xpos":
            self.gl = nn.Parameter(torch.logit(torch.tensor([0.9, 0.97, 0.99, 0.997])))
        if typ == "sandwich":
            self.c = P([4.0] * 4)
        if typ == "dape":
            self.W1, self.W2 = nn.Parameter(torch.zeros(8, 16)), nn.Parameter(torch.zeros(16, 4))
        if typ == "drope":
            self.wc = nn.Linear(d, 4, bias=True)
        if typ == "distt":
            self.a = P([-2.0] * 4)
        if typ == "kbias":
            self.kb = nn.Linear(d, 4, bias=False)
        if typ == "agev":
            self.E = nn.Parameter(torch.zeros(4, 10, db // 4))
        if typ == "dualt":
            self.wdt = nn.Linear(d, 4, bias=True)
        if typ == "pconv":
            self.cq, self.ck, self.cu = (nn.Parameter(torch.zeros(3, db)) for _ in range(3))
        if typ == "rglru":
            self.wa, self.wx = nn.Linear(d, db, bias=True), nn.Linear(d, db, bias=True)
            self.lam = nn.Parameter(torch.logit(torch.linspace(0.9, 0.999, db)))
        if typ == "trend":
            self.lg = nn.Parameter(torch.logit(torch.linspace(0.8, 0.99, db)))

    def post_init(self):                                                  # the model re-init draws every 2-D tensor
        with torch.no_grad():
            if hasattr(self, "rb"):
                self.rb.zero_()
            if hasattr(self, "W1"):
                self.W1.normal_(0.0, 0.3); self.W2.zero_()
            for n in ("cq", "ck", "cu"):
                if hasattr(self, n):
                    p = getattr(self, n); p.zero_(); p[0].fill_(1.0)
            for n, b in (("wgp", 0.0), ("dualt_", 0.0), ("wdt", 0.0), ("drope_", 0.0)):
                if hasattr(self, n):
                    m = getattr(self, n); nn.init.zeros_(m.weight); nn.init.constant_(m.bias, b)
            if hasattr(self, "wc"):
                nn.init.zeros_(self.wc.weight); nn.init.constant_(self.wc.bias, 0.5413)   # softplus = 1: plain RoPE
            if hasattr(self, "wm"):
                nn.init.zeros_(self.wm.weight)
                self.wm.bias.copy_(torch.log(torch.expm1(torch.tensor([1.0, 4.0, 16.0, 64.0]))))
            if hasattr(self, "kb"):
                nn.init.zeros_(self.kb.weight)

    _sm = staticmethod(Wild2._sm)
    _scan = staticmethod(Wild2._scan)
    _shift = staticmethod(Wild._shift)

    @staticmethod
    def _rot(z, ang):
        h = z.shape[-1] // 2
        c, s = torch.cos(ang), torch.sin(ang)
        return torch.cat([z[..., :h] * c - z[..., h:] * s, z[..., :h] * s + z[..., h:] * c], -1)

    def forward(self, x):
        B, T, _ = x.shape
        t, db, H = self.typ, self.db, self.H
        hs = db // H
        u = self.wi(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        incl = age >= 0
        agef = age.clamp(min=0).float()
        if t == "rglru":
            la = (8.0 * torch.sigmoid(self.wa(x)) * F.logsigmoid(self.lam)).clamp(min=-0.7)
            b = torch.sqrt((1 - torch.exp(2 * la)).clamp(min=1e-6)) * torch.sigmoid(self.wx(x)) * u
            y = self._scan(la, b)
        elif t == "sdft":
            n = torch.arange(32, device=dev).float()
            fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=dev)
            ang = 2 * math.pi * fr[:, None] * (31 - n)[None] / 32                    # weight j sees age 31 - j
            w = torch.stack([torch.cos(ang), torch.sin(ang)], 1).reshape(8, 1, 32)    # [4 freq x re/im, 1, 32]
            z = F.pad(u[..., :8].transpose(1, 2), (31, 0))
            o = F.conv1d(z, w.repeat(8, 1, 1), groups=8).view(B, 8, 4, 2, T)
            y = torch.sqrt(o.pow(2).sum(3) + 1e-6).reshape(B, 32, T).transpose(1, 2) / 32
        elif t == "acf":
            v = u[..., :8]
            p = torch.cat([v * self._shift(v, l) for l in (1, 2, 4, 8)], -1)
            y = self._scan(torch.full_like(p, math.log(0.9)), p) * 0.1
        elif t == "trend":
            lam = torch.sigmoid(self.lg)[:, None, None]                              # [db,1,1]
            W = torch.exp(agef[None] * torch.log(lam)) * incl[None]                   # [db,T,T]
            tau = -agef[None]
            S0, S1, S2 = W.sum(-1), (W * tau).sum(-1), (W * tau ** 2).sum(-1)         # [db,T]
            Su = torch.einsum("cts,bsc->btc", W, u)
            Stu = torch.einsum("cts,bsc->btc", W * tau, u)
            y = (S0.T * Stu - S1.T * Su) / (S0.T * S2.T - S1.T ** 2 + 1e-3)
        else:
            q, k = self.wq(x), self.wk(x)
            if t == "pconv":
                cv = lambda z, c: c[0] * z + c[1] * self._shift(z, 1) + c[2] * self._shift(z, 2)
                q, k, u = cv(q, self.cq), cv(k, self.ck), cv(u, self.cu)
            if t == "dkey":
                k = k - self._shift(k, 1)
            hv = lambda z: z.view(B, T, H, -1).transpose(1, 2)
            q, k, uh = hv(q), hv(k), hv(u)
            if t in ("rope", "drope"):
                om = 10000.0 ** (-torch.arange(hs // 2, device=dev).float() / (hs // 2))
                if t == "rope":
                    ang = ii.float()[:, None] * om
                else:
                    clk = F.softplus(self.wc(x)).cumsum(1).transpose(1, 2)            # [B,H,T]
                    ang = clk[..., None] * om
                q, k = self._rot(q, ang), self._rot(k, ang)
            if t == "l1att":
                s = -torch.cdist(q.contiguous(), k.contiguous(), p=1) * hs ** -0.5
            else:
                s = q @ k.transpose(-1, -2) * hs ** -0.5                             # [B,H,T,T]
            v4 = lambda p: p[:, None, None]
            if t in ("lapc", "cauchyc", "gaussg", "dog", "mixprior"):
                bw = torch.exp(self.ls)
                g1 = -((agef - v4(self.mu)) ** 2) / (2 * v4(bw) ** 2)
                if t == "lapc":
                    s = s + (-(agef - v4(self.mu)).abs() / v4(bw))
                elif t == "cauchyc":
                    s = s - torch.log1p(((agef - v4(self.mu)) / v4(bw)) ** 2)
                elif t == "gaussg":
                    s = s + 2 * torch.sigmoid(self.wgp(x)).transpose(1, 2)[..., None] * g1
                elif t == "dog":
                    g2 = -((agef - v4(self.mu)) ** 2) / (2 * (2.5 * v4(bw)) ** 2)
                    s = s + self.c[0] * torch.exp(g1) - self.c[1] * torch.exp(g2)
                else:
                    g2 = -((agef - v4(self.mu2)) ** 2) / (2 * v4(torch.exp(self.ls2)) ** 2)
                    s = s + torch.logaddexp(g1 + v4(F.logsigmoid(self.mw)), g2 + v4(F.logsigmoid(-self.mw)))
            elif t == "gaussl":
                s = s - (torch.log1p(agef) - v4(self.m)) ** 2 / (2 * v4(torch.exp(self.ls)) ** 2)
            elif t == "gausscq":
                mu = F.softplus(self.wm(x)).transpose(1, 2)[..., None]               # [B,H,T,1]
                s = s - (agef - mu) ** 2 / (2 * v4(torch.exp(self.ls)) ** 2)
            elif t == "sgauss":
                sig = v4(torch.exp(self.lr_)) * (ii + 1).float()[None, :, None]
                s = s - 0.5 * (agef / sig) ** 2
            elif t == "relfull":
                s = s + self.rb[:, age.clamp(0, T - 1)]
            elif t == "xpos":
                s = s * torch.exp(agef * v4(F.logsigmoid(self.gl)))
            elif t == "sandwich":
                om = 10000.0 ** (-torch.arange(16, device=dev).float() / 16)
                s = s + v4(self.c) * torch.cos(agef[..., None] * om).mean(-1)
            elif t == "dape":
                al = -v4(2.0 ** -torch.arange(1, 5, device=dev).float()) * agef     # [H,T,T]
                z = torch.cat([s, al[None].expand(B, H, T, T)], 1)                   # [B,2H,T,T]
                hdd = F.relu(torch.einsum("bctu,ce->betu", z, self.W1))
                s = s + al + torch.einsum("betu,eh->bhtu", hdd, self.W2)
            elif t == "distt":
                s = s / (1 + v4(F.softplus(self.a)) * torch.log1p(agef))
            elif t == "kbias":
                s = s + self.kb(x).transpose(1, 2)[:, :, None, :]
            elif t == "capb":
                s = 5.0 * torch.tanh(s / 5.0)
            if t == "sm1":
                z = torch.cat([s.masked_fill(~incl, float("-inf")), torch.zeros_like(s[..., :1])], -1)
                A = z.softmax(-1)[..., :-1]
            elif t == "ent15":
                z = (s / 2).masked_fill(~incl, -1e4)
                with torch.no_grad():
                    hi = z.max(-1, keepdim=True).values; lo = hi - 1.0
                    for _ in range(18):
                        mid = (lo + hi) / 2
                        big = F.relu(z - mid).pow(2).sum(-1, keepdim=True) > 1
                        lo, hi = torch.where(big, mid, lo), torch.where(big, hi, mid)
                A = F.relu(z - (lo + hi) / 2).pow(2)
                A = A / A.sum(-1, keepdim=True).clamp(min=1e-9)
            elif t == "dualt":
                g = torch.sigmoid(self.wdt(x)).transpose(1, 2)[..., None]
                A = g * self._sm(s * 2.0, incl) + (1 - g) * self._sm(s * 0.5, incl)
            else:
                A = self._sm(s, incl)
            if t == "cent":
                A = A - incl.float() / (ii + 1).float()[:, None]
            y = A @ uh
            if t == "agev":
                bk = torch.floor(torch.log2(agef + 1)).long().clamp(max=9)          # [T,T]
                Ab = torch.zeros(B, H, T, 10, device=dev, dtype=A.dtype).scatter_add_(-1, bk[None, None].expand(B, H, T, T), A)
                y = y + torch.einsum("bhtk,hkd->bhtd", Ab, self.E)
            if t == "attvar":
                m2 = A @ (uh * uh)
                y = torch.cat([y, (m2 - y * y).clamp(min=0.0)], -1)
            y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y * F.silu(self.gate(x)))


class Wild5(nn.Module):
    """round-19 side branch (option wb:<typ>), 4 heads, each op aimed at a measured failure of the Engram bed
    (XS 600: exact in-context repeats still cost 0.83 nats on code; positions 64-255 are only 0.01 nats better than 16-63).
    succk: induction head matching the last TWO tokens, k_s from (x_{s-1}, x_s), returns u_{s+1} (Known: induction head;
    two-token key ours) | succ2: successor heads, heads 0-1 return u_{s+1}, heads 2-3 return u_{s+2} (ours) | sharps:
    successor head on cosine q,k with learned temperature, init 10 (ours) | cauchys: cauchyc distance prior + content,
    returns u_{s+1} (ours) | farsoft: softmax attention over keys at age >= 16 only (ours) | farsucc: successor head over
    keys at age >= 16 only (ours)."""
    NEW = set("succk succ2 sharps cauchys farsoft farsucc".split())

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d = cfg.d_model
        self.typ, self.db, self.H = typ, db, 4
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, db, bias=False)
        self.wo = nn.Linear(db, d, bias=False)
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "succk":
            self.wq2, self.wk2 = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "sharps":
            self.lt = nn.Parameter(torch.full((4,), math.log(10.0)))
        if typ == "cauchys":
            self.mu = nn.Parameter(torch.tensor([1.0, 4.0, 16.0, 64.0]))
            self.ls = nn.Parameter(torch.log(torch.tensor([1.0, 2.0, 8.0, 32.0])))

    def post_init(self):
        with torch.no_grad():
            if hasattr(self, "lt"):
                self.lt.fill_(math.log(10.0))

    _shift = staticmethod(Wild._shift)
    _msoft = staticmethod(Wild._msoft)

    @staticmethod
    def _lead(u, l):                                                     # value at s = u_{s+l}
        return torch.cat([u[:, l:], torch.zeros_like(u[:, :l])], 1)

    def forward(self, x):
        B, T, _ = x.shape
        t, db, H = self.typ, self.db, self.H
        hs = db // H
        u = self.wi(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        agef = age.clamp(min=0).float()
        q, k = self.wq(x), self.wk(x)
        if t == "succk":
            q, k = q + self.wq2(self._shift(x, 1)), k + self.wk2(self._shift(x, 1))
        hv = lambda z: z.view(B, T, H, -1).transpose(1, 2)
        q, k = hv(q), hv(k)
        if t == "sharps":
            s = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2) * torch.exp(self.lt)[None, :, None, None]
        else:
            s = q @ k.transpose(-1, -2) * hs ** -0.5                          # [B,H,T,T]
        if t == "cauchys":
            v4 = lambda p: p[:, None, None]
            s = s - torch.log1p(((agef - v4(self.mu)) / v4(torch.exp(self.ls))) ** 2)
        if t == "farsoft":
            A = self._msoft(s, (age >= 16)[None, None].expand(B, H, T, T))
            y = A @ hv(u)
        elif t == "succ2":
            m1 = (age >= 1)[None, None].expand(B, 2, T, T); m2 = (age >= 2)[None, None].expand(B, 2, T, T)
            y = torch.cat([self._msoft(s[:, :2], m1) @ hv(self._lead(u, 1))[:, :2],
                           self._msoft(s[:, 2:], m2) @ hv(self._lead(u, 2))[:, 2:]], 1)
        else:                                                                 # succk sharps cauchys farsucc: successor value
            mn = 16 if t == "farsucc" else 1
            y = self._msoft(s, (age >= mn)[None, None].expand(B, H, T, T)) @ hv(self._lead(u, 1))
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y * F.silu(self.gate(x)))


class Wild6(nn.Module):
    """round-21 side branch (option wb:<typ>), 4 heads, built to test rule 2 (the Engram bed barely uses context past
    position 64: p2 - p1 = -0.011 to -0.016 nats). Each attention op is a per-head spec (value lead, min age, max age)
    plus score flags. far8/far32/far64/band/ladder: softmax over far keys only, value u (ours) | fsucc*/sladder:
    successor value u_{s+1} over far keys (ours) | near*/nsucc*: near-only controls (Known: sliding window) |
    succfar/succnear: 2 successor heads + 2 far (or near) heads | lead2far/leadmix/sv2/sv2far: other value leads |
    logage/logup: + a_h log(1+age) score bias, init 0 / +0.5 (ours) | ssmaxa: score x (1 + c_h log(1+t)) (Known: SSMax, additive form) |
    cosfar: cosine score, learned temperature | t5b: learned log-bucket age bias (Known: T5 bias) | sink*: extra
    zero-value logit so a head can attend to nothing (Known: attention sink / softmax-1) | topk*: top-8 keys only |
    kkfar: two-token key successor head over age >= 16 | fmean*/fmax64: running mean / max of u over age >= 64 or 16
    (ours) | chunkm: softmax over means of finished 16-token chunks (Known: compressive memory) | chunks: chunk keys,
    value = mean of the next chunk (ours). A leading s on a flag op = successor value."""
    F = (0, 16, None)
    SPEC = dict(
        far8=[(0, 8, None)] * 4, far32=[(0, 32, None)] * 4, far64=[(0, 64, None)] * 4,
        band=[(0, 16, 64)] * 4, ladder=[(0, 1, None), (0, 8, None), (0, 32, None), (0, 64, None)],
        fsucc8=[(1, 8, None)] * 4, fsucc32=[(1, 32, None)] * 4, fsucc64=[(1, 64, None)] * 4,
        sladder=[(1, 1, None), (1, 8, None), (1, 32, None), (1, 64, None)],
        near8=[(0, 0, 8)] * 4, near32=[(0, 0, 32)] * 4, nsucc8=[(1, 1, 8)] * 4, nsucc32=[(1, 1, 32)] * 4,
        succfar=[(1, 1, None)] * 2 + [(0, 16, None)] * 2, succnear=[(1, 1, None)] * 2 + [(0, 0, 8)] * 2,
        lead2far=[(2, 16, None)] * 4, leadmix=[(1, 1, None), (2, 2, None), (3, 3, None), (4, 4, None)],
        sv2=[("v2", 2, None)] * 4, sv2far=[("v2", 16, None)] * 4,
        logage=[(0, 0, None)] * 4, slogage=[(1, 1, None)] * 4, logup=[(0, 0, None)] * 4, slogup=[(1, 1, None)] * 4,
        ladderlog=[(0, 1, None), (0, 8, None), (0, 32, None), (0, 64, None)],
        sladderlog=[(1, 1, None), (1, 8, None), (1, 32, None), (1, 64, None)],
        sladdercos=[(1, 1, None), (1, 8, None), (1, 32, None), (1, 64, None)],
        ssmaxa=[(0, 0, None)] * 4, sssmax=[(1, 1, None)] * 4, cosfar=[(0, 16, None)] * 4, scosfar=[(1, 16, None)] * 4,
        t5b=[(0, 0, None)] * 4, st5b=[(1, 1, None)] * 4,
        sinkfar=[(0, 16, None)] * 4, sinksucc=[(1, 1, None)] * 4, sinkall=[(0, 0, None)] * 4,
        topkfar=[(0, 16, None)] * 4, topksucc=[(1, 1, None)] * 4, kkfar=[(1, 16, None)] * 4)
    FLAGS = dict(logage="log0", slogage="log0", logup="logp", slogup="logp", ladderlog="log0", sladderlog="log0",
                 sladdercos="cos", ssmaxa="ssm", sssmax="ssm", cosfar="cos", scosfar="cos", t5b="t5", st5b="t5",
                 sinkfar="sink", sinksucc="sink", sinkall="sink", topkfar="topk", topksucc="topk", kkfar="kk")
    POOL = set("fmean64 fmax64 fmean16 chunkm chunks".split())
    NEW = set(SPEC) | POOL

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d = cfg.d_model
        self.typ, self.db, self.H = typ, db, 4
        self.fl = self.FLAGS.get(typ, "")
        self.wi = nn.Linear(d, db, bias=False)
        self.gate = nn.Linear(d, db, bias=False)
        self.wo = nn.Linear(db, d, bias=False)
        if typ in ("fmean64", "fmax64", "fmean16"):
            return
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if self.fl == "kk":
            self.wq2, self.wk2 = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if self.fl in ("log0", "logp", "ssm", "cos", "sink"):
            self.hp = nn.Parameter(torch.zeros(4))
        if self.fl == "t5":
            self.tb = nn.Parameter(torch.zeros(4, 16))

    def post_init(self):
        with torch.no_grad():
            if hasattr(self, "hp"):
                self.hp.fill_({"logp": 0.5, "ssm": 0.1, "cos": math.log(10.0)}.get(self.fl, 0.0))
            if hasattr(self, "tb"):
                self.tb.zero_()

    _shift = staticmethod(Wild._shift)
    _msoft = staticmethod(Wild._msoft)
    _lead = staticmethod(Wild5._lead)

    def forward(self, x):
        B, T, _ = x.shape
        t, db, H = self.typ, self.db, self.H
        hs = db // H
        u = self.wi(x)
        dev = x.device
        hv = lambda z: z.view(B, z.shape[1], H, -1).transpose(1, 2)
        if t in ("fmean64", "fmean16", "fmax64"):
            L = 16 if t == "fmean16" else 64
            us = self._shift(u, L)
            if t == "fmax64":
                y = torch.cummax(us, 1).values
            else:
                cnt = (torch.arange(T, device=dev) - L + 1).clamp(min=1).float()[None, :, None]
                y = us.cumsum(1) / cnt
            return self.wo(y * F.silu(self.gate(x)))
        q, k = self.wq(x), self.wk(x)
        if t in ("chunkm", "chunks"):
            C = 16; nc = T // C
            kc = k[:, :nc * C].view(B, nc, C, -1).mean(2); uc = u[:, :nc * C].view(B, nc, C, -1).mean(2)
            ii = torch.arange(T, device=dev)[:, None]; cc = torch.arange(nc, device=dev)[None, :]
            if t == "chunks":
                uc = torch.cat([uc[:, 1:], torch.zeros_like(uc[:, :1])], 1)
                m = (cc + 2) * C <= ii + 1                                     # chunk and its successor finished
            else:
                m = (cc + 1) * C <= ii + 1
            s = hv(q) @ hv(kc).transpose(-1, -2) * hs ** -0.5                   # [B,H,T,nc]
            y = self._msoft(s, m[None, None].expand(B, H, T, nc)) @ hv(uc)
            y = y.transpose(1, 2).reshape(B, T, -1)
            return self.wo(y * F.silu(self.gate(x)))
        if self.fl == "kk":
            q, k = q + self.wq2(self._shift(x, 1)), k + self.wk2(self._shift(x, 1))
        q, k = hv(q), hv(k)
        ii = torch.arange(T, device=dev)
        age = ii[:, None] - ii[None, :]
        agef = age.clamp(min=0).float()
        if self.fl == "cos":
            s = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2) * torch.exp(self.hp)[None, :, None, None]
        else:
            s = q @ k.transpose(-1, -2) * hs ** -0.5
        if self.fl in ("log0", "logp"):
            s = s + self.hp[None, :, None, None] * torch.log1p(agef)
        if self.fl == "ssm":
            s = s * (1 + self.hp[None, :, None, None] * torch.log1p(ii.float())[None, None, :, None])
        if self.fl == "t5":
            bk = torch.log2(agef + 1).floor().clamp(max=15).long()
            s = s + self.tb[:, bk][None]
        spec = self.SPEC[t]
        m = torch.stack([(age >= a) & (age < (b if b is not None else T + 1)) for _, a, b in spec])[None].expand(B, H, T, T)
        if self.fl == "topk":
            sm = s.masked_fill(~m, float("-inf"))
            kth = sm.topk(min(8, T), -1).values[..., -1:]
            m = m & (s >= kth)
        if self.fl == "sink":
            w = s.masked_fill(~m, float("-inf"))
            w = torch.cat([w, self.hp[None, :, None, None].expand(B, H, T, 1)], -1).softmax(-1)
            A = w[..., :-1]
        else:
            A = self._msoft(s, m)
        vals = []
        for h, (l, _, _) in enumerate(spec):
            v = self._lead(u, 1) + self._lead(u, 2) if l == "v2" else (self._lead(u, l) if l else u)
            vals.append(hv(v)[:, h])
        y = (A * 1.0) @ torch.stack(vals, 1)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y * F.silu(self.gate(x)))


class Wild7(nn.Module):
    """round-22 published comparison for succ (gate 0, 2026-11-23): kvshift = KV Shifting Attention (Known: arXiv
    2411.19574) as a 4-head softmax side branch: k_s <- a1_h k_s + a2_h k_{s-1}, v_s <- b1_h u_s + b2_h u_{s-1},
    init a1 = b1 = 1, a2 = b2 = 0, causal incl. s = t. Matching the previous token of a key and reading u_s is the same
    match-then-copy-next circuit as succ, learned instead of fixed.
    Round 22 (built from succ's measured misses, 2026-11-23): kksucc = successor head whose query/key also see the
    previous token (copy conflicts: last0 1.20 nats, and m4 = m0, a longer match does not help) | succh8 = successor
    head with 8 heads (more slots for conflicting copies; use wbd:64 for 8 channels per head)."""
    NEW = {"kvshift", "kksucc", "succh8"}

    def __init__(self, cfg, typ, db=32):
        super().__init__()
        d = cfg.d_model
        self.typ, self.db, self.H = typ, db, (8 if typ == "succh8" else 4)
        self.wi, self.gate, self.wo = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False), nn.Linear(db, d, bias=False)
        self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "kvshift":
            self.ab = nn.Parameter(torch.zeros(4, 4))
        if typ == "kksucc":
            self.wq2, self.wk2 = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)

    def post_init(self):
        with torch.no_grad():
            if hasattr(self, "ab"):
                self.ab.zero_(); self.ab[0].fill_(1.0); self.ab[2].fill_(1.0)   # rows: a1, a2, b1, b2

    _shift = staticmethod(Wild._shift)
    _msoft = staticmethod(Wild._msoft)

    def forward(self, x):
        B, T, _ = x.shape
        H, hs = self.H, self.db // self.H
        hv = lambda z: z.view(B, T, H, hs).transpose(1, 2)
        if self.typ in ("kksucc", "succh8"):
            q, k = self.wq(x), self.wk(x)
            if self.typ == "kksucc":
                q, k = q + self.wq2(self._shift(x, 1)), k + self.wk2(self._shift(x, 1))
            u = self.wi(x)
            vs = hv(torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1))     # value at s = u_{s+1}
            s = hv(q) @ hv(k).transpose(-1, -2) * hs ** -0.5
            ii = torch.arange(T, device=x.device)
            m = (ii[:, None] - ii[None, :]) > 0
            y = (self._msoft(s, m[None, None].expand(B, H, T, T)) @ vs).transpose(1, 2).reshape(B, T, -1)
            return self.wo(y * F.silu(self.gate(x)))
        u, q, k = hv(self.wi(x)), hv(self.wq(x)), hv(self.wk(x))
        sh = lambda z: torch.cat([torch.zeros_like(z[:, :, :1]), z[:, :, :-1]], 2)
        a = self.ab[:, None, :, None, None]                                    # [4,1,H,1,1]
        k = a[0] * k + a[1] * sh(k)
        v = a[2] * u + a[3] * sh(u)
        s = q @ k.transpose(-1, -2) * hs ** -0.5
        m = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        y = (s.masked_fill(~m, float("-inf")).softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(y * F.silu(self.gate(x)))


class Branch(nn.Module):
    """round-11 side branch added to a base mixer: u = W_i x (db channels), y = op(u) over past positions,
    y *= silu(W_g x) (brg:1), out = W_o y (W_o zero-init before the model re-init).
    cmean: running mean of u | cmax: running max | clog: running sum / log(2+t) (count features, ours for the three)
    ema: fixed multi-scale EMA, decays brsc (Known: MEGA) | aft: static-query softmax pooling (Known: AFT-simple)
    swa: 1-head sliding-window softmax attention, window brw (Known: SWA hybrid) | chunk: softmax attention over
    means of finished chunks of length brC (Known: compressive memory) | lin: elu+1 linear attention (Known: Katharopoulos)"""
    def __init__(self, cfg, typ, db=32, gate=1, w=16, C=8, sc="0.5:0.9:0.99", nh=1):
        super().__init__()
        d = cfg.d_model
        self.typ, self.db, self.w, self.C, self.nh = typ, db, w, C, nh
        self.wi = nn.Linear(d, db, bias=False)
        self.wo = nn.Linear(db, d, bias=False); nn.init.zeros_(self.wo.weight)
        if gate:
            self.gate = nn.Linear(d, db, bias=False)
        if typ in ("swa", "chunk", "lin"):
            self.wq, self.wk = nn.Linear(d, db, bias=False), nn.Linear(d, db, bias=False)
        if typ == "aft":
            self.ws = nn.Linear(d, nh, bias=False)
        if typ == "ema":
            lam = torch.tensor([float(v) for v in sc.split(":")])
            self.register_buffer("lam", lam, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        u = self.wi(x)
        dev = x.device
        ii = torch.arange(T, device=dev)
        incl = ii[:, None] >= ii[None, :]
        t = self.typ
        if t == "cmean":
            y = u.cumsum(1) / (ii + 1).float()[None, :, None]
        elif t == "cmax":
            y = torch.cummax(u, 1).values
        elif t == "clog":
            y = u.cumsum(1) / torch.log(ii.float() + 2.0)[None, :, None]
        elif t == "ema":
            g = self.lam.numel(); cs = self.db // g
            age = (ii[:, None] - ii[None, :]).clamp(min=0).float()
            ys = []
            for n in range(g):
                lam = self.lam[n]
                M = ((1 - lam) * lam ** age).masked_fill(~incl, 0.0)
                ys.append(M @ u[..., n * cs:(n + 1) * cs] if n < g - 1 else M @ u[..., n * cs:])
            y = torch.cat(ys, -1)
        elif t == "aft":
            nh = self.nh; hs = self.db // nh
            s = self.ws(x).transpose(1, 2)                                     # [B,nh,T]
            W = s[:, :, None, :].expand(B, nh, T, T).masked_fill(~incl, float("-inf")).softmax(-1)
            y = (W @ u.view(B, T, nh, hs).transpose(1, 2)).transpose(1, 2).reshape(B, T, self.db)
        else:
            q, k = self.wq(x), self.wk(x)
            if t == "swa":
                m = incl & (ii[:, None] - ii[None, :] < self.w)
                y = F.scaled_dot_product_attention(q[:, None], k[:, None], u[:, None], attn_mask=m)[:, 0]
            elif t == "chunk":
                C = self.C; N = T // C
                kc, vc = k[:, :N * C].reshape(B, N, C, -1).mean(2), u[:, :N * C].reshape(B, N, C, -1).mean(2)
                m = torch.arange(N, device=dev)[None, :] < (ii // C)[:, None]
                sc_ = (q @ kc.transpose(-1, -2)) * self.db ** -0.5
                has = m.any(-1)[None, :, None]
                W = sc_.masked_fill(~m[None], float("-inf"))
                W = torch.where(has, W, torch.zeros_like(W)).softmax(-1) * has
                y = W @ vc
            else:                                                              # lin
                fq, fk = F.elu(q) + 1, F.elu(k) + 1
                A = (fq @ fk.transpose(-1, -2)).masked_fill(~incl[None], 0.0)
                y = (A @ u) / A.sum(-1, keepdim=True).clamp(min=1e-6)
        if hasattr(self, "gate"):
            y = y * F.silu(self.gate(x))
        return self.wo(y)
