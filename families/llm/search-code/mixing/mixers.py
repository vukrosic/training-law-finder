"""Token-mixing blocks for the mixing search (replaces the attention block; dsv4 MoE stays fixed).

Select with --set mix=<name>; extra options with --set mxo=key:val,key:val (parsed by opt()).
Every mixer has forward(x, cos, sin) -> [B, T, d], like model.Attention.

Baselines (written from the papers' equations and the official code they cite):
  mla    DeepSeek-V2/V3 Multi-head Latent Attention: low-rank joint KV latent (kv_lora) with RMSNorm,
         per-head no-RoPE q/k part + decoupled RoPE part (one shared RoPE key), scale (nope+rope)^-1/2
         (DeepSeek-V3 inference/model.py MLA).
  gdn    Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025; fla/layers/gated_deltanet.py):
         short conv + SiLU on q,k,v, L2-normed q,k, scalar decay g = -exp(A_log) softplus(a(x)+dt_bias),
         beta = sigmoid(b(x)), S_t = a_t (I - b k k^T) S_{t-1} + b v k^T, gated RMSNorm (SiLU gate) output.
  kda    Kimi Delta Attention (Kimi Linear, arXiv 2510.26692; fla/layers/kda.py): as gdn but the
         decay is per key channel from a low-rank projection, and the output gate is a low-rank
         sigmoid gate on a per-head RMSNorm.
  mamba2 Mamba-2 SSD (Dao & Gu 2024; mamba_ssm/modules/mamba2_simple.py): in_proj -> z,x,B,C,dt,
         causal conv on xBC, scalar-per-head decay exp(dt A), D skip, gated RMSNorm, out_proj.
New ideas are registered from ideas_mix.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

REG = {}


def register(name):
    def f(cls):
        REG[name] = cls
        return cls
    return f


def opt(cfg):
    o = {}
    s = str(getattr(cfg, "mxo", "") or "")
    for kv in s.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            for f in (int, float):
                try:
                    v = f(v)
                    break
                except ValueError:
                    pass
            o[k] = v
    return o


CTX = {}                                          # side inputs from model.forward (token ids)


def make(cfg, layer):
    import ideas_mix  # noqa: F401  (registers the idea mixers)
    return REG[cfg.mix](cfg, layer, opt(cfg))


class RMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(dim, T, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(torch.arange(T).float(), inv)
    return ang.cos(), ang.sin()


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class ShortConv(nn.Module):
    """causal depthwise conv over time + SiLU (fla ShortConvolution, kernel 4). Written as shift-and-add
    (same math as Conv1d(groups=C)); the CPU depthwise Conv1d backward was 10x slower."""
    def __init__(self, ch, k=4, act=True, init="normal"):
        super().__init__()
        self.k, self.act, self.init = k, act, init     # init: "normal" std 0.02 (fla _init_weights) or "kaiming" (mamba_ssm)
        w = torch.empty(ch, 1, k)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))             # Conv1d default init
        self.w = nn.Parameter(w.view(ch, k).flip(-1).contiguous())   # w[:, j] multiplies x[t-j]
        self._kw = self.w.detach().clone()

    def post_init(self):
        with torch.no_grad():
            if self.init == "kaiming":
                self.w.copy_(self._kw)
            elif self.init == "ident":                 # starts as the identity (no mixing)
                self.w.zero_()
                self.w[:, 0] = 1.0
            else:
                self.w.normal_(0, 0.02)

    def forward(self, x):                              # x [B, T, C]
        y = x * self.w[:, 0]
        for j in range(1, self.k):
            y = y + F.pad(x[:, :-j], (0, 0, j, 0)) * self.w[:, j]
        return F.silu(y) if self.act else y


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def dt_bias_init(n, lo=1e-3, hi=1e-1):
    dt = torch.exp(torch.rand(n) * (math.log(hi) - math.log(lo)) + math.log(lo)).clamp(min=1e-4)
    return inv_softplus(dt)


def DR(*a, **kw):
    return delta_par(*a, **kw)


def l2n(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def causal_mask(T, dev):
    return torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)


# ------------------------------------------------------------------ delta-rule recurrence
def delta_rule(q, k, v, beta, logdecay, chan=False, S0=None):
    """q,k [B,H,T,dk] (normed), v [B,H,T,dv], beta [B,H,T], logdecay [B,H,T] (scalar) or [B,H,T,dk] (chan).
    S_t = diag(a_t) S_{t-1};  S_t += beta k (v - S_t^T k)^T;  o_t = S_t^T q.   (state S is dk x dv)"""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv) if S0 is None else S0
    a = logdecay.exp()
    outs = []
    for t in range(T):
        if chan:
            S = S * a[:, :, t, :, None]
        else:
            S = S * a[:, :, t, None, None]
        kt = k[:, :, t]
        old = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, beta[:, :, t, None] * (v[:, :, t] - old))
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, 2)


def delta_par(q, k, v, beta, logdecay, chan=False):
    """exact parallel (one-chunk UT / WY) form of delta_rule with S0 = 0, for short sequences.
    S_t = sum_i D(t,i) k_i u_i^T,  u = (I + A)^-1 (beta v),  A_ti = beta_t <k_t, D(t,i) k_i> (i<t)."""
    B, H, T, dk = q.shape
    dev = q.device
    strict = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
    incl = torch.ones(T, T, dtype=torch.bool, device=dev).tril(0)
    G = logdecay.cumsum(2)
    if chan:                                                   # G [B,H,T,dk]
        diff = G[:, :, :, None, :] - G[:, :, None, :, :]       # [B,H,t,i,dk]
        diff = diff.masked_fill(~incl[:, :, None], float("-inf")).exp()
        kk = torch.einsum("bhtc,bhic,bhtic->bhti", k, k, diff)
        qk = torch.einsum("bhtc,bhic,bhtic->bhti", q, k, diff)
    else:
        dec = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
        kk = (k @ k.transpose(-1, -2)) * dec
        qk = (q @ k.transpose(-1, -2)) * dec
    A = (kk * beta[..., :, None]).masked_fill(~strict, 0.0)
    I = torch.eye(T, device=dev).expand_as(A)
    u = torch.linalg.solve_triangular(I + A, beta[..., None] * v, upper=False, unitriangular=True)
    return qk.masked_fill(~incl, 0.0) @ u


# ------------------------------------------------------------------ baselines
@register("mla")
class MLA(nn.Module):
    def __init__(self, cfg, layer, o):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        dh = d // H
        self.H = H
        self.nope = o.get("nope", dh)
        self.rd = o.get("rope", dh // 2)
        self.vd = o.get("vd", dh)
        self.r = o.get("kv_lora", d // 2)
        self.wq = nn.Linear(d, H * (self.nope + self.rd), bias=False)
        self.wkv_a = nn.Linear(d, self.r + self.rd, bias=False)
        self.kv_norm = RMS(self.r)
        self.wkv_b = nn.Linear(self.r, H * (self.nope + self.vd), bias=False)
        self.o = nn.Linear(H * self.vd, d, bias=False)
        c, s = rope_tables(self.rd, cfg.seq_len)
        self.register_buffer("rc", c, persistent=False)
        self.register_buffer("rs", s, persistent=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H = self.H
        q = self.wq(x).view(B, T, H, self.nope + self.rd).transpose(1, 2)
        qn, qr = q.split([self.nope, self.rd], -1)
        kv = self.wkv_a(x)
        c, kr = kv.split([self.r, self.rd], -1)
        if self.rd:
            qr = rope(qr, self.rc[:T], self.rs[:T])
            kr = rope(kr.unsqueeze(1), self.rc[:T], self.rs[:T])
        kr = kr.unsqueeze(1).expand(B, H, T, self.rd) if kr.dim() == 3 else kr.expand(B, H, T, self.rd)
        kv = self.wkv_b(self.kv_norm(c)).view(B, T, H, self.nope + self.vd).transpose(1, 2)
        kn, v = kv.split([self.nope, self.vd], -1)
        qq = torch.cat([qn, qr], -1)
        kk = torch.cat([kn, kr], -1)
        y = F.scaled_dot_product_attention(qq, kk, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, H * self.vd))


class _DeltaBase(nn.Module):
    """shared parts of gdn / kda: q,k,v projections + short conv, beta."""
    def __init__(self, cfg, o, ev=1.0):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H = o.get("heads", H)
        self.dk = o.get("dk", d // self.H)
        self.dv = int(self.dk * o.get("ev", ev))
        self.q = nn.Linear(d, self.H * self.dk, bias=False)
        self.k = nn.Linear(d, self.H * self.dk, bias=False)
        self.v = nn.Linear(d, self.H * self.dv, bias=False)
        self.cq, self.ck, self.cv = ShortConv(self.H * self.dk), ShortConv(self.H * self.dk), ShortConv(self.H * self.dv)
        self.b = nn.Linear(d, self.H, bias=False)
        self.o = nn.Linear(self.H * self.dv, d, bias=False)
        self.onorm = RMS(self.dv)

    def qkvb(self, x):
        B, T, _ = x.shape
        H = self.H
        q = l2n(self.cq(self.q(x)).view(B, T, H, self.dk)).transpose(1, 2)
        k = l2n(self.ck(self.k(x)).view(B, T, H, self.dk)).transpose(1, 2)
        v = self.cv(self.v(x)).view(B, T, H, self.dv).transpose(1, 2)
        beta = self.b(x).sigmoid().transpose(1, 2)
        return q, k, v, beta


@register("gdn")
class GatedDeltaNet(_DeltaBase):
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, o)
        d = cfg.d_model
        self.a = nn.Linear(d, self.H, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(0, 16)))
        self.dt_bias = nn.Parameter(dt_bias_init(self.H))
        self.g = nn.Linear(d, self.H * self.dv, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v, beta = self.qkvb(x)
        g = (-self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)).transpose(1, 2)    # [B,H,T]
        y = DR(q * self.dk ** -0.5, k, v, beta, g)                                # [B,H,T,dv]
        y = self.onorm(y.transpose(1, 2)) * F.silu(self.g(x).view(B, T, self.H, self.dv))
        return self.o(y.reshape(B, T, -1))


@register("kda")
class KimiDeltaAttention(_DeltaBase):
    def __init__(self, cfg, layer, o):
        super().__init__(cfg, o)
        d = cfg.d_model
        self.f1 = nn.Linear(d, self.dv, bias=False)
        self.f2 = nn.Linear(self.dv, self.H * self.dk, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(dt_bias_init(self.H * self.dk))
        self.g1 = nn.Linear(d, self.dv, bias=False)
        self.g2 = nn.Linear(self.dv, self.H * self.dv, bias=True)
        nn.init.zeros_(self.g2.bias)
        self.og = o.get("og", "sig")                   # diagnostics only: "silu" = GDN-style full gate
        if self.og == "silu":
            self.gf = nn.Linear(d, self.H * self.dv, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H = self.H
        q, k, v, beta = self.qkvb(x)
        f = (self.f2(self.f1(x)) + self.dt_bias).view(B, T, H, self.dk)
        g = (-self.A_log.exp()[:, None] * F.softplus(f)).transpose(1, 2)                  # [B,H,T,dk]
        y = DR(q * self.dk ** -0.5, k, v, beta, g, chan=True)
        if self.og == "silu":
            return self.o((self.onorm(y.transpose(1, 2)) * F.silu(self.gf(x).view(B, T, H, self.dv))).reshape(B, T, -1))
        y = self.onorm(y.transpose(1, 2)) * torch.sigmoid(self.g2(self.g1(x)).view(B, T, H, self.dv))
        return self.o(y.reshape(B, T, -1))


@register("mamba2")
class Mamba2(nn.Module):
    def __init__(self, cfg, layer, o):
        super().__init__()
        d = cfg.d_model
        self.di = int(o.get("expand", 2) * d)
        self.N = o.get("dstate", 16)
        self.P = o.get("headdim", 32)
        self.H = self.di // self.P
        self.inp = nn.Linear(d, 2 * self.di + 2 * self.N + self.H, bias=False)
        self.conv = ShortConv(self.di + 2 * self.N, init=o.get("conv_init", "kaiming"))
        self.dt_bias = nn.Parameter(dt_bias_init(self.H))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMS(self.di)
        self.out = nn.Linear(self.di, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        z, xbc, dt = self.inp(x).split([self.di, self.di + 2 * self.N, self.H], -1)
        xbc = self.conv(xbc)
        xs, Bm, Cm = xbc.split([self.di, self.N, self.N], -1)
        dt = F.softplus(dt + self.dt_bias)                                  # [B,T,H]
        la = dt * -self.A_log.exp()                                         # log decay per step
        cum = la.cumsum(1).transpose(1, 2)                                  # [B,H,T]
        seg = cum[..., :, None] - cum[..., None, :]                         # [B,H,T(i),T(j)]
        seg = seg.masked_fill(causal_mask(T, x.device), float("-inf"))
        L = seg.exp()
        cb = Cm @ Bm.transpose(1, 2)                                        # [B,T,T]
        W = L * cb[:, None] * dt.transpose(1, 2)[:, :, None, :]             # [B,H,T,T]
        xh = xs.view(B, T, self.H, self.P).transpose(1, 2)                  # [B,H,T,P]
        y = W @ xh + self.D[:, None, None] * xh
        y = y.transpose(1, 2).reshape(B, T, self.di)
        y = self.norm(y * F.silu(z))
        return self.out(y)


@register("kimilin")
class KimiLinear(nn.Module):
    """Kimi Linear layout (arXiv 2510.26692): KDA layers with a full MLA layer every few layers
    (paper 3 KDA : 1 MLA, MLA without RoPE). At 3 layers: KDA, KDA, MLA-NoPE (mla_every:3)."""
    def __init__(self, cfg, layer, o):
        super().__init__()
        every = o.get("mla_every", 3)
        if (layer + 1) % every == 0:
            o2 = dict(o, rope=0) if o.get("nope_mla", 1) else o
            self.m = MLA(cfg, layer, o2)
        else:
            self.m = KimiDeltaAttention(cfg, layer, o)

    def forward(self, x, cos, sin):
        return self.m(x, cos, sin)
