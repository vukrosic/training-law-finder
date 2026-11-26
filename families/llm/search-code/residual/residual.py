"""Residual-stream variants (search: residual). Selected by cfg.res; the default "pre" is the bed's pre-norm.

Every block is split into 2 sublayers F(h) = attn(n1(h)) and F(h) = ffn(n2(h)) (each with its own pre-norm,
as in the bed). A residual scheme decides what each sublayer reads and how its output is written back.

Baselines (written from the papers' equations):
  pre      x <- x + F(x)                                                    (bed default)
  hc       Hyper-Connections (Zhu et al., arXiv 2409.19606), dynamic DHC with tanh: n streams H [n,...];
           A_m(H) = s_a*tanh(norm(H) W_m) + A_m, A_r(H) = s_a*tanh(norm(H) W_r) + A_r,
           B(H) = s_b*tanh(norm(H) W_b) + B; h = sum_i A_m[i] H_i; H'_k = sum_i A_r[i,k] H_i + B[k] F(h).
           Init: A_m = e_{j mod n}, A_r = I, B = 1, W = 0, s = 0.01 (paper's Pre-Norm-equivalent init).
           res_dyn=false gives static SHC. Readout: sum of streams.
  mhc      mHC (DeepSeek, arXiv 2512.24880): x~ = RMSNorm(vec(H)) [nD];
           H_pre = sigmoid(a_pre x~ P_pre + b_pre), H_post = 2 sigmoid(a_post x~ P_post + b_post),
           H_res = Sinkhorn-Knopp(exp(a_res mat(x~ P_res) + b_res)), 20 iterations (doubly stochastic);
           H' = H_res H + H_post^T F(H_pre H). a init 0.01, P init 0; b_pre +-1 so sublayer j starts reading
           stream j mod n (breaks stream symmetry as HC's init does); b_post 0; b_res = res_c * I.
  attnres  Attention Residuals (Kimi team, arXiv 2603.15031): values v_0 = embedding, v_i = F_i output;
           input of sublayer l = sum_i softmax_i(w_l . RMSNorm(v_i)) v_i, pseudo-query w_l init 0.
           res_block=S: Block AttnRes, S sublayers per block summed; attention over block sums + partial sum.
  vres     Value residual / ResFormer (arXiv 2410.17897): V_n <- l1 V_n + l2 V_1 in every attention after
           the first; l = 0.5, 0.5 (res_learn=true: learned, init 0.5). Residual stream stays pre-norm.
  dense    DenseFormer (arXiv 2402.02622): after block i, x_i = sum_{j<=i} a_ij X_j, X_0 = embedding,
           X_j = block j output; a_ii = 1, rest 0. res_gran=sub applies DWA after every sublayer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Pre(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()

    def forward(self, x, subs):
        for f in subs:
            x = x + f(x)
        return x


class HC(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        n, d = cfg.res_n, cfg.d_model
        self.n, self.dyn = n, cfg.res_dyn
        am = torch.zeros(L, n)
        am[torch.arange(L), torch.arange(L) % n] = 1.0
        self.am = nn.Parameter(am)
        self.ar = nn.Parameter(torch.eye(n).repeat(L, 1, 1))
        self.b = nn.Parameter(torch.ones(L, n))
        if self.dyn:
            self.nw = nn.Parameter(torch.ones(L, d))
            self.wm = nn.Parameter(torch.zeros(L, d, n + 2))        # [W_m (1) | W_r (n) | W_b (1)]
            self.sa = nn.Parameter(torch.full((L,), 0.01))
            self.sb = nn.Parameter(torch.full((L,), 0.01))

    def forward(self, x, subs):
        n = self.n
        H = x.unsqueeze(0).expand(n, *x.shape)                       # [n,B,T,D]
        for j, f in enumerate(subs):
            if self.dyn:
                Hn = rms(H) * self.nw[j]
                z = torch.tanh(torch.einsum("nbtd,de->nbte", Hn, self.wm[j]))   # [n,B,T,n+2]
                am = self.am[j][:, None, None] + self.sa[j] * z[..., 0]       # [n,B,T]
                ar = self.ar[j][:, None, None, :] + self.sa[j] * z[..., 1:n + 1]  # [n(i),B,T,n(k)]
                # B(H) = s_b * tanh(H_bar W_b)^T + B: one write weight per stream
                b = self.b[j][:, None, None] + self.sb[j] * z[..., n + 1]
                h = (am.unsqueeze(-1) * H).sum(0)
                y = f(h)
                H = torch.einsum("ibtk,ibtd->kbtd", ar, H) + b.unsqueeze(-1) * y.unsqueeze(0)
            else:
                h = torch.einsum("n,nbtd->btd", self.am[j], H)
                y = f(h)
                H = torch.einsum("ik,ibtd->kbtd", self.ar[j], H) + self.b[j][:, None, None, None] * y
        return H.sum(0)


def sinkhorn(logits, iters):
    m = torch.exp(logits - logits.amax((-1, -2), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True)
        m = m / m.sum(-2, keepdim=True)
    return m


class MHC(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        n, d = cfg.res_n, cfg.d_model
        self.n, self.iters = n, cfg.res_sk
        self.nw = nn.Parameter(torch.ones(L, n * d))
        self.P = nn.Parameter(torch.zeros(L, n * d, n + n + n * n))
        self.a = nn.Parameter(torch.full((L, 3), 0.01))
        bp = -torch.ones(L, n)
        bp[torch.arange(L), torch.arange(L) % n] = 1.0
        self.b_pre = nn.Parameter(bp)
        self.b_post = nn.Parameter(torch.zeros(L, n))
        self.b_res = nn.Parameter(torch.eye(n).repeat(L, 1, 1) * cfg.res_c)

    def forward(self, x, subs):
        n = self.n
        B, T, D = x.shape
        H = x.unsqueeze(2).expand(B, T, n, D)                       # [B,T,n,D]
        for j, f in enumerate(subs):
            xt = rms(H.reshape(B, T, n * D)) * self.nw[j]
            z = xt @ self.P[j]
            pre = torch.sigmoid(self.a[j, 0] * z[..., :n] + self.b_pre[j])
            post = 2 * torch.sigmoid(self.a[j, 1] * z[..., n:2 * n] + self.b_post[j])
            res = sinkhorn(self.a[j, 2] * z[..., 2 * n:].view(B, T, n, n) + self.b_res[j], self.iters)
            h = torch.einsum("btn,btnd->btd", pre, H)
            y = f(h)
            H = torch.einsum("btkn,btnd->btkd", res, H) + post.unsqueeze(-1) * y.unsqueeze(2)
        return H.sum(2)


class AttnRes(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        d = cfg.d_model
        self.S = max(1, cfg.res_block)
        self.q = nn.Parameter(torch.zeros(L + 1, d))                 # +1: final readout query
        self.kw = nn.Parameter(torch.ones(L + 1, d))

    def mix(self, j, vals):
        V = torch.stack(vals, 0)                                     # [m,B,T,D]
        s = (rms(V) * self.kw[j] * self.q[j]).sum(-1)                # [m,B,T]
        a = s.softmax(0)
        return (a.unsqueeze(-1) * V).sum(0)

    def forward(self, x, subs):
        blocks, part = [x], None                                     # completed block sums, current partial
        for j, f in enumerate(subs):
            vals = blocks + ([part] if part is not None else [])
            y = f(self.mix(j, vals))
            part = y if part is None else part + y
            if (j + 1) % self.S == 0:
                blocks.append(part)
                part = None
        vals = blocks + ([part] if part is not None else [])
        return self.mix(len(subs), vals)


class Dense(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        self.per = 1 if cfg.res_gran == "sub" else 2
        nd = L // self.per
        a = torch.zeros(nd, nd + 1)
        a[torch.arange(nd), torch.arange(nd) + 1] = 1.0
        self.a = nn.Parameter(a)

    def forward(self, x, subs):
        X = [x]
        for j, f in enumerate(subs):
            x = x + f(x)
            if (j + 1) % self.per == 0:
                i = (j + 1) // self.per
                X.append(x)
                x = sum(self.a[i - 1, k] * X[k] for k in range(i + 1))
        return x


SCHEMES = {"pre": Pre, "vres": Pre, "hc": HC, "mhc": MHC, "attnres": AttnRes, "dense": Dense}


class Peri(nn.Module):
    """Peri-LN (Kim et al., arXiv 2502.02732; init search's family, used here as a reference baseline):
    x <- x + Norm_out(F(Norm_in(x))), output RMSNorm with vector gain init 1, plus an embedding RMSNorm
    (res_c > 0: embedding norm on; res_c = 0: off)."""
    def __init__(self, cfg, L):
        super().__init__()
        d = cfg.d_model
        self.w = nn.Parameter(torch.ones(L, d))
        self.emb = cfg.res_c > 0
        if self.emb:
            self.we = nn.Parameter(torch.ones(d))

    def forward(self, x, subs):
        if self.emb:
            x = rms(x) * self.we
        for j, f in enumerate(subs):
            x = x + rms(f(x)) * self.w[j]
        return x


SCHEMES["peri"] = Peri


class CompleteP(nn.Module):
    """CompleteP (Dey et al., arXiv 2505.01618, alpha = 1), used as a reference baseline: pre-LN residual with every
    branch scaled by m_L^-1 = L_base / L (L = blocks, L_base = rk blocks; Table 1). At alpha = 1 the hidden Adam lr and
    init have no depth term, so they are unchanged; the residual-block Adam epsilon scaling (eps_base m_L^-1) is not
    applied (optimizer path not owned here). rk = L blocks reduces exactly to pre."""
    def __init__(self, cfg, L):
        super().__init__()
        self.m = float(cfg.rk) / (L // 2)

    def forward(self, x, subs):
        for f in subs:
            x = x + self.m * f(x)
        return x


SCHEMES["completep"] = CompleteP


class NAG(nn.Module):
    """NAG, norm-agnostic residual (Figliolia & Millidge, arXiv 2606.16112, Eq. 10-17):
    rho = RMS(R), Rbar = R / rho; f centred; f_perp = f - <f,Rbar>/<Rbar,Rbar> Rbar;
    m = (sum_i p_i sigmoid(Rbar.w_i + b_i))^beta, C directions (p softmax, beta > 0 learned, init 1);
    R' = rho * Rbar + rho * alpha_l * m * N_out(f_perp), N_out -> unit RMS; alpha_l trainable, init 1/sqrt(l).
    Our sublayers keep their own pre-norm, so f(Rbar) = f(R). C = res_n (paper ablations use 32).
    The bed's final norm removes rho before the head (the paper scales logits by rho instead)."""
    def __init__(self, cfg, L):
        super().__init__()
        d, C = cfg.d_model, max(1, cfg.res_n)
        self.alpha = nn.Parameter(1.0 / torch.arange(1, L + 1).float().sqrt())
        self.w = nn.Parameter(torch.randn(L, C, d) / d)
        self.bb = nn.Parameter(torch.full((L, C), 3.0))        # gate starts near 1 (sigmoid(3) = 0.95)
        self.pl = nn.Parameter(torch.zeros(L, C))
        self.lb = nn.Parameter(torch.zeros(L))                 # beta = exp(lb)

    def forward(self, x, subs):
        R = x
        for j, f in enumerate(subs):
            rho = R.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            Rb = R / rho
            y = f(R)
            y = y - y.mean(-1, keepdim=True)
            y = y - (y * Rb).sum(-1, keepdim=True) / (Rb * Rb).sum(-1, keepdim=True) * Rb
            g = torch.sigmoid(torch.einsum("btd,cd->btc", Rb, self.w[j]) + self.bb[j])
            m = (g * self.pl[j].softmax(-1)).sum(-1, keepdim=True).clamp_min(1e-6) ** self.lb[j].exp()
            R = R + rho * self.alpha[j] * m * rms(y)
        return R


SCHEMES["nag"] = NAG


def make(cfg, L):
    if cfg.res in SCHEMES:
        return SCHEMES[cfg.res](cfg, L)
    import inventions_r2
    if cfg.res in inventions_r2.IDEAS:
        return inventions_r2.make(cfg, L)
    import inventions_r4
    if cfg.res in inventions_r4.IDEAS:
        return inventions_r2.wrap(inventions_r4.make(cfg, L), cfg, L)
    import inventions_r5
    if cfg.res in inventions_r5.IDEAS:
        return inventions_r2.wrap(inventions_r5.make(cfg, L), cfg, L)
    import inventions_r6
    if cfg.res in inventions_r6.IDEAS:
        return inventions_r2.wrap(inventions_r6.make(cfg, L), cfg, L)
    import inventions_r7
    if cfg.res in inventions_r7.IDEAS:
        return inventions_r2.wrap(inventions_r7.make(cfg, L), cfg, L)
    import inventions_r12
    if cfg.res in inventions_r12.IDEAS:
        return inventions_r2.wrap(inventions_r12.make(cfg, L), cfg, L)
    import inventions_r8
    if cfg.res in inventions_r8.IDEAS:
        return inventions_r2.wrap(inventions_r8.make(cfg, L), cfg, L)
    import inventions_res
    return inventions_r2.wrap(inventions_res.make(cfg, L), cfg, L)
