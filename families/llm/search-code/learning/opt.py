"""Matrix-weight learning rules for the learning search.

Matrix weights = every hidden weight with 2+ dims except the embedding and the output head:
attention qkv (split into q, k, v), attention o, router gate, shared-expert w1/w3/w2 and the
routed-expert w1/w3/w2 (one matrix per expert). Everything else stays on the bed's default
scalar optimizer. Every rule sees a batch of matrices in (out, in) layout.

Baselines (written from the papers' equations / official code):
  muon      Muon (Jordan et al. 2024) with the Moonlight update scale 0.2*sqrt(max(out,in))
            (Liu et al. 2025), Nesterov momentum 0.95, quintic Newton-Schulz 5 steps
  muonclip  Kimi K2 MuonClip: muon + QK-clip after each step: per head h,
            gamma = min(1, tau / S_max^h), W_q^h *= sqrt(gamma), W_k^h *= sqrt(gamma), tau = 100
  normuon   NorMuon (Li et al. 2025): O = NS(M); v = b2 v + (1-b2) mean_cols(O*O);
            O' = O / (sqrt(v)+eps); step = 0.2 * lr * sqrt(m n) / ||O'||_F
  soap      SOAP (Vyas et al. 2024): the default scalar optimizer's update in the eigenbasis of
            Shampoo's L = E[G G^T], R = E[G^T G]; betas (0.95, 0.95), basis every 10 steps,
            first step only builds the preconditioner (official code)
New rules live in RULES (added by round).
"""
import math
import torch


def _svd(M, full_matrices=False):
    """SVD on the CPU: torch.linalg.svd on CUDA takes about 20 s per step at XS sizes."""
    U, S, Vh = torch.linalg.svd(M.detach().cpu(), full_matrices=full_matrices)
    return U.to(M.device), S.to(M.device), Vh.to(M.device)


def _eigh(M):
    """eigh on the CPU for the same reason as _svd."""
    e, Q = torch.linalg.eigh(M.detach().cpu())
    return e.to(M.device), Q.to(M.device)

NS_ABC = (3.4445, -4.7750, 2.0315)


def ns5(G, steps=5, eps=1e-7, abc=NS_ABC):
    a, b, c = abc
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.transpose(-1, -2)
    return X


def kind_of(name, p):
    if p.dim() == 3:
        return "exp"
    if name.endswith("attn.qkv.weight"):
        return "qkv"
    return "lin"


def to_mats(t, kind):
    if kind == "exp":
        return t.transpose(1, 2)                    # (E, in, out) -> (E, out, in)
    if kind == "qkv":
        return t.view(3, t.shape[0] // 3, t.shape[1])
    return t.unsqueeze(0)


def from_mats(m, kind, shape):
    if kind == "exp":
        return m.transpose(1, 2)
    return m.reshape(shape)


def mats_params(model):
    out = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2 and not n.startswith("emb") and not n.startswith("head"):
            out.append((n, p))
    return out


class MatOpt(torch.optim.Optimizer):
    def __init__(self, named, cfg, model=None):
        self.names = {id(p): n for n, p in named}
        self.kinds = {id(p): kind_of(n, p) for n, p in named}
        super().__init__([p for _, p in named], dict(lr=cfg.mlr, base=cfg.mlr))
        self.cfg, self.model, self.t = cfg, model, 0
        self.rule = RULES[cfg.opt]
        self.smax = {}
        if cfg.opt == "muonclip" and model is not None:
            install_qk_probe(model, self.smax)
        if hasattr(self.rule, "setup"):
            self.rule.setup(self, model)

    @torch.no_grad()
    def step(self):
        self.t += 1
        for gr in self.param_groups:
            lr = gr["lr"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                k = self.kinds[id(p)]
                G = to_mats(p.grad, k).float()
                W = to_mats(p.data, k)
                st = self.state[p]
                U = self.rule(self, G, W, st, lr, p)        # full update (already times lr)
                if U is None:
                    continue
                if self.cfg.mwd:
                    W.mul_(1 - lr * self.cfg.mwd)
                W.sub_(U.to(W.dtype))
        if self.cfg.opt == "muonclip":
            qk_clip(self.model, self.smax, self.cfg.tau)
        if hasattr(self.rule, "post"):
            self.rule.post(self)


def moon_scale(G):
    return 0.2 * math.sqrt(max(G.shape[-2], G.shape[-1]))


def nesterov(st, G, beta, key="m"):
    if key not in st:
        st[key] = torch.zeros_like(G)
    st[key].lerp_(G, 1 - beta)
    return G.lerp(st[key], beta)


# ---------------------------------------------------------------- baselines
def r_muon(o, G, W, st, lr, p):
    M = nesterov(st, G, o.cfg.mom)
    return lr * moon_scale(G) * ns5(M)


def r_normuon(o, G, W, st, lr, p):
    M = nesterov(st, G, o.cfg.mom)
    O = ns5(M)
    if "v" not in st:
        st["v"] = torch.zeros(O.shape[:-1] + (1,), device=O.device)
    b2 = o.cfg.b2
    st["v"].mul_(b2).add_((O * O).mean(-1, keepdim=True), alpha=1 - b2)
    O2 = O / (st["v"].sqrt() + 1e-8)
    m, n = G.shape[-2], G.shape[-1]
    scale = 0.2 * math.sqrt(m * n) / (O2.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    return lr * scale * O2


def r_soap(o, G, W, st, lr, p):
    b1, b2, sb, f = o.cfg.mom, 0.95, 0.95, int(o.cfg.soap_f)
    if "L" not in st:
        st["L"] = torch.zeros(G.shape[:-1] + (G.shape[-2],), device=G.device)
        st["R"] = torch.zeros(G.shape[:-2] + (G.shape[-1], G.shape[-1]), device=G.device)
        st["L"].lerp_(G @ G.transpose(-1, -2), 1 - sb)
        st["R"].lerp_(G.transpose(-1, -2) @ G, 1 - sb)
        st["QL"] = _eigh(st["L"] + 1e-30 * torch.eye(G.shape[-2], device=G.device))[1]
        st["QR"] = _eigh(st["R"] + 1e-30 * torch.eye(G.shape[-1], device=G.device))[1]
        st["ea"] = torch.zeros_like(G)
        st["es"] = torch.zeros_like(G)
        st["n"] = 0
        return None
    st["n"] += 1
    QL, QR = st["QL"], st["QR"]
    Gp = QL.transpose(-1, -2) @ G @ QR
    st["ea"].mul_(b1).add_(Gp, alpha=1 - b1)
    st["es"].mul_(b2).add_(Gp * Gp, alpha=1 - b2)
    n = st["n"]
    step = lr * math.sqrt(1 - b2 ** n) / (1 - b1 ** n)
    U = QL @ (st["ea"] / (st["es"].sqrt() + 1e-8)) @ QR.transpose(-1, -2)
    st["L"].lerp_(G @ G.transpose(-1, -2), 1 - sb)
    st["R"].lerp_(G.transpose(-1, -2) @ G, 1 - sb)
    if n % f == 0:
        ea = QL @ st["ea"] @ QR.transpose(-1, -2)
        st["QL"] = _eigh(st["L"] + 1e-30 * torch.eye(G.shape[-2], device=G.device))[1]
        st["QR"] = _eigh(st["R"] + 1e-30 * torch.eye(G.shape[-1], device=G.device))[1]
        st["ea"] = st["QL"].transpose(-1, -2) @ ea @ st["QR"]
    return step * U


# ---------------------------------------------------------------- QK-clip (Kimi K2)
def install_qk_probe(model, smax):
    from model import rope
    for i, b in enumerate(model.blocks):
        att = b.attn

        def hook(mod, inp, out, i=i, att=att):
            with torch.no_grad():
                B, T, _ = out.shape
                q, k, _ = out.view(B, T, 3, att.h, att.dh).permute(2, 0, 3, 1, 4)
                q, k = rope(q, model.cos[:T], model.sin[:T]), rope(k, model.cos[:T], model.sin[:T])
                s = (q @ k.transpose(-1, -2)) / att.dh ** 0.5
                s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
                smax[i] = s.amax(dim=(0, 2, 3))            # per head
        att.qkv.register_forward_hook(hook)


@torch.no_grad()
def qk_clip(model, smax, tau):
    for i, b in enumerate(model.blocks):
        if i not in smax:
            continue
        att = b.attn
        g = (tau / smax[i]).clamp(max=1.0)
        model.cfg.smax_peak = max(getattr(model.cfg, "smax_peak", 0.0), float(smax[i].max()))
        if (g < 1).any():
            w = att.qkv.weight.view(3, att.h, att.dh, -1)
            w[0].mul_(g.sqrt().view(-1, 1, 1))
            w[1].mul_(g.sqrt().view(-1, 1, 1))
            model.cfg.qk_clips = getattr(model.cfg, "qk_clips", 0) + int((g < 1).sum())


RULES = dict(muon=r_muon, muonclip=r_muon, normuon=r_normuon, soap=r_soap)


# ================================================================ round 1 (2026-11-22)
def K(o, d):
    return o.cfg.k1 if o.cfg.k1 else d


def rms(D):
    return D.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-12


def unit(D):
    """scale each matrix to the Moonlight-Muon update RMS (0.2)"""
    return 0.2 * D / rms(D)


def spec(M, p):
    U, S, Vh = _svd(M, full_matrices=False)
    S = S / (S.amax(-1, keepdim=True) + 1e-12)
    return (U * S.pow(p).unsqueeze(-2)) @ Vh


def mom(o, st, G):
    return nesterov(st, G, o.cfg.mom)


def ema(st, key, x, b):
    if key not in st:
        st[key] = torch.zeros_like(x)
    st[key].mul_(b).add_(x, alpha=1 - b)
    return st[key]


def r_svd_polar(o, G, W, st, lr, p):
    return lr * unit(spec(mom(o, st, G), 0.0))


def r_spow(o, G, W, st, lr, p):
    return lr * unit(spec(mom(o, st, G), K(o, 0.25)))


def r_dpolar(o, G, W, st, lr, p):
    M = mom(o, st, G)
    U, S, Vh = _svd(M, full_matrices=False)
    s1 = S.amax(-1, keepdim=True)
    S2 = S / torch.sqrt(S * S + K(o, 0.01) * s1 * s1)
    return lr * unit((U * S2.unsqueeze(-2)) @ Vh)


def r_colnorm(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    v = ema(st, "v", (O * O).mean(-2, keepdim=True), o.cfg.b2)
    return lr * unit(O / (v.sqrt() + 1e-8))


def r_binorm(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    O = O / (O.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)
    O = O / (O.pow(2).mean(-2, keepdim=True).sqrt() + 1e-8)
    return lr * unit(O)


def r_adapre(o, G, W, st, lr, p):
    v = ema(st, "v", G * G, 0.95)
    M = mom(o, st, G)
    return lr * unit(ns5(M / (v.sqrt() + 1e-12)))


def r_postmom(o, G, W, st, lr, p):
    O = ns5(G)
    return lr * unit(nesterov(st, O, o.cfg.mom))


def r_dualmom(o, G, W, st, lr, p):
    M = mom(o, st, G)
    S = ema(st, "slow", G, 0.99)
    return lr * unit(ns5(M / rms(M) + K(o, 0.5) * S / rms(S)))


def r_mars(o, G, W, st, lr, p):
    g0 = st.get("gp", torch.zeros_like(G))
    st["gp"] = G.clone()
    b = o.cfg.mom
    c = G + K(o, 0.025) * b / (1 - b) * (G - g0)
    n = c.norm(dim=(-2, -1), keepdim=True)
    c = torch.where(n > 1, c / n, c)
    return lr * unit(ns5(mom(o, st, c)))


def r_cautious(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    mask = (O * G > 0).float()
    return lr * unit(O * mask / (mask.mean(dim=(-2, -1), keepdim=True) + 1e-3))


def r_normaware(o, G, W, st, lr, p):
    M = mom(o, st, G)
    n = M.norm(dim=(-2, -1), keepdim=True)
    return lr * unit(ns5(M)) * (n / (n.mean() + 1e-12)).pow(K(o, 0.5))


def r_timenorm(o, G, W, st, lr, p):
    M = mom(o, st, G)
    n = M.norm(dim=(-2, -1), keepdim=True)
    e = ema(st, "n", n, 0.9) / (1 - 0.9 ** (st.get("t", 0) + 1))
    st["t"] = st.get("t", 0) + 1
    return lr * unit(ns5(M)) * (n / (e + 1e-12)).pow(K(o, 0.5))


def r_jointexp(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "exp":
        return lr * unit(ns5(M))
    E, a, b = M.shape
    O = ns5(M.reshape(E * a, b)).reshape(E, a, b)
    return lr * 0.2 * O / rms(O.reshape(1, E * a, b)).reshape(1, 1, 1)


def r_expsplit(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "exp":
        return lr * unit(ns5(M))
    Mb = M.mean(0, keepdim=True)
    D = unit(ns5(Mb)).expand_as(M) + K(o, 1.0) * unit(ns5(M - Mb))
    return lr * unit(D)


def _setup_load(o, model):
    o.mod_of = {}
    for m in model.moe_layers():
        for w in (m.w1, m.w2, m.w3):
            o.mod_of[id(w)] = m


def r_loadscale(o, G, W, st, lr, p):
    U = lr * unit(ns5(mom(o, st, G)))
    m = o.mod_of.get(id(p))
    if m is None or getattr(m, "last_load", None) is None:
        return U
    ld = m.last_load
    f = (ld / (ld.mean() + 1e-9)).clamp(min=1e-3).pow(K(o, 0.5))
    return U * f.view(-1, 1, 1)


r_loadscale.setup = _setup_load


def r_keller(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return lr * 1.6 * max(1.0, G.shape[-2] / G.shape[-1]) ** 0.5 * ns5(M)


def _name_mult(o, p, key, f):
    return f if key in o.names[id(p)] else 1.0


def r_routerlr(o, G, W, st, lr, p):
    return lr * _name_mult(o, p, "ffn.gate", K(o, 0.3)) * unit(ns5(mom(o, st, G)))


def r_attnlr(o, G, W, st, lr, p):
    return lr * _name_mult(o, p, ".attn.", K(o, 2.0)) * unit(ns5(mom(o, st, G)))


def r_depthlr(o, G, W, st, lr, p):
    i = int(o.names[id(p)].split(".")[1])
    L = o.cfg.n_layers
    f = 1 + K(o, -0.3) * (i - (L - 1) / 2)
    return lr * max(f, 0.1) * unit(ns5(mom(o, st, G)))


def r_relupd(o, G, W, st, lr, p):
    f = (rms(W.float()) / 0.02).pow(K(o, 1.0))
    return lr * f * unit(ns5(mom(o, st, G)))


def r_condcap(o, G, W, st, lr, p):
    return lr * unit(ns5(mom(o, st, G)))


def _post_condcap(o):
    c = K(o, 10.0)
    for gr in o.param_groups:
        for p in gr["params"]:
            k = o.kinds[id(p)]
            W = to_mats(p.data, k)
            U, S, Vh = _svd(W.float(), full_matrices=False)
            med = S.median(-1, keepdim=True).values
            S2 = torch.minimum(S, c * med)
            if (S2 < S).any():
                W.copy_(((U * S2.unsqueeze(-2)) @ Vh).to(W.dtype))


r_condcap.post = _post_condcap


def r_soapns(o, G, W, st, lr, p):
    U = r_soap(o, G, W, st, 1.0, p)
    if U is None:
        return lr * unit(ns5(G))
    return lr * unit(ns5(U))


def r_tgram(o, G, W, st, lr, p):
    M = mom(o, st, G)
    L = ema(st, "L", M @ M.transpose(-1, -2), 0.95)
    e, Q = _eigh(L)
    e = e.clamp(min=0)
    d = (e + 1e-3 * e.amax(-1, keepdim=True) + 1e-30).rsqrt()
    return lr * unit((Q * d.unsqueeze(-2)) @ Q.transpose(-1, -2) @ M)


def r_decor(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    if "up" in st:
        u = st["up"]
        c = (O * u).sum(dim=(-2, -1), keepdim=True) / (u.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        O = O - K(o, 0.5) * c * u
    st["up"] = O.clone()
    return lr * unit(O)


def r_rowgain(o, G, W, st, lr, p):
    M = mom(o, st, G)
    r = M.norm(dim=-1, keepdim=True)
    return lr * unit(ns5(M) * (r / (r.mean(-2, keepdim=True) + 1e-12)).pow(K(o, 0.5)))


def r_colgain(o, G, W, st, lr, p):
    M = mom(o, st, G)
    r = M.norm(dim=-2, keepdim=True)
    return lr * unit(ns5(M) * (r / (r.mean(-1, keepdim=True) + 1e-12)).pow(K(o, 0.5)))


def r_rowtrust(o, G, W, st, lr, p):
    w = W.float().norm(dim=-1, keepdim=True)
    return lr * unit(ns5(mom(o, st, G)) * (w / (w.mean(-2, keepdim=True) + 1e-12)).pow(K(o, 1.0)))


def r_spike(o, G, W, st, lr, p):
    M = mom(o, st, G)
    n = G.norm(dim=(-2, -1), keepdim=True)
    t = st.get("t", 0) + 1
    st["t"] = t
    e = ema(st, "n", n, 0.9) / (1 - 0.9 ** t)
    return lr * unit(ns5(M)) / (1 + K(o, 1.0) * (n / (e + 1e-12) - 1).clamp(min=0))


def r_hsign(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return lr * unit(unit(ns5(M)) + K(o, 0.3) * unit(torch.sign(M)))


def r_adammix(o, G, W, st, lr, p):
    M = mom(o, st, G)
    v = ema(st, "v", G * G, 0.95)
    m = ema(st, "m2", G, 0.9)
    return lr * unit(unit(ns5(M)) + K(o, 0.3) * unit(m / (v.sqrt() + 1e-12)))


def r_sphere(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    Wf = W.float()
    c = (O * Wf).sum(dim=(-2, -1), keepdim=True) / (Wf.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return lr * unit(O - K(o, 1.0) * c * Wf)


def r_panneal(o, G, W, st, lr, p):
    return lr * unit(spec(mom(o, st, G), K(o, 0.5) * o.t / o.cfg.T))


def r_pdecay(o, G, W, st, lr, p):
    return lr * unit(spec(mom(o, st, G), K(o, 0.5) * (1 - o.t / o.cfg.T)))


def r_momflip(o, G, W, st, lr, p):
    if "m" in st:
        flip = ((G * st["m"]).sum(-1, keepdim=True) < 0).float()
        st["m"].mul_(1 - flip).add_(flip * G)
    return lr * unit(ns5(mom(o, st, G)))


def r_qkjoint(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "qkv":
        return lr * unit(ns5(M))
    d = M.shape[-1]
    qk = ns5(M[:2].reshape(2 * d, d)).reshape(2, d, d)
    return lr * unit(torch.cat([unit(qk.unsqueeze(0))[0], unit(ns5(M[2:]))], 0))


def r_qkvjoint(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "qkv":
        return lr * unit(ns5(M))
    d = M.shape[-1]
    return lr * unit(ns5(M.reshape(1, 3 * d, d))).reshape(3, d, d)


def r_headwise(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "qkv":
        return lr * unit(ns5(M))
    h = o.cfg.n_heads
    d = M.shape[-1]
    Mh = M.reshape(3 * h, d // h, d)
    return lr * unit(ns5(Mh).reshape(3, d, d).unsqueeze(0))[0]


def r_plain(o, G, W, st, lr, p):
    return lr * unit(ns5(mom(o, st, G)))


def _post_lookahead(o):
    k, a = 5, K(o, 0.5)
    if o.t % k:
        return
    for gr in o.param_groups:
        for p in gr["params"]:
            st = o.state[p]
            if "slow" not in st:
                st["slow"] = p.data.clone()
                continue
            st["slow"].add_(a * (p.data - st["slow"]))
            p.data.copy_(st["slow"])


def r_lookahead(o, G, W, st, lr, p):
    return r_plain(o, G, W, st, lr, p)


r_lookahead.post = _post_lookahead


def _post_emaw(o):
    b = K(o, 0.9)
    for gr in o.param_groups:
        for p in gr["params"]:
            st = o.state[p]
            if "wema" not in st:
                st["wema"] = p.data.clone()
            st["wema"].mul_(b).add_(p.data, alpha=1 - b)
            if o.t == o.cfg.T:
                p.data.copy_(st["wema"])


def r_emaw(o, G, W, st, lr, p):
    return r_plain(o, G, W, st, lr, p)


r_emaw.post = _post_emaw


def r_ns3(o, G, W, st, lr, p):
    return lr * unit(ns5(mom(o, st, G), steps=3))


RULES.update(svd_polar=r_svd_polar, spow=r_spow, dpolar=r_dpolar, colnorm=r_colnorm, binorm=r_binorm,
             adapre=r_adapre, postmom=r_postmom, dualmom=r_dualmom, mars=r_mars, cautious=r_cautious,
             normaware=r_normaware, timenorm=r_timenorm, jointexp=r_jointexp, expsplit=r_expsplit,
             loadscale=r_loadscale, keller=r_keller, routerlr=r_routerlr, attnlr=r_attnlr, depthlr=r_depthlr,
             relupd=r_relupd, condcap=r_condcap, soapns=r_soapns, tgram=r_tgram, decor=r_decor,
             rowgain=r_rowgain, colgain=r_colgain, rowtrust=r_rowtrust, spike=r_spike, hsign=r_hsign,
             adammix=r_adammix, sphere=r_sphere, panneal=r_panneal, pdecay=r_pdecay, momflip=r_momflip,
             qkjoint=r_qkjoint, qkvjoint=r_qkvjoint, headwise=r_headwise, lookahead=r_lookahead,
             emaw=r_emaw, ns3=r_ns3, plain=r_plain)


# ================================================================ round 2 (2026-11-22)
# All round 2+ rules keep Muon's size: the final direction is rescaled to the RMS of Muon's
# Newton-Schulz output O for that matrix, times lr * 0.2 * sqrt(max(out, in)).
def fin(G, O, D, lr):
    return lr * moon_scale(G) * D * (rms(O) / rms(D))


def rowcos(a, b, dim=-1):
    return (a * b).sum(dim, keepdim=True) / (a.norm(dim=dim, keepdim=True) * b.norm(dim=dim, keepdim=True) + 1e-12)


def flip_rows(st, G, thr=0.0, dim=-1):
    if "m" in st:
        f = (rowcos(G, st["m"], dim) < thr).float()
        st["m"].mul_(1 - f).add_(f * G)


def r2_muon(o, G, W, st, lr, p):
    return lr * moon_scale(G) * ns5(mom(o, st, G))


def r2_momflip(o, G, W, st, lr, p):
    flip_rows(st, G)
    return r2_muon(o, G, W, st, lr, p)


def _caut(O, G):
    mask = (O * G > 0).float()
    return O * mask / (mask.mean(dim=(-2, -1), keepdim=True) + 1e-3)


def _sph(O, W, k=1.0):
    Wf = W.float()
    c = (O * Wf).sum(dim=(-2, -1), keepdim=True) / (Wf.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return O - k * c * Wf


def r2_cautious(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, _caut(O, G), lr)


def r2_sphere(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, _sph(O, W, K(o, 1.0)), lr)


def r2_elemflip(o, G, W, st, lr, p):
    if "m" in st:
        f = (G * st["m"] < 0).float()
        st["m"].mul_(1 - f).add_(f * G)
    return r2_muon(o, G, W, st, lr, p)


def r2_colflip(o, G, W, st, lr, p):
    flip_rows(st, G, 0.0, dim=-2)
    return r2_muon(o, G, W, st, lr, p)


def r2_softflip(o, G, W, st, lr, p):
    if "m" in st:
        st["m"].mul_(rowcos(G, st["m"]).clamp(min=0))
    return r2_muon(o, G, W, st, lr, p)


def r2_cosmom(o, G, W, st, lr, p):
    bmax = K(o, 0.9)
    if "m" not in st:
        st["m"] = G.clone()
    b = bmax * rowcos(G, st["m"]).clamp(min=0)
    st["m"].mul_(b).add_((1 - b) * G)
    M = G * (1 - b) + st["m"] * b
    return lr * moon_scale(G) * ns5(M)


def r2_matflip(o, G, W, st, lr, p):
    if "m" in st:
        f = ((G * st["m"]).sum(dim=(-2, -1), keepdim=True) < 0).float()
        st["m"].mul_(1 - f).add_(f * G)
    return r2_muon(o, G, W, st, lr, p)


def r2_flipsphere(o, G, W, st, lr, p):
    flip_rows(st, G)
    return r2_sphere(o, G, W, st, lr, p)


def r2_cautsphere(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, _sph(_caut(O, G), W), lr)


def r2_flipcaut(o, G, W, st, lr, p):
    flip_rows(st, G)
    return r2_cautious(o, G, W, st, lr, p)


def r2_snrmom(o, G, W, st, lr, p):
    m1 = ema(st, "g1", G, 0.9)
    m2 = ema(st, "g2", G * G, 0.9)
    snr = (m1.pow(2).sum(-1, keepdim=True) / (m2.sum(-1, keepdim=True) + 1e-20)).clamp(0, 1)
    b = K(o, 0.9) * snr
    if "m" not in st:
        st["m"] = G.clone()
    st["m"].mul_(b).add_((1 - b) * G)
    return lr * moon_scale(G) * ns5(G * (1 - b) + st["m"] * b)


def r2_halfsphere(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, _sph(O, W, 0.5), lr)


def r2_rowsphere(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    Wf = W.float()
    c = (O * Wf).sum(-1, keepdim=True) / (Wf.pow(2).sum(-1, keepdim=True) + 1e-12)
    return fin(G, O, O - c * Wf, lr)


def r2_heavyball(o, G, W, st, lr, p):
    M = ema(st, "m", G, o.cfg.mom)
    return lr * moon_scale(G) * ns5(M)


def _mom_b(st, G, b):
    if "m" not in st:
        st["m"] = torch.zeros_like(G)
    st["m"].lerp_(G, 1 - b)
    return G.lerp(st["m"], b)


def r2_momwarm(o, G, W, st, lr, p):
    b = 0.3 + (o.cfg.mom - 0.3) * min(1.0, o.t / 50)
    return lr * moon_scale(G) * ns5(_mom_b(st, G, b))


def r2_momdecay(o, G, W, st, lr, p):
    b = 0.9 - 0.4 * o.t / o.cfg.T
    return lr * moon_scale(G) * ns5(_mom_b(st, G, b))


def r2_twoorth(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, O + K(o, 0.5) * ns5(G), lr)


def r2_flipthr(o, G, W, st, lr, p):
    flip_rows(st, G, K(o, 0.2))
    return r2_muon(o, G, W, st, lr, p)


def r2_lookahead(o, G, W, st, lr, p):
    return r2_muon(o, G, W, st, lr, p)


r2_lookahead.post = _post_lookahead


# ---- explore
def r2_rowcenter(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    return fin(G, O, ns5(M - M.mean(-2, keepdim=True)), lr)


def r2_colcenter(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    return fin(G, O, ns5(M - M.mean(-1, keepdim=True)), lr)


def r2_deflate(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    U, S, Vh = _svd(M, full_matrices=False)
    top = U[..., :, :1] @ Vh[..., :1, :]
    return fin(G, O, O - K(o, 0.5) * top, lr)


def r2_lowrank(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    U, S, Vh = _svd(M, full_matrices=False)
    r = max(1, S.shape[-1] // 2)
    return fin(G, O, U[..., :, :r] @ Vh[..., :r, :], lr)


def r2_langevin(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, O + K(o, 0.1) * rms(O) * torch.randn_like(O), lr)


def _scalar_step(o, G, st, lr):
    """the bed's default scalar optimizer (betas 0.9/0.95, eps 1e-8) at the scalar lr schedule"""
    st["n"] = st.get("n", 0) + 1
    m = ema(st, "sm", G, 0.9)
    v = ema(st, "sv", G * G, 0.95)
    n = st["n"]
    lr_s = o.cfg.lr * lr / o.cfg.mlr
    return lr_s * (m / (1 - 0.9 ** n)) / ((v / (1 - 0.95 ** n)).sqrt() + 1e-8)


def _scalar_if(key):
    def r(o, G, W, st, lr, p):
        if key(o, p):
            return _scalar_step(o, G, st, lr)
        return r2_muon(o, G, W, st, lr, p)
    return r


r2_router_sc = _scalar_if(lambda o, p: "ffn.gate" in o.names[id(p)])
r2_expert_sc = _scalar_if(lambda o, p: o.kinds[id(p)] == "exp")
r2_attn_sc = _scalar_if(lambda o, p: ".attn." in o.names[id(p)])


def r2_shared2(o, G, W, st, lr, p):
    return _name_mult(o, p, "ffn.shared", K(o, 2.0)) * r2_muon(o, G, W, st, lr, p)


def r2_expmomshare(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    if o.kinds[id(p)] == "exp":
        a = K(o, 0.3)
        st["m"].mul_(1 - a).add_(a * st["m"].mean(0, keepdim=True))
    return U


def r2_specclip(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    wn = torch.linalg.matrix_norm(W.float(), ord=2, keepdim=True)
    un = torch.linalg.matrix_norm(U, ord=2, keepdim=True)
    return U * (K(o, 0.02) * wn / (un + 1e-12)).clamp(max=1.0)


def r2_signortho(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    return fin(G, O, ns5(torch.sign(M)), lr)


def r2_negpow(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    U, S, Vh = _svd(M, full_matrices=False)
    S = (S / (S.amax(-1, keepdim=True) + 1e-12)).clamp(min=1e-2)
    return fin(G, O, (U * S.pow(K(o, -0.25)).unsqueeze(-2)) @ Vh, lr)


def r2_invload(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    m = o.mod_of.get(id(p))
    if m is None or getattr(m, "last_load", None) is None:
        return U
    ld = m.last_load
    f = (ld / (ld.mean() + 1e-9)).clamp(min=0.05).pow(K(o, -0.5)).clamp(max=3.0)
    return U * f.view(-1, 1, 1)


r2_invload.setup = _setup_load


def r2_bbstep(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    f = torch.ones(U.shape[:-2] + (1, 1))
    if "pu" in st:
        s, y = -st["pu"], G - st["pg"]
        sy = (s * y).sum(dim=(-2, -1), keepdim=True)
        bb = (s * s).sum(dim=(-2, -1), keepdim=True) / sy.clamp(min=1e-20)
        bb = torch.where(sy > 0, bb, torch.full_like(bb, float("nan")))
        e = st.get("bbe")
        if e is None:
            st["bbe"] = torch.nan_to_num(bb, nan=1.0)
        else:
            st["bbe"] = torch.where(torch.isnan(bb), e, 0.9 * e + 0.1 * bb)
            f = torch.nan_to_num(bb / st["bbe"], nan=1.0).clamp(0.5, 2.0)
    st["pu"], st["pg"] = U.clone(), G.clone()
    return U * f


def r2_hyperlr(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    f = torch.ones(U.shape[:-2] + (1, 1))
    if "pu" in st:
        c = -(G * st["pu"]).sum(dim=(-2, -1), keepdim=True) / (G.norm(dim=(-2, -1), keepdim=True) * st["pu"].norm(dim=(-2, -1), keepdim=True) + 1e-12)
        st["h"] = st.get("h", torch.zeros_like(c)) * 0.9 + 0.1 * c
        f = (1 + K(o, 1.0) * st["h"]).clamp(0.5, 2.0)
    st["pu"] = U.clone()
    return U * f


def r2_wdinside(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    Wf = W.float()
    return fin(G, O, ns5(M + K(o, 0.1) * Wf * M.norm(dim=(-2, -1), keepdim=True) / (Wf.norm(dim=(-2, -1), keepdim=True) + 1e-12)), lr)


def r2_noloadskip(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    if o.kinds[id(p)] == "exp":
        live = (G.abs().amax(dim=(-2, -1), keepdim=True) > 0).float()
        U = U * live
    return U


def r2_lars(o, G, W, st, lr, p):
    U = r2_muon(o, G, W, st, lr, p)
    r = K(o, 0.05) * W.float().norm(dim=(-2, -1), keepdim=True) / (U.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    return U * r.clamp(max=1.0)


def r2_normgrad(o, G, W, st, lr, p):
    Gn = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    return lr * moon_scale(G) * ns5(mom(o, st, Gn))


RULES.update(
    m2=r2_muon, momflip2=r2_momflip, cautious2=r2_cautious, sphere2=r2_sphere, elemflip=r2_elemflip,
    colflip=r2_colflip, softflip=r2_softflip, cosmom=r2_cosmom, matflip=r2_matflip, flipsphere=r2_flipsphere,
    cautsphere=r2_cautsphere, flipcaut=r2_flipcaut, snrmom=r2_snrmom, halfsphere=r2_halfsphere,
    rowsphere=r2_rowsphere, heavyball=r2_heavyball, momwarm=r2_momwarm, momdecay=r2_momdecay,
    twoorth=r2_twoorth, flipthr=r2_flipthr, lookahead2=r2_lookahead,
    rowcenter=r2_rowcenter, colcenter=r2_colcenter, deflate=r2_deflate, lowrank=r2_lowrank,
    langevin=r2_langevin, router_sc=r2_router_sc, expert_sc=r2_expert_sc, attn_sc=r2_attn_sc,
    shared2=r2_shared2, expmomshare=r2_expmomshare, specclip=r2_specclip, signortho=r2_signortho,
    negpow=r2_negpow, invload=r2_invload, bbstep=r2_bbstep, hyperlr=r2_hyperlr, wdinside=r2_wdinside,
    noloadskip=r2_noloadskip, lars=r2_lars, normgrad=r2_normgrad)


# ================================================================ round 3 (2026-11-22)
# Exploit: shrink the top singular direction(s) of Muon's update (round 2 leader: deflate).
def _top(M, r=1):
    U, S, Vh = _svd(M, full_matrices=False)
    return U[..., :, :r] @ Vh[..., :r, :], S


def _defl(o, G, st, M, k, r=1, src=None):
    O = ns5(M)
    T, _ = _top(M if src is None else src, r)
    return fin(G, O, O - k * T, o._lr)


def _wrap(f):
    def r(o, G, W, st, lr, p):
        o._lr = lr
        return f(o, G, W, st, lr, p)
    return r


def r3_defl1(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 1.0))


def r3_defl75(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 0.75))


def r3_defl2(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 0.5), r=2)


def r3_defl4(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 0.5), r=4)


def r3_deflg(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 0.5), src=G)


def r3_deflema(o, G, W, st, lr, p):
    slow = ema(st, "slow", G, 0.95)
    return _defl(o, G, st, mom(o, st, G), K(o, 0.5), src=slow)


def r3_histdefl(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    h = ema(st, "h", O, 0.9)
    T, _ = _top(h)
    return fin(G, O, O - K(o, 0.5) * T, lr)


def r3_deflgap(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    T, S = _top(M)
    k = (1 - S[..., 1:2] / (S[..., :1] + 1e-12)).clamp(0, 1).unsqueeze(-1)
    return fin(G, O, O - K(o, 1.0) * k * T, lr)


def r3_negpow5(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), _negspec(M, K(o, -0.5)), lr)


def _negspec(M, pw):
    U, S, Vh = _svd(M, full_matrices=False)
    S = (S / (S.amax(-1, keepdim=True) + 1e-12)).clamp(min=1e-2)
    return (U * S.pow(pw).unsqueeze(-2)) @ Vh


def r3_negpow1(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), _negspec(M, K(o, -0.1)), lr)


def r3_deflneg(o, G, W, st, lr, p):
    M = mom(o, st, G)
    U, S, Vh = _svd(M, full_matrices=False)
    S = (S / (S.amax(-1, keepdim=True) + 1e-12)).clamp(min=1e-2).pow(-0.25)
    S[..., 0] = S[..., 0] * (1 - K(o, 0.5))
    return fin(G, ns5(M), (U * S.unsqueeze(-2)) @ Vh, lr)


def r3_deflpi(o, G, W, st, lr, p):
    """deflate with the top singular pair from one warm-started power iteration per step"""
    M = mom(o, st, G)
    O = ns5(M)
    if "v" not in st:
        st["v"] = torch.randn(M.shape[:-2] + (M.shape[-1], 1), device=M.device)
    v = M.transpose(-1, -2) @ (M @ st["v"])
    v = v / (v.norm(dim=-2, keepdim=True) + 1e-12)
    st["v"] = v
    u = M @ v
    u = u / (u.norm(dim=-2, keepdim=True) + 1e-12)
    return fin(G, O, O - K(o, 0.5) * (u @ v.transpose(-1, -2)), lr)


def r3_defllars(o, G, W, st, lr, p):
    U = _defl(o, G, st, mom(o, st, G), 0.5)
    r = K(o, 0.05) * W.float().norm(dim=(-2, -1), keepdim=True) / (U.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    return U * r.clamp(max=1.0)


def r3_deflnorm(o, G, W, st, lr, p):
    Gn = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    M = mom(o, st, Gn)
    O = ns5(M)
    T, _ = _top(M)
    return lr * moon_scale(G) * (O - K(o, 0.5) * T) * (rms(O) / rms(O - K(o, 0.5) * T))


def r3_deflskip(o, G, W, st, lr, p):
    U = _defl(o, G, st, mom(o, st, G), K(o, 0.5))
    if o.kinds[id(p)] == "exp":
        U = U * (G.abs().amax(dim=(-2, -1), keepdim=True) > 0).float()
    return U


def r3_deflwarm(o, G, W, st, lr, p):
    b = 0.3 + (o.cfg.mom - 0.3) * min(1.0, o.t / 50)
    return _defl(o, G, st, _mom_b(st, G, b), K(o, 0.5))


def r3_deflattn(o, G, W, st, lr, p):
    if ".attn." in o.names[id(p)]:
        return _defl(o, G, st, mom(o, st, G), K(o, 0.5))
    return r2_muon(o, G, W, st, lr, p)


def r3_deflffn(o, G, W, st, lr, p):
    if ".attn." not in o.names[id(p)]:
        return _defl(o, G, st, mom(o, st, G), K(o, 0.5))
    return r2_muon(o, G, W, st, lr, p)


def r3_deflanneal(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 1.0) * (1 - o.t / o.cfg.T))


def r3_deflup(o, G, W, st, lr, p):
    return _defl(o, G, st, mom(o, st, G), K(o, 1.0) * o.t / o.cfg.T)


# Explore: new mechanisms.
def r3_antiprev(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    D = O
    if "up" in st:
        up = st["up"]
        c = (O * up).sum(dim=(-2, -1), keepdim=True) / (up.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        D = O - K(o, 0.5) * c * up
    st["up"] = O
    return fin(G, O, D, lr)


def r3_agree(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    m = 1.0
    if "up" in st:
        up = st["up"]
        c = (O * up).sum(dim=(-2, -1), keepdim=True) / (O.norm(dim=(-2, -1), keepdim=True) * up.norm(dim=(-2, -1), keepdim=True) + 1e-12)
        m = (1 + K(o, 1.0) * c).clamp(0.5, 2.0)
    st["up"] = O
    return m * lr * moon_scale(G) * O


def r3_gclip(o, G, W, st, lr, p):
    s = G.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() * K(o, 3.0)
    return r2_muon(o, torch.maximum(torch.minimum(G, s), -s), W, st, lr, p)


def r3_oclip(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    s = rms(O) * K(o, 2.5)
    return fin(G, O, torch.maximum(torch.minimum(O, s), -s), lr)


def r3_wdeflate(o, G, W, st, lr, p):
    """remove the update's component along the weight's own top singular pair u1 v1^T"""
    O = ns5(mom(o, st, G))
    U, S, Vh = _svd(W.float(), full_matrices=False)
    u, v = U[..., :, :1], Vh[..., :1, :]
    c = (u.transpose(-1, -2) @ O @ v.transpose(-1, -2))
    return fin(G, O, O - K(o, 1.0) * c * (u @ v), lr)


def r3_rownormin(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), ns5(M / (M.norm(dim=-1, keepdim=True) + 1e-12)), lr)


def r3_colnormin(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), ns5(M / (M.norm(dim=-2, keepdim=True) + 1e-12)), lr)


def r3_dcenter(o, G, W, st, lr, p):
    M = mom(o, st, G)
    C = M - M.mean(-1, keepdim=True) - M.mean(-2, keepdim=True) + M.mean(dim=(-2, -1), keepdim=True)
    return fin(G, ns5(M), ns5(C), lr)


def r3_grokfast(o, G, W, st, lr, p):
    slow = ema(st, "slow", G, 0.98)
    return r2_muon(o, G + K(o, 2.0) * slow, W, st, lr, p)


def r3_efb(o, G, W, st, lr, p):
    e = st.get("e", torch.zeros_like(G))
    M = mom(o, st, G + e)
    O = ns5(M)
    s = (M * O).sum(dim=(-2, -1), keepdim=True) / (O.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
    st["e"] = K(o, 0.5) * (M - s * O)
    return fin(G, O, O, lr)


def r3_nsclean(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    X = O
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-1, -2)
    for _ in range(2):
        X = 1.5 * X - 0.5 * (X @ X.transpose(-1, -2)) @ X
    if tr:
        X = X.transpose(-1, -2)
    return fin(G, O, X, lr)


def r3_dualmom2(o, G, W, st, lr, p):
    f = nesterov(st, G, 0.3, "mf")
    s = nesterov(st, G, 0.9, "ms")
    O = ns5(mom(o, st, G))
    return fin(G, O, 0.5 * ns5(f) + 0.5 * ns5(s), lr)


def r3_pospow(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), spec(M, K(o, 0.25)), lr)


def r3_gnratio(o, G, W, st, lr, p):
    n = G.norm(dim=(-2, -1), keepdim=True)
    e = ema(st, "gn", n, 0.9) / (1 - 0.9 ** o.t)
    m = (n / (e + 1e-12)).pow(-K(o, 0.5)).clamp(0.5, 2.0)
    return m * r2_muon(o, G, W, st, lr, p)


def r3_tailboost(o, G, W, st, lr, p):
    M = mom(o, st, G)
    U, S, Vh = _svd(M, full_matrices=False)
    w = torch.ones_like(S)
    w[..., S.shape[-1] // 2:] = K(o, 1.5)
    return fin(G, ns5(M), (U * w.unsqueeze(-2)) @ Vh, lr)


def r3_specwd(o, G, W, st, lr, p):
    """Muon + weight decay on the weight's top singular value only"""
    U, S, Vh = _svd(W.float(), full_matrices=False)
    top = S[..., :1].unsqueeze(-1) * (U[..., :, :1] @ Vh[..., :1, :])
    return r2_muon(o, G, W, st, lr, p) + K(o, 0.5) * lr * top


def r3_nest2(o, G, W, st, lr, p):
    b = o.cfg.mom
    if "m" not in st:
        st["m"] = torch.zeros_like(G)
    st["m"].lerp_(G, 1 - b)
    M = G + min(2 * b, 1.6) * (st["m"] - G)
    return lr * moon_scale(G) * ns5(M)


def r3_odrop(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    mask = (torch.rand_like(O) > K(o, 0.1)).float()
    return fin(G, O, O * mask, lr)


def r3_nscubic(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), ns5(M, steps=10, abc=(1.5, -0.5, 0.0)), lr)


def r3_ocenter(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    return fin(G, O, O - O.mean(-1, keepdim=True), lr)


_R3 = dict(defl1=r3_defl1, defl75=r3_defl75, defl2=r3_defl2, defl4=r3_defl4, deflg=r3_deflg,
           deflema=r3_deflema, histdefl=r3_histdefl, deflgap=r3_deflgap, negpow5=r3_negpow5,
           negpow1=r3_negpow1, deflneg=r3_deflneg, deflpi=r3_deflpi, defllars=r3_defllars,
           deflnorm=r3_deflnorm, deflskip=r3_deflskip, deflwarm=r3_deflwarm, deflattn=r3_deflattn,
           deflffn=r3_deflffn, deflanneal=r3_deflanneal, deflup=r3_deflup,
           antiprev=r3_antiprev, agree=r3_agree, gclip=r3_gclip, oclip=r3_oclip, wdeflate=r3_wdeflate,
           rownormin=r3_rownormin, colnormin=r3_colnormin, dcenter=r3_dcenter, grokfast=r3_grokfast,
           efb=r3_efb, nsclean=r3_nsclean, dualmom2=r3_dualmom2, pospow=r3_pospow, gnratio=r3_gnratio,
           tailboost=r3_tailboost, specwd=r3_specwd, nest2=r3_nest2, odrop=r3_odrop, nscubic=r3_nscubic,
           ocenter=r3_ocenter)
RULES.update({k: _wrap(v) for k, v in _R3.items()})


# ================================================================ round 4 (2026-11-22)
# Exploit: momentum schedules (the part of deflwarm that grows with training length), with new
# twists: rule-set targets, per-expert step counts, adaptive per-matrix momentum, cool-downs.
def _warm(o, t, n=50, b0=0.3, tgt=None):
    tgt = o.cfg.mom if tgt is None else tgt
    return b0 + (tgt - b0) * min(1.0, t / n)


def _mw(o, G, b):
    return lambda st: _mom_b(st, G, b)


def _mwrule(bfun):
    def r(o, G, W, st, lr, p):
        return lr * moon_scale(G) * ns5(_mom_b(st, G, bfun(o, st, G, p)))
    return r


r4_mw25 = _mwrule(lambda o, st, G, p: _warm(o, o.t, 25))
r4_mw100 = _mwrule(lambda o, st, G, p: _warm(o, o.t, 100))
r4_mwfrac = _mwrule(lambda o, st, G, p: _warm(o, o.t, o.cfg.T / 3))
r4_mw0 = _mwrule(lambda o, st, G, p: _warm(o, o.t, 50, 0.0))
r4_mw5 = _mwrule(lambda o, st, G, p: _warm(o, o.t, 50, 0.5))
r4_mw95 = _mwrule(lambda o, st, G, p: _warm(o, o.t, o.cfg.T / 3, 0.3, 0.95))
r4_mwcos = _mwrule(lambda o, st, G, p: 0.3 + (o.cfg.mom - 0.3) * 0.5 * (1 - math.cos(math.pi * min(1.0, o.t / 50))))
r4_mwsqrt = _mwrule(lambda o, st, G, p: o.cfg.mom * math.sqrt(min(1.0, o.t / 50)))
r4_mwpoly = _mwrule(lambda o, st, G, p: min(o.cfg.mom, o.t / (o.t + 3)))
r4_mwpoly1 = _mwrule(lambda o, st, G, p: min(o.cfg.mom, o.t / (o.t + 1)))
r4_mwpoly10 = _mwrule(lambda o, st, G, p: min(o.cfg.mom, o.t / (o.t + 10)))


def _cool(o):
    b = _warm(o, o.t)
    c = 0.8 * o.cfg.T
    if o.t > c:
        b = o.cfg.mom - (o.cfg.mom - 0.5) * (o.t - c) / (o.cfg.T - c)
    return b


r4_mwcool = _mwrule(lambda o, st, G, p: _cool(o))
r4_kindwarm = _mwrule(lambda o, st, G, p: _warm(o, o.t, 25 if ".attn." in o.names[id(p)] else 100))


def _count_beta(o, st, G, c=3.0):
    """per-matrix momentum min(target, n/(n+c)), n = number of steps this matrix got a nonzero gradient"""
    live = (G.abs().amax(dim=(-2, -1), keepdim=True) > 0).float()
    st["cnt"] = st.get("cnt", torch.zeros_like(live)) + live
    b = (st["cnt"] / (st["cnt"] + c)).clamp(max=o.cfg.mom)
    return torch.where(live > 0, b, torch.ones_like(b))      # no token -> keep the buffer as is


def r4_expcount(o, G, W, st, lr, p):
    if o.kinds[id(p)] == "exp":
        b = _count_beta(o, st, G)
    else:
        b = _warm(o, o.t)
    return lr * moon_scale(G) * ns5(_mom_b(st, G, b))


def r4_polycount(o, G, W, st, lr, p):
    return lr * moon_scale(G) * ns5(_mom_b(st, G, _count_beta(o, st, G)))


def r4_deflannwarm(o, G, W, st, lr, p):
    M = _mom_b(st, G, _warm(o, o.t))
    O = ns5(M)
    T, _ = _top(M)
    return fin(G, O, O - K(o, 1.0) * (1 - o.t / o.cfg.T) * T, lr)


def r4_earlydefl(o, G, W, st, lr, p):
    M = _mom_b(st, G, _warm(o, o.t))
    O = ns5(M)
    if o.t > o.cfg.T / 3:
        return lr * moon_scale(G) * O
    T, _ = _top(M)
    return fin(G, O, O - 0.5 * T, lr)


def r4_cosbeta(o, G, W, st, lr, p):
    """per-matrix momentum moved by how well new gradients agree with the buffer"""
    if "m" in st:
        c = (G * st["m"]).sum(dim=(-2, -1), keepdim=True) / (G.norm(dim=(-2, -1), keepdim=True) * st["m"].norm(dim=(-2, -1), keepdim=True) + 1e-12)
        e = ema(st, "ce", c, 0.9)
        b = (o.cfg.mom + 0.4 * (e / (1 - 0.9 ** (o.t - 1)) - 0.25)).clamp(0.3, 0.95)
    else:
        b = 0.3
    return lr * moon_scale(G) * ns5(_mom_b(st, G, b))


def r4_mreset(o, G, W, st, lr, p):
    if o.t == max(2, o.cfg.T // 10) and "m" in st:
        st["m"].zero_()
    return lr * moon_scale(G) * ns5(mom(o, st, G))


def r4_mwnor(o, G, W, st, lr, p):
    M = _mom_b(st, G, _warm(o, o.t))
    O = ns5(M)
    v = ema(st, "v", (O * O).mean(-1, keepdim=True), o.cfg.b2)
    O2 = O / (v.sqrt() + 1e-8)
    m, n = G.shape[-2], G.shape[-1]
    return lr * 0.2 * math.sqrt(m * n) / (O2.norm(dim=(-2, -1), keepdim=True) + 1e-12) * O2


# Explore
def r4_sinkin(o, G, W, st, lr, p):
    M = mom(o, st, G)
    X = M
    for _ in range(2):
        X = X / (X.norm(dim=-1, keepdim=True) + 1e-12)
        X = X / (X.norm(dim=-2, keepdim=True) + 1e-12)
    return fin(G, ns5(M), ns5(X), lr)


def r4_uextrap(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    D = O + K(o, 0.3) * (O - st["up"]) if "up" in st else O
    st["up"] = O
    return fin(G, O, D, lr)


def r4_gdefl(o, G, W, st, lr, p):
    U, S, Vh = _svd(G, full_matrices=False)
    G2 = G - K(o, 0.5) * S[..., :1].unsqueeze(-1) * (U[..., :, :1] @ Vh[..., :1, :])
    return lr * moon_scale(G) * ns5(mom(o, st, G2))


def r4_topreset(o, G, W, st, lr, p):
    if "m" in st:
        U, S, Vh = _svd(G, full_matrices=False)
        u, v = U[..., :, :1], Vh[..., :1, :]
        c = u.transpose(-1, -2) @ st["m"] @ v.transpose(-1, -2)
        st["m"].sub_(torch.clamp(c, max=0) * (u @ v))
    return lr * moon_scale(G) * ns5(mom(o, st, G))


def r4_ns6(o, G, W, st, lr, p):
    return lr * moon_scale(G) * ns5(mom(o, st, G), steps=6)


def r4_qkdefl(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    if o.kinds[id(p)] != "qkv":
        return lr * moon_scale(G) * O
    T, _ = _top(M)
    D = O.clone()
    D[:2] = O[:2] - 0.5 * T[:2]
    return fin(G, O, D, lr)


def _kind_ramp(key):
    def r(o, G, W, st, lr, p):
        U = r2_muon(o, G, W, st, lr, p)
        return U * min(1.0, o.t / 50) if key(o, p) else U
    return r


r4_routerwarm = _kind_ramp(lambda o, p: "ffn.gate" in o.names[id(p)])
r4_expwarm = _kind_ramp(lambda o, p: o.kinds[id(p)] == "exp")


def r4_wflat(o, G, W, st, lr, p):
    Wf = W.float()
    Wt = ns5(Wf) * Wf.norm(dim=(-2, -1), keepdim=True) / math.sqrt(min(W.shape[-2], W.shape[-1]))
    return r2_muon(o, G, W, st, lr, p) + K(o, 0.5) * lr * (Wf - Wt)


def _layer(o, p):
    n = o.names[id(p)]
    if n.startswith("blocks."):
        return int(n.split(".")[1])
    return 0


def r4_kindmom(o, G, W, st, lr, p):
    b = o.cfg.mom + (0.1 if ".attn." in o.names[id(p)] else -0.1)
    return lr * moon_scale(G) * ns5(_mom_b(st, G, min(b, 0.95)))


def r4_depthmom(o, G, W, st, lr, p):
    L = max(1, o.model.cfg.n_layers - 1) if hasattr(o.model.cfg, "n_layers") else 2
    b = o.cfg.mom - 0.1 + 0.2 * _layer(o, p) / L
    return lr * moon_scale(G) * ns5(_mom_b(st, G, min(b, 0.95)))


def r4_topmom(o, G, W, st, lr, p):
    """top-4 left singular subspace of the momentum uses momentum 0.95, the rest the tuned value"""
    Mf = nesterov(st, G, o.cfg.mom, "m")
    Ms = nesterov(st, G, 0.95, "ms")
    U, S, Vh = _svd(Mf, full_matrices=False)
    P = U[..., :, :4] @ U[..., :, :4].transpose(-1, -2)
    M = P @ Ms + Mf - P @ Mf
    return fin(G, ns5(Mf), ns5(M), lr)


def r4_gatedefl(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    T, S = _top(M)
    gate = ((S[..., :1] / (S[..., 1:2] + 1e-12)) > 1.5).float().unsqueeze(-1)
    return fin(G, O, O - 0.5 * gate * T, lr)


def r4_rowdamp(o, G, W, st, lr, p):
    M = mom(o, st, G)
    rn = M.norm(dim=-1, keepdim=True)
    s = (3 * rn.median(dim=-2, keepdim=True).values / (rn + 1e-12)).clamp(max=1.0)
    return fin(G, ns5(M), ns5(M * s), lr)


def r4_expstack(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    if o.kinds[id(p)] != "exp":
        return lr * moon_scale(G) * O
    E, m, n = M.shape
    Os = ns5(M.reshape(1, E * m, n)).reshape(E, m, n)
    return fin(G, O, Os, lr)


def r4_gnpos(o, G, W, st, lr, p):
    n = G.norm(dim=(-2, -1), keepdim=True)
    e = ema(st, "gn", n, 0.9) / (1 - 0.9 ** o.t)
    return (n / (e + 1e-12)).pow(K(o, 0.5)).clamp(0.5, 2.0) * r2_muon(o, G, W, st, lr, p)


def r4_ns2stage(o, G, W, st, lr, p):
    return lr * moon_scale(G) * ns5(mom(o, st, ns5(G)))


def r4_randdefl(o, G, W, st, lr, p):
    """control for defl4: halve a random rank-4 block instead of the top-4 block"""
    O = ns5(mom(o, st, G))
    Q = torch.linalg.qr(torch.randn(O.shape[:-1] + (4,), device=O.device))[0]
    R = torch.linalg.qr(torch.randn(O.shape[:-2] + (O.shape[-1], 4), device=O.device))[0]
    return fin(G, O, O - 0.5 * Q @ (Q.transpose(-1, -2) @ O @ R) @ R.transpose(-1, -2), lr)


_R4 = dict(mw25=r4_mw25, mw100=r4_mw100, mwfrac=r4_mwfrac, mw0=r4_mw0, mw5=r4_mw5, mw95=r4_mw95,
           mwcos=r4_mwcos, mwsqrt=r4_mwsqrt, mwpoly=r4_mwpoly, mwpoly1=r4_mwpoly1, mwpoly10=r4_mwpoly10,
           mwcool=r4_mwcool, kindwarm=r4_kindwarm, expcount=r4_expcount, polycount=r4_polycount,
           deflannwarm=r4_deflannwarm, earlydefl=r4_earlydefl, cosbeta=r4_cosbeta, mreset=r4_mreset,
           mwnor=r4_mwnor,
           sinkin=r4_sinkin, uextrap=r4_uextrap, gdefl=r4_gdefl, topreset=r4_topreset, ns6=r4_ns6,
           qkdefl=r4_qkdefl, routerwarm=r4_routerwarm, expwarm=r4_expwarm, wflat=r4_wflat,
           kindmom=r4_kindmom, depthmom=r4_depthmom, topmom=r4_topmom, gatedefl=r4_gatedefl,
           rowdamp=r4_rowdamp, expstack=r4_expstack, gnpos=r4_gnpos, ns2stage=r4_ns2stage,
           randdefl=r4_randdefl)
RULES.update({k: _wrap(v) for k, v in _R4.items()})


def r4_vodefl(o, G, W, st, lr, p):
    """deflate only the attention value and output matrices (complement of qkdefl)"""
    M = mom(o, st, G)
    O = ns5(M)
    n = o.names[id(p)]
    if ".attn." not in n:
        return lr * moon_scale(G) * O
    T, _ = _top(M)
    D = O - 0.5 * T
    if o.kinds[id(p)] == "qkv":
        D[:2] = O[:2]
    return fin(G, O, D, lr)


def r4_rankwarm(o, G, W, st, lr, p):
    """rank curriculum: keep only the top fraction min(1, t/50) of singular directions early on"""
    M = mom(o, st, G)
    O = ns5(M)
    f = min(1.0, o.t / 50)
    if f >= 1.0:
        return lr * moon_scale(G) * O
    U, S, Vh = _svd(M, full_matrices=False)
    r = max(1, round(f * S.shape[-1]))
    return fin(G, O, U[..., :, :r] @ Vh[..., :r, :], lr)


RULES.update(vodefl=_wrap(r4_vodefl), rankwarm=_wrap(r4_rankwarm))


# ================================================================ round 5 (2026-11-22)
def _sel(o, p, parts):
    """1/0 per matrix in the batch: which of q, k, v, o, ffn (router + experts + shared) it is in parts"""
    n, k = o.names[id(p)], o.kinds[id(p)]
    if k == "qkv":
        return torch.tensor([float(x in parts) for x in "qkv"]).view(3, 1, 1)
    if n.endswith("attn.o.weight"):
        return torch.tensor(float("o" in parts)).view(1, 1, 1)
    return torch.tensor(float("ffn" in parts)).view(1, 1, 1)


def _norwarm_O(o, st, O):
    v = ema(st, "v", (O * O).mean(-1, keepdim=True), o.cfg.b2)
    return O / (v.sqrt() + 1e-8)


def _make5(warm=None, parts="", k=0.5, r=1, nor=False, kanneal=False, tgt=None):
    """momentum warmup (attention steps, ffn steps) or None; deflate on parts; NorMuon row scaling"""
    def rule(o, G, W, st, lr, p):
        o._p = p
        if warm is None:
            b = o.cfg.mom
        else:
            na, nf = warm
            attn = ".attn." in o.names[id(p)]
            t0 = (tgt[0] if attn else tgt[1]) if tgt else o.cfg.mom
            n = na if attn else nf
            b = t0 if n == 0 else _warm(o, o.t, n, 0.3, t0)
        M = _mom_b(st, G, b)
        O = ns5(M)
        D = O
        if parts:
            s = _sel(o, p, parts).to(G.device)
            if s.sum() > 0:
                kk = k * (1 - o.t / o.cfg.T) if kanneal else k
                T, _ = _top(M, r)
                D = O - kk * s * T
        if nor:
            D = _norwarm_O(o, st, D)
            m, n2 = G.shape[-2], G.shape[-1]
            return lr * 0.2 * math.sqrt(m * n2) / (D.norm(dim=(-2, -1), keepdim=True) + 1e-12) * D
        if D is O:
            return lr * moon_scale(G) * O
        return fin(G, O, D, lr)
    return rule


_R5E = dict(
    vowarm=_make5((100, 100), "vo"), vowarmnor=_make5((100, 100), "vo", nor=True),
    kindnor=_make5((25, 100), nor=True), kindvo=_make5((25, 100), "vo"),
    kindwarm10=_make5((10, 100)), kindwarm200=_make5((25, 200)),
    ffnwarm=_make5((0, 100)), attnwarm=_make5((50, 0)),
    vo25w=_make5((100, 100), "vo", k=0.25), vodefl4w=_make5((100, 100), "vo", r=4),
    odeflw=_make5((100, 100), "o"), vdeflw=_make5((100, 100), "v"),
    kindtgt=_make5((25, 100), tgt=(0.9, 0.8)), mwnor100=_make5((100, 100), nor=True),
    allin=_make5((25, 100), "vo", nor=True), voanneal=_make5((100, 100), "vo", k=1.0, kanneal=True),
    explow=_make5((25, 100), tgt=(0.85, 0.7)), norfrac=None, kindtgt2=_make5((25, 100), tgt=(0.8, 0.9)),
    vo75w=_make5((100, 100), "vo", k=0.75))


def r5_norfrac(o, G, W, st, lr, p):
    M = _mom_b(st, G, _warm(o, o.t, o.cfg.T / 3))
    O = _norwarm_O(o, st, ns5(M))
    m, n = G.shape[-2], G.shape[-1]
    return lr * 0.2 * math.sqrt(m * n) / (O.norm(dim=(-2, -1), keepdim=True) + 1e-12) * O


_R5E["norfrac"] = r5_norfrac


# Explore (150 steps)
def _mult(parts, f):
    def r(o, G, W, st, lr, p):
        s = _sel(o, p, parts).to(G.device)
        return (1 + (f - 1) * s) * r2_muon(o, G, W, st, lr, p)
    return r


r5_volr = _mult("vo", 0.8)
r5_qklr = _mult("qk", 1.25)
r5_attnlr2 = _mult("qkvo", 0.8)
r5_explr = _mult("ffn", 1.25)
r5_odefl = _make5(None, "o")
r5_vdefl = _make5(None, "v")
r5_vodefl2 = _make5(None, "vo", r=2)


def r5_routerlr2(o, G, W, st, lr, p):
    return _name_mult(o, p, "ffn.gate", 0.5) * r2_muon(o, G, W, st, lr, p)


def r5_sharedlr(o, G, W, st, lr, p):
    return _name_mult(o, p, "ffn.shared", 0.5) * r2_muon(o, G, W, st, lr, p)


def r5_keller2(o, G, W, st, lr, p):
    """Keller's original scale max(1, out/in)^0.5, matched to Moonlight's update RMS on d x d matrices"""
    m, n = G.shape[-2], G.shape[-1]
    d = o.model.cfg.d_model if o.model is not None else 64
    return lr * 0.2 * math.sqrt(d) * max(1, m / n) ** 0.5 * ns5(mom(o, st, G))


def r5_softpolar(o, G, W, st, lr, p):
    """directions below the median singular value keep their relative size; above the median -> 1"""
    M = mom(o, st, G)
    U, S, Vh = _svd(M, full_matrices=False)
    med = S.median(dim=-1, keepdim=True).values
    w = (S / (med + 1e-12)).clamp(max=1.0)
    return fin(G, ns5(M), (U * w.unsqueeze(-2)) @ Vh, lr)


def r5_nsspec(o, G, W, st, lr, p):
    """Newton-Schulz started from the spectral norm (power-iteration estimate) instead of Frobenius"""
    M = mom(o, st, G)
    O = ns5(M)
    if "pv" not in st:
        st["pv"] = torch.randn(M.shape[:-2] + (M.shape[-1], 1), device=M.device)
    v = st["pv"]
    for _ in range(2):
        v = M.transpose(-1, -2) @ (M @ v)
        v = v / (v.norm(dim=-2, keepdim=True) + 1e-12)
    st["pv"] = v
    s1 = (M @ v).norm(dim=-2, keepdim=True)
    X = M / (1.05 * s1 + 1e-12)
    a, b, c = NS_ABC
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-1, -2)
    for _ in range(5):
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.transpose(-1, -2)
    return fin(G, O, X, lr)


def r5_vojoint(o, G, W, st, lr, p):
    """v and o of a layer as one pair: stack M_v and M_o^T (both d x d) for a joint Newton-Schulz"""
    M = mom(o, st, G)
    O = ns5(M)
    n = o.names[id(p)]
    if n.endswith("attn.o.weight") or n.endswith("attn.qkv.weight"):
        key = n.split(".attn.")[0]
        buf = o.__dict__.setdefault("_vo", {})
        if n.endswith("qkv.weight"):
            buf[key] = M[2].clone()          # v first in the step order
            return lr * moon_scale(G) * O
        if key in buf:
            J = torch.cat([buf[key], M[0].transpose(0, 1)], dim=0)
            Oj = ns5(J)[buf[key].shape[0]:].transpose(0, 1).unsqueeze(0)
            return fin(G, O, Oj, lr)
    return lr * moon_scale(G) * O


def r5_headvodefl(o, G, W, st, lr, p):
    """deflate the top direction of each head's block of v (rows) and o (columns)"""
    M = mom(o, st, G)
    O = ns5(M)
    n = o.names[id(p)]
    h = o.model.cfg.n_heads if o.model is not None else 4
    if n.endswith("attn.qkv.weight"):
        Mv = M[2]
        d = Mv.shape[0]
        B = Mv.view(h, d // h, -1)
        T, _ = _top(B)
        D = O.clone()
        D[2] = O[2] - 0.5 * T.reshape(d, -1)
        return fin(G, O, D, lr)
    if n.endswith("attn.o.weight"):
        Mo = M[0]
        d = Mo.shape[1]
        B = Mo.view(-1, h, d // h).permute(1, 0, 2)
        T, _ = _top(B)
        D = O - 0.5 * T.permute(1, 0, 2).reshape(1, -1, d)
        return fin(G, O, D, lr)
    return lr * moon_scale(G) * O


def r5_adaptk(o, G, W, st, lr, p):
    M = mom(o, st, G)
    O = ns5(M)
    s = _sel(o, p, "vo").to(G.device)
    if s.sum() == 0:
        return lr * moon_scale(G) * O
    T, S = _top(M)
    k = (1 - (S.median(dim=-1, keepdim=True).values / (S[..., :1] + 1e-12)).sqrt()).clamp(0, 1).unsqueeze(-1)
    return fin(G, O, O - k * s * T, lr)


def r5_elemnor(o, G, W, st, lr, p):
    """elementwise second moment of the orthogonal update (AdaMuon-style), Muon's RMS kept"""
    O = ns5(mom(o, st, G))
    v = ema(st, "ev", O * O, o.cfg.b2) / (1 - o.cfg.b2 ** o.t)
    return fin(G, O, O / (v.sqrt() + 1e-8), lr)


def r5_specnormcap(o, G, W, st, lr, p):
    """step x min(1, initial top singular value of W / current one): stop the weight's spectral norm growing"""
    if "pv" not in st:
        st["pv"] = torch.randn(W.shape[:-2] + (W.shape[-1], 1), device=W.device)
    v = st["pv"]
    Wf = W.float()
    v = Wf.transpose(-1, -2) @ (Wf @ v)
    v = v / (v.norm(dim=-2, keepdim=True) + 1e-12)
    st["pv"] = v
    s1 = (Wf @ v).norm(dim=-2, keepdim=True)
    if "s0" not in st:
        st["s0"] = s1.clone()
    m = (K(o, 2.0) * st["s0"] / (s1 + 1e-12)).clamp(max=1.0)
    return m * r2_muon(o, G, W, st, lr, p)


def r5_globclip(o, G, W, st, lr, p):
    """clip each matrix's gradient norm at 2 x its running average before momentum"""
    n = G.norm(dim=(-2, -1), keepdim=True)
    e = st.get("gn", n.clone())
    st["gn"] = 0.9 * e + 0.1 * torch.minimum(n, 2 * e)
    G2 = G * (2 * e / (n + 1e-12)).clamp(max=1.0)
    return r2_muon(o, G2, W, st, lr, p)


def r5_erankstep(o, G, W, st, lr, p):
    """step x (effective rank of the momentum / its running average)^0.5"""
    M = mom(o, st, G)
    Fn = M.norm(dim=(-2, -1), keepdim=True)
    S = torch.linalg.svdvals(M.detach().cpu()).to(M.device)
    er = (S.sum(-1, keepdim=True).unsqueeze(-1) / (Fn + 1e-12)) ** 2
    e = ema(st, "er", er, 0.9) / (1 - 0.9 ** o.t)
    return (er / (e + 1e-12)).pow(0.5).clamp(0.5, 2.0) * lr * moon_scale(G) * ns5(M)


def r5_vomom(o, G, W, st, lr, p):
    s = _sel(o, p, "vo").to(G.device)
    b = o.cfg.mom - 0.2 * s
    return lr * moon_scale(G) * ns5(_mom_b(st, G, b))


_R5X = dict(volr=r5_volr, qklr=r5_qklr, attnlr2=r5_attnlr2, explr=r5_explr, routerlr2=r5_routerlr2,
            sharedlr=r5_sharedlr, softpolar=r5_softpolar, nsspec=r5_nsspec, vojoint=r5_vojoint,
            headvodefl=r5_headvodefl, adaptk=r5_adaptk, elemnor=r5_elemnor, specnormcap=r5_specnormcap,
            globclip=r5_globclip, erankstep=r5_erankstep, vomom=r5_vomom, odefl=r5_odefl, vdefl=r5_vdefl,
            vodefl2=r5_vodefl2, keller2=r5_keller2)
RULES.update({k: _wrap(v) for k, v in {**_R5E, **_R5X}.items()})


# ================================================================ round 6 (2026-11-22)
# Exploit (600 steps): momentum warmup + value/output deflation, pushed with NorMuon, query/key step,
# adaptive cut strength, per-head cuts, cut rank, and cut schedules.
def _headT(o, p, M):
    """per-head top direction of v (row blocks) or o (column blocks), shaped like M; zeros elsewhere"""
    n = o.names[id(p)]
    h = o.model.cfg.n_heads if o.model is not None else 4
    T = torch.zeros_like(M)
    if n.endswith("attn.qkv.weight"):
        d = M.shape[-2]
        Tb, _ = _top(M[2].view(h, d // h, -1))
        T[2] = Tb.reshape(d, -1)
    elif n.endswith("attn.o.weight"):
        d = M.shape[-1]
        Tb, _ = _top(M[0].view(-1, h, d // h).permute(1, 0, 2))
        T[0] = Tb.permute(1, 0, 2).reshape(-1, d)
    return T


def _make6(warm=(100, 100), parts="vo", k=0.5, r=4, nor=False, qk=1.0, adapt=False, head=False,
           kfun=None, tgt=None):
    def rule(o, G, W, st, lr, p):
        attn = ".attn." in o.names[id(p)]
        if warm is None:
            b = o.cfg.mom
        else:
            t0 = (tgt[0] if attn else tgt[1]) if tgt else o.cfg.mom
            n = warm[0] if attn else warm[1]
            b = t0 if n == 0 else _warm(o, o.t, n, 0.3, t0)
        M = _mom_b(st, G, b)
        O = ns5(M)
        D = O
        s = _sel(o, p, parts).to(G.device) if parts else torch.zeros(1, 1, 1, device=G.device)
        if s.sum() > 0:
            kk = k * (kfun(o.t / o.cfg.T) if kfun else 1.0)
            if head:
                T = _headT(o, p, M)
            else:
                T, S = _top(M, r)
                if adapt:
                    kk = kk * 2 * (1 - (S.median(dim=-1, keepdim=True).values
                                        / (S[..., :1] + 1e-12)).sqrt()).clamp(0, 1).unsqueeze(-1)
            D = O - kk * s * T
        m, n2 = G.shape[-2], G.shape[-1]
        if nor:
            D = _norwarm_O(o, st, D)
            U = lr * 0.2 * math.sqrt(m * n2) / (D.norm(dim=(-2, -1), keepdim=True) + 1e-12) * D
        elif D is O:
            U = lr * moon_scale(G) * O
        else:
            U = fin(G, O, D, lr)
        if qk != 1.0:
            U = (1 + (qk - 1) * _sel(o, p, "qk").to(G.device)) * U
        return U
    return rule


_R6E = dict(
    v4nor=_make6(nor=True), v4k65=_make6(k=0.65), v8w=_make6(r=8), v2w=_make6(r=2),
    v4qk=_make6(qk=1.25), v4qknor=_make6(qk=1.25, nor=True), v4adapt=_make6(adapt=True),
    adaptw=_make6(r=1, adapt=True), headw=_make6(head=True), v4w150=_make6(warm=(150, 150)),
    v4w50=_make6(warm=(50, 50)), v4kind=_make6(warm=(25, 100)), v4nor75=_make6(k=0.75, nor=True),
    v4mom9=_make6(tgt=(0.9, 0.9)), v4voffn=_make6(parts="voffn", k=0.25),
    v4all=_make6(k=0.75, nor=True, qk=1.25), v4kup=_make6(kfun=lambda f: min(1.0, 2 * f) * 1.5),
    v4sqrt=_make6(k=0.75, kfun=lambda f: (1 - f) ** 0.5), v4qk15=_make6(qk=1.5),
    headqk=_make6(head=True, qk=1.25))


# Explore (150 steps)
def r6_mixraw(o, G, W, st, lr, p):
    """keep some of the raw momentum's shape: O + 0.3 x M scaled to O's RMS"""
    M = mom(o, st, G)
    O = ns5(M)
    return fin(G, O, O + 0.3 * M * rms(O) / rms(M), lr)


def r6_ns3(o, G, W, st, lr, p):
    M = mom(o, st, G)
    return fin(G, ns5(M), ns5(M, steps=3), lr)


def r6_colnor(o, G, W, st, lr, p):
    """NorMuon-style second moment per input column instead of per output row"""
    O = ns5(mom(o, st, G))
    v = ema(st, "cv", (O * O).mean(-2, keepdim=True), o.cfg.b2)
    return fin(G, O, O / (v.sqrt() + 1e-8), lr)


def r6_rcnor(o, G, W, st, lr, p):
    O = ns5(mom(o, st, G))
    v = ema(st, "rv", (O * O).mean(-1, keepdim=True), o.cfg.b2)
    c = ema(st, "cv", (O * O).mean(-2, keepdim=True), o.cfg.b2)
    return fin(G, O, O / (v.sqrt() + 1e-8) / (c.sqrt() + 1e-8), lr)


def r6_dirmom(o, G, W, st, lr, p):
    """momentum on the orthogonalized direction: average O_t over steps (beta 0.5) on top of Muon"""
    O = ns5(mom(o, st, G))
    A = ema(st, "oa", O, 0.5)
    return fin(G, O, A, lr)


def r6_pullback(o, G, W, st, lr, p):
    """pull each matrix 5% toward a slow average of its own weights (lookahead inside the update)"""
    U = r2_muon(o, G, W, st, lr, p)
    Wf = W.float()
    if "slow" not in st:
        st["slow"] = Wf.clone()
    st["slow"].lerp_(Wf, 0.05)
    return U + 0.05 * (Wf - st["slow"]) * (o.t % 10 == 0)


def r6_gnlr(o, G, W, st, lr, p):
    """step x (gradient norm / its running average)^0.25"""
    n = G.norm(dim=(-2, -1), keepdim=True)
    e = ema(st, "gn", n, 0.9) / (1 - 0.9 ** o.t)
    return (n / (e + 1e-12)).pow(0.25).clamp(0.7, 1.4) * r2_muon(o, G, W, st, lr, p)


def r6_layerup(o, G, W, st, lr, p):
    """deeper layers take bigger steps: 0.85x at layer 0 up to 1.15x at the last layer"""
    i = int(o.names[id(p)].split(".")[1])
    L = o.model.cfg.n_layers if o.model is not None else 3
    return (0.85 + 0.3 * i / max(1, L - 1)) * r2_muon(o, G, W, st, lr, p)


def r6_noisek(o, G, W, st, lr, p):
    """value/output cut strength from gradient noise: k = 0.5 x ||G - m|| / (||G|| + ||m||)"""
    s = _sel(o, p, "vo").to(G.device)
    noise = None
    if "m" in st:
        noise = (G - st["m"]).norm(dim=(-2, -1), keepdim=True) / (
            G.norm(dim=(-2, -1), keepdim=True) + st["m"].norm(dim=(-2, -1), keepdim=True) + 1e-12)
    M = mom(o, st, G)
    O = ns5(M)
    if s.sum() == 0 or noise is None:
        return lr * moon_scale(G) * O
    T, _ = _top(M)
    return fin(G, O, O - 1.0 * noise * s * T, lr)


def r6_onoise(o, G, W, st, lr, p):
    """add Gaussian noise at 10% of the update RMS"""
    M = mom(o, st, G)
    O = ns5(M)
    return fin(G, O, O + 0.1 * rms(O) * torch.randn_like(O), lr)


def r6_vowd(o, G, W, st, lr, p):
    """weight decay 0.1 (times lr) on value and output only"""
    s = _sel(o, p, "vo").to(G.device)
    return r2_muon(o, G, W, st, lr, p) + lr * 0.1 * s * W.float()


def r6_wdall(o, G, W, st, lr, p):
    return r2_muon(o, G, W, st, lr, p) + lr * 0.1 * W.float()


r6_qkvo = _make6(warm=None, parts="vo", r=2, qk=1.25)
r6_adapthead = _make6(warm=None, head=True, k=0.75)
r6_adapt4 = _make6(warm=None, r=4, adapt=True)
r6_vonor = _make6(warm=None, r=2, nor=True)
r6_qk15 = _mult("qk", 1.5)
r6_olr12 = _mult("o", 1.2)


def r6_flipdefl(o, G, W, st, lr, p):
    """value/output: cut the top direction only when it agrees with last step's top direction"""
    M = mom(o, st, G)
    O = ns5(M)
    s = _sel(o, p, "vo").to(G.device)
    if s.sum() == 0:
        return lr * moon_scale(G) * O
    T, _ = _top(M)
    prev = st.get("pt", T)
    st["pt"] = T.clone()
    agree = ((T * prev).sum(dim=(-2, -1), keepdim=True).abs() > 0.5).float()
    return fin(G, O, O - 0.5 * agree * s * T, lr)


def r6_boostlow(o, G, W, st, lr, p):
    """value/output: halve top 2 and grow bottom quarter of singular directions x1.3"""
    M = mom(o, st, G)
    O = ns5(M)
    s = _sel(o, p, "vo").to(G.device)
    if s.sum() == 0:
        return lr * moon_scale(G) * O
    U, S, Vh = _svd(M, full_matrices=False)
    r = S.shape[-1]
    w = torch.ones_like(S)
    w[..., :2] = 0.5
    w[..., r - r // 4:] = 1.3
    D = (U * w.unsqueeze(-2)) @ Vh
    return fin(G, O, O + s * (D - O), lr)


_R6X = dict(mixraw=r6_mixraw, ns3=r6_ns3, colnor=r6_colnor, rcnor=r6_rcnor, dirmom=r6_dirmom,
            pullback=r6_pullback, gnlr=r6_gnlr, layerup=r6_layerup, noisek=r6_noisek, onoise=r6_onoise,
            vowd=r6_vowd, wdall=r6_wdall, qkvo=r6_qkvo, adapthead=r6_adapthead, adapt4=r6_adapt4,
            vonor=r6_vonor, qk15=r6_qk15, olr12=r6_olr12, flipdefl=r6_flipdefl, boostlow=r6_boostlow)
RULES.update({k: _wrap(v) for k, v in {**_R6E, **_R6X}.items()})


# ================================================================ round 7 (2026-11-22)
# Per-expert update rules for MoE experts (pivot 13:35). Non-expert matrices take plain Muon.
# share s_e = tokens routed to expert e this step / mean over the layer's experts, shape (E, 1, 1).
def _share(o, p, G):
    m = getattr(o, "mod_of", {}).get(id(p))
    if m is None or getattr(m, "last_load", None) is None:
        return None
    ld = m.last_load.to(G.device)
    return (ld / (ld.mean() + 1e-9)).view(-1, 1, 1)


def _nrm(f):
    """factor normalized to mean 1 over the layer's experts (total layer step unchanged)"""
    return f / (f.mean() + 1e-12)


def _wrapS(f):
    r = _wrap(f)
    r.setup = _setup_load
    return r


def _expf(ffun):
    """Muon, expert update x ffun(o, st, G, s) (factor (E,1,1))"""
    def rule(o, G, W, st, lr, p):
        U = r2_muon(o, G, W, st, lr, p)
        if o.kinds[id(p)] != "exp":
            return U
        s = _share(o, p, G)
        if s is None:
            return U
        return ffun(o, st, G, s, p) * U
    return rule


def _ldema(st, s, b=0.9):
    e = st.get("lde", torch.ones_like(s))
    st["lde"] = b * e + (1 - b) * s
    return st["lde"]


def _ldcum(st, s):
    st["ldc"] = st.get("ldc", torch.zeros_like(s)) + s
    return st["ldc"] / (st["ldc"].mean() + 1e-12)


def _w2(o, p):
    return o.names[id(p)].endswith("w2")


# A. load-share step scaling (each normalized so the layer's mean factor is 1)
r7_ldp25 = _expf(lambda o, st, G, s, p: _nrm(s.clamp(min=0.05) ** 0.25))
r7_ldp50 = _expf(lambda o, st, G, s, p: _nrm(s.clamp(min=0.05) ** 0.5))
r7_ldn25 = _expf(lambda o, st, G, s, p: _nrm(s.clamp(min=0.05) ** -0.25))
r7_ldema = _expf(lambda o, st, G, s, p: _nrm(_ldema(st, s) ** 0.5))
r7_ldcum = _expf(lambda o, st, G, s, p: _nrm(_ldcum(st, s).clamp(min=0.05) ** 0.5))
r7_ldsat = _expf(lambda o, st, G, s, p: _nrm(s / (s + 0.5)))
r7_ldclip = _expf(lambda o, st, G, s, p: _nrm(s.clamp(0.5, 1.5)))
r7_ldw2 = _expf(lambda o, st, G, s, p: _nrm(s.clamp(min=0.05) ** 0.5) if _w2(o, p) else torch.ones_like(s))
r7_ldw13 = _expf(lambda o, st, G, s, p: torch.ones_like(s) if _w2(o, p) else _nrm(s.clamp(min=0.05) ** 0.5))
r7_ldraw = _expf(lambda o, st, G, s, p: s.clamp(min=0.05) ** 0.25)          # not normalized
r7_ldrank = _expf(lambda o, st, G, s, p: 1 + 0.2 * torch.sign(s - s.median()))


def r7_ldage(o, st, G, s, p):
    """catch-up: experts with fewer total tokens so far take bigger steps"""
    return _nrm(_ldcum(st, s).clamp(min=0.05) ** -0.25)


r7_ldage = _expf(r7_ldage)


# B. second moments pooled across the experts of one matrix
def _poolnor(mode, mix=1.0):
    def rule(o, G, W, st, lr, p):
        M = mom(o, st, G)
        O = ns5(M)
        if o.kinds[id(p)] != "exp":
            return lr * moon_scale(G) * O
        if mode == "row":
            x = (O * O).mean(-1, keepdim=True)
        elif mode == "col":
            x = (O * O).mean(-2, keepdim=True)
        else:
            x = O * O
        own = x
        pool = x.mean(0, keepdim=True).expand_as(x)
        v = ema(st, "pv2", mix * pool + (1 - mix) * own, o.cfg.b2) / (1 - o.cfg.b2 ** o.t)
        return fin(G, O, O / (v.sqrt() + 1e-8), lr)
    return rule


r7_shvrow = _poolnor("row")
r7_shvmix = _poolnor("row", 0.5)
r7_shvown = _poolnor("row", 0.0)        # control: own row statistics on experts only
r7_shvcol = _poolnor("col")
r7_shvel = _poolnor("el")
r7_shvelmix = _poolnor("el", 0.5)


def _gnf(pw, src="g"):
    def f(o, st, G, s, p):
        X = G if src == "g" else st["m"]
        n = X.norm(dim=(-2, -1), keepdim=True)
        return _nrm((n / (n.mean() + 1e-12)).clamp(min=0.05) ** pw)
    return f


r7_shgn = _expf(_gnf(0.5))
r7_shgninv = _expf(_gnf(-0.5))
r7_shmn = _expf(_gnf(0.5, "m"))


# C. joint orthogonalization / shared factors across experts
def _joint(fn):
    def rule(o, G, W, st, lr, p):
        M = mom(o, st, G)
        O = ns5(M)
        if o.kinds[id(p)] != "exp":
            return lr * moon_scale(G) * O
        return fin(G, O, fn(M, O), lr)
    return rule


def _jo_in(M, O):
    E, m, n = M.shape
    return ns5(M.permute(1, 0, 2).reshape(1, m, E * n)).reshape(m, E, n).permute(1, 0, 2)


def _jo_out(M, O):
    E, m, n = M.shape
    return ns5(M.reshape(1, E * m, n)).reshape(E, m, n)


r7_join = _joint(_jo_in)
r7_joout = _joint(_jo_out)
r7_joblend = _joint(lambda M, O: O / rms(O) + _jo_in(M, O) / rms(_jo_in(M, O)))
r7_jocommon = _joint(lambda M, O: ns5(M - M.mean(0, keepdim=True)) + 0.5 * ns5(M.mean(0, keepdim=True)))
r7_jocenter = _joint(lambda M, O: ns5(M - 0.5 * M.mean(0, keepdim=True)))


def _gramR(M, O):
    """shared right whitening: M_e C^-1/4, C = mean_e M_e^T M_e (a Shampoo factor shared by the experts)"""
    C = (M.transpose(-1, -2) @ M).mean(0)
    C = C / (C.diagonal().mean() + 1e-12) + 1e-3 * torch.eye(C.shape[0], device=C.device)
    L, Q = _eigh(C)
    P = (Q * L.clamp(min=1e-6).pow(-0.25)) @ Q.T
    return ns5(M @ P)


def _gramL(M, O):
    C = (M @ M.transpose(-1, -2)).mean(0)
    C = C / (C.diagonal().mean() + 1e-12) + 1e-3 * torch.eye(C.shape[0], device=C.device)
    L, Q = _eigh(C)
    P = (Q * L.clamp(min=1e-6).pow(-0.25)) @ Q.T
    return ns5(P @ M)


r7_jogramr = _joint(_gramR)
r7_jograml = _joint(_gramL)


# D. token-counted momentum
def _tokmom(kind):
    def rule(o, G, W, st, lr, p):
        if o.kinds[id(p)] != "exp":
            return r2_muon(o, G, W, st, lr, p)
        s = _share(o, p, G)
        if s is None:
            return r2_muon(o, G, W, st, lr, p)
        b = o.cfg.mom
        if kind == "lazy":
            live = (s > 0).float()
            if "m" not in st:
                st["m"] = torch.zeros_like(G)
            st["m"].copy_(live * st["m"].lerp(G, 1 - b) + (1 - live) * st["m"])
            M = G.lerp(st["m"], b)
            return live * lr * moon_scale(G) * ns5(M)
        if kind == "pow":
            be = b ** s.clamp(min=0.05)                          # few tokens -> longer memory
        elif kind == "inv":
            be = b ** (1 / s.clamp(min=0.25))                    # control: the reverse
        else:                                                    # token-weighted EMA
            be = 1 - ((1 - b) * s).clamp(max=1.0)
        return lr * moon_scale(G) * ns5(_mom_b(st, G, be))
    return rule


r7_lazy = _tokmom("lazy")
r7_mompow = _tokmom("pow")
r7_mominv = _tokmom("inv")
r7_momtok = _tokmom("tok")


# E. noise-relative steps and expert diversity / sharing
def _relnoise(o, st, G, s, p):
    m = st["m"]
    # st["m"] already includes this step; use the gap between G and it
    nz = (G - m).norm(dim=(-2, -1), keepdim=True) / (G.norm(dim=(-2, -1), keepdim=True) + m.norm(dim=(-2, -1), keepdim=True) + 1e-12)
    return _nrm((1 - nz).clamp(min=0.05))


r7_relnoise = _expf(_relnoise)


def _snr(o, st, G, s, p):
    g2 = ema(st, "g2", G.pow(2).sum(dim=(-2, -1), keepdim=True), 0.9) / (1 - 0.9 ** o.t)
    r = st["m"].norm(dim=(-2, -1), keepdim=True) / (g2.sqrt() + 1e-12)
    return _nrm(r.clamp(min=0.05) ** 0.5)


r7_snr = _expf(_snr)


def _div(c, share=False, onmom=False):
    def rule(o, G, W, st, lr, p):
        M = mom(o, st, G)
        if o.kinds[id(p)] != "exp":
            return lr * moon_scale(G) * ns5(M)
        s = _share(o, p, G)
        w = c if not share or s is None else c * (1 - s.clamp(max=1.0))
        if onmom:
            M2 = M + w * M.mean(0, keepdim=True)
            return fin(G, ns5(M), ns5(M2), lr)
        O = ns5(M)
        return fin(G, O, O + w * O.mean(0, keepdim=True), lr)
    return rule


r7_expdiv = _div(-0.3)
r7_expdiv5 = _div(-0.5)
r7_expshare = _div(0.3, share=True)
r7_momshare = _div(0.3, share=True, onmom=True)
r7_mommean = _div(0.3, onmom=True)


def r7_expdefl(o, G, W, st, lr, p):
    M = mom(o, st, G)
    if o.kinds[id(p)] != "exp":
        return lr * moon_scale(G) * ns5(M)
    return _defl(o, G, st, M, 0.5)


def r7_ldkeep(o, G, W, st, lr, p):
    """low-load experts keep part of their previous orthogonal update: D = a O + (1 - a) O_prev, a = min(s, 1)"""
    M = mom(o, st, G)
    O = ns5(M)
    if o.kinds[id(p)] != "exp":
        return lr * moon_scale(G) * O
    s = _share(o, p, G)
    prev = st.get("op", O)
    st["op"] = O.clone()
    a = torch.ones_like(O[..., :1, :1]) if s is None else s.clamp(max=1.0)
    return fin(G, O, a * O + (1 - a) * prev, lr)


def r7_ldshv(o, G, W, st, lr, p):
    """pooled row second moment + EMA-load step scaling"""
    U = r7_shvrow(o, G, W, st, lr, p)
    if o.kinds[id(p)] != "exp":
        return U
    s = _share(o, p, G)
    return U if s is None else _nrm(_ldema(st, s) ** 0.5) * U


def r7_lazyld(o, G, W, st, lr, p):
    U = r7_lazy(o, G, W, st, lr, p)
    if o.kinds[id(p)] != "exp":
        return U
    s = _share(o, p, G)
    return U if s is None else _nrm(_ldema(st, s) ** 0.25) * U


_R7 = dict(ldp25=r7_ldp25, ldp50=r7_ldp50, ldn25=r7_ldn25, ldema=r7_ldema, ldcum=r7_ldcum, ldsat=r7_ldsat,
           ldclip=r7_ldclip, ldw2=r7_ldw2, ldw13=r7_ldw13, ldraw=r7_ldraw, ldrank=r7_ldrank, ldage=r7_ldage,
           shvrow=r7_shvrow, shvmix=r7_shvmix, shvown=r7_shvown, shvcol=r7_shvcol, shvel=r7_shvel,
           shvelmix=r7_shvelmix, shgn=r7_shgn, shgninv=r7_shgninv, shmn=r7_shmn,
           join=r7_join, joout=r7_joout, joblend=r7_joblend, jocommon=r7_jocommon, jocenter=r7_jocenter,
           jogramr=r7_jogramr, jograml=r7_jograml,
           lazy=r7_lazy, mompow=r7_mompow, mominv=r7_mominv, momtok=r7_momtok,
           relnoise=r7_relnoise, snr=r7_snr, expdiv=r7_expdiv, expdiv5=r7_expdiv5, expshare=r7_expshare,
           momshare=r7_momshare, mommean=r7_mommean, expdefl=r7_expdefl, ldkeep=r7_ldkeep, ldshv=r7_ldshv,
           lazyld=r7_lazyld)
RULES.update({k: _wrapS(v) for k, v in _R7.items()})


# DeepSeek-V4 Muon (arXiv 2606.19348, Algorithm 1 as relayed): Nesterov momentum, Frobenius-normalized
# hybrid Newton-Schulz (8 steps of (3.4445, -4.7750, 2.0315), then 2 steps of (2, -1.5, 0.5)),
# update RMS 0.18: scale 0.18 * sqrt(max(m, n)); weight decay 0.1 via mwd=0.1 on the command line.
def ns_hybrid(G, eps=1e-7):
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for i in range(10):
        a, b, c = NS_ABC if i < 8 else (2.0, -1.5, 0.5)
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.transpose(-1, -2)
    return X


def r_muonv4(o, G, W, st, lr, p):
    M = nesterov(st, G, o.cfg.mom)
    return lr * 0.18 * math.sqrt(max(G.shape[-2], G.shape[-1])) * ns_hybrid(M)


RULES["muonv4"] = r_muonv4


# ---- Round 9 (2026-11-22, 64-expert bed): momentum borrowing across a layer's experts and stale-step fixes.
# Muon's Newton-Schulz turns any momentum into a full-size step, so an expert that got no tokens still takes a
# full step from its decayed old momentum. lazy (round 7) freezes it; round 9 varies the weight, source and target.
def _bor(c=0.3, wfun="lin", src="mean", where="mom", lazy=False, state=False, sim=False, only=None, dead="full"):
    def rule(o, G, W, st, lr, p):
        nm = o.names[id(p)]
        if o.kinds[id(p)] != "exp" or (only and not any(x in nm for x in only)):
            return r2_muon(o, G, W, st, lr, p)
        s = _share(o, p, G)
        if s is None:
            return r2_muon(o, G, W, st, lr, p)
        b = o.cfg.mom
        sc = s.clamp(max=1.0)
        live = (s > 0).float()
        if wfun == "lin":
            w = c * (1 - sc)
        elif wfun == "sq":
            w = c * (1 - sc) ** 2
        elif wfun == "inv":
            w = c * sc
        elif wfun == "ema":
            se = ema(st, "se", s, 0.9) / (1 - 0.9 ** o.t)
            w = c * (1 - se.clamp(max=1.0))
        elif wfun == "zero":
            w = c * (1 - live)
        else:
            w = torch.zeros_like(s)

        def ref(X):
            if src == "wmean":
                return (s * X).sum(0, keepdim=True) / (s.sum() + 1e-9)
            if src == "top":
                hi = (s >= 1).float()
                return (hi * X).sum(0, keepdim=True) / (hi.sum() + 1e-9)
            return X.mean(0, keepdim=True)
        Gi = G + w * ref(G) if where == "grad" else G
        if "m" not in st:
            st["m"] = torch.zeros_like(G)
        new = st["m"].lerp(Gi, 1 - b)
        st["m"].copy_(live * new + (1 - live) * st["m"] if lazy else new)
        if state:
            st["m"].lerp_(ref(st["m"]).expand_as(st["m"]), w.expand_as(st["m"]))
        M = Gi.lerp(st["m"], b)
        if where == "mom" and wfun != "none":
            R = ref(M)
            ww = w * rowcos(M.flatten(1), R.flatten(1)).view(-1, 1, 1).clamp(min=0) if sim else w
            O = ns5(M)
            U = fin(G, O, ns5(M + ww * R), lr)
        else:
            U = lr * moon_scale(G) * ns5(M)
        if lazy or dead == "skip":
            U = U * live
        elif dead == "half":
            U = U * (0.5 + 0.5 * live)
        elif dead == "mag":
            n = M.norm(dim=(-2, -1), keepdim=True)
            ne = ema(st, "mn", n, 0.9) / (1 - 0.9 ** o.t)
            U = U * (n / (ne + 1e-12)).clamp(max=1.0)
        return U
    return rule


_R9 = dict(
    bor_c15=_bor(0.15), bor_c5=_bor(0.5), bor_c8=_bor(0.8), bor_sq=_bor(0.5, "sq"),
    bor_inv=_bor(0.3, "inv"), bor_neg=_bor(-0.3), bor_wmean=_bor(0.3, src="wmean"), bor_top=_bor(0.3, src="top"),
    bor_grad=_bor(0.3, where="grad"), bor_state=_bor(0.1, state=True, where="none"), bor_sim=_bor(0.5, sim=True),
    bor_w2=_bor(0.3, only=("w2",)), bor_w13=_bor(0.3, only=("w1", "w3")), bor_lazy=_bor(0.3, lazy=True),
    bor_ema=_bor(0.3, "ema"), lazyskip=_bor(0.0, "none", dead="skip"), lazyhalf=_bor(0.0, "none", dead="half"),
    lazymag=_bor(0.0, "none", dead="mag"), bor_lazy5=_bor(0.5, lazy=True), bor_zero=_bor(0.3, "zero"),
)
RULES.update({k: _wrapS(v) for k, v in _R9.items()})
