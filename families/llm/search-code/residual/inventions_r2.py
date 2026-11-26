"""Round-2 residual ideas (search: residual). Standard init (no BIGSVD, no pe=). Knobs rk, rk2.

Exploit family (dg*): pre-equivalent dynamic depth gates. Sublayer l reads h_l = sum_i w_li(x) v_i over the
values v_0 = embedding, v_i = earlier sublayer outputs (or block sums), with every w = 1 at init, so the start
is exactly pre-norm. Generalises the read side of HC / AttnRes / DenseFormer without their init change.
Explore family: multi-stream, depth-attention variants, write routing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Base(nn.Module):
    def __init__(self, cfg, L):
        super().__init__()
        self.cfg, self.L, self.d = cfg, L, cfg.d_model
        self.k, self.k2 = float(cfg.rk), float(cfg.rk2)


# ======================= exploit: depth gates =======================
class DG(Base):
    """generic depth gate. opts:
    mode: sig (2 sigmoid(s)), neg (1 + 2 tanh(s)), soft (m * softmax_i(s)), static (2 sigmoid(b_li)),
          mix (2 sigmoid(b_li + s)), chan (static per-channel 2 sigmoid(B_li)), dynchan (1 + tanh(k_i * Q_l))
    S: block size for values; where: all / attn / ffn / final / emb; heads; curq; keyproj; temp; sharedq.
    rk = parameter-scale multiplier (effective lr boost of the gate params)."""
    opts = {}

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        o = dict(mode="sig", S=1, where="all", heads=1, curq=False, keyproj=0, temp=False, sharedq=False)
        o.update(s.opts)
        s.o = o
        d, nq = s.d, (1 if o["sharedq"] else L + 1)
        nv = L + 1
        s.q = nn.Parameter(torch.zeros(nq, d))
        if o["mode"] in ("static", "mix"):
            s.b = nn.Parameter(torch.zeros(L + 1, nv))
        if o["mode"] == "chan":
            s.B = nn.Parameter(torch.zeros(L + 1, nv, d))
        if o["curq"]:
            s.a = nn.Parameter(torch.zeros(L + 1, d))
        if o["keyproj"]:
            s.P = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        if o["temp"]:
            s.tau = nn.Parameter(torch.zeros(L + 1))

    def gated(s, l):
        w = s.o["where"]
        if w == "all":
            return True
        if w == "final":
            return l == s.L
        if w == "attn":
            return l < s.L and l % 2 == 0
        if w == "ffn":
            return l < s.L and l % 2 == 1
        return True                                                   # emb: handled in mix

    def mix(s, l, vals):
        V = torch.stack(vals, 0)                                      # [m,B,T,D]
        m = V.shape[0]
        if not s.gated(l):
            return V.sum(0)
        o, k = s.o, s.k
        q = s.q[0 if o["sharedq"] else l] * k
        K = rms(V)
        if o["keyproj"]:
            K = K @ s.P
        mode = o["mode"]
        if mode == "chan":
            w = 2 * torch.sigmoid(s.B[l, :m] * k)[:, None, None, :]
            return (w * V).sum(0)
        if mode == "dynchan":
            w = 1 + torch.tanh(K * q)
            return (w * V).sum(0)
        h = o["heads"]
        if o["curq"]:
            q = q + s.a[l] * k * rms(V.sum(0))                        # [B,T,D]
        sc = (K * q).view(m, *V.shape[1:3], h, -1).sum(-1)            # [m,B,T,h]
        if o["temp"]:
            sc = sc * torch.exp(s.tau[l])
        if mode == "sig":
            w = 2 * torch.sigmoid(sc)
        elif mode == "neg":
            w = 1 + 2 * torch.tanh(sc)
        elif mode == "soft":
            w = sc.softmax(0) * m
        elif mode == "static":
            w = (2 * torch.sigmoid(s.b[l, :m] * k))[:, None, None, None].expand(m, *sc.shape[1:])
        elif mode == "mix":
            w = 2 * torch.sigmoid(s.b[l, :m, None, None, None] * k + sc)
        if o["where"] == "emb":                                       # only the embedding's weight is gated
            w = torch.cat([w[:1], torch.ones_like(w[1:])], 0)
        Vh = V.view(m, *V.shape[1:3], h, -1)
        return (w.unsqueeze(-1) * Vh).sum(0).flatten(-2)

    def forward(s, x, subs):
        S = s.o["S"]
        blocks, part = [x], None
        for j, f in enumerate(subs):
            vals = blocks + ([part] if part is not None else [])
            y = f(s.mix(j, vals))
            part = y if part is None else part + y
            if (j + 1) % S == 0:
                blocks.append(part)
                part = None
        vals = blocks + ([part] if part is not None else [])
        return s.mix(s.L, vals)


def dg(name, **o):
    return type(name, (DG,), {"opts": o})


class DGWrite(Base):
    """write-side dynamic gate: x += 2 sigmoid(q_l . rms(y) * rk) * y, q zero init (pre at init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L, cfg.d_model))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            y = f(x)
            x = x + 2 * torch.sigmoid((rms(y) * s.q[j] * s.k).sum(-1, keepdim=True)) * y
        return x


# ======================= explore: multi-stream =======================
class MSRound(Base):
    """n = rk2 streams; embedding in stream 0; sublayer j writes stream j mod n; reads sum_i a_ji H_i,
    a learned static init 1 (pre at init); readout learned init 1. rk = param-scale multiplier."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.n = max(2, int(s.k2))
        s.a = nn.Parameter(torch.zeros(L + 1, s.n))

    def forward(s, x, subs):
        H = [x] + [torch.zeros_like(x) for _ in range(s.n - 1)]
        for j, f in enumerate(subs):
            w = 1 + s.a[j] * s.k
            y = f(sum(w[i] * H[i] for i in range(s.n)))
            H[j % s.n] = H[j % s.n] + y
        w = 1 + s.a[s.L] * s.k
        return sum(w[i] * H[i] for i in range(s.n))


class MSKind3(Base):
    """3 streams: embedding / attention writes / FFN writes. Read weights per sublayer per stream =
    1 + a + rk * tanh(rms(H_i) . u) (static + dynamic, pre at init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.a = nn.Parameter(torch.zeros(L + 1, 3))
        s.u = nn.Parameter(torch.zeros(L + 1, 3, cfg.d_model))

    def read(s, l, H):
        V = torch.stack(H, 0)
        z = torch.tanh((rms(V) * s.u[l][:, None, None, :]).sum(-1, keepdim=True))
        w = 1 + s.a[l][:, None, None, None] + s.k * z
        return (w * V).sum(0)

    def forward(s, x, subs):
        H = [x, torch.zeros_like(x), torch.zeros_like(x)]
        for j, f in enumerate(subs):
            y = f(s.read(j, H))
            H[1 + j % 2] = H[1 + j % 2] + y
        return s.read(s.L, H)


class MSEma(Base):
    """2 streams: plain sum x and a depth-leaky copy e <- rk * e + y. Each sublayer reads x + b_l * e,
    b zero init (pre at init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L + 1))

    def forward(s, x, subs):
        e = torch.zeros_like(x)
        for j, f in enumerate(subs):
            y = f(x + s.b[j] * e)
            x = x + y
            e = s.k * e + y
        return x + s.b[s.L] * e


class HCStatic(Base):
    """static Hyper-Connections variants. opts: uni (read init mix (1-rk2)*onehot + rk2/n), lr (param scale
    rk), leak (A_r init diag(1, rk2, ...)), share (one set of connections for all sublayers), block (connect
    once per block). n = 4 unless opts give n."""
    opts = {}

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        o = dict(n=4, uni=0.0, leak=1.0, share=False, block=False)
        o.update(s.opts)
        if "uni" in s.opts and s.opts["uni"] == "rk2":
            o["uni"] = s.k2
        if "leak" in s.opts and s.opts["leak"] == "rk2":
            o["leak"] = s.k2
        s.o, n = o, o["n"]
        s.n = n
        nl = 1 if o["share"] else L
        am = torch.zeros(nl, n)
        am[torch.arange(nl), torch.arange(nl) % n] = 1.0
        am = (1 - o["uni"]) * am + o["uni"] / n
        ar = torch.eye(n).repeat(nl, 1, 1)
        ar[:, 1:, 1:] *= o["leak"]
        s.am0, s.ar0 = nn.Parameter(am, requires_grad=False), nn.Parameter(ar, requires_grad=False)
        s.am = nn.Parameter(torch.zeros(nl, n))
        s.ar = nn.Parameter(torch.zeros(nl, n, n))
        s.b = nn.Parameter(torch.zeros(nl, n))

    def forward(s, x, subs):
        H = x.unsqueeze(0).expand(s.n, *x.shape)
        k = s.k
        for j, f in enumerate(subs):
            i = 0 if s.o["share"] else j
            if s.o["block"] and j % 2 == 1:
                y = f(h + y_prev)
                H = H + (1 + s.b[i - 1] * k)[:, None, None, None] * y
                continue
            am = s.am0[i] + s.am[i] * k
            ar = s.ar0[i] + s.ar[i] * k
            h = torch.einsum("n,nbtd->btd", am, H)
            y = f(h)
            y_prev = y
            H = torch.einsum("ik,ibtd->kbtd", ar, H) + (1 + s.b[i] * k)[:, None, None, None] * y
        return H.sum(0)


def hcs(name, **o):
    return type(name, (HCStatic,), {"opts": o})


class HCSoft(Base):
    """n = rk2 streams, round-robin writes; read = n * softmax_i(q_l . rms(H_i) * rk) H_i (uniform -> pre at init)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.n = max(2, int(s.k2))
        s.q = nn.Parameter(torch.zeros(L + 1, cfg.d_model))

    def read(s, l, H):
        V = torch.stack(H, 0)
        w = ((rms(V) * s.q[l] * s.k).sum(-1, keepdim=True)).softmax(0) * s.n
        return (w * V).sum(0)

    def forward(s, x, subs):
        H = [x] + [torch.zeros_like(x) for _ in range(s.n - 1)]
        for j, f in enumerate(subs):
            y = f(s.read(j, H))
            H[j % s.n] = H[j % s.n] + y
        return s.read(s.L, H)


# ======================= explore: depth attention variants =======================
class AResX(Base):
    """AttnRes (Kimi) variants. opts: S block size, heads (rk2), hyb (h = sum + beta*m*mix, beta init rk),
    alibi (recency bias slope init rk), only (attn/ffn reads by AttnRes, other sublayers read the plain sum),
    sep (separate values for attention and FFN sums per block), qmul (query param scale rk), center (keys
    centred over depth)."""
    opts = {}

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        o = dict(S=1, heads=1, hyb=False, alibi=False, only=None, sep=False, qmul=False, center=False)
        o.update(s.opts)
        s.o = o
        d = cfg.d_model
        s.q = nn.Parameter(torch.zeros(L + 1, d))
        s.kw = nn.Parameter(torch.ones(L + 1, d))
        if o["hyb"]:
            s.beta = nn.Parameter(torch.full((L + 1,), s.k))
        if o["alibi"]:
            s.sl = nn.Parameter(torch.full((L + 1,), s.k))
        s.h = max(1, int(s.k2)) if o["heads"] == "rk2" else 1

    def mix(s, l, vals):
        V = torch.stack(vals, 0)
        m = V.shape[0]
        o = s.o
        if o["only"] and l < s.L and (l % 2 == 0) != (o["only"] == "attn"):
            return V.sum(0)
        q = s.q[l] * (s.k if o["qmul"] else 1.0)
        K = V - V.mean(0, keepdim=True) if o["center"] else V
        sc = (rms(K) * s.kw[l] * q).view(m, *V.shape[1:3], s.h, -1).sum(-1)   # [m,B,T,h]
        if o["alibi"]:
            sc = sc + s.sl[l] * torch.arange(m, device=V.device, dtype=V.dtype).view(m, 1, 1, 1) / max(1, m - 1)
        a = sc.softmax(0)
        out = (a.unsqueeze(-1) * V.view(m, *V.shape[1:3], s.h, -1)).sum(0).flatten(-2)
        if o["hyb"]:
            out = V.sum(0) + s.beta[l] * m * out
        return out

    def forward(s, x, subs):
        S = s.o["S"]
        if s.o["sep"]:
            blocks, pa, pf = [x], None, None
            for j, f in enumerate(subs):
                vals = blocks + [p for p in (pa, pf) if p is not None]
                y = f(s.mix(j, vals))
                if j % 2 == 0:
                    pa = y if pa is None else pa + y
                else:
                    pf = y if pf is None else pf + y
                if (j + 1) % S == 0:
                    blocks += [p for p in (pa, pf) if p is not None]
                    pa = pf = None
            return s.mix(s.L, blocks + [p for p in (pa, pf) if p is not None])
        blocks, part = [x], None
        for j, f in enumerate(subs):
            vals = blocks + ([part] if part is not None else [])
            y = f(s.mix(j, vals))
            part = y if part is None else part + y
            if (j + 1) % S == 0:
                blocks.append(part)
                part = None
        return s.mix(s.L, blocks + ([part] if part is not None else []))


def ar(name, **o):
    return type(name, (AResX,), {"opts": o})


class AResWrite(Base):
    """write-side depth attention: after each block the stream becomes m * softmax-mix of all block states
    (dynamic DenseFormer; uniform weights at init would rescale, so mix = x + beta*(m*mix - x), beta init rk)."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.q = nn.Parameter(torch.zeros(L, cfg.d_model))
        s.beta = nn.Parameter(torch.full((L,), s.k))

    def forward(s, x, subs):
        X = [x]
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                X.append(x)
                V = torch.stack(X, 0)
                a = (rms(V) * s.q[j]).sum(-1, keepdim=True).softmax(0)
                x = x + s.beta[j] * ((a * V).sum(0) - x)
        return x


# ======================= explore: write routing / other =======================
class DenseDyn(Base):
    """DenseFormer per sublayer with token-dependent weights a_ik + rk * tanh(u_ik . rms(X_k))."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        a = torch.zeros(L, L + 1)
        a[torch.arange(L), torch.arange(L) + 1] = 1.0
        s.a = nn.Parameter(a)
        s.u = nn.Parameter(torch.zeros(L, L + 1, cfg.d_model))

    def forward(s, x, subs):
        X = [x]
        for j, f in enumerate(subs):
            x = x + f(x)
            X.append(x)
            V = torch.stack(X, 0)
            n = V.shape[0]
            w = s.a[j, :n, None, None, None] + s.k * torch.tanh((rms(V) * s.u[j, :n, None, None, :]).sum(-1, keepdim=True))
            x = (w * V).sum(0)
        return x


class Parallel(Base):
    """parallel block (GPT-J / PaLM): attention and FFN read the same x; FFN also reads rk * y_attn."""
    def forward(s, x, subs):
        for j in range(0, len(subs), 2):
            ya = subs[j](x)
            yf = subs[j + 1](x + s.k * ya)
            x = x + ya + yf
        return x


class StochDepth(Base):
    """stochastic depth (Huang et al. 2016) control: in training each sublayer is skipped with prob rk,
    kept outputs scaled by 1 / (1 - rk)."""
    def forward(s, x, subs):
        for f in subs:
            y = f(x)
            if s.training and s.k > 0:
                keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= s.k).to(x.dtype)
                y = y * keep / (1 - s.k)
            x = x + y
        return x


class Transport(Base):
    """learned linear transport of the stream once per block: x <- x + (x U_l) V_l^T, rank rk2,
    V zero init (pre at init), U scaled by rk."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        r = max(1, int(s.k2))
        s.U = nn.Parameter(torch.randn(L // 2, cfg.d_model, r) / math.sqrt(cfg.d_model))
        s.V = nn.Parameter(torch.zeros(L // 2, cfg.d_model, r))

    def forward(s, x, subs):
        for j, f in enumerate(subs):
            x = x + f(x)
            if j % 2 == 1:
                b = j // 2
                x = x + (x @ s.U[b] * s.k) @ s.V[b].t()
        return x


class SubX0(Base):
    """each sublayer reads x - b_l * x0 (b zero init, scale rk): lets deeper reads drop the raw embedding."""
    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        s.b = nn.Parameter(torch.zeros(L + 1))

    def forward(s, x, subs):
        x0 = x
        for j, f in enumerate(subs):
            x = x + f(x - s.k * s.b[j] * x0)
        return x - s.k * s.b[s.L] * x0


IDEAS = dict(
    # exploit: depth gates (20)
    dgsig=dg("DGSig"), dgsigblk=dg("DGSigBlk", S=2), dgstatic=dg("DGStatic", mode="static"),
    dgchan=dg("DGChan", mode="chan"), dgdynchan=dg("DGDynChan", mode="dynchan"), dgneg=dg("DGNeg", mode="neg"),
    dgcur=dg("DGCur", curq=True), dgfinal=dg("DGFinal", where="final"), dgwrite=DGWrite,
    dgffn=dg("DGFfn", where="ffn"), dgattn=dg("DGAttn", where="attn"), dgsoft=dg("DGSoft", mode="soft"),
    dgsoftblk=dg("DGSoftBlk", mode="soft", S=2), dgemb=dg("DGEmb", where="emb"), dgmix=dg("DGMix", mode="mix"),
    dgheads=dg("DGHeads", heads=4), dgsharedq=dg("DGSharedQ", sharedq=True), dgkeyproj=dg("DGKeyProj", keyproj=1),
    dgtemp=dg("DGTemp", temp=True), dgcurblk=dg("DGCurBlk", curq=True, S=2),
    # explore (20)
    msround=MSRound, mskind3=MSKind3, msema=MSEma, hcuni=hcs("HCUni", uni="rk2"), hclr=hcs("HCLr"),
    hcleak=hcs("HCLeak", n=2, leak="rk2"), hcshare=hcs("HCShare", share=True), hcblock=hcs("HCBlock", block=True),
    hcsoft=HCSoft, aresmh=ar("AResMH", heads="rk2", S=2), areshyb=ar("AResHyb", hyb=True, S=2),
    aresalibi=ar("AResAlibi", alibi=True, S=2), aressep=ar("AResSep", sep=True, S=2),
    aresqmul=ar("AResQMul", qmul=True, S=2), arescenter=ar("AResCenter", center=True, S=2),
    areswrite=AResWrite, densedyn=DenseDyn, parallel=Parallel,
    transport=Transport, subx0=SubX0,
)


class PeriWrap(nn.Module):
    """rs=peri: every sublayer output passes Peri-LN's output RMSNorm (vector gain init 1) and the embedding
    gets RMSNorm with gain, before the round-2 scheme sees them (ideas stacked on the Peri-LN baseline)."""
    def __init__(s, inner, cfg, L):
        super().__init__()
        s.inner = inner
        s.w = nn.Parameter(torch.ones(L, cfg.d_model))
        s.we = nn.Parameter(torch.ones(cfg.d_model))

    def forward(s, x, subs):
        for a in ("ids", "blocks_ref", "rope"):
            if hasattr(s, a):
                object.__setattr__(s.inner, a, getattr(s, a))
        wr = [(lambda h, f=f, j=j: rms(f(h)) * s.w[j]) for j, f in enumerate(subs)]
        return s.inner(rms(x) * s.we, wr)


class NAGWrap(nn.Module):
    """rs=nag: every sublayer output becomes NAG's write (arXiv 2606.16112 Eq. 14) computed from the scheme's
    read h: RMS(h) * alpha_j * m_j(hbar) * RMSNorm(perp(centre(F(h)), hbar)). For a pre-equivalent scheme
    (h = plain stream at init) this is exactly res=nag at init. C = res_n gate directions."""
    def __init__(s, inner, cfg, L):
        super().__init__()
        import residual
        s.inner, s.nag = inner, residual.NAG(cfg, L)

    def forward(s, x, subs):
        for a in ("ids", "blocks_ref", "rope"):
            if hasattr(s, a):
                object.__setattr__(s.inner, a, getattr(s, a))
        g = s.nag

        def w(h, f, j):
            rho = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            hb = h / rho
            y = f(h)
            y = y - y.mean(-1, keepdim=True)
            y = y - (y * hb).sum(-1, keepdim=True) / (hb * hb).sum(-1, keepdim=True) * hb
            gt = torch.sigmoid(torch.einsum("btd,cd->btc", hb, g.w[j]) + g.bb[j])
            m = (gt * g.pl[j].softmax(-1)).sum(-1, keepdim=True).clamp_min(1e-6) ** g.lb[j].exp()
            return rho * g.alpha[j] * m * rms(y)
        return s.inner(x, [(lambda h, f=f, j=j: w(h, f, j)) for j, f in enumerate(subs)])


def make(cfg, L):
    return wrap(IDEAS[cfg.res](cfg, L), cfg, L)


def wrap(m, cfg, L):
    rs = getattr(cfg, "rs", "")
    if rs == "peri":
        return PeriWrap(m, cfg, L)
    if rs == "nag":
        return NAGWrap(m, cfg, L)
    return m
