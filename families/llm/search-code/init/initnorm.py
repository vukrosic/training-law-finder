"""INIT SEARCH: new initialization and normalization ideas, selected by --set idea=<name> ik=<knob>.

model.py calls model_setup(model, cfg) at the end of GPT.__init__ when cfg.idea != "none".
An idea may: replace norm modules (block n1/n2, final norm), re-initialize weights, set
model.data_hook(train_tokens) (called once by train.py before training), model.init_step_hook()
(after every optimizer step), model.head_hook(x) (replaces head(norm(x))).
Token ids for the current batch: model.ctx["ids"] (set by GPT.forward before the blocks run).
No per-dimension statistic ever mixes positions after t (causal cumulative stats or detached EMAs).
"""
import json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "tokenizer", "bpe8192.json")
IDEAS = {}


def idea(f):
    IDEAS[f.__name__] = f
    return f


def model_setup(m, cfg):
    """idea "a+b": a gets knob ik, b gets ik2 (b sees cfg.ik = ik2 while it is set up)"""
    name = getattr(cfg, "idea", "none")
    if name == "none":
        return
    if getattr(cfg, "ik3", "none") == "nag":     # round 7+: stack the idea on the NAG lab baseline (C = 8); the idea keeps ik
        import types as _t
        IDEAS["nag"](m, _t.SimpleNamespace(**{**vars(cfg), "ik": 8}))
    parts, k1 = name.split("+"), cfg.ik
    for j, nm in enumerate(parts):
        cfg.ik = k1 if j == 0 else cfg.ik2
        IDEAS[nm](m, cfg)
    cfg.ik = k1


# ---------------------------------------------------------------- helpers
def add_hook(m, name, f):
    """chain hooks so that combined ideas ("a+b") keep both"""
    prev = getattr(m, name, None)
    if prev is None:
        setattr(m, name, f)
    else:
        def both(*a):
            prev(*a)
            f(*a)
        setattr(m, name, both)


def ms(x):
    return x.pow(2).mean(-1, keepdim=True)


def hkey(ids, nb, V):
    return ids if nb >= V else (ids * 2654435761 + 12345) % nb


def prev_ids(ids):
    return torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1)


def cummean(x, dim=1):
    n = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype).view(-1, *([1] * (x.dim() - dim - 1)))
    return x.cumsum(dim) / n


def block_norms(m):
    for b in m.blocks:
        yield b, "n1"
        yield b, "n2"


def swap(m, where, make):
    """where: 'blocks', 'final', 'all', 'first' (block 0 only)."""
    if where in ("blocks", "all", "first"):
        for i, b in enumerate(m.blocks):
            if where == "first" and i > 0:
                continue
            for nm in ("n1", "n2"):
                setattr(b, nm, make(m))
    if where in ("final", "all"):
        m.norm = make(m)
    if where == "out":                       # Peri-LN output norms (needs norm_mode=peri)
        for b in m.blocks:
            b.no1, b.no2 = make(m), make(m)


class Ema:
    """detached running statistic, updated in train mode only"""

    def __init__(self, rate):
        self.rate, self.v = rate, None

    def upd(self, val, train):
        """returns the value from BEFORE this batch (no information from the current batch, so no
        future-token leak); on the very first call returns the detached current value"""
        val = val.detach()
        old = self.v if self.v is not None else val
        if train:
            self.v = val.clone() if self.v is None else self.v.lerp(val, self.rate)
        return old


# ---------------------------------------------------------------- generic custom norm
class Norm(nn.Module):
    """RMSNorm with a pluggable denominator / gain. fn(self, x) -> y"""

    def __init__(self, m, d, fn, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps, self.fn, self.ctx, self.d = 1e-6, fn, m.ctx, d
        for k, v in kw.items():
            setattr(self, k, v)

    def forward(self, x):
        return self.fn(self, x)


def rmsn(x, eps=1e-6):
    return x * torch.rsqrt(ms(x) + eps)


# ================================================================ family 1: id-keyed norm gains
def _gain_table(m, cfg, rows, cols, init=1.0):
    t = nn.Embedding(rows, cols)
    nn.init.constant_(t.weight, init)
    return t


def _keyed(m, cfg, key, nb):
    V = cfg.vocab

    def k(ids):
        if key == "cur":
            return hkey(ids, nb, V)
        if key == "prev":
            return hkey(prev_ids(ids), nb, V)
        if key == "pos":
            return torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        if key == "freq":
            return m.freq_bucket[ids]
    return k


def _idgain(m, cfg, key, where, nb, scalar=False):
    V, d = cfg.vocab, cfg.d_model
    rows = {"pos": cfg.seq_len, "freq": nb}.get(key, min(nb, V))
    kf = _keyed(m, cfg, key, nb)

    def make(mm):
        n = Norm(mm, d, lambda s, x: rmsn(x) * s.w * s.tab(kf(s.ctx["ids"])))
        n.tab = _gain_table(mm, cfg, rows, 1 if scalar else d)
        return n
    swap(m, where, make)


def _freq_hook(m, nb):
    def hook(train):
        fr = np.bincount(np.asarray(train[:4_000_000]), minlength=m.cfg.vocab)
        rk = np.empty(m.cfg.vocab, np.int64)
        rk[np.argsort(-fr)] = np.arange(m.cfg.vocab)
        b = np.floor(nb * np.log1p(rk) / np.log1p(m.cfg.vocab)).clip(0, nb - 1)
        m.freq_bucket.copy_(torch.from_numpy(b).long())
    m.register_buffer("freq_bucket", torch.zeros(m.cfg.vocab, dtype=torch.long), persistent=False)
    add_hook(m, "data_hook", hook)


@idea
def idgain(m, cfg):        # block-norm gain vector looked up by current token id (ik: table rows)
    _idgain(m, cfg, "cur", "blocks", int(cfg.ik or 8192))


@idea
def idgain_fin(m, cfg):    # final-norm gain vector per current id
    _idgain(m, cfg, "cur", "final", int(cfg.ik or 8192))


@idea
def idscalar(m, cfg):      # block-norm scalar gain per current id
    _idgain(m, cfg, "cur", "blocks", int(cfg.ik or 8192), scalar=True)


@idea
def prevgain(m, cfg):      # block-norm gain per previous token id
    _idgain(m, cfg, "prev", "blocks", int(cfg.ik or 8192))


@idea
def posgain(m, cfg):       # block-norm gain per position (ik 0: all blocks, 1: first block)
    _idgain(m, cfg, "pos", "first" if cfg.ik == 1 else "blocks", 0)


@idea
def freqgain(m, cfg):      # block-norm gain per log-frequency-rank bucket (ik buckets)
    nb = int(cfg.ik or 8)
    _freq_hook(m, nb)
    _idgain(m, cfg, "freq", "blocks", nb)


@idea
def idgain_lr(m, cfg):     # gain = w * (1 + A[id] B), rank ik, A zero init
    r, d, V = int(cfg.ik or 4), cfg.d_model, cfg.vocab

    def make(mm):
        n = Norm(mm, d, lambda s, x: rmsn(x) * s.w * (1 + s.A(s.ctx["ids"]) @ s.B))
        n.A = nn.Embedding(V, r)
        nn.init.zeros_(n.A.weight)
        n.B = nn.Parameter(torch.randn(r, d) / math.sqrt(r))
        return n
    swap(m, "blocks", make)


@idea
def idqk(m, cfg):          # per-head QK-norm whose gain (both q and k) is also scaled per current id
    nb, V = int(cfg.ik or 8192), cfg.vocab
    for b in m.blocks:
        a = b.attn
        a.qk_tab = nn.Embedding(min(nb, V), a.dh)
        nn.init.ones_(a.qk_tab.weight)
        a.qn_w, a.kn_w = nn.Parameter(torch.ones(a.dh)), nn.Parameter(torch.ones(a.dh))
        a.qk = "idqk"
        a.ctx_ref = m.ctx

        def qk(a, q, k, nb=nb):
            g = a.qk_tab(hkey(a.ctx_ref["ids"], nb, V))[:, None]        # [B,1,T,dh]
            return rmsn(q) * a.qn_w * g, rmsn(k) * a.kn_w * g
        a.qk_fn = qk


@idea
def outid(m, cfg):         # Peri-style output norm on each branch, gain per current id
    nb, V, d = int(cfg.ik or 8192), cfg.vocab, cfg.d_model
    for b in m.blocks:
        for nm in ("no1", "no2"):
            n = Norm(m, d, lambda s, x: rmsn(x) * s.w * s.tab(hkey(s.ctx["ids"], nb, V)))
            n.tab = _gain_table(m, cfg, min(nb, V), d)
            setattr(b, nm, n)
        b.mode = "peri"                                               # x + o(F(n(x)))


# ================================================================ family 2: magnitude-aware norms
def _ref(s, x):
    """detached EMA of this norm's mean input rms (train mode)"""
    r = ms(x).mean().sqrt()
    return s.ema.upd(r, s.training)


@idea
def magnorm(m, cfg):       # y = Norm(x) w + u * (log rms(x) - log ref); keeps the magnitude as a feature
    d = cfg.d_model

    def f(s, x):
        lr = 0.5 * torch.log(ms(x) + 1e-12) - torch.log(_ref(s, x) + 1e-12)
        return rmsn(x) * s.w + s.u * lr
    def make(mm):
        n = Norm(mm, d, f, ema=Ema(0.05))
        n.u = nn.Parameter(torch.zeros(d))
        return n
    swap(m, "all" if cfg.ik == 1 else "blocks", make)


@idea
def partnorm(m, cfg):      # y = x / (rms^p ref^(1-p)): partial normalization (ik = p)
    p = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: x / ((ms(x) + 1e-12) ** (p / 2) * _ref(s, x) ** (1 - p)) * s.w,
                                       ema=Ema(0.05)))


@idea
def capnorm(m, cfg):       # y = x / max(rms, c ref): never amplify small vectors past c ref
    c = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model,
                                       lambda s, x: x / torch.maximum(ms(x).sqrt(), c * _ref(s, x)) * s.w, ema=Ema(0.05)))


@idea
def lpnorm(m, cfg):        # normalize by the Lp norm (Gaussian-calibrated), p = ik
    p = cfg.ik
    kp = (2 ** (p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)) ** (1 / p)   # E|z|^p^(1/p)
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model,
                                       lambda s, x: x / ((x.abs().pow(p).mean(-1, keepdim=True) + 1e-12) ** (1 / p) / kp) * s.w))


@idea
def seqnorm(m, cfg):       # denominator rms_tok^(1-a) * rms_seq^a, rms_seq over positions <= t (ik = a)
    a = cfg.ik

    def f(s, x):
        mt = ms(x) + 1e-12
        return x / (mt ** ((1 - a) / 2) * cummean(mt) ** (a / 2)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def latinhib(m, cfg):      # y = Norm(x - lam * mean_{s<=t} x_s): subtract the causal running mean (ik = lam)
    lam = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(x - lam * cummean(x)) * s.w))


@idea
def groupnorm(m, cfg):     # RMS over groups of g dims (ik = g)
    g = int(cfg.ik)

    def f(s, x):
        sh = x.shape
        xg = x.view(*sh[:-1], sh[-1] // g, g)
        return (xg * torch.rsqrt(ms(xg) + 1e-6)).view(sh) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def statgain(m, cfg):      # gain = w * (1 + tanh(W [log rms - log ref, log max|x|/rms])), W zero init
    d = cfg.d_model

    def f(s, x):
        r = (ms(x) + 1e-12).sqrt()
        feat = torch.cat([torch.log(r) - torch.log(_ref(s, x) + 1e-12),
                          torch.log(x.abs().amax(-1, keepdim=True) / r + 1e-6) - math.log(math.sqrt(2 * math.log(d)))], -1)
        return x / r * s.w * (1 + torch.tanh(s.lin(feat)))
    def make(mm):
        n = Norm(mm, d, f, ema=Ema(0.05))
        n.lin = nn.Linear(2, d)
        nn.init.zeros_(n.lin.weight); nn.init.zeros_(n.lin.bias)
        return n
    swap(m, "all" if cfg.ik == 1 else "blocks", make)


@idea
def loggain(m, cfg):       # gain parametrized as exp(g), g init 0 (ik 1: final too)
    def make(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * torch.exp(s.w))
        nn.init.zeros_(n.w)
        return n
    swap(m, "all" if cfg.ik == 1 else "blocks", make)


@idea
def gainrand(m, cfg):      # norm gains start log-normal with sigma ik (dimensions start unequal)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for mod in m.modules():
            if type(mod).__name__ == "RMSNorm" and mod.w.dim() == 1 and mod.w.numel() == cfg.d_model:
                mod.w.copy_(torch.exp(cfg.ik * torch.randn(cfg.d_model, generator=g) - cfg.ik ** 2 / 2))


@idea
def finalgain(m, cfg):     # control: final-norm gain starts at ik (logit temperature at init)
    with torch.no_grad():
        m.norm.w.fill_(cfg.ik)


@idea
def knorm(m, cfg):         # normalize keys only (per head, gain ik); query magnitude = per-token temperature
    for b in m.blocks:
        a = b.attn
        a.kn_w = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.qk = "knorm"
        a.qk_fn = lambda a, q, k: (q, rmsn(k) * a.kn_w)


@idea
def normnoise(m, cfg):     # train only: Gaussian noise (std ik) added to block-norm outputs
    sd = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model,
                                       lambda s, x: (rmsn(x) + (sd * torch.randn_like(x) if s.training else 0)) * s.w))


@idea
def normdrop(m, cfg):      # train only: with prob ik a token is scaled by the EMA ref instead of its own rms
    p = cfg.ik

    def f(s, x):
        y = rmsn(x)
        if s.training:
            alt = x / (_ref(s, x) + 1e-12)
            keep = (torch.rand(x.shape[:-1] + (1,), device=x.device) < p)
            y = torch.where(keep, alt, y)
        else:
            _ref(s, x)
        return y * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f, ema=Ema(0.05)))


# ================================================================ family 3: top-of-network collapse
@idea
def dimstd_fin(m, cfg):    # final: per-dim standardization with causal stats (prior = EMA, 8 pseudo-tokens), blend ik
    lam, d, n0 = cfg.ik, cfg.d_model, 8.0

    def f(s, x):
        y = rmsn(x)
        mu_e = s.em.upd(y.mean((0, 1)), s.training)
        va_e = s.ev.upd(y.var((0, 1)), s.training)
        n = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype)[:, None]
        mu = (n0 * mu_e + y.cumsum(1)) / (n0 + n)
        m2 = (n0 * (va_e + mu_e ** 2) + (y * y).cumsum(1)) / (n0 + n)
        z = (y - mu) * torch.rsqrt((m2 - mu * mu).clamp(min=1e-4))
        return ((1 - lam) * y + lam * z) * s.w
    m.norm = Norm(m, d, f, em=Ema(0.05), ev=Ema(0.05))


@idea
def emastd_fin(m, cfg):    # final: divide each dim by its EMA std (detached), mean-preserving (ik = EMA keep)
    d = cfg.d_model

    def f(s, x):
        y = rmsn(x)
        v = s.ev.upd(y.var((0, 1)), s.training)
        sc = torch.rsqrt(v + 1e-4)
        return y * (sc / sc.pow(2).mean().sqrt()) * s.w
    m.norm = Norm(m, d, f, ev=Ema(1 - cfg.ik))


@idea
def emawhite_fin(m, cfg):  # final: whiten with EMA covariance (detached), shrinkage ik toward isotropic
    d, rho = cfg.d_model, cfg.ik

    def f(s, x):
        y = rmsn(x)
        yf = y.reshape(-1, d)
        C = s.ec.upd(yf.t() @ yf / yf.shape[0], s.training)
        if s.training or s.Wm is None:
            with torch.no_grad():
                Cs = (1 - rho) * C + rho * torch.eye(d, device=C.device) * C.diagonal().mean()
                e, U = torch.linalg.eigh(Cs.double())
                W = (U * e.clamp(min=1e-8).rsqrt()) @ U.t()
                s.Wm = (W * (C.diagonal().mean().sqrt())).float()
        return (y @ s.Wm) * s.w
    m.norm = Norm(m, d, f, ec=Ema(0.05), Wm=None)


@idea
def meansub_ema(m, cfg):   # subtract the EMA mean residual vector (detached) before normalizing (ik 1: final only)
    def make(mm):
        return Norm(mm, cfg.d_model, lambda s, x: rmsn(x - s.em.upd(x.mean((0, 1)), s.training)) * s.w, em=Ema(0.05))
    swap(m, "final" if cfg.ik == 1 else "all", make)


@idea
def embsphere(m, cfg):     # embedding rows projected to norm ik * 0.02 sqrt(d) after every step
    R = cfg.ik * 0.02 * math.sqrt(cfg.d_model)

    @torch.no_grad()
    def proj():
        m.emb.weight.copy_(F.normalize(m.emb.weight, dim=1) * R)
    proj()
    add_hook(m, "init_step_hook", proj)


@idea
def eqrow(m, cfg):         # head rows (ik 1: and embedding rows) projected to their mean norm after every step
    @torch.no_grad()
    def proj():
        for W in ([m.head.weight] + ([m.emb.weight] if cfg.ik == 1 else [])):
            W.copy_(F.normalize(W, dim=1) * W.norm(dim=1).mean())
    add_hook(m, "init_step_hook", proj)


# ================================================================ family 4: tokenizer-structured embedding init
def _tok_strings():
    v = json.load(open(TOK))["model"]
    toks = [""] * len(v["vocab"])
    for s, i in v["vocab"].items():
        toks[i] = s
    return toks, v["merges"]


def _bytecode(d, decay, from_end, seed=3):
    toks, _ = _tok_strings()
    g = torch.Generator().manual_seed(seed)
    chars = sorted({c for t in toks for c in t})
    R = torch.randn(len(chars), d, generator=g)
    P = torch.randn(16, d, generator=g) * 0.5                      # weak position-in-token code
    ci = {c: i for i, c in enumerate(chars)}
    out = torch.zeros(len(toks), d)
    for i, t in enumerate(toks):
        if not t:
            continue
        seq = t[::-1] if from_end else t
        for j, c in enumerate(seq):
            out[i] += decay ** j * (R[ci[c]] + P[min(j, 15)])
    out[0] = torch.randn(d, generator=g)
    return out / out.std()


def _blend(W, code, lam):
    with torch.no_grad():
        s = W.std()
        W.copy_(((1 - lam) * W / s + lam * code) / math.sqrt((1 - lam) ** 2 + lam ** 2) * s)


@idea
def byteemb(m, cfg):       # input embedding starts as a decayed sum of its characters' random codes (ik = blend)
    _blend(m.emb.weight, _bytecode(cfg.d_model, 0.7, False), cfg.ik)


@idea
def bytehead(m, cfg):      # output head rows start from the same character codes (ik = blend)
    _blend(m.head.weight, _bytecode(cfg.d_model, 0.7, False), cfg.ik)


@idea
def edgebytes(m, cfg):     # emb from the token's LAST characters, head from its FIRST characters (ik = decay)
    _blend(m.emb.weight, _bytecode(cfg.d_model, cfg.ik, True, seed=3), 0.7)
    _blend(m.head.weight, _bytecode(cfg.d_model, cfg.ik, False, seed=4), 0.7)


@idea
def mergeemb(m, cfg):      # embedding follows the BPE merge tree: e(ab) = (e(a)+e(b))/sqrt2 mixed with noise ik
    toks, merges = _tok_strings()
    d, eps = cfg.d_model, cfg.ik
    idx = {t: i for i, t in enumerate(toks)}
    g = torch.Generator().manual_seed(5)
    E = torch.randn(len(toks), d, generator=g)
    for a, b in merges:
        ab = a + b
        if ab in idx and a in idx and b in idx:
            E[idx[ab]] = math.sqrt(1 - eps ** 2) * (E[idx[a]] + E[idx[b]]) / math.sqrt(2) + eps * torch.randn(d, generator=g)
    with torch.no_grad():
        m.emb.weight.copy_(E / E.std() * m.emb.weight.std())


@idea
def embfreq(m, cfg):       # embedding row norms scaled by (frequency rank + 1)^ik, mean square kept
    def hook(train):
        fr = np.bincount(np.asarray(train[:4_000_000]), minlength=cfg.vocab)
        rk = np.empty(cfg.vocab, np.int64)
        rk[np.argsort(-fr)] = np.arange(cfg.vocab)
        sc = torch.from_numpy((rk + 1.0) ** cfg.ik).float()
        sc = sc / sc.pow(2).mean().sqrt()
        with torch.no_grad():
            m.emb.weight.mul_(sc[:, None].to(m.emb.weight.device))
    add_hook(m, "data_hook", hook)


@idea
def unibias(m, cfg):       # control (known trick): learned head bias that starts at ik * log unigram (centred)
    m.hbias = nn.Parameter(torch.zeros(cfg.vocab))

    def hook(train):
        fr = np.bincount(np.asarray(train[:4_000_000]), minlength=cfg.vocab) + 0.5
        lp = np.log(fr / fr.sum())
        with torch.no_grad():
            m.hbias.copy_(torch.from_numpy(cfg.ik * (lp - lp.mean())).float())
    add_hook(m, "data_hook", hook)
    m.head_hook = lambda x: m.head(m.norm(x)) + m.hbias


# ================================================================ family 5: weight structure at init
def _block_mats(m):
    for b in m.blocks:
        yield b.attn.qkv.weight
        yield b.attn.o.weight
        f = b.ffn
        if getattr(f, "shared", None) is not None:
            for mod in f.shared.modules():
                if isinstance(mod, nn.Linear):
                    yield mod.weight


def _expert_mats(m):
    for b in m.blocks:
        f = b.ffn
        for nm in ("w1", "w3", "w2"):
            p = getattr(f, nm, None)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                for e in range(p.shape[0]):
                    yield p.data[e]


@idea
def powerspec(m, cfg):     # every block matrix: random singular vectors, singular values ~ i^-ik, Frobenius kept
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for W in list(_block_mats(m)) + list(_expert_mats(m)):
            o, i = W.shape
            r = min(o, i)
            U = torch.linalg.qr(torch.randn(o, r, generator=g))[0]
            Vv = torch.linalg.qr(torch.randn(i, r, generator=g))[0]
            s = torch.arange(1, r + 1).float() ** (-cfg.ik)
            M = (U * s) @ Vv.t()
            W.copy_(M * W.norm() / M.norm())


@idea
def lowrank(m, cfg):       # every block matrix: rank-ik product + 30% noise, Frobenius kept
    g = torch.Generator().manual_seed(9)
    r = int(cfg.ik)
    with torch.no_grad():
        for W in list(_block_mats(m)) + list(_expert_mats(m)):
            o, i = W.shape
            M = torch.randn(o, r, generator=g) @ torch.randn(r, i, generator=g)
            M = M / M.norm() * math.sqrt(0.91) + torch.randn(o, i, generator=g) / math.sqrt(o * i) * 0.3
            W.copy_(M / M.norm() * W.norm())


@idea
def zerokey(m, cfg):       # W_k = 0 (ik 1: W_q = 0 too): attention starts as a causal mean
    with torch.no_grad():
        for b in m.blocks:
            d = cfg.d_model
            b.attn.qkv.weight[d:2 * d].zero_()
            if cfg.ik == 1:
                b.attn.qkv.weight[:d].zero_()


@idea
def sharedqk(m, cfg):      # W_k = a W_q + sqrt(1-a^2) noise: queries start matching similar keys (ik = a)
    a = cfg.ik
    with torch.no_grad():
        for b in m.blocks:
            d = cfg.d_model
            W = b.attn.qkv.weight
            W[d:2 * d].copy_(a * W[:d] + math.sqrt(max(0.0, 1 - a * a)) * W[d:2 * d])


@idea
def ffnmirror(m, cfg):     # FFN down-projection starts as ik * (up-projection)^T (shared and routed experts)
    with torch.no_grad():
        for b in m.blocks:
            f = b.ffn
            sh = getattr(f, "shared", None)
            if sh is not None and hasattr(sh, "w2"):
                s = sh.w2.weight.std()
                sh.w2.weight.copy_(cfg.ik * sh.w3.weight.t() / sh.w3.weight.std() * s)
            if isinstance(getattr(f, "w2", None), nn.Parameter) and f.w2.dim() == 3:
                s = f.w2.std()
                f.w2.copy_(cfg.ik * f.w3.transpose(1, 2) / f.w3.std() * s)


# ================================================================ round 2: output-norm (Peri-LN) family
def _peri(m, cfg, make_out, make_in=None):
    """turn every block into x + no(F(n(x))) with custom output (and optional input) norms"""
    for b in m.blocks:
        b.mode = "peri"
        b.no1, b.no2 = make_out(m), make_out(m)
        if make_in is not None:
            b.n1, b.n2 = make_in(m), make_in(m)
    if getattr(cfg, "peri_emb", True) and not hasattr(m, "emb_norm"):
        m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)


@idea
def ratio_out(m, cfg):     # branch output = Norm(F) * g * rms(x)/ref: write size follows the residual's size (ik: g init)
    d = cfg.d_model

    def fin(s, x):
        s.ctx["rin"] = ms(x).sqrt()
        return rmsn(x) * s.w

    def fout(s, x):
        r = s.ctx["rin"]
        return rmsn(x) * s.w * (r / _ref(s, r.pow(2).expand_as(x[..., :1])))
    def mo(mm):
        n = Norm(mm, d, fout, ema=Ema(0.05))
        nn.init.constant_(n.w, cfg.ik or 1.0)
        return n
    _peri(m, cfg, mo, lambda mm: Norm(mm, d, fin))


@idea
def outcap(m, cfg):        # output norm only shrinks: F / max(rms F, c ref)
    c = cfg.ik
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model,
                                  lambda s, x: x / torch.maximum(ms(x).sqrt(), c * _ref(s, x)) * s.w, ema=Ema(0.05)))


@idea
def outdetach(m, cfg):     # Peri-LN with the output-norm denominator detached (ik 1: input norm detached too)
    f = lambda s, x: x * torch.rsqrt(ms(x).detach() + 1e-6) * s.w
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, f),
          (lambda mm: Norm(mm, cfg.d_model, f)) if cfg.ik == 1 else None)


@idea
def outgain0(m, cfg):      # Peri-LN with output-norm gains starting at ik (branches start small but normalized)
    def mo(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w)
        nn.init.constant_(n.w, cfg.ik)
        return n
    _peri(m, cfg, mo)


@idea
def outscalar(m, cfg):     # Peri-LN whose output norm has one scalar gain (init ik) instead of a vector
    def mo(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w[:1])
        nn.init.constant_(n.w, cfg.ik or 1.0)
        return n
    _peri(m, cfg, mo)


@idea
def embnorm(m, cfg):       # control: bed pre-norm + RMSNorm on the embedding output only (gain init ik)
    m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)
    nn.init.constant_(m.emb_norm.w, cfg.ik or 1.0)


@idea
def peri_attn(m, cfg):     # diagnostic: Peri-LN output norm after attention only (FFN write unnormalized)
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w))
    for b in m.blocks:
        b.no2 = nn.Identity()


@idea
def peri_ffn(m, cfg):      # diagnostic: Peri-LN output norm after the FFN/MoE only (attention write unnormalized)
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w))
    for b in m.blocks:
        b.no1 = nn.Identity()


@idea
def outsoft(m, cfg):       # smooth output norm F / sqrt(ms F + (c ref)^2): small writes pass, large ones are normalized (ik = c)
    c = cfg.ik
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model,
                                  lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05)))


# ---------------------------------------------------------------- round 2 (standard init): output-norm placement
def _rmsw(init):
    def mk(mm):
        n = Norm(mm, mm.cfg.d_model, lambda s, x: rmsn(x) * s.w)
        nn.init.constant_(n.w, init)
        return n
    return mk


@idea
def outdepth(m, cfg):      # Peri-LN with output gains init ik/sqrt(2L) (scaled-residual rule applied to normalized writes)
    _peri(m, cfg, _rmsw(cfg.ik / math.sqrt(2 * len(m.blocks))))


@idea
def resscale(m, cfg):      # Peri-LN with output gains init ik/sqrt(layer index + 1): later layers write less at start
    _peri(m, cfg, _rmsw(1.0))
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            b.no1.w.fill_(cfg.ik / math.sqrt(i + 1)); b.no2.w.fill_(cfg.ik / math.sqrt(i + 1))


@idea
def outlast(m, cfg):       # Peri-LN output norms in the last block only (+ embedding norm); ik 1: no embedding norm
    if cfg.ik == 1:
        cfg.peri_emb = False
    _peri(m, cfg, _rmsw(1.0))
    for b in m.blocks[:-1]:
        b.no1, b.no2 = nn.Identity(), nn.Identity()


@idea
def outfirst(m, cfg):      # Peri-LN output norms in the first block only (+ embedding norm); ik 1: no embedding norm
    if cfg.ik == 1:
        cfg.peri_emb = False
    _peri(m, cfg, _rmsw(1.0))
    for b in m.blocks[1:]:
        b.no1, b.no2 = nn.Identity(), nn.Identity()


@idea
def outmean(m, cfg):       # Peri-LN whose output norms also remove the mean (LayerNorm-style writes; ik 1: input norms too)
    f = lambda s, x: rmsn(x - x.mean(-1, keepdim=True)) * s.w
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, f), (lambda mm: Norm(mm, cfg.d_model, f)) if cfg.ik == 1 else None)


@idea
def nofinal(m, cfg):       # Peri-LN without the final norm (writes are already bounded); ik = scalar logit gain init
    _peri(m, cfg, _rmsw(1.0))
    m.norm = Norm(m, 1, lambda s, x: x * s.w)
    nn.init.constant_(m.norm.w, cfg.ik or 1.0)


@idea
def embscalar(m, cfg):     # Peri-LN whose embedding norm has one scalar gain (init ik)
    _peri(m, cfg, _rmsw(1.0))
    m.emb_norm = Norm(m, 1, lambda s, x: rmsn(x) * s.w)
    nn.init.constant_(m.emb_norm.w, cfg.ik or 1.0)


@idea
def embgain(m, cfg):       # Peri-LN with the embedding-norm gain vector starting at ik
    _peri(m, cfg, _rmsw(1.0))
    nn.init.constant_(m.emb_norm.w, cfg.ik)


# ---------------------------------------------------------------- round 2: vocabulary-side centring
@idea
def headcenter(m, cfg):    # after every step subtract ik * the mean head row (the shared logit direction); grows with vocab
    @torch.no_grad()
    def f():
        W = m.head.weight
        W.sub_(cfg.ik * W.mean(0, keepdim=True))
    add_hook(m, "init_step_hook", f)


@idea
def embcenter(m, cfg):     # after every step subtract ik * the mean embedding row (common component of all inputs)
    @torch.no_grad()
    def f():
        W = m.emb.weight
        W.sub_(cfg.ik * W.mean(0, keepdim=True))
    add_hook(m, "init_step_hook", f)


# ---------------------------------------------------------------- round 2 explore: attention / input-norm / residual
@idea
def vnorm(m, cfg):         # RMSNorm on attention values per head (gain init ik): bounds what each head can write
    for b in m.blocks:
        a = b.attn
        a.vn_w = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.v_fn = lambda a, v: rmsn(v) * a.vn_w


@idea
def attemp(m, cfg):        # learnable per-head query temperature, init ik (attention sharpness starts at ik x default)
    for b in m.blocks:
        a = b.attn
        a.tq = nn.Parameter(torch.full((a.h, 1, 1), float(cfg.ik)))
        prev = a.qk_fn if a.qk not in ("none", "full", "head") else None
        a.qk = "attemp"
        a.qk_fn = (lambda a, q, k, prev=prev: (lambda qq, kk: (qq * a.tq, kk))(*(prev(a, q, k) if prev else (q, k))))


@idea
def ingain(m, cfg):        # block input-norm gains start at ik (pre-norm or Peri)
    with torch.no_grad():
        for b in m.blocks:
            for nm in ("n1", "n2"):
                w = getattr(getattr(b, nm), "weight", None)
                if w is None:
                    w = getattr(b, nm).w
                w.fill_(cfg.ik)


@idea
def insoft(m, cfg):        # smooth input norms x / sqrt(ms x + (c ref)^2): tiny inputs are not blown up (ik = c)
    c = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model,
                                       lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05)))


@idea
def headzero(m, cfg):      # control (known practice): head starts at ik x its usual init (ik 0 = zero head)
    with torch.no_grad():
        m.head.weight.mul_(cfg.ik)


@idea
def resfix(m, cfg):        # residual rms held at ik x its block-input rms: x_out = rmsn(x_out) * ik * rms(x_in) (no growth)
    for b in m.blocks:
        b.register_forward_hook(lambda mod, inp, out: rmsn(out) * (cfg.ik * ms(inp[0]).sqrt()))


@idea
def dimstd_blk(m, cfg):    # block input norms: per-dim standardization with causal stats blended in (ik), as dimstd_fin
    lam, d, n0 = cfg.ik, cfg.d_model, 8.0

    def f(s, x):
        y = rmsn(x)
        mu_e = s.em.upd(y.mean((0, 1)), s.training)
        va_e = s.ev.upd(y.var((0, 1)), s.training)
        n = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype)[:, None]
        mu = (n0 * mu_e + y.cumsum(1)) / (n0 + n)
        m2 = (n0 * (va_e + mu_e ** 2) + (y * y).cumsum(1)) / (n0 + n)
        z = (y - mu) * torch.rsqrt((m2 - mu * mu).clamp(min=1e-4))
        return ((1 - lam) * y + lam * z) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, d, f, em=Ema(0.05), ev=Ema(0.05)))


# ---------------------------------------------------------------- round 2c: more output-norm forms (on Peri-LN)
@idea
def outlp(m, cfg):         # Peri-LN output norm by the Lp norm (Gaussian-calibrated), p = ik
    p = cfg.ik
    kp = (2 ** (p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)) ** (1 / p)
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model,
                                  lambda s, x: x / ((x.abs().pow(p).mean(-1, keepdim=True) + 1e-12) ** (1 / p) / kp) * s.w))


@idea
def outmax(m, cfg):        # Peri-LN output norm by max |F| (L-infinity) times ik (bounds the largest coordinate write)
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model,
                                  lambda s, x: x / (x.abs().amax(-1, keepdim=True) + 1e-6) * cfg.ik * s.w))


@idea
def outgroup(m, cfg):      # Peri-LN output norm per group of ik channels (group RMS), so one group cannot drown the others
    G = int(cfg.ik)

    def f(s, x):
        sh = x.shape
        xg = x.reshape(*sh[:-1], sh[-1] // G, G)
        return (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(sh) * s.w
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, f))


@idea
def tiedgain(m, cfg):      # Peri-LN with each branch's input and output norm sharing one gain vector (init ik)
    _peri(m, cfg, _rmsw(1.0))
    for b in m.blocks:
        for ni, no in (("n1", "no1"), ("n2", "no2")):
            n = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)
            nn.init.constant_(n.w, cfg.ik or 1.0)
            setattr(b, ni, n)
            o = Norm(m, cfg.d_model, lambda s, x, n=n: rmsn(x) * n.w)
            del o.w
            setattr(b, no, o)


# ---------------------------------------------------------------- round 3 explore: every-step mechanisms (on Peri-LN)
@idea
def outpart(m, cfg):       # Peri-LN output norm partial: F / (rms^p ref^(1-p)), p = ik (p 0: one running scale, per-token size kept)
    p = cfg.ik
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model,
                                  lambda s, x: x / ((ms(x) + 1e-12) ** (p / 2) * _ref(s, x) ** (1 - p)) * s.w, ema=Ema(0.05)))


class _TokGrad(torch.autograd.Function):
    """forward identity; backward rescales each token's gradient toward the batch-mean token-gradient rms (blend k)"""
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        r = ms(g).sqrt()
        sc = (r.mean() / (r + 1e-12)) ** ctx.k
        return g * sc, None


@idea
def tokgrad(m, cfg):       # Peri-LN branch inputs: per-token gradient sizes equalized in backward (blend ik): rare/odd tokens don't dominate
    k = cfg.ik
    for b in m.blocks:
        for nm in ("n1", "n2"):
            old = getattr(b, nm)
            n = Norm(m, cfg.d_model, lambda s, x: _TokGrad.apply(s.inner(x), k))
            del n.w
            n.inner = old                                        # registered submodule: its gain still trains
            setattr(b, nm, n)


@idea
def poscenter(m, cfg):     # block input norms see x minus ik * its causal running mean over earlier positions (per dim)
    c = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(x - c * cummean(x)) * s.w))


@idea
def finprefix(m, cfg):     # final norm denominator rms^(1-ik) * (causal prefix mean rms)^ik: a token's logit scale keeps its size relative to its context
    a = cfg.ik

    def f(s, x):
        r = (ms(x) + 1e-6).sqrt()
        return x / (r ** (1 - a) * cummean(r) ** a) * s.w
    m.norm = Norm(m, cfg.d_model, f)


@idea
def wproj(m, cfg):         # after every step, rows of every block matrix pulled toward their init norm (blend ik): fixed weight norms, free directions
    ws = [mod.weight for b in m.blocks for mod in b.modules() if isinstance(mod, nn.Linear)]
    ws += [p for b in m.blocks for n, p in b.named_parameters() if p.dim() == 3]          # stacked expert weights
    tgt = [w.detach().norm(dim=-1, keepdim=True).mean().item() for w in ws]
    k = cfg.ik

    @torch.no_grad()
    def proj():
        for w, t in zip(ws, tgt):
            nrm = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            w.mul_((t / nrm) ** k)
    add_hook(m, "init_step_hook", proj)


@idea
def sharegain(m, cfg):     # Peri-LN with one output-norm gain vector shared by all branches (init ik): one global write scale per dim
    g = nn.Parameter(torch.full((cfg.d_model,), float(cfg.ik or 1.0)))
    m.share_g = g

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * mm.share_g)
        del n.w
        return n
    _peri(m, cfg, mk)


@idea
def qkpart(m, cfg):        # partial QK-norm: q, k divided by rms^p (per head, p = ik), gains init 1 (p 1 = QK-norm, 0 = none)
    p = cfg.ik
    for b in m.blocks:
        a = b.attn
        a.qp_w = nn.Parameter(torch.ones(a.dh))
        a.kp_w = nn.Parameter(torch.ones(a.dh))
        a.qk = "part"
        a.qk_fn = lambda a, q, k: (q / (ms(q) + 1e-6) ** (p / 2) * a.qp_w, k / (ms(k) + 1e-6) ** (p / 2) * a.kp_w)


@idea
def outtanh(m, cfg):       # Peri-LN output norm then per-coordinate soft cap c tanh(y / c), c = ik: no single coordinate write can spike
    c = cfg.ik
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, lambda s, x: c * torch.tanh(rmsn(x) / c) * s.w))


@idea
def outlearnp(m, cfg):     # outpart with a learnable exponent per branch (init ik, kept in [0, 1] by a sigmoid)
    p0 = min(max(float(cfg.ik), 0.02), 0.98)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x / ((ms(x) + 1e-12) ** (torch.sigmoid(s.pl) / 2)
                                                     * _ref(s, x) ** (1 - torch.sigmoid(s.pl))) * s.w, ema=Ema(0.05))
        n.pl = nn.Parameter(torch.tensor(math.log(p0 / (1 - p0))))
        return n
    _peri(m, cfg, mk)


@idea
def gainbound(m, cfg):     # Peri-LN; after every step all norm gains clamped to [-ik, ik] (bounded per-dim scales, every step)
    c = cfg.ik
    gs = [mod.w for mod in m.modules() if type(mod).__name__ in ("RMSNorm", "Norm") and isinstance(getattr(mod, "w", None), nn.Parameter)]

    @torch.no_grad()
    def clamp():
        for g in gs:
            g.clamp_(-c, c)
    add_hook(m, "init_step_hook", clamp)


@idea
def tokgrad_out(m, cfg):   # Peri-LN branch outputs: per-token gradient sizes equalized in backward (blend ik)
    k = cfg.ik
    for b in m.blocks:
        for nm in ("no1", "no2"):
            old = getattr(b, nm)
            n = Norm(m, cfg.d_model, lambda s, x: s.inner(_TokGrad.apply(x, k)))
            del n.w
            n.inner = old
            setattr(b, nm, n)


@idea
def dualnorm(m, cfg):      # block input = rmsn(x) w + ik * x / ref: the normalized read plus a magnitude-keeping read
    c = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w + c * x / (_ref(s, x) + 1e-6), ema=Ema(0.05)))


# ---------------------------------------------------------------- round 3: final / embedding norm forms
def _fsoft(c):
    return lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w


def _fcap(c):
    return lambda s, x: x / torch.maximum(ms(x).sqrt(), c * _ref(s, x)) * s.w


def _fpart(p):
    return lambda s, x: x / ((ms(x) + 1e-12) ** (p / 2) * _ref(s, x) ** (1 - p)) * s.w


@idea
def finsoft(m, cfg):       # smooth final norm x / sqrt(ms x + (c ref)^2), c = ik (low-norm tokens get small logits = less confident)
    m.norm = Norm(m, cfg.d_model, _fsoft(cfg.ik), ema=Ema(0.05))


@idea
def fincap(m, cfg):        # final norm x / max(rms, c ref), c = ik
    m.norm = Norm(m, cfg.d_model, _fcap(cfg.ik), ema=Ema(0.05))


@idea
def finpart(m, cfg):       # partial final norm x / (rms^p ref^(1-p)), p = ik: token norm partly sets logit confidence
    m.norm = Norm(m, cfg.d_model, _fpart(cfg.ik), ema=Ema(0.05))


@idea
def embpart(m, cfg):       # Peri-LN embedding norm made partial (p = ik): rare-token embedding size partly kept
    m.emb_norm = Norm(m, cfg.d_model, _fpart(cfg.ik), ema=Ema(0.05))


@idea
def embsoft(m, cfg):       # Peri-LN embedding norm made smooth (c = ik)
    m.emb_norm = Norm(m, cfg.d_model, _fsoft(cfg.ik), ema=Ema(0.05))


@idea
def outnoise(m, cfg):      # train only: Gaussian noise (std ik) added after Peri-LN output norms
    sd = cfg.ik
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, lambda s, x: (rmsn(x) + (sd * torch.randn_like(x) if s.training else 0)) * s.w))


@idea
def finnoise(m, cfg):      # train only: Gaussian noise (std ik) added after the final norm (before the head)
    sd = cfg.ik
    m.norm = Norm(m, cfg.d_model, lambda s, x: (rmsn(x) + (sd * torch.randn_like(x) if s.training else 0)) * s.w)


# ================================================================ round 4 (O33): 28 untried mechanisms + 12 variants
def _att_probs(a, q, k):
    T = q.shape[-2]
    sc = a.att_scale if a.att_scale is not None else math.sqrt(a.dh)
    s = (q @ k.transpose(-1, -2)) / sc
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    return s, mask


@idea
def attcount(m, cfg):      # attention output times n_eff^(ik/2), n_eff = 1/sum p^2: spread-out attention no longer shrinks the read
    k_ = cfg.ik

    def f(a, q, k, v):
        s, mask = _att_probs(a, q, k)
        p = s.masked_fill(mask, float("-inf")).softmax(-1)
        neff = 1.0 / p.pow(2).sum(-1, keepdim=True)
        return (p @ v) * neff.detach() ** (k_ / 2)
    for b in m.blocks:
        b.attn.att_fn = f


@idea
def logitrms(m, cfg):      # attention scores divided by their rms over visible keys, times a learned per-head temperature (init ik)
    for b in m.blocks:
        a = b.attn
        a.lt = nn.Parameter(torch.full((a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            s, mask = _att_probs(a, q, k)
            n = (~mask).sum(-1, keepdim=True).to(s.dtype)
            sm = s.masked_fill(mask, 0.0)
            mu = sm.sum(-1, keepdim=True) / n
            var = ((sm - mu).masked_fill(mask, 0.0).pow(2)).sum(-1, keepdim=True) / n
            z = (s - mu) * torch.rsqrt(var + 1e-4) * a.lt
            return z.masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def headbal(m, cfg):       # per-head attention outputs rescaled toward the mean head rms (blend ik): no head drowns the others in o-proj
    k_ = cfg.ik

    def f(a, y):
        r = ms(y).mean(-2, keepdim=True).add(1e-8).sqrt()            # (B, h, 1, 1) per-head rms over positions
        return y * (r.mean(1, keepdim=True) / r).detach() ** k_
    for b in m.blocks:
        b.attn.o_fn = f


@idea
def qktie(m, cfg):         # QK-norm per head with ONE gain vector shared by q and k (init ik): the logit scale is gain^2
    for b in m.blocks:
        a = b.attn
        a.tg = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.qk = "tie"
        a.qk_fn = lambda a, q, k: (rmsn(q) * a.tg, rmsn(k) * a.tg)


def _peri_fn(m, cfg, post):
    """Peri-LN block forward with a per-branch hook post(b, j, x_in, y_normed) -> write"""
    def fwd(b, x, cos, sin):
        y = b.no1(b.attn(b.n1(x), cos, sin))
        x = x + post(b, 0, x, y)
        y = b.no2(b.ffn(b.n2(x)))
        return x + post(b, 1, x, y)
    for b in m.blocks:
        b.mode, b.fwd = "fn", fwd
        b.no1, b.no2 = RMSN(cfg.d_model), RMSN(cfg.d_model)
    if not hasattr(m, "emb_norm"):
        m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)


class RMSN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rmsn(x) * self.w


@idea
def rmsgate(m, cfg):       # Peri-LN; each write gated by sigmoid(a + u (log rms(x) - log ref)): a branch can stay quiet on small/large streams
    _peri_fn(m, cfg, lambda b, j, x, y: y * torch.sigmoid(b.rg_a[j] + b.rg_u[j] * (0.5 * torch.log(ms(x) + 1e-12)
                                                                                  - torch.log(b.rg_e[j].upd(ms(x).mean().sqrt(), b.training) + 1e-12))))
    for b in m.blocks:
        b.rg_a = nn.Parameter(torch.full((2,), float(cfg.ik)))
        b.rg_u = nn.Parameter(torch.zeros(2))
        b.rg_e = [Ema(0.05), Ema(0.05)]


@idea
def branchbal(m, cfg):     # Peri-LN; per dim, attention and FFN write gains are (cos t, sin t) * sqrt2 * ik: the block trades one for the other
    _peri_fn(m, cfg, lambda b, j, x, y: y * (math.sqrt(2) * cfg.ik) * (torch.cos(b.bb_t) if j == 0 else torch.sin(b.bb_t)))
    for b in m.blocks:
        b.bb_t = nn.Parameter(torch.full((cfg.d_model,), math.pi / 4))


@idea
def sharedrms(m, cfg):     # Peri-LN; FFN input norm denominator = (block-input rms)^a (current rms)^(1-a), a = ik: one shared scale per block
    a_ = cfg.ik

    def fwd(b, x, cos, sin):
        r = (ms(x) + 1e-6) ** (-a_ / 2)
        x = x + b.no1(b.attn(b.n1(x), cos, sin))
        h = x * r * (ms(x) + 1e-6) ** (-(1 - a_) / 2) * b.n2.w
        return x + b.no2(b.ffn(h))
    for b in m.blocks:
        b.mode, b.fwd = "fn", fwd
        b.no1, b.no2 = RMSN(cfg.d_model), RMSN(cfg.d_model)
    if not hasattr(m, "emb_norm"):
        m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)


@idea
def rotgain(m, cfg):       # block-norm gains act in a learned rotated basis: y = ((rmsn(x) R) * w) R^T, R = exp(A - A^T), A init ik * 0.02 randn
    d = cfg.d_model

    def mk(mm):
        n = Norm(mm, d, lambda s, x: ((rmsn(x) @ s.R()) * s.w) @ s.R().t())
        n.A = nn.Parameter(cfg.ik * 0.02 * torch.randn(d, d))
        n.R = lambda n=n: torch.matrix_exp(n.A - n.A.t())
        return n
    swap(m, "blocks", mk)


@idea
def posmax(m, cfg):        # block-norm denominator sqrt(max(ms_t, ik * ms_{t-1})): a token cannot be blown up far past its predecessor
    c = cfg.ik

    def f(s, x):
        q = ms(x)
        qp = torch.cat([q[:, :1], q[:, :-1]], 1)
        return x * torch.rsqrt(torch.maximum(q, c * qp) + 1e-6) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def learneps(m, cfg):      # block norms with a learned epsilon: x / sqrt(ms + softplus(e) ref^2), softplus(e) init ik
    e0 = math.log(math.expm1(max(cfg.ik, 1e-4)))

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + F.softplus(s.e) * _ref(s, x) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
        n.e = nn.Parameter(torch.tensor(e0))
        return n
    swap(m, "blocks", mk)


@idea
def halfrms(m, cfg):       # train only: the rms is measured on a random fraction ik of the dims (stochastic denominator); eval uses all
    fr = cfg.ik

    def f(s, x):
        if s.training:
            keep = (torch.rand(x.shape[-1], device=x.device) < fr).to(x.dtype)
            q = (x.pow(2) * keep).sum(-1, keepdim=True) / keep.sum().clamp_min(1)
        else:
            q = ms(x)
        return x * torch.rsqrt(q + 1e-6) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def mednorm(m, cfg):       # block norms divide by median |x| / 0.6745 (robust scale), blended with rms by ik
    a = cfg.ik

    def f(s, x):
        med = x.abs().median(-1, keepdim=True).values / 0.6745
        r = (ms(x) + 1e-6).sqrt()
        return x / (med.clamp_min(1e-6) ** a * r ** (1 - a)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def trimnorm(m, cfg):      # block norms: rms of x clipped at +-c rms (c = ik): outlier coordinates don't shrink every other coordinate
    c = cfg.ik

    def f(s, x):
        r = (ms(x) + 1e-6).sqrt().detach()
        return x * torch.rsqrt(ms(torch.maximum(torch.minimum(x, c * r), -c * r)) + 1e-6) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def gainmean(m, cfg):      # all norm gains used as w / mean|w| * ik: gains only reallocate scale across dims, total fixed
    for mod in list(m.modules()):
        if type(mod).__name__ in ("RMSNorm",):
            mod.forward = (lambda mod: lambda x: rmsn(x, mod.eps) * mod.w / mod.w.abs().mean().clamp_min(1e-6) * cfg.ik)(mod)


@idea
def gainpull(m, cfg):      # after every step all norm gains pulled toward 1 by a fraction ik (gain decay to 1, not to 0)
    k_ = cfg.ik
    gs = [mod.w for mod in m.modules() if type(mod).__name__ in ("RMSNorm", "Norm", "RMSN") and isinstance(getattr(mod, "w", None), nn.Parameter)]

    @torch.no_grad()
    def pull():
        for g in gs:
            g.lerp_(torch.ones_like(g), k_)
    add_hook(m, "init_step_hook", pull)


@idea
def rowdecor(m, cfg):      # after every step, attention matrices' rows decorrelated: W -= ik * offdiag(W W^T) W / mean diag
    ws = [b.attn.o.weight for b in m.blocks] + [mod.weight for b in m.blocks for n_, mod in b.attn.named_modules()
                                                 if isinstance(mod, nn.Linear) and n_ != "o"]
    k_ = cfg.ik

    @torch.no_grad()
    def dec():
        for w in ws:
            G = w @ w.t()
            dg = G.diagonal().mean().clamp_min(1e-8)
            w.sub_(k_ * (G - torch.diag(G.diagonal())) @ w / dg / w.shape[0])
    add_hook(m, "init_step_hook", dec)


@idea
def logitnorm(m, cfg):     # LM logits divided by their rms over the vocabulary, times a learned scalar temperature (init ik)
    m.lnt = nn.Parameter(torch.tensor(float(cfg.ik)))

    def hh(x):
        z = m.head(m.norm(x))
        z = z - z.mean(-1, keepdim=True)
        return z * torch.rsqrt(ms(z) + 1e-6) * m.lnt
    m.head_hook = hh


@idea
def postemp(m, cfg):       # LM logits times (1 + a log(1 + t)), a learned scalar init ik: confidence can grow with context length
    m.pta = nn.Parameter(torch.tensor(float(cfg.ik)))

    def hh(x):
        T = x.shape[1]
        lt = torch.log1p(torch.arange(T, device=x.device, dtype=x.dtype)) / math.log(T)
        return m.head(m.norm(x)) * (1 + m.pta * lt)[None, :, None]
    m.head_hook = hh


@idea
def normbias(m, cfg):      # block norms subtract a learned per-dim shift before normalizing: y = rmsn(x - b) w (b init 0; ik 1: final too)
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x - s.b) * s.w)
        n.b = nn.Parameter(torch.zeros(cfg.d_model))
        return n
    swap(m, "all" if cfg.ik == 1 else "blocks", mk)


@idea
def dimwrms(m, cfg):       # block norms: denominator sqrt(mean a_i x_i^2), a = softmax(l) * d learned (init uniform; lr scale ik)
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * torch.rsqrt((x.pow(2) * (s.l * cfg.ik).softmax(-1)).sum(-1, keepdim=True) + 1e-6) * s.w)
        n.l = nn.Parameter(torch.zeros(cfg.d_model))
        return n
    swap(m, "blocks", mk)


@idea
def inclip(m, cfg):        # block inputs clipped per coordinate at +-c rms (detached) before the norm, c = ik
    c = cfg.ik

    def f(s, x):
        r = (ms(x) + 1e-6).sqrt().detach()
        return rmsn(torch.maximum(torch.minimum(x, c * r), -c * r)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def coordnorm(m, cfg):     # block norms per coordinate: y_i = x_i sqrt(1+k) / sqrt(ms + k x_i^2), k = ik: big coordinates saturate
    k_ = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: x * math.sqrt(1 + k_) * torch.rsqrt(ms(x) + k_ * x * x + 1e-6) * s.w))


@idea
def powread(m, cfg):       # block norms read a power-compressed stream: rmsn(sign(x) |x/rms|^a), a = ik
    a = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, lambda s, x: rmsn(torch.sign(x) * (rmsn(x).abs() + 1e-6) ** a) * s.w))


class _GScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.c, None


@idea
def depthgrad(m, cfg):     # Peri-LN; the gradient into block l's branches is scaled by (l+1)^-ik (forward unchanged): depth-wise step sizes
    for i, b in enumerate(m.blocks):
        c = float((i + 1) ** -cfg.ik)
        for nm in ("no1", "no2"):
            old = getattr(b, nm)
            n = Norm(m, cfg.d_model, lambda s, x, c=c: s.inner(_GScale.apply(x, c)))
            del n.w
            n.inner = old
            setattr(b, nm, n)


@idea
def gainlr(m, cfg):        # every norm gain reparametrized w = 1 + ik * g (g init 0): gain steps ik times larger/smaller
    k_ = cfg.ik
    for mod in list(m.modules()):
        if type(mod).__name__ == "RMSNorm":
            with torch.no_grad():
                mod.w.zero_()
            mod.forward = (lambda mod: lambda x: rmsn(x, mod.eps) * (1 + k_ * mod.w))(mod)


@idea
def normwarm(m, cfg):      # block norms partial with p ramping 0 -> 1 over the first ik steps: early on, per-token size is kept
    st = {"t": 0}
    T = max(float(cfg.ik), 1.0)

    def f(s, x):
        p = min(st["t"] / T, 1.0)
        if p >= 1.0:
            return rmsn(x) * s.w
        return x / ((ms(x) + 1e-12) ** (p / 2) * _ref(s, x) ** (1 - p)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f, ema=Ema(0.05)))

    def tick():
        st["t"] += 1
    add_hook(m, "init_step_hook", tick)


@idea
def projnorm(m, cfg):      # block norms divide by the rms of P x, P a learned d x r map (r = ik, init orthonormal): size measured in a learned subspace
    r = int(cfg.ik)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x @ s.P) + 1e-6) * s.w)
        q, _ = torch.linalg.qr(torch.randn(cfg.d_model, r))
        n.P = nn.Parameter(q * math.sqrt(cfg.d_model / r) / math.sqrt(cfg.d_model / r))
        return n
    swap(m, "blocks", mk)


@idea
def vcenter(m, cfg):       # attention values minus ik * their causal running mean over positions: heads read deviations from the context average
    c = cfg.ik
    for b in m.blocks:
        b.attn.v_fn = lambda a, v: v - c * cummean(v, 2)


# ---------------------------------------------------------------- round 4 variants of killed ideas (new place / strength)
@idea
def finmag(m, cfg):        # magnorm at the final norm (magnitude kept as a feature for the head), u init ik
    d = cfg.d_model

    def f(s, x):
        lr = 0.5 * torch.log(ms(x) + 1e-12) - torch.log(_ref(s, x) + 1e-12)
        return rmsn(x) * s.w + s.u * lr
    m.norm = Norm(m, d, f, ema=Ema(0.05))
    m.norm.u = nn.Parameter(torch.full((d,), float(cfg.ik)))


@idea
def finlp(m, cfg):         # final norm by the Lp norm (p = ik)
    p = cfg.ik
    kp = (2 ** (p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)) ** (1 / p)
    m.norm = Norm(m, cfg.d_model, lambda s, x: x / ((x.abs().pow(p).mean(-1, keepdim=True) + 1e-12) ** (1 / p) / kp) * s.w)


@idea
def finlatin(m, cfg):      # final norm after subtracting ik * the causal running mean (latinhib at the top)
    lam = cfg.ik
    m.norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x - lam * cummean(x)) * s.w)


@idea
def fingroup(m, cfg):      # final norm per group of ik dims
    G = int(cfg.ik)

    def f(s, x):
        sh = x.shape
        xg = x.reshape(*sh[:-1], sh[-1] // G, G)
        return (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(sh) * s.w
    m.norm = Norm(m, cfg.d_model, f)


@idea
def finseq(m, cfg):        # final norm denominator rms_tok^(1-a) * (causal rms over positions <= t)^a (seqnorm at the top), a = ik
    a = cfg.ik

    def f(s, x):
        q = ms(x)
        return x / ((q + 1e-6) ** ((1 - a) / 2) * (cummean(q) + 1e-6) ** (a / 2)) * s.w
    m.norm = Norm(m, cfg.d_model, f)


@idea
def embcap(m, cfg):        # Peri-LN embedding norm as a cap x / max(rms, c ref) (c = ik)
    m.emb_norm = Norm(m, cfg.d_model, _fcap(cfg.ik), ema=Ema(0.05))


@idea
def embnoise(m, cfg):      # train only: Gaussian noise (std ik) after the Peri-LN embedding norm
    sd = cfg.ik
    m.emb_norm = Norm(m, cfg.d_model, lambda s, x: (rmsn(x) + (sd * torch.randn_like(x) if s.training else 0)) * s.w)


@idea
def qnorm(m, cfg):         # normalize queries only (per head, gain ik): key size = per-key salience (knorm's mirror)
    for b in m.blocks:
        a = b.attn
        a.qn_w = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.qk = "qnorm"
        a.qk_fn = lambda a, q, k: (rmsn(q) * a.qn_w, k)


@idea
def endtie(m, cfg):        # Peri-LN embedding norm and final norm share one gain vector (init ik): the stream's read-in and read-out scales are tied
    g = nn.Parameter(torch.full((cfg.d_model,), float(cfg.ik)))
    m.end_g = g
    m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * m.end_g)
    m.norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * m.end_g)
    del m.emb_norm.w, m.norm.w


@idea
def outrot(m, cfg):        # Peri-LN output-norm gains act in a learned rotated basis (rotgain at the writes), A init ik * 0.02 randn
    d = cfg.d_model

    def mk(mm):
        n = Norm(mm, d, lambda s, x: ((rmsn(x) @ s.R()) * s.w) @ s.R().t())
        n.A = nn.Parameter(cfg.ik * 0.02 * torch.randn(d, d))
        n.R = lambda n=n: torch.matrix_exp(n.A - n.A.t())
        return n
    _peri(m, cfg, mk)


@idea
def embmag(m, cfg):        # Peri-LN embedding norm keeps log-magnitude as a feature (magnorm at the input), u init ik
    d = cfg.d_model

    def f(s, x):
        lr = 0.5 * torch.log(ms(x) + 1e-12) - torch.log(_ref(s, x) + 1e-12)
        return rmsn(x) * s.w + s.u * lr
    m.emb_norm = Norm(m, d, f, ema=Ema(0.05))
    m.emb_norm.u = nn.Parameter(torch.full((d,), float(cfg.ik)))


@idea
def finmed(m, cfg):        # final norm by median |x| / 0.6745 blended with rms (mednorm at the top), blend ik
    a = cfg.ik

    def f(s, x):
        med = x.abs().median(-1, keepdim=True).values / 0.6745
        return x / (med.clamp_min(1e-6) ** a * (ms(x) + 1e-6).sqrt() ** (1 - a)) * s.w
    m.norm = Norm(m, cfg.d_model, f)


# ================================================================ round 5 (O33): 28 untried mechanisms
@idea
def adagain(m, cfg):       # block norms with input-conditioned gains: y = rmsn(x) w (1 + tanh(x^ U) V^T), rank r = ik, V init 0
    r = int(cfg.ik)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: (lambda y: y * s.w * (1 + torch.tanh(y @ s.U) @ s.V.t()))(rmsn(x)))
        n.U = nn.Parameter(torch.randn(cfg.d_model, r) / math.sqrt(cfg.d_model))
        n.V = nn.Parameter(torch.zeros(cfg.d_model, r))
        return n
    swap(m, "blocks", mk)


@idea
def ctxgain(m, cfg):       # block norms with context-conditioned gains from the causal mean of normalized inputs (rank ik, V init 0)
    r = int(cfg.ik)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: (lambda y: y * s.w * (1 + torch.tanh(cummean(y) @ s.U) @ s.V.t()))(rmsn(x)))
        n.U = nn.Parameter(torch.randn(cfg.d_model, r) / math.sqrt(cfg.d_model))
        n.V = nn.Parameter(torch.zeros(cfg.d_model, r))
        return n
    swap(m, "blocks", mk)


@idea
def dirremove(m, cfg):     # block norms remove the stream's component along ik learned directions before normalizing (learned outlier dims out)
    r = int(cfg.ik)

    def f(s, x):
        Q, _ = torch.linalg.qr(s.Ud)
        return rmsn(x - (x @ Q) @ Q.t() * torch.sigmoid(s.c)) * s.w
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.Ud = nn.Parameter(torch.randn(cfg.d_model, r))
        n.c = nn.Parameter(torch.tensor(-2.0))
        return n
    swap(m, "blocks", mk)


@idea
def qkcap(m, cfg):         # q and k per head shrunk only when rms > c (c = ik): free below the cap, bounded logits above
    c = cfg.ik
    for b in m.blocks:
        a = b.attn
        a.qk = "cap"
        a.qk_fn = lambda a, q, k: (q / torch.clamp(ms(q).sqrt() / c, min=1.0), k / torch.clamp(ms(k).sqrt() / c, min=1.0))


@idea
def headscale(m, cfg):     # learned per-head scalar on the attention output before o-proj, init ik (LayerScale per head)
    for b in m.blocks:
        a = b.attn
        a.hs = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))
        a.o_fn = lambda a, y: y * a.hs


@idea
def streamcenter(m, cfg):  # after every block the stream's mean over dims is removed (blend ik): a stream with no common offset
    c = cfg.ik
    for b in m.blocks:
        b.register_forward_hook(lambda mod, inp, out: out - c * out.mean(-1, keepdim=True))


@idea
def streamcap(m, cfg):     # after every block, per-coordinate soft cap of the stream at c * its rms (c = ik): no runaway coordinate
    c = cfg.ik

    def h(mod, inp, out):
        r = (ms(out) + 1e-6).sqrt().detach()
        return c * r * torch.tanh(out / (c * r))
    for b in m.blocks:
        b.register_forward_hook(h)


@idea
def groupgain(m, cfg):     # Peri-LN with every block norm gain shared across groups of ik dims (coarser gains)
    G = int(cfg.ik)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.g.repeat_interleave(G))
        del n.w
        n.g = nn.Parameter(torch.ones(cfg.d_model // G))
        return n
    _peri(m, cfg, mk, mk)


@idea
def signnorm(m, cfg):      # block norms normalize positive and negative parts separately, blended with rmsn by ik
    a = cfg.ik

    def f(s, x):
        p, q = x.clamp(min=0), (-x).clamp(min=0)
        y = (p * torch.rsqrt(ms(p) + 1e-6) - q * torch.rsqrt(ms(q) + 1e-6)) * 0.5
        return ((1 - a) * rmsn(x) + a * rmsn(y)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f))


@idea
def qpow(m, cfg):          # queries rescaled by rms(q)^a, a learned per head (init ik): query magnitude sets sharpness more or less than linearly
    for b in m.blocks:
        a = b.attn
        a.qa = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))
        a.qk = "qpow"
        a.qk_fn = lambda a, q, k: (q * (ms(q) + 1e-6) ** (a.qa / 2), k)


@idea
def periblend(m, cfg):     # each write is (1-g) F + g Norm(F) with g = sigmoid(l) learned per branch (init ik): Peri-LN vs pre-norm learned
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: (lambda g: (1 - g) * x + g * rmsn(x) * s.w)(torch.sigmoid(s.l)))
        n.l = nn.Parameter(torch.tensor(float(cfg.ik)))
        return n
    _peri(m, cfg, mk)


class _WS(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd

    def forward(self, w):
        return (w - w.mean(-1, keepdim=True)) / (w.std(-1, keepdim=True) + 1e-8) * self.sd


@idea
def wstd(m, cfg):          # weight standardization: every block Linear row zero-mean, std fixed at ik x its init std (forward reparam)
    import torch.nn.utils.parametrize as P
    for b in m.blocks:
        for mod in list(b.modules()):
            if isinstance(mod, nn.Linear):
                sd = mod.weight.detach().std().item() * cfg.ik
                P.register_parametrization(mod, "weight", _WS(sd))


@idea
def earlygain(m, cfg):     # block and final norms use a separate gain vector for the first ik positions (sink / no-context tokens)
    k_ = int(cfg.ik)

    def f(s, x):
        y = rmsn(x)
        T = x.shape[1]
        msk = (torch.arange(T, device=x.device) < k_).to(x.dtype)[:, None]
        return y * (s.w * (1 - msk) + s.we * msk)
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.we = nn.Parameter(torch.ones(cfg.d_model))
        return n
    swap(m, "all", mk)


@idea
def ropenorm(m, cfg):      # q, k: each rotary pair normalized to length sqrt(2) g (only the angle carries content), blend ik with raw
    a = cfg.ik

    def pn(t, g):
        sh = t.shape
        tp = t.reshape(*sh[:-1], sh[-1] // 2, 2)
        tn = tp * torch.rsqrt(tp.pow(2).mean(-1, keepdim=True) + 1e-6)
        return ((1 - a) * t + a * tn.reshape(sh)) * g
    for b in m.blocks:
        at = b.attn
        at.rg = nn.Parameter(torch.ones(at.dh))
        at.qk = "ropenorm"
        at.qk_fn = lambda at, q, k: (pn(q, at.rg), pn(k, at.rg))


@idea
def freezenorm(m, cfg):    # after ik steps block norms stop dividing by each token's rms and use the running reference instead
    st = {"t": 0}
    T = float(cfg.ik)

    def f(s, x):
        if st["t"] < T:
            r = _ref(s, x)
            return rmsn(x) * s.w
        return x / (s.ema.v if s.ema.v is not None else _ref(s, x)) * s.w
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f, ema=Ema(0.05)))
    add_hook(m, "init_step_hook", lambda: st.__setitem__("t", st["t"] + 1))


@idea
def normlinear(m, cfg):    # block-norm gain generalized to a low-rank linear map: y = (rmsn(x) w)(I + U V^T), rank ik, V init 0
    r = int(cfg.ik)

    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: (lambda y: y + (y @ s.U) @ s.V.t())(rmsn(x) * s.w))
        n.U = nn.Parameter(torch.randn(cfg.d_model, r) / math.sqrt(cfg.d_model))
        n.V = nn.Parameter(torch.zeros(cfg.d_model, r))
        return n
    swap(m, "blocks", mk)


@idea
def topkwrite(m, cfg):     # Peri-LN writes keep only their top fraction ik of coordinates by |value| (sparse writes), renormalized
    fr = cfg.ik

    def f(s, x):
        k_ = max(1, int(round(fr * x.shape[-1])))
        th = x.abs().topk(k_, -1).values[..., -1:]
        y = x * (x.abs() >= th)
        return rmsn(y) * s.w
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, f))


def _hadamard(d):
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


@idea
def hadgain(m, cfg):       # block-norm gains act in a fixed Hadamard basis (outlier dims spread over all coordinates), blend ik
    H = _hadamard(cfg.d_model)
    a = cfg.ik

    def f(s, x):
        y = rmsn(x)
        return (1 - a) * y * s.w + a * ((y @ s.H) * s.w) @ s.H.t()
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.register_buffer("H", H.clone())
        return n
    swap(m, "blocks", mk)


@idea
def normmix(m, cfg):       # block norms: per-token learned mix of RMS and max-abs denominators, gate from the normalized token (init ik)
    def f(s, x):
        y = rmsn(x)
        g = torch.sigmoid(y @ s.u + s.c)[..., None]
        mx = x.abs().amax(-1, keepdim=True) / math.sqrt(2 * math.log(x.shape[-1]))
        r = (ms(x) + 1e-6).sqrt()
        return x / ((1 - g) * r + g * mx + 1e-6) * s.w
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.u = nn.Parameter(torch.zeros(cfg.d_model))
        n.c = nn.Parameter(torch.tensor(float(cfg.ik)))
        return n
    swap(m, "blocks", mk)


@idea
def qkbias(m, cfg):        # QK-norm per head plus a learned query offset after the norm (init 0): a content-free attention prior per head; ik = gain init
    for b in m.blocks:
        a = b.attn
        a.qg = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.kg = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.qb = nn.Parameter(torch.zeros(1, a.h, 1, a.dh))
        a.qk = "qkbias"
        a.qk_fn = lambda a, q, k: (rmsn(q) * a.qg + a.qb, rmsn(k) * a.kg)


@idea
def ksalience(m, cfg):     # QK-norm per head, plus a_h * log rms(k_raw) added to every score: key magnitude lost by the norm returns as salience (a init ik)
    for b in m.blocks:
        a = b.attn
        a.qg = nn.Parameter(torch.ones(a.dh))
        a.kg = nn.Parameter(torch.ones(a.dh))
        a.ka = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            sal = 0.5 * torch.log(ms(k) + 1e-6).transpose(-1, -2)          # (B, h, 1, T)
            q, k = rmsn(q) * a.qg, rmsn(k) * a.kg
            s_, mask = _att_probs(a, q, k)
            s_ = s_ + a.ka * sal
            return s_.masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def qsharp(m, cfg):        # QK-norm per head, scores times (rms(q_raw)/ref)^a_h: the norm keeps a per-query sharpness knob (a init ik)
    for b in m.blocks:
        a = b.attn
        a.qg = nn.Parameter(torch.ones(a.dh))
        a.kg = nn.Parameter(torch.ones(a.dh))
        a.qa2 = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            rq = ms(q) + 1e-6
            sh = (rq / rq.mean((-2, -1), keepdim=True).detach()) ** (a.qa2 / 2)
            q, k = rmsn(q) * a.qg * sh, rmsn(k) * a.kg
            s_, mask = _att_probs(a, q, k)
            return s_.masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def attpostemp(m, cfg):    # attention scores times (1 + a_h log(1+t)/log T), a_h learned per head (init ik): sharpness grows with context
    for b in m.blocks:
        a = b.attn
        a.pa = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            s_, mask = _att_probs(a, q, k)
            T = q.shape[-2]
            lt = (torch.log1p(torch.arange(T, device=q.device, dtype=q.dtype)) / math.log(T))[:, None]
            return (s_ * (1 + a.pa * lt)).masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def ffnraw(m, cfg):        # Peri-LN, but the FFN reads x / ref (running scale) instead of a per-token rmsn: FFN sees token magnitude (blend ik)
    a = cfg.ik
    for b in m.blocks:
        n = Norm(m, cfg.d_model, lambda s, x: ((1 - a) * rmsn(x) + a * x / (_ref(s, x) + 1e-6)) * s.w, ema=Ema(0.05))
        b.n2 = n


@idea
def squash(m, cfg):        # block norms keep a saturating magnitude: y = rmsn(x) w * r/(r + c ref) * (1 + c), c = ik
    c = cfg.ik

    def f(s, x):
        r = (ms(x) + 1e-6).sqrt()
        return rmsn(x) * s.w * (r / (r + c * _ref(s, x))) * (1 + c)
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model, f, ema=Ema(0.05)))


@idea
def predtemp(m, cfg):      # LM logits times exp(u . x^ + b): a per-token confidence predicted from the final normalized stream (u init 0, b = ik)
    m.pt_u = nn.Parameter(torch.zeros(cfg.d_model))
    m.pt_b = nn.Parameter(torch.tensor(float(cfg.ik)))

    def hh(x):
        y = m.norm(x)
        return m.head(y) * torch.exp(rmsn(x) @ m.pt_u + m.pt_b)[..., None]
    m.head_hook = hh


@idea
def dyntemp(m, cfg):       # attention scores per head times exp(W_h . x^_query): per-token, per-head sharpness from the query token (W init 0, bias ik)
    for b in m.blocks:
        a = b.attn
        a.dt_w = nn.Parameter(torch.zeros(a.h, a.dh))
        a.dt_b = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            s_, mask = _att_probs(a, q, k)
            tq = torch.exp(torch.einsum("bhtd,hd->bht", rmsn(q), a.dt_w)[..., None] + a.dt_b)
            return (s_ * tq).masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def ropegain(m, cfg):      # QK-norm per head with gains initialized by rotary band: pair i gain (1 + ik * (i/(P-1) - 0.5)) (tilt toward slow or fast bands)
    for b in m.blocks:
        a = b.attn
        P_ = a.dh // 2
        g = (1 + cfg.ik * (torch.arange(P_).float() / max(P_ - 1, 1) - 0.5)).repeat_interleave(2)
        a.qg = nn.Parameter(g.clone())
        a.kg = nn.Parameter(g.clone())
        a.qk = "ropegain"
        a.qk_fn = lambda a, q, k: (rmsn(q) * a.qg, rmsn(k) * a.kg)


# ---------------------------------------------------------------- round 5: 5 more untried mechanisms
@idea
def outpow(m, cfg):        # Peri-LN output norm with a learned power: F * rms(F)^-p, p per branch init ik (p=1 Peri-LN, p=0 pre-norm)
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * (ms(x) + 1e-6) ** (-s.p / 2) * s.w)
        n.p = nn.Parameter(torch.tensor(float(cfg.ik)))
        return n
    _peri(m, cfg, mk)


@idea
def headnorm(m, cfg):      # per-head RMSNorm of the attention read before o-proj (gain init ik): every head writes at one size
    for b in m.blocks:
        a = b.attn
        a.hg = nn.Parameter(torch.full((a.dh,), float(cfg.ik)))
        a.o_fn = lambda a, y: rmsn(y) * a.hg


@idea
def streamrms(m, cfg):     # after every block the stream is pulled toward rms = s_l (learned, init its first-batch rms), blend ik
    a = cfg.ik
    for b in m.blocks:
        b.srms = nn.Parameter(torch.tensor(-1.0))

        def h(mod, inp, out):
            r = (ms(out) + 1e-6).sqrt()
            if mod.srms.item() < 0:
                with torch.no_grad():
                    mod.srms.fill_(r.mean().item())
            return (1 - a) * out + a * out / r * mod.srms
        b.register_forward_hook(h)


@idea
def qkeq(m, cfg):          # init W_k = W_q (blend ik with the random W_k): attention starts as self-similarity, each token attends to similar tokens
    with torch.no_grad():
        for b in m.blocks:
            W = b.attn.qkv.weight
            d = W.shape[1]
            W[d:2 * d] = (1 - cfg.ik) * W[d:2 * d] + cfg.ik * W[:d]


@idea
def voinit(m, cfg):        # init W_o = W_v^T rescaled to W_o's norm (blend ik): each head starts as a copy of attended tokens
    with torch.no_grad():
        for b in m.blocks:
            a = b.attn
            d = a.qkv.weight.shape[1]
            Wv = a.qkv.weight[2 * d:]
            Wo = a.o.weight
            t = Wv.t() * (Wo.norm() / Wv.norm())
            Wo.copy_((1 - cfg.ik) * Wo + cfg.ik * t)


# ---------------------------------------------------------------- round 5: changed variants of round-2 losers (O33)
@idea
def outdetach2(m, cfg):    # outdetach variant: output-norm denominator only partly detached, ms = (1-a) ms + a sg(ms), a = ik
    a = cfg.ik
    f = lambda s, x: (lambda v: x * torch.rsqrt((1 - a) * v + a * v.detach() + 1e-6) * s.w)(ms(x))
    _peri(m, cfg, lambda mm: Norm(mm, cfg.d_model, f))


@idea
def outfirst2(m, cfg):     # outfirst/outlast variant: output norms on every block except the last (last block writes raw); ik 0: no embedding norm
    if cfg.ik == 0:                 # ik 0: also no embedding norm
        cfg.peri_emb = False
    _peri(m, cfg, _rmsw(1.0))
    b = m.blocks[-1]
    b.no1, b.no2 = nn.Identity(), nn.Identity()


@idea
def outscalar2(m, cfg):    # outscalar variant: per-branch scalar gain in log space, w = exp(l) * vector gain, l init log ik
    def mo(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * s.w * torch.exp(s.l))
        n.l = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        return n
    _peri(m, cfg, mo)


@idea
def outcap2(m, cfg):       # outcap variant: learned cap per branch, F / max(rms F, exp(l) ref), l init log ik
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x / torch.maximum(ms(x).sqrt(), torch.exp(s.l) * _ref(s, x)) * s.w, ema=Ema(0.05))
        n.l = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        return n
    _peri(m, cfg, mk)


@idea
def outsoft2(m, cfg):      # outsoft variant: learned softness per branch, F / sqrt(ms F + (exp(l) ref)^2), l init log ik
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (torch.exp(s.l) * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
        n.l = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        return n
    _peri(m, cfg, mk)


@idea
def resscale2(m, cfg):     # resscale variant: output gains init ik/(layer+1) (steeper) on FFN writes only, attention writes at 1
    _peri(m, cfg, _rmsw(1.0))
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            b.no2.w.fill_(cfg.ik / (i + 1))


@idea
def outdepth2(m, cfg):     # outdepth variant: attention write gains init ik/sqrt(2L), FFN write gains 1
    _peri(m, cfg, _rmsw(1.0))
    with torch.no_grad():
        for b in m.blocks:
            b.no1.w.fill_(cfg.ik / math.sqrt(2 * len(m.blocks)))


# ================================================================ round 6 (manager 17:50): 8 changed variants, 4 nofinal exploits, 28 untried
# ---- changed variants of killed round-2 ideas (O33)
@idea
def outgainrise(m, cfg):   # outgain0 variant: output gains init ik*(l+1)/L (later layers write more, first layers start small)
    _peri(m, cfg, _rmsw(1.0))
    L = len(m.blocks)
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            b.no1.w.fill_(cfg.ik * (i + 1) / L); b.no2.w.fill_(cfg.ik * (i + 1) / L)


@idea
def vnormfull(m, cfg):     # vnorm variant: values normalized per token across ALL heads together (gain init ik), heads keep relative size
    for b in m.blocks:
        a = b.attn
        a.vf_w = nn.Parameter(torch.full((a.h, 1, a.dh), float(cfg.ik)))
        a.v_fn = lambda a, v: v * torch.rsqrt(v.pow(2).mean((1, 3), keepdim=True) + 1e-6) * a.vf_w


@idea
def attdepth(m, cfg):      # attemp variant: fixed (not learned) score temperature 1 + ik*l/(L-1) per layer, deeper layers sharper
    L = len(m.blocks)
    for i, b in enumerate(m.blocks):
        a = b.attn
        a.td = 1 + cfg.ik * i / max(L - 1, 1)
        a.qk = "attdepth"
        a.qk_fn = lambda a, q, k: (q * a.td, k)


@idea
def insoft2(m, cfg):       # insoft variant: learned softness per input norm, x / sqrt(ms x + (exp(l) ref)^2), l init log ik
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (torch.exp(s.l) * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
        n.l = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        return n
    swap(m, "blocks", mk)


@idea
def ingain_att(m, cfg):    # ingain variant: only the attention input-norm gains start at ik (FFN input gains stay 1)
    with torch.no_grad():
        for b in m.blocks:
            b.n1.w.fill_(cfg.ik)


@idea
def resfixlearn(m, cfg):   # resfix variant: block output rms = exp(l_b) x block input rms, l_b learned per block, init log ik
    for b in m.blocks:
        b.rf_l = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        b.register_forward_hook(lambda mod, inp, out: rmsn(out) * torch.exp(mod.rf_l) * ms(inp[0]).sqrt())


@idea
def tiedinv(m, cfg):       # tiedgain variant: output gain = ik / input gain (shared vector): a branch's net scale is fixed, only its shape learns
    _peri(m, cfg, _rmsw(1.0))
    for b in m.blocks:
        for ni, no in (("n1", "no1"), ("n2", "no2")):
            n = getattr(b, ni)
            o = Norm(m, cfg.d_model, lambda s, x, n=n: rmsn(x) * cfg.ik / (n.w.abs() + 0.05))
            del o.w
            setattr(b, no, o)


@idea
def outmean_ffn(m, cfg):   # outmean variant: mean removal only on FFN (MoE) writes, attention writes plain Peri (ik: gain init)
    _peri(m, cfg, _rmsw(1.0))
    for b in m.blocks:
        b.no2 = Norm(m, cfg.d_model, lambda s, x: rmsn(x - x.mean(-1, keepdim=True)) * s.w)
        nn.init.constant_(b.no2.w, cfg.ik)


# ---- nofinal exploits (final-norm family, <= 30%)
@idea
def nofinal_vec(m, cfg):   # nofinal with a per-dim vector logit gain (init ik) instead of a scalar
    _peri(m, cfg, _rmsw(1.0))
    m.norm = Norm(m, cfg.d_model, lambda s, x: x * s.w)
    nn.init.constant_(m.norm.w, cfg.ik)


@idea
def nofinal_soft(m, cfg):  # final norm -> soft norm x / sqrt(ms x + (c ref)^2): small streams pass raw, large ones normalized (c = ik)
    _peri(m, cfg, _rmsw(1.0))
    c = cfg.ik
    m.norm = Norm(m, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))


@idea
def dytfinal(m, cfg):      # final norm -> DyT (2503.10622) w * tanh(a x), a learned, init ik, on Peri-LN
    _peri(m, cfg, _rmsw(1.0))
    m.norm = Norm(m, cfg.d_model, lambda s, x: torch.tanh(s.a * x) * s.w)
    m.norm.a = nn.Parameter(torch.tensor(float(cfg.ik)))


@idea
def finalblend(m, cfg):    # final = (1-g) x + g rmsn(x), g = sigmoid(l) learned (init l = ik): the model picks how much final norm it keeps
    _peri(m, cfg, _rmsw(1.0))
    m.norm = Norm(m, cfg.d_model, lambda s, x: (lambda g: ((1 - g) * x + g * rmsn(x)) * s.w)(torch.sigmoid(s.l)))
    m.norm.l = nn.Parameter(torch.tensor(float(cfg.ik)))


# ---- untried mechanisms
@idea
def pcaremove(m, cfg):     # block input norms remove the stream's running top principal direction (EMA power iteration, blend ik)
    a = cfg.ik

    def f(s, x):
        if s.training:
            with torch.no_grad():
                xf = x.reshape(-1, x.shape[-1])
                u = xf.t() @ (xf @ s.u) / xf.shape[0]
                s.u.copy_(F.normalize(0.9 * s.u * (s.u @ u) + 0.1 * u, dim=0) if s.u.abs().sum() > 0 else F.normalize(u, dim=0))
        return rmsn(x - a * (x @ s.u)[..., None] * s.u) * s.w
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.register_buffer("u", F.normalize(torch.randn(cfg.d_model), dim=0))
        return n
    swap(m, "blocks", mk)


@idea
def ctxrms(m, cfg):        # block input denominators blend the token's ms with the causal mean ms of its context (blend ik)
    a = cfg.ik
    swap(m, "blocks", lambda mm: Norm(mm, cfg.d_model,
                                      lambda s, x: (lambda q: x * torch.rsqrt((1 - a) * q + a * cummean(q) + 1e-6))(ms(x)) * s.w))


@idea
def shrinknorm(m, cfg):    # block input norms soft-threshold small normalized coordinates, threshold softplus(t) learned (init ik)
    def f(s, x):
        y = rmsn(x)
        t = F.softplus(s.t)
        return torch.sign(y) * F.relu(y.abs() - t) * s.w
    def mk(mm):
        n = Norm(mm, cfg.d_model, f)
        n.t = nn.Parameter(torch.tensor(math.log(math.expm1(cfg.ik))))
        return n
    swap(m, "blocks", mk)


@idea
def signgain(m, cfg):      # block input norms with separate gains for positive and negative normalized coordinates (neg gain init ik)
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: (lambda y: torch.where(y > 0, y * s.w, y * s.wn))(rmsn(x)))
        n.wn = nn.Parameter(torch.full((cfg.d_model,), float(cfg.ik)))
        return n
    swap(m, "blocks", mk)


@idea
def headcompete(m, cfg):   # attention heads normalized, then share a write budget: head h scaled by H softmax(l)_h (l init 0) times ik
    for b in m.blocks:
        a = b.attn
        a.hc = nn.Parameter(torch.zeros(1, a.h, 1, 1))
        a.o_fn = lambda a, y: rmsn(y) * a.h * torch.softmax(a.hc, 1) * cfg.ik


def _shared_gain(m, cfg, names):
    g = nn.Parameter(torch.ones(cfg.d_model))
    m.sh_gain = g
    for b in m.blocks:
        for nm in names:
            n = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * m.sh_gain * s.w)
            del n.w
            n.w = nn.Parameter(torch.tensor(float(cfg.ik)))
            setattr(b, nm, n)


@idea
def gainshare(m, cfg):     # all block input norms share one gain vector across depth, times a per-norm scalar (init ik)
    _shared_gain(m, cfg, ("n1", "n2"))


@idea
def outshare(m, cfg):      # all Peri output norms share one gain vector across depth, times a per-norm scalar (init ik)
    _peri(m, cfg, _rmsw(1.0))
    _shared_gain(m, cfg, ("no1", "no2"))


@idea
def vcap(m, cfg):          # values soft-capped per head, c tanh(v/c), c learned per head (init ik): no single value dominates a read
    for b in m.blocks:
        a = b.attn
        a.vc = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))
        a.v_fn = lambda a, v: a.vc * torch.tanh(v / a.vc)


@idea
def kcenter(m, cfg):       # key s minus the mean of keys 0..s (causal key centring, blend ik): keys carry what is new at s
    for b in m.blocks:
        a = b.attn
        a.qk = "kcenter"
        a.qk_fn = lambda a, q, k: (q, k - cfg.ik * cummean(k, 2))


@idea
def cosattn(m, cfg):       # cosine attention (unit q, k) with a learned log-temperature per head, init log(ik)
    for b in m.blocks:
        a = b.attn
        a.lt = nn.Parameter(torch.full((1, a.h, 1, 1), math.log(cfg.ik)))
        a.qk = "cos"
        a.qk_fn = lambda a, q, k: (F.normalize(q, dim=-1) * torch.exp(a.lt) * math.sqrt(a.dh), F.normalize(k, dim=-1))


@idea
def attcap(m, cfg):        # attention-score soft cap c tanh(s/c) with c learned per head (init ik) (Gemma 2 cap, learned)
    for b in m.blocks:
        a = b.attn
        a.sc_c = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            s_, mask = _att_probs(a, q, k)
            s_ = a.sc_c * torch.tanh(s_ / a.sc_c)
            return s_.masked_fill(mask, float("-inf")).softmax(-1) @ v
        a.att_fn = f


@idea
def dytin(m, cfg):         # block input norms -> DyT w tanh(a x) (2503.10622), a learned per norm, init ik; Peri output norms kept
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: torch.tanh(s.a * x) * s.w)
        n.a = nn.Parameter(torch.tensor(float(cfg.ik)))
        return n
    swap(m, "blocks", mk)


@idea
def qktanh(m, cfg):        # q and k bounded per coordinate: c tanh(q/c), c learned per head (init ik), instead of a norm
    for b in m.blocks:
        a = b.attn
        a.qc = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))
        a.qk = "qktanh"
        a.qk_fn = lambda a, q, k: (a.qc * torch.tanh(q / a.qc), a.qc * torch.tanh(k / a.qc))


def _block_linears(m):
    for b in m.blocks:
        for mod in b.modules():
            if isinstance(mod, nn.Linear):
                yield mod


@idea
def orthinit(m, cfg):      # every block Linear re-initialized orthogonal at ik x its original Frobenius norm
    with torch.no_grad():
        for mod in _block_linears(m):
            W = mod.weight
            fn = W.norm()
            nn.init.orthogonal_(W)
            W.mul_(cfg.ik * fn / W.norm())


@idea
def specinit(m, cfg):      # spectral-condition init (2310.17813): every block Linear scaled to spectral norm ik sqrt(out/in)
    with torch.no_grad():
        for mod in _block_linears(m):
            W = mod.weight
            W.mul_(cfg.ik * math.sqrt(W.shape[0] / W.shape[1]) / torch.linalg.matrix_norm(W, 2))


@idea
def qkzero(m, cfg):        # W_q starts at ik x its init (ik 0: zero queries, uniform attention at step 0)
    with torch.no_grad():
        for b in m.blocks:
            d = b.attn.qkv.weight.shape[1]
            b.attn.qkv.weight[:d].mul_(cfg.ik)


@idea
def embortho(m, cfg):      # embedding columns orthonormal (rows spread evenly over R^d), scaled to the original norm x ik
    with torch.no_grad():
        W = m.emb.weight
        fn = W.norm()
        nn.init.orthogonal_(W)
        W.mul_(cfg.ik * fn / W.norm())


@idea
def tieinit(m, cfg):       # head starts as the embedding (scaled to the head's norm, blend ik), untied afterwards
    with torch.no_grad():
        E, H = m.emb.weight, m.head.weight
        H.copy_((1 - cfg.ik) * H + cfg.ik * E * (H.norm() / E.norm()))


@idea
def mupatt(m, cfg):        # attention scale ik/d_head instead of 1/sqrt(d_head) (muP attention), ik = multiplier
    for b in m.blocks:
        b.attn.att_scale = b.attn.dh / cfg.ik


@idea
def qkgainshare(m, cfg):   # QK-norm per head whose q and k gain vectors are shared by all layers (init ik)
    m.qg_sh = nn.Parameter(torch.full((cfg.d_model // cfg.n_heads,), float(cfg.ik)))
    m.kg_sh = nn.Parameter(torch.full((cfg.d_model // cfg.n_heads,), float(cfg.ik)))
    for b in m.blocks:
        a = b.attn
        a.qk = "qkgs"
        a.qk_fn = lambda a, q, k: (rmsn(q) * m.qg_sh, rmsn(k) * m.kg_sh)


class _RowNorm(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = nn.Parameter(g)

    def forward(self, w):
        return F.normalize(w, dim=1) * self.g[:, None]


@idea
def wnorm_o(m, cfg):       # weight norm (Salimans) on attention o-proj rows: direction and length learned separately (length init ik x init)
    import torch.nn.utils.parametrize as P
    for b in m.blocks:
        W = b.attn.o.weight
        P.register_parametrization(b.attn.o, "weight", _RowNorm(W.detach().norm(dim=1) * cfg.ik))


@idea
def headcos(m, cfg):       # cosine LM head: unit-norm head rows times their mean norm times ik (one learned scale s)
    m.hc_s = nn.Parameter(torch.tensor(float(cfg.ik)))

    def hh(x):
        W = m.head.weight
        Wn = F.normalize(W, dim=1) * W.norm(dim=1).mean().detach()
        return m.norm(x) @ Wn.t() * m.hc_s
    m.head_hook = hh


@idea
def headaxisnorm(m, cfg):  # q and k normalized per dim ACROSS heads (each coordinate shared fairly between heads), blend ik
    a_ = cfg.ik
    for b in m.blocks:
        a = b.attn
        a.qk = "hax"
        f = lambda t: (1 - a_) * t + a_ * t * torch.rsqrt(t.pow(2).mean(1, keepdim=True) + 1e-6)
        a.qk_fn = lambda a, q, k, f=f: (f(q), f(k))


@idea
def softmax1(m, cfg):      # attention softmax with an extra learned null logit per head (init ik): heads may attend to nothing
    for b in m.blocks:
        a = b.attn
        a.nl = nn.Parameter(torch.full((1, a.h, 1, 1), float(cfg.ik)))

        def f(a, q, k, v):
            s_, mask = _att_probs(a, q, k)
            s_ = s_.masked_fill(mask, float("-inf"))
            z = torch.cat([s_, a.nl.expand(*s_.shape[:-1], 1)], -1).softmax(-1)
            return z[..., :-1] @ v
        a.att_fn = f


@idea
def qkcenter(m, cfg):      # q and k mean over head dims removed before the dot product (LayerNorm-style centring, blend ik)
    for b in m.blocks:
        a = b.attn
        a.qk = "qkc"
        a.qk_fn = lambda a, q, k: (q - cfg.ik * q.mean(-1, keepdim=True), k - cfg.ik * k.mean(-1, keepdim=True))


@idea
def logitcap(m, cfg):      # LM logits soft-capped c tanh(z/c), c = exp(l) learned, init ik (Gemma 2 final cap, learned)
    m.lc_l = nn.Parameter(torch.tensor(math.log(cfg.ik)))

    def hh(x):
        z = m.head(m.norm(x))
        c = torch.exp(m.lc_l)
        return c * torch.tanh(z / c)
    m.head_hook = hh


@idea
def gainfloor(m, cfg):     # block input-norm gains used as |w| + ik (w init 1 - ik): no gain can reach zero or flip sign
    def mk(mm):
        n = Norm(mm, cfg.d_model, lambda s, x: rmsn(x) * (s.w.abs() + cfg.ik))
        nn.init.constant_(n.w, 1 - cfg.ik)
        return n
    swap(m, "blocks", mk)


@idea
def stalenorm(m, cfg):     # FFN input norm divides by the BLOCK input's rms (computed before the attention write), blend ik with its own
    a_ = cfg.ik
    for b in m.blocks:
        st = {}

        def f1(s, x, st=st):
            st["r"] = ms(x)
            return rmsn(x) * s.w
        def f2(s, x, st=st):
            return x * torch.rsqrt((1 - a_) * ms(x) + a_ * st["r"] + 1e-6) * s.w
        b.n1, b.n2 = Norm(m, cfg.d_model, f1), Norm(m, cfg.d_model, f2)


# ---------------------------------------------------------------- lab baseline: NAG (arXiv 2606.16112)
def _nag_fwd(b, x, cos, sin):
    R = x
    subs = ((lambda h: b.attn(b.n1(h), cos, sin)), (lambda h: b.ffn(b.n2(h))))
    for j, f in enumerate(subs):
        rho = R.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        Rb = R / rho
        y = f(R)                                         # sublayers keep their own pre-norm, so f(Rbar) = f(R)
        y = y - y.mean(-1, keepdim=True)
        y = y - (y * Rb).sum(-1, keepdim=True) / (Rb * Rb).sum(-1, keepdim=True) * Rb
        g = torch.sigmoid(torch.einsum("btd,cd->btc", Rb, b.nag_w[j]) + b.nag_b[j])
        mm = (g * b.nag_p[j].softmax(-1)).sum(-1, keepdim=True).clamp_min(1e-6) ** b.nag_lb[j].exp()
        R = R + rho * b.nag_a[j] * mm * rmsn(y)
    return R


def _nagv_fwd(b, x, cos, sin):
    """NAG block (same as _nag_fwd) with option switches in b.nv (round 7). Defaults reproduce _nag_fwd exactly."""
    o = b.nv
    R = x
    subs = ((lambda h: b.attn(h, cos, sin), b.n1), (lambda h: b.ffn(h), b.n2))
    for j, (f, n) in enumerate(subs):
        if o.get("skip") == j:                                       # this sublayer is a plain pre-norm add
            R = R + f(n(R))
            continue
        q = R.pow(2).mean(-1, keepdim=True).add(1e-6)
        if j == 0:
            b.nv_q0 = q
        if o.get("stalerho") and j == 1:
            q = (1 - o["stalerho"]) * q + o["stalerho"] * b.nv_q0
        if o.get("rhosoft"):
            q = q + (o["rhosoft"] * b.nv_rref[j].clone()) ** 2
            if b.training:
                with torch.no_grad():
                    b.nv_rref[j] = 0.95 * b.nv_rref[j] + 0.05 * q.mean().sqrt()
        if o.get("rhoref"):
            q = (1 - o["rhoref"]) * q + o["rhoref"] * q.mean().detach()
        if o.get("ctxrho"):
            q = (1 - o["ctxrho"]) * q + o["ctxrho"] * cummean(q)
        if o.get("rhol1"):
            q = (1 - o["rhol1"]) * q + o["rhol1"] * (R.abs().mean(-1, keepdim=True) * 1.2533) ** 2 + 1e-6
        rho = q.sqrt()
        if o.get("rhodet"):
            rho = rho.detach() * (rho / rho.detach()) ** (1 - o["rhodet"])
        Rb = R / rho
        inp = o.get("inp", "norm")
        if o.get("incenter"):
            R_in = R - o["incenter"] * cummean(R)
        else:
            R_in = R
        if o.get("par") and j == 1:
            R_in = (1 - o["par"]) * R_in + o["par"] * b.nv_R0
        if j == 0:
            b.nv_R0 = R
        h = n(R_in) if inp == "norm" else (R / (o.get("refc", 1.0) * b.nv_ref[j].clone()) if inp == "ref" else (1 - o["inb"]) * n(R) + o["inb"] * R)
        if inp == "ref" and b.training:
            with torch.no_grad():
                b.nv_ref[j] = 0.95 * b.nv_ref[j] + 0.05 * rho.mean()
        if o.get("indet"):
            r_ = ms(R_in).add(1e-6).sqrt()
            h = h * (r_ / r_.detach()) ** o["indet"]
        y = f(h)
        c = o.get("center", 1.0)
        if "censub" in o and o["censub"] != j:
            c = 0.0
        if o.get("seqcenter"):
            y = y - c * cummean(y)
        elif o.get("group"):
            G = int(o["group"])
            yg = y.unflatten(-1, (-1, G))
            y = (yg - c * yg.mean(-1, keepdim=True)).flatten(-2)
        else:
            y = y - c * y.mean(-1, keepdim=True)
        pr = o.get("perp", 1.0)
        dirv = F.normalize(cummean(Rb), dim=-1) * math.sqrt(Rb.shape[-1]) if o.get("ctxperp") else Rb
        if hasattr(b, "nv_pg"):
            pr = torch.sigmoid(b.nv_pg[j])
        if "perpsub" in o and o["perpsub"] != j:
            pr = 0.0
        y = y - pr * (y * dirv).sum(-1, keepdim=True) / (dirv * dirv).sum(-1, keepdim=True) * dirv
        gsrc = o["gatey"] * rmsn(y) + (1 - o["gatey"]) * Rb if o.get("gatey") else Rb
        gl = torch.einsum("btd,cd->btc", gsrc, b.nag_w[j]) + b.nag_b[j]
        if hasattr(b, "nv_gp"):
            T_ = R.shape[1]
            gl = gl + (torch.log1p(torch.arange(T_, device=R.device, dtype=R.dtype)) / math.log(T_))[:, None] * b.nv_gp[j]
        g = torch.sigmoid(gl)
        mm = (g * b.nag_p[j].softmax(-1)).sum(-1, keepdim=True).clamp_min(1e-6) ** b.nag_lb[j].exp()
        on = o.get("out", "rms")
        if o.get("yclip"):
            r_ = ms(y).add(1e-6).sqrt() * o["yclip"]
            y = torch.maximum(torch.minimum(y, r_), -r_)
        if o.get("perp2") and j == 1:
            dv = b.nv_y0
            y = y - o["perp2"] * (y * dv).sum(-1, keepdim=True) / (dv * dv).sum(-1, keepdim=True).add(1e-6) * dv
        if on == "rms" and o.get("ydet"):
            r_ = ms(y).add(1e-6).sqrt()
            yn = y / r_.detach() * (r_.detach() / r_) ** (1 - o["ydet"])
        elif on == "rms" and o.get("l1out"):
            r_ = (1 - o["l1out"]) * ms(y).add(1e-6).sqrt() + o["l1out"] * y.abs().mean(-1, keepdim=True).mul(1.2533).add(1e-6)
            yn = y / r_
        elif on == "rms" and o.get("wgroup"):
            yg = y.unflatten(-1, (-1, int(o["wgroup"])))
            yn = rmsn(yg).flatten(-2)
        elif on == "rms":
            yn = rmsn(y)
        elif on == "soft":
            yn = y * torch.rsqrt(ms(y) + (o["c"] * b.nv_yref[j].clone()) ** 2 + 1e-12)
            if b.training:
                with torch.no_grad():
                    b.nv_yref[j] = 0.95 * b.nv_yref[j] + 0.05 * ms(y).mean().sqrt()
        elif on == "cap":
            yn = rmsn(y)
            yn = o["c"] * torch.tanh(yn / o["c"])
        if hasattr(b, "nv_g"):
            yn = yn * b.nv_g[j]
        a = b.nag_a[j] * (b.nv_av[j] if hasattr(b, "nv_av") else 1.0)
        if o.get("acap"):
            a = o["acap"] * torch.tanh(a * mm / o["acap"]) / mm.clamp_min(1e-6)
        sc = rho ** torch.exp(b.nv_p[j]) if hasattr(b, "nv_p") else rho
        if j == 0:
            b.nv_y0 = yn
        if o.get("interp"):
            R = R * (1 - (o["interp"] * a * mm).clamp(max=0.5))
        R = R + sc * a * mm * yn
        if o.get("fixrms"):
            R = R * torch.rsqrt(ms(R) + 1e-6) * b.nv_s[j]
    return R


def _nagv(m, cfg, **opt):
    import types as _t
    nag(m, _t.SimpleNamespace(**{**vars(cfg), "ik": 8}))
    for b in m.blocks:
        b.fwd, b.nv = _nagv_fwd, dict(opt)
        b.register_buffer("nv_ref", torch.ones(2)); b.register_buffer("nv_yref", torch.ones(2)); b.register_buffer("nv_rref", torch.ones(2))
    return m.blocks


@idea
def nv_check(m, cfg):      # parity control: _nagv_fwd with default options must equal NAG
    _nagv(m, cfg)


@idea
def nv_noin(m, cfg):       # NAG with the sublayer input norm blended away (ik: 1 = sublayers read the raw stream R)
    _nagv(m, cfg, inp="blend", inb=cfg.ik)


@idea
def nv_refin(m, cfg):      # NAG, sublayers read R / (ik x running mean rho) (no per-token input norm; token magnitude kept)
    _nagv(m, cfg, inp="ref", refc=cfg.ik or 1.0)


@idea
def nv_nocenter(m, cfg):   # NAG with the write's mean removal scaled by ik (0 = no centring)
    _nagv(m, cfg, center=cfg.ik)


@idea
def nv_perp(m, cfg):       # NAG with the projection off the stream direction scaled by ik (0 = no projection)
    _nagv(m, cfg, perp=cfg.ik)


@idea
def nv_softout(m, cfg):    # NAG with a soft write norm y / sqrt(ms y + (c ref)^2), c = ik: small writes stay small
    _nagv(m, cfg, out="soft", c=cfg.ik)


@idea
def nv_capout(m, cfg):     # NAG with the normalized write soft-capped per coordinate at c (= ik)
    _nagv(m, cfg, out="cap", c=cfg.ik)


@idea
def nv_vecgain(m, cfg):    # NAG with a per-dim gain vector on the normalized write (init ik)
    for b in _nagv(m, cfg):
        b.nv_g = nn.Parameter(torch.full((2, cfg.d_model), float(cfg.ik)))


@idea
def nv_rhopow(m, cfg):     # NAG writes scaled by rho^p, p = exp(l) learned per sublayer, init p = ik (1 = NAG)
    for b in _nagv(m, cfg):
        b.nv_p = nn.Parameter(torch.full((2,), math.log(cfg.ik)))


@idea
def nv_gatey(m, cfg):      # NAG gates read a blend of the normalized write and the stream (blend ik, 1 = write only): the write sizes itself
    _nagv(m, cfg, gatey=cfg.ik)


@idea
def nv_skip(m, cfg):       # NAG on one sublayer only: ik 0 = attention write is a plain pre-norm add, ik 1 = FFN (MoE) write plain
    _nagv(m, cfg, skip=int(cfg.ik))


@idea
def nv_gatebias(m, cfg):   # NAG gate bias init ik instead of 3 (gates start half-open: write size learned from the start)
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_b.fill_(cfg.ik)


@idea
def nv_alpha1(m, cfg):     # NAG with alpha init ik for every sublayer instead of 1/sqrt(l)
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_a.fill_(cfg.ik)


@idea
def nv_alphavec(m, cfg):   # NAG alpha times a learned per-dim vector (init ik)
    for b in _nagv(m, cfg):
        b.nv_av = nn.Parameter(torch.full((2, cfg.d_model), float(cfg.ik)))


@idea
def nv_fixrms(m, cfg):     # NAG stream renormalized after every sublayer to a learned rms s (init ik): a hypersphere-like stream
    for b in _nagv(m, cfg, fixrms=True):
        b.nv_s = nn.Parameter(torch.full((2,), float(cfg.ik)))


@idea
def nv_ctxrho(m, cfg):     # NAG rho blends the token's stream rms with its causal context mean (blend ik)
    _nagv(m, cfg, ctxrho=cfg.ik)


@idea
def nv_seqcenter(m, cfg):  # NAG write centred over positions (causal mean of writes, scaled ik) instead of over dims
    _nagv(m, cfg, seqcenter=True, center=cfg.ik)


@idea
def nv_ctxperp(m, cfg):    # NAG projection removes the causal-mean stream direction instead of the token's own (scaled ik)
    _nagv(m, cfg, ctxperp=True, perp=cfg.ik)


@idea
def nv_attonly(m, cfg):    # NAG on attention only; the FFN (MoE) write is a plain pre-norm add, ik unused
    _nagv(m, cfg, skip=1)


@idea
def nv_ffnonly(m, cfg):    # NAG on the FFN only; attention write is a plain pre-norm add, ik unused
    _nagv(m, cfg, skip=0)


@idea
def nv_groupcenter(m, cfg): # NAG write centred within groups of ik dims (local mean removal) instead of over all dims
    _nagv(m, cfg, group=cfg.ik)


@idea
def nv_rhoref(m, cfg):     # NAG rho blends the token's stream ms with the batch mean ms (blend ik): writes keep per-token stream size less
    _nagv(m, cfg, rhoref=cfg.ik)


@idea
def nv_rhosoft(m, cfg):    # NAG rho = sqrt(ms R + (c ref)^2), c = ik: small streams are not inflated
    _nagv(m, cfg, rhosoft=cfg.ik)


@idea
def nv_incenter(m, cfg):   # NAG sublayers read the stream minus its causal mean over positions (blend ik): context-common part removed
    _nagv(m, cfg, incenter=cfg.ik)


@idea
def nv_acap(m, cfg):       # NAG write coefficient alpha*m soft-capped at c (= ik): no sublayer can dominate the stream
    _nagv(m, cfg, acap=cfg.ik)


@idea
def nv_gatepos(m, cfg):    # NAG gate logits plus w * log(1+t)/log T, w learned per gate direction (init ik): write size follows position
    for b in _nagv(m, cfg):
        b.nv_gp = nn.Parameter(torch.full((2, b.nag_b.shape[1]), float(cfg.ik)))


@idea
def nv_stalerho(m, cfg):   # NAG FFN rho blends in the block-input rho (computed before the attention write), blend ik
    _nagv(m, cfg, stalerho=cfg.ik)


@idea
def nv_projgain(m, cfg):   # NAG projection strength learned per sublayer, sigmoid(l), l init ik
    for b in _nagv(m, cfg):
        b.nv_pg = nn.Parameter(torch.full((2,), float(cfg.ik)))


@idea
def nv_share(m, cfg):      # NAG gate parameters shared between the attention and FFN sublayer of each block, shared gate bias init ik
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_b.fill_(cfg.ik)
        for nm in ("nag_w", "nag_b", "nag_p", "nag_lb"):
            P_ = getattr(b, nm)
            setattr(b, nm, nn.Parameter(P_.detach()[:1].clone()))
        b.nv["share"] = True
        b.fwd = lambda b, x, cos, sin: _nagv_share(b, x, cos, sin)


def _nagv_share(b, x, cos, sin):
    for nm in ("nag_w", "nag_b", "nag_p", "nag_lb"):
        b.__dict__.setdefault("_sh", {})
    w, bb, p_, lb = b.nag_w, b.nag_b, b.nag_p, b.nag_lb
    class _V:
        pass
    v = _V()
    v.nag_w, v.nag_b, v.nag_p, v.nag_lb = w.expand(2, *w.shape[1:]), bb.expand(2, -1), p_.expand(2, -1), lb.expand(2)
    for nm in ("nag_w", "nag_b", "nag_p", "nag_lb"):
        b._parameters.pop(nm)
        object.__setattr__(b, nm, getattr(v, nm))
    try:
        return _nagv_fwd(b, x, cos, sin)
    finally:
        for nm, P_ in (("nag_w", w), ("nag_b", bb), ("nag_p", p_), ("nag_lb", lb)):
            object.__delattr__(b, nm) if nm in b.__dict__ else None
            b._parameters[nm] = P_


@idea
def nv_fixalpha(m, cfg):   # NAG alpha frozen at its init 1/sqrt(l) x ik (write scale not learned)
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_a.mul_(cfg.ik)
        b.nag_a.requires_grad_(False)


@idea
def nv_alphadepth(m, cfg): # NAG alpha init ik/l instead of 1/sqrt(l) (steeper depth decay)
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_a.copy_(cfg.ik / b.nag_a.detach().pow(-2))


# ---------------------------------------------------------------- round 8: more NAG internals, final norm on NAG, init scales on NAG
@idea
def nv_beta(m, cfg):       # NAG gate exponent beta init ik instead of 1 (sharper / softer write gating at start)
    for b in _nagv(m, cfg):
        with torch.no_grad():
            b.nag_lb.fill_(math.log(cfg.ik))


@idea
def nv_rhodet(m, cfg):     # NAG rho gradient blocked (fraction ik): the write scale follows the stream without pulling it
    _nagv(m, cfg, rhodet=cfg.ik)


@idea
def nv_ydet(m, cfg):       # NAG write-norm denominator gradient blocked (fraction ik)
    _nagv(m, cfg, ydet=cfg.ik)


@idea
def nv_indet(m, cfg):      # NAG sublayer input-norm denominator gradient blocked (fraction ik)
    _nagv(m, cfg, indet=cfg.ik)


@idea
def nv_l1out(m, cfg):      # NAG write normalized by a blend (ik) of rms and scaled mean |y| (robust to outlier dims)
    _nagv(m, cfg, l1out=cfg.ik)


@idea
def nv_rhol1(m, cfg):      # NAG rho from a blend (ik) of stream rms and scaled mean |R|
    _nagv(m, cfg, rhol1=cfg.ik)


@idea
def nv_wgroup(m, cfg):     # NAG write normalized per group of ik dims (each group gets unit rms)
    _nagv(m, cfg, wgroup=cfg.ik)


@idea
def nv_perp2(m, cfg):      # NAG FFN write also projected off the attention write direction (scale ik): the two writes stay distinct
    _nagv(m, cfg, perp2=cfg.ik)


@idea
def nv_interp(m, cfg):     # NAG update shrinks the old stream by ik*alpha*m before adding the write (convex-like update)
    _nagv(m, cfg, interp=cfg.ik)


def nv_par(m, cfg):        # NOT RUN (residual's family: which earlier stream a layer reads). NAG FFN reads the block-input stream (blend ik)
    _nagv(m, cfg, par=cfg.ik)


@idea
def nv_yclip(m, cfg):      # NAG write coordinates clipped at ik x their rms before the write norm (outlier dims cannot set the scale)
    _nagv(m, cfg, yclip=cfg.ik)


@idea
def nv_perpsub(m, cfg):    # NAG projection only on one sublayer (ik 0 attention, 1 FFN)
    _nagv(m, cfg, perpsub=int(cfg.ik))


@idea
def nv_censub(m, cfg):     # NAG write centring only on one sublayer (ik 0 attention, 1 FFN)
    _nagv(m, cfg, censub=int(cfg.ik))


@idea
def nv_gbdepth(m, cfg):    # NAG gate bias init ik - 0.5 l over the 2L sublayers (deeper writes start more closed)
    for i, b in enumerate(_nagv(m, cfg)):
        with torch.no_grad():
            for j in range(2):
                b.nag_b[j].fill_(cfg.ik - 0.5 * (2 * i + j))


def _nag_final(m, cfg, fn, **kw):
    import types as _t
    if getattr(cfg, "ik3", "none") != "nag":
        nag(m, _t.SimpleNamespace(**{**vars(cfg), "ik": 8}))
    m.norm = Norm(m, cfg.d_model, fn, **kw)


@idea
def nf_pow(m, cfg):        # NAG, final norm keeps part of the magnitude: rmsn(x) w (rms/ref)^a, a learned, init ik
    a = nn.Parameter(torch.tensor(float(cfg.ik)))
    _nag_final(m, cfg, lambda s, x: rmsn(x) * s.w * ((ms(x) + 1e-6).sqrt() / (_ref(s, x) + 1e-6)) ** s.a, ema=Ema(0.05))
    m.norm.a = a


@idea
def nf_soft(m, cfg):       # NAG, soft final norm x w / sqrt(ms x + (c ref)^2), c = ik: small streams give soft logits
    c = cfg.ik
    _nag_final(m, cfg, lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))


@idea
def nf_partial(m, cfg):    # NAG, final norm on the first (1 - ik) of dims; the rest pass divided by the running ref rms
    d = cfg.d_model
    k = int(round(d * (1 - cfg.ik)))

    def f(s, x):
        r = _ref(s, x)
        return torch.cat([rmsn(x[..., :k]), x[..., k:] / (r + 1e-6)], -1) * s.w
    _nag_final(m, cfg, f, ema=Ema(0.05))


@idea
def nf_ctx(m, cfg):        # NAG, final norm divides by a blend (ik) of token ms and the causal mean ms over positions
    k = cfg.ik
    _nag_final(m, cfg, lambda s, x: x * torch.rsqrt((1 - k) * ms(x) + k * cummean(ms(x)) + 1e-6) * s.w)


@idea
def nf_ema(m, cfg):        # NAG, final norm divides by a blend (ik) of token ms and the running ref ms (batch-level scale)
    k = cfg.ik
    _nag_final(m, cfg, lambda s, x: x * torch.rsqrt((1 - k) * ms(x) + k * _ref(s, x) ** 2 + 1e-6) * s.w, ema=Ema(0.05))


@idea
def nf_detach(m, cfg):     # NAG, final-norm denominator gradient blocked (fraction ik)
    k = cfg.ik

    def f(s, x):
        r = (ms(x) + 1e-6).sqrt()
        return x / r.detach() * (r.detach() / r) ** (1 - k) * s.w
    _nag_final(m, cfg, f)


@idea
def nf_center(m, cfg):     # NAG, final norm is LayerNorm-style: mean over dims removed (blend ik) before the rms
    k = cfg.ik
    _nag_final(m, cfg, lambda s, x: rmsn(x - k * x.mean(-1, keepdim=True)) * s.w)


@idea
def nf_group(m, cfg):      # NAG, final norm per group of ik dims
    G = int(cfg.ik)
    _nag_final(m, cfg, lambda s, x: rmsn(x.unflatten(-1, (-1, G))).flatten(-2) * s.w)


@idea
def nf_cap(m, cfg):        # NAG, final rmsn then per-coordinate soft cap c tanh(x/c), c = ik
    c = cfg.ik
    _nag_final(m, cfg, lambda s, x: c * torch.tanh(rmsn(x) / c) * s.w)


@idea
def nf_mag(m, cfg):        # NAG, final norm plus u (log rms - log ref) added along a learned direction u (init 0): magnitude as a feature (ik unused scale of u init)
    d = cfg.d_model
    _nag_final(m, cfg, lambda s, x: rmsn(x) * s.w + (0.5 * torch.log(ms(x) + 1e-12) - torch.log(_ref(s, x) + 1e-12)) * s.u, ema=Ema(0.05))
    m.norm.u = nn.Parameter(torch.full((d,), float(cfg.ik)) / math.sqrt(d))


@idea
def nf_bias(m, cfg):       # NAG, final norm plus a learned bias vector (LayerNorm-style shift), init N(0, ik^2)
    d = cfg.d_model
    _nag_final(m, cfg, lambda s, x: rmsn(x) * s.w + s.bv)
    m.norm.bv = nn.Parameter(torch.randn(d) * cfg.ik)


def _nag_base(m, cfg):
    import types as _t
    if getattr(cfg, "ik3", "none") != "nag":
        nag(m, _t.SimpleNamespace(**{**vars(cfg), "ik": 8}))


@idea
def ni_emb(m, cfg):        # NAG with embedding init scaled by ik
    _nag_base(m, cfg)
    with torch.no_grad():
        m.emb.weight.mul_(cfg.ik)


@idea
def ni_embrow(m, cfg):     # NAG with every embedding row rescaled to the mean row norm times ik (equal-norm token vectors)
    _nag_base(m, cfg)
    with torch.no_grad():
        w = m.emb.weight
        w.mul_(w.norm(dim=1).mean() * cfg.ik / w.norm(dim=1, keepdim=True).clamp_min(1e-8))


@idea
def ni_head(m, cfg):       # NAG with head init scaled by ik (small: near-uniform first predictions)
    _nag_base(m, cfg)
    with torch.no_grad():
        m.head.weight.mul_(cfg.ik)


@idea
def ni_qk(m, cfg):         # NAG with query and key projection init scaled by ik
    _nag_base(m, cfg)
    with torch.no_grad():
        for b in m.blocks:
            d = b.attn.o.weight.shape[0]
            b.attn.qkv.weight[: 2 * d].mul_(cfg.ik)


@idea
def ni_v(m, cfg):          # NAG with value projection init scaled by ik
    _nag_base(m, cfg)
    with torch.no_grad():
        for b in m.blocks:
            d = b.attn.o.weight.shape[0]
            b.attn.qkv.weight[2 * d:].mul_(cfg.ik)


@idea
def ni_depth(m, cfg):      # NAG with block l Linears scaled by (l+1)^-ik
    _nag_base(m, cfg)
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            for mod in b.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.mul_((i + 1.0) ** -cfg.ik)


# ---------------------------------------------------------------- round 9 (O36: each idea names a measured baseline failure)
def _soft_out(c):
    def mk(mm):
        return Norm(mm, mm.cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (c * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
    return mk


@idea
def osfg(m, cfg):          # leader as one idea: Peri-LN with soft output norms (floor c = ik x running ref rms) and final-norm gain 2
    _peri(m, cfg, _soft_out(cfg.ik))
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def osfg_null(m, cfg):     # wrapper null for osfg: same code path, floor 0 and final gain 1 (= Peri-LN)
    _peri(m, cfg, _soft_out(0.0))


@idea
def os_learnc(m, cfg):     # osfg with the floor c learned per output norm (init ik): each branch chooses how much it normalizes
    def mk(mm):
        n = Norm(mm, mm.cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (s.lc.exp() * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
        n.lc = nn.Parameter(torch.tensor(math.log(cfg.ik)))
        return n
    _peri(m, cfg, mk)
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def os_ctx(m, cfg):        # osfg with the floor from this sequence's causal mean branch rms (no cross-batch running statistic), c = ik
    c = cfg.ik

    def mk(mm):
        return Norm(mm, mm.cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + c * c * cummean(ms(x)).detach() + 1e-12) * s.w)
    _peri(m, cfg, mk)
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def os_in(m, cfg):         # osfg with soft input norms too (floor c = ik): small streams are not inflated on the way in either
    c = cfg.ik
    _peri(m, cfg, _soft_out(c), _soft_out(c))
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def os_fsoft(m, cfg):      # osfg (c 2) with a soft final norm instead of gain 2: x w / sqrt(ms x + (k ref)^2), w init 2, k = ik
    _peri(m, cfg, _soft_out(2.0))
    k = cfg.ik
    m.norm = Norm(m, cfg.d_model, lambda s, x: x * torch.rsqrt(ms(x) + (k * _ref(s, x)) ** 2 + 1e-12) * s.w, ema=Ema(0.05))
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def os_fpow(m, cfg):       # osfg (c 2) with a final norm that keeps part of the magnitude: rmsn(x) w (rms/ref)^a, a learned, init ik; w init 2
    _peri(m, cfg, _soft_out(2.0))
    m.norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w * ((ms(x) + 1e-6).sqrt() / (_ref(s, x) + 1e-6)) ** s.a, ema=Ema(0.05))
    m.norm.a = nn.Parameter(torch.tensor(float(cfg.ik)))
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def nag_embnorm(m, cfg):   # NAG plus an RMS norm on the token embedding (gain init ik): rare-token rows enter at the same size as frequent ones
    _nag_base(m, cfg)
    m.emb_norm = Norm(m, cfg.d_model, lambda s, x: rmsn(x) * s.w)
    nn.init.constant_(m.emb_norm.w, cfg.ik)


@idea
def nag_softfg(m, cfg):    # NAG with the leader's soft floor on its write norm (c = ik) and final-norm gain 2
    _nagv(m, cfg, out="soft", c=cfg.ik)
    with torch.no_grad():
        m.norm.w.fill_(2.0)


@idea
def nag(m, cfg):           # LAB BASELINE NAG, norm-agnostic residual (Figliolia & Millidge, arXiv 2606.16112, Eq. 10-17),
    """ported from the residual search (residual/arena/env/residual.py, gate-0 checked there):
    R' = rho Rbar + rho alpha_l m N_out(f_perp), alpha_l init 1/sqrt(l) over the 2L sublayers, C = ik gate directions
    (default 8), gate bias 3, beta = exp(0) = 1. Bed final norm kept."""
    d, C = cfg.d_model, int(cfg.ik or 8)
    for i, b in enumerate(m.blocks):
        b.mode, b.fwd = "fn", _nag_fwd
        b.nag_a = nn.Parameter(1.0 / torch.tensor([2 * i + 1.0, 2 * i + 2.0]).sqrt())
        b.nag_w = nn.Parameter(torch.randn(2, C, d) / d)
        b.nag_b = nn.Parameter(torch.full((2, C), 3.0))
        b.nag_p = nn.Parameter(torch.zeros(2, C))
        b.nag_lb = nn.Parameter(torch.zeros(2))


@idea
def nagnofinal(m, cfg):    # nofinal's changed variant (O33): NAG baseline (C = 8) with the final norm replaced by one scalar logit gain init ik
    import types as _t
    nag(m, _t.SimpleNamespace(**{**vars(cfg), "ik": 8}))
    m.norm = Norm(m, 1, lambda s, x: x * s.w)
    nn.init.constant_(m.norm.w, cfg.ik or 1.0)


# ================================================================ round 2: corpus-statistic attention init
def _window_logp(train, V, W, n, alpha=0.1, pmi=False):
    """C[a, b] = log P(x_{t+1} = b | a occurs at x_{t-W..t-1}), smoothed, row-centred (dense V x V float32);
    pmi: minus log P(b) (the unigram part, which the bigram path already supplies)"""
    t = np.asarray(train[:n]).astype(np.int64)
    M = np.zeros(V * V, np.float32)
    for g in range(1, W + 1):                       # a at t+1-1-g ... target at t+1
        np.add.at(M, t[:-g - 1] * V + t[g + 1:], 1.0)
    M = torch.from_numpy(M).view(V, V)
    rs = M.sum(1, keepdim=True)
    cs = M.sum(0, keepdim=True)
    M.add_(alpha).log_().sub_(torch.log(rs + alpha * V))
    if pmi:
        M -= torch.log((cs + alpha) / (cs.sum() + alpha * V))
    M -= M.mean(1, keepdim=True)
    return M


@idea
def ctxov(m, cfg):
    """BIGSVD + first-layer attention value/output map from corpus context statistics (ours).
    Needs emb_init=true emb_var=bigsvd. With near-uniform attention the first attention layer outputs
    W_o W_v mean_s n1(e(x_s)); we set M = W_o W_v so that head . M n(e_a) / rms(e) ~ g * kappa * C[a, b],
    kappa = the scale at which the head already reads the bigram. ik = g, window = ik2 (default 8)."""
    V, d = cfg.vocab, cfg.d_model
    Wn = int(cfg.ik2 or 8)

    def hook(train):
        A, H = m.emb.weight.detach().float().cpu().clone(), m.head.weight.detach().float().cpu().clone()   # stats on CPU
        C = _window_logp(train, V, Wn, min(cfg.ht_n, 2_000_000), pmi=getattr(cfg, "ik3", "none") == "pmi")
        An = A * torch.rsqrt(A.pow(2).mean(1, keepdim=True) + 1e-12)          # what n1 gives attention
        # token-frequency-weighted samples (rare rows would dominate a uniform sample)
        smp = torch.from_numpy(np.asarray(train[1_000_000:1_040_001]).astype(np.int64))
        ia, ib = smp[:-1], torch.from_numpy(np.asarray(train[3_000_000:3_040_000]).astype(np.int64))
        re = A[ia].pow(2).mean().sqrt()                                     # typical residual rms at init
        # kappa: how big bigram logits are per unit of bigram log-prob (head . e / rms(e) vs C)
        lg = (H[ib] * An[ia]).sum(1)                                        # bigram logits for random pairs
        kap = lg.std() / C[ia, ib].std()
        Hp, Ap = torch.linalg.pinv(H), torch.linalg.pinv(An)                # d x V each
        M = Hp @ (C.t() @ Ap.t())                                           # d x d: H M An^T ~ C^T
        del C
        M = M * cfg.ik * kap * re
        with torch.no_grad():
            a = m.blocks[0].attn
            a.qkv.weight[2 * d:].copy_(torch.eye(d))                         # W_v = I
            a.o.weight.copy_(M)                                             # W_o = M
            a.qkv.weight[d:2 * d].zero_()                                   # W_k = 0: uniform causal attention
    add_hook(m, "data_hook", hook)
