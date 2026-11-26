"""TARGET SEARCH: training targets / losses. Eval is unchanged (next-token CE through the main head).

cfg.tgt picks the target; knobs ta, tb, tc, td (floats), tm (string), tsched (weight schedule over training).
Training-only modules (MTP modules, extra heads) live in model.tmods and are never used at eval.

Baselines
  ntp    plain next-token cross-entropy
  mtp    DeepSeek-V3 multi-token prediction (arXiv 2412.19437 sec 2.2, eqs 21-25): D sequential modules;
         module k: h'_i = M_k [RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))], h^k = TRM_k(h'), P^k = OutHead(h^k),
         shared embedding and output head, loss L = L_main + mtp_lam / D * sum_k CE^k.
"""
import copy, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S = {}                                   # run state: tables, teacher, etc.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")


# ---------------------------------------------------------------- modules
def make_modules(cfg, ctx):
    from model import Block, RMSNorm
    t, d, V = cfg.tgt, cfg.d_model, cfg.vocab
    m = nn.Module()
    add = False
    if getattr(cfg, "crec", 0) > 0:      # add-on: current-token reconstruction head (stackable)
        m.rnorm, m.rhead = RMSNorm(d), nn.Linear(d, V, bias=False)
        add = True
    if t != "mtp" and cfg.mtp_depth > 0:  # add-on: DeepSeek-V3 MTP modules under another target
        D = cfg.mtp_depth
        m.blocks = nn.ModuleList([Block(cfg, cfg.n_layers + k, ctx) for k in range(D)])
        m.hn = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        m.en = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        m.proj = nn.ModuleList([nn.Linear(2 * d, d, bias=False) for _ in range(D)])
        m.fn = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        add = True
    if getattr(cfg, "wy", "none") in WY_HEAD or getattr(cfg, "wy", "none") in W3_HEAD or getattr(cfg, "wy", "none") in W4_HEAD or getattr(cfg, "wy", "none") in W5_HEAD:   # add-on: round w2 heads (stackable)
        _wy_modules(m, cfg, d)
        add = True
    if t == "mtp":
        D = cfg.mtp_depth
        m.blocks = nn.ModuleList([Block(cfg, cfg.n_layers + k, ctx) for k in range(D)])
        m.hn = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        m.en = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        m.proj = nn.ModuleList([nn.Linear(2 * d, d, bias=False) for _ in range(D)])
        m.fn = nn.ModuleList([RMSNorm(d) for _ in range(D)])
        return m
    if t in HEADV:                       # extra vocab head from the final hidden
        m.norm, m.head = RMSNorm(d), nn.Linear(d, V, bias=False)
        if t == "mtp_lin":
            m.proj = nn.Linear(2 * d, d, bias=False)
        return m
    if t in HEADD:                       # extra d-dim projection
        m.norm, m.lin = RMSNorm(d), nn.Linear(d, d, bias=False)
        return m
    if t in HEAD1:                       # scalar head
        m.norm, m.lin = RMSNorm(d), nn.Linear(d, 1, bias=True)
        return m
    if t == "cluster":
        m.norm, m.lin = RMSNorm(d), nn.Linear(d, 64, bias=False)
        return m
    if t == "mtp_share":
        m.hn, m.en, m.fn = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        m.proj = nn.Linear(2 * d, d, bias=False)
        return m
    if t == "posrec":
        m.norm, m.lin = RMSNorm(d), nn.Linear(d, cfg.seq_len, bias=False)
        return m
    if t == "mtp_lin_tied":
        m.proj, m.fn = nn.Linear(2 * d, d, bias=False), RMSNorm(d)
        return m
    return m if add else None


HEADV = {"skip2", "prev_rec", "cur_rec", "fbce", "mtp_lin", "cur_rec_sg", "cur_rec_all", "cur_rec_l1", "cur_rec_rare",
         "ntp2", "rand_tgt", "bagrec"}
HEADD = {"nexth", "infonce", "embreg", "futsum", "cur_rec_cos", "infonce2", "infonce_cur", "infonce_prev", "ctxmean"}
HEAD1 = {"copyflag", "nextfreq", "tokcount"}


# ---------------------------------------------------------------- tables
def _counts(train, V, n=20_000_000):
    a = np.asarray(train[:n]).astype(np.int64)
    uni = np.bincount(a, minlength=V).astype(np.float64)
    return a, uni


class _CSR:
    """Row lookups of a CSR count table without scipy (the GPU box has none). Same file format as scipy.sparse.save_npz;
    M[rows].toarray() gives the same float32 rows as scipy."""

    def __init__(self, data, indices, indptr, shape):
        self.data, self.indices, self.indptr, self.shape = data, indices, indptr, tuple(int(v) for v in shape)
        self._rows = None

    def __getitem__(self, rows):
        r = _CSR(self.data, self.indices, self.indptr, self.shape)
        r._rows = np.asarray(rows).reshape(-1)
        return r

    def toarray(self):
        rows = self._rows
        lo, hi = self.indptr[rows], self.indptr[rows + 1]
        nrow = (hi - lo).astype(np.int64)
        out = np.zeros((len(rows), self.shape[1]), dtype=self.data.dtype)
        tot = int(nrow.sum())
        if tot:
            rr = np.repeat(np.arange(len(rows)), nrow)
            start = np.repeat(lo.astype(np.int64) - np.concatenate([[0], np.cumsum(nrow)[:-1]]), nrow)
            idx = start + np.arange(tot)
            out[rr, self.indices[idx]] = self.data[idx]
        return out


def bigram_csr(train, V, name):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{name}_bigram.npz")
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    if sp is None:
        if not os.path.exists(f):
            a, _ = _counts(train, V)
            u, c = np.unique(a[:-1] * V + a[1:], return_counts=True)
            r, col = u // V, u % V
            indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=V))]).astype(np.int32)
            np.savez(f, indices=col.astype(np.int32), indptr=indptr, format=np.bytes_(b"csr"),
                     shape=np.array([V, V]), data=c.astype(np.float32))
        z = np.load(f)
        return _CSR(z["data"], z["indices"], z["indptr"], z["shape"])
    if os.path.exists(f):
        return sp.load_npz(f)
    a, _ = _counts(train, V)
    code = a[:-1] * V + a[1:]
    u, c = np.unique(code, return_counts=True)
    M = sp.csr_matrix((c.astype(np.float32), (u // V, u % V)), shape=(V, V))
    sp.save_npz(f, M)
    return M


def clusters(train, V, name, k=64):
    f = os.path.join(CACHE, f"{name}_clusters{k}.npy")
    if os.path.exists(f):
        return np.load(f)
    import scipy.sparse.linalg as sla
    M = bigram_csr(train, V, name)
    # rows = previous-token context vectors of each id (column of the bigram matrix), log-scaled
    C = M.T.tocsr().astype(np.float64)
    C.data = np.log1p(C.data)
    U, s, _ = sla.svds(C, k=32)
    X = U * s
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(0)
    cen = X[rng.choice(V, k, replace=False)]
    for _ in range(25):
        lab = np.argmax(X @ cen.T, 1)
        for j in range(k):
            if (lab == j).any():
                cen[j] = X[lab == j].mean(0)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9
    np.save(f, lab)
    return lab


def setup(model, train, cfg, steps, dev):
    S.clear()
    V = cfg.vocab
    name = "pystd" if len(train) < 30_000_000 else "fineweb"
    t = cfg.tgt
    NGK = {"obig": "obig", "otri": "otri", "oskip": "oskip", "ctrig": "ctrig", "oquad": "oquad",
           "ocache": "obig", "oind": "obig", "obig_unl": "obig",
           "otri_conf": "otri", "otri_ent": "otri", "otri_logit": "otri", "ent_match": "otri",
           "embpath_obig": "obig", "embpath_otri": "otri", "embpath_oquad": "oquad", "embotri_u": "otri", "otri_x": "otri", "oquad_x": "oquad", "wx": "ctrig"}
    NGK.update({m: "ctrig" for m in CC_MODES})
    if t in NGK:
        S["ng"] = NGram(NGK[t], train, cfg, name, dev)
    if t in CC_MODES:
        _cc_setup(train, cfg, name, dev)
    if t in ("bigmid",):
        S["big"] = bigram_csr(train, V, name)
        _, uni = _counts(train, V)
        S["uni"] = torch.from_numpy((uni + 1) / (uni + 1).sum()).float().to(dev)
    if t in ("cur_rec_rare",):
        _, uni = _counts(train, V)
        rk = np.empty(V, dtype=np.int64); rk[np.argsort(-uni)] = np.arange(V)
        S["rare"] = torch.from_numpy(rk >= 256).to(dev)
    if t in ("ls_uni", "freqw", "nextfreq", "hardrare"):
        _, uni = _counts(train, V)
        p = (uni + 1) / (uni + 1).sum()
        S["uni"] = torch.from_numpy(p).float().to(dev)
    if t in ("bigram_kd", "embpath_kd"):
        M = bigram_csr(train, V, name)
        S["big"] = M
        _, uni = _counts(train, V)
        S["uni"] = torch.from_numpy((uni + 1) / (uni + 1).sum()).float().to(dev)
    if t == "cluster":
        S["clu"] = torch.from_numpy(clusters(train, V, name)).long().to(dev)
    if t == "ema_kd":
        te = copy.deepcopy(model)
        for p in te.parameters():
            p.requires_grad_(False)
        S["teacher"] = te
    if getattr(cfg, "wy", "none") != "none":
        _wy_setup(model, train, cfg, dev)
    if t == "wx":
        _wx_setup(train, cfg, name, dev)
    if t == "logit_ema":
        S["rowp"] = torch.full((V, V), 1.0 / V, device=dev) if V <= 8192 else None


def sched(cfg, step, total):
    f = step / max(1, total - 1)
    s = cfg.tsched
    if s == "const":
        return 1.0
    if s == "decay":
        return 1.0 - f
    if s == "warm":
        return f
    if s == "early":
        return 1.0 if f < 0.5 else 0.0
    if s == "late":
        return 0.0 if f < 0.5 else 1.0
    if s == "v3":                         # DeepSeek-V3: lam for the first 68% of tokens, then lam/3
        return 1.0 if f < 0.68 else 1.0 / 3
    if s == "cos":
        return 0.5 * (1 + math.cos(math.pi * f))
    raise ValueError(s)


def soft_term(logits, qfn, B, chunk=4):
    """Mean soft cross-entropy to a teacher q, with low memory: q is built per chunk of sequences and only the
    gradient G = softmax - q (one B x T x V tensor) is kept. Returns a scalar whose value and gradient match
    mean(-(log_softmax(logits) * q).sum(-1))."""
    G = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    val = 0.0
    with torch.no_grad():
        for b0 in range(0, B, chunk):
            lg = logits[b0:b0 + chunk].float()
            q = qfn(b0, min(B, b0 + chunk))
            lp = F.log_softmax(lg, -1)
            val += -(lp * q).sum().item()
            G[b0:b0 + chunk] = lp.exp_() - q
            del lp, q
    n = logits.shape[0] * logits.shape[1]
    sur = (logits.float() * G).sum() / n
    return sur - sur.detach() + val / n


def soft_ce(logits, target_p):
    return -(target_p * F.log_softmax(logits.float(), -1)).sum(-1)


# ---------------------------------------------------------------- loss
def train_loss(model, xy, cfg, step, total):
    """returns (ntp loss for logging, training loss, parts). Adds the stackable add-ons (crec, MTP under another tgt)."""
    wy = getattr(cfg, "wy", "none")
    if wy != "none":
        S["wy"]["frac"] = step / max(1, total)
        _wy_pre(model, xy, cfg)
    ntp, tl, parts = _train_loss(model, xy, cfg, step, total)
    tm, V = model.tmods, cfg.vocab
    h = model.ctx["h"]
    if getattr(cfg, "crec", 0) > 0:
        lg = tm.rhead(tm.rnorm(h)).float()
        aux = F.cross_entropy(lg.reshape(-1, V), xy[:, :-1].reshape(-1))
        tl = tl + cfg.crec * aux
        parts["crec"] = round(aux.item(), 4)
    if cfg.tgt != "mtp" and cfg.mtp_depth > 0:
        tl = tl + cfg.mtp_lam * _mtp(model, xy, cfg, h)
    if getattr(cfg, "ctied", 0) > 0:          # current-token reconstruction through Emb^T, stackable
        aux = F.cross_entropy((model.norm(h) @ model.emb.weight.T).float().reshape(-1, V), xy[:, :-1].reshape(-1))
        tl = tl + cfg.ctied * aux
        parts["ctied"] = round(aux.item(), 4)
    if getattr(cfg, "vseq", 0) > 0:          # variance floor over time inside each sequence
        hn = model.norm(h)
        tl = tl + cfg.vseq * F.relu(cfg.vics - hn.std(1)).mean()
    if getattr(cfg, "vsq", 0) > 0:           # squared-hinge variance floor
        hh = model.norm(h).reshape(-1, h.shape[-1])
        tl = tl + cfg.vsq * (F.relu(cfg.vics - hh.std(0)) ** 2).mean()
    if getattr(cfg, "vhead", 0) > 0:         # variance floor on the output-head rows (spread over the vocab)
        W = model.head.weight
        tl = tl + cfg.vhead * F.relu(cfg.vics / W.shape[1] ** 0.5 - W.std(0)).mean()
    if getattr(cfg, "vraw", 0) > 0:          # variance floor on the final hidden before the norm
        tl = tl + cfg.vraw * F.relu(cfg.vics - h.reshape(-1, h.shape[-1]).std(0)).mean()
    if getattr(cfg, "vic", 0) > 0 or getattr(cfg, "vcov", 0) > 0:   # VICReg variance floor (+ covariance) on final normed hidden
        hh = model.norm(h).reshape(-1, h.shape[-1])
        if cfg.vic > 0:
            tl = tl + cfg.vic * F.relu(cfg.vics - hh.std(0)).mean()
        if cfg.vcov > 0:
            tl = tl + cfg.vcov * _offcov(hh)
    if wy != "none":
        tl = _wy_post(model, xy, cfg, tl, parts)
    return ntp, tl, parts


def _offcov(hh):
    z = hh - hh.mean(0)
    C = (z.T @ z) / (z.shape[0] - 1)
    off = C - torch.diag(torch.diag(C))
    return (off ** 2).sum() / hh.shape[1]


def _rms(hh):
    return hh * torch.rsqrt(hh.pow(2).mean(-1, keepdim=True) + 1e-6)


def _vfloor(hh, s=1.0):
    hh = hh.reshape(-1, hh.shape[-1])
    return F.relu(s - hh.std(0)).mean()


def _mtp(model, xy, cfg, h):
    tm, V = model.tmods, cfg.vocab
    T = xy.shape[1] - 1
    D = cfg.mtp_depth
    hp, extra = h, []
    for k in range(D):
        L = T - 1 - k
        e = model.emb(xy[:, k + 1:k + 1 + L])
        hin = tm.proj[k](torch.cat([tm.hn[k](hp[:, :L]), tm.en[k](e)], -1))
        tm.blocks[k].ffn.cur_ids = xy[:, k + 1:k + 1 + L]
        hk = tm.blocks[k](hin, model.cos[:L], model.sin[:L])
        lg = model.head(tm.fn[k](hk))
        extra.append(F.cross_entropy(lg.float().reshape(-1, V), xy[:, k + 2:k + 2 + L].reshape(-1)))
        hp = hk
    return sum(extra) / D


def shape_teacher(q, cfg):
    if cfg.tc and cfg.tc != 1.0:                  # temperature on the teacher
        q = q ** (1.0 / cfg.tc)
        q = q / q.sum(-1, keepdim=True)
    if cfg.td and cfg.td > 0:                     # keep only the teacher's top-k
        k = int(cfg.td)
        v, i = torch.topk(q, k, -1)
        q = torch.zeros_like(q).scatter(-1, i, v)
        q = q / q.sum(-1, keepdim=True)
    return q


def _train_loss(model, xy, cfg, step, total):
    x, y = xy[:, :-1], xy[:, 1:]
    logits, nll = model(x, y)
    model.ctx["lg"], model.ctx["nll"] = logits, nll
    ntp = nll.mean()
    t = cfg.tgt
    if t == "ntp":
        return ntp, ntp, {}
    w = sched(cfg, step, total)
    a = cfg.ta * w
    c = model.ctx
    h = c["h"]
    B, T = x.shape
    V = cfg.vocab
    tm = model.tmods
    lp = None

    def logp():
        return F.log_softmax(logits.float(), -1)

    if t == "mtp":
        D = cfg.mtp_depth
        hp, extra = h, []
        for k in range(D):
            L = T - 1 - k
            e = model.emb(xy[:, k + 1:k + 1 + L])
            hin = tm.proj[k](torch.cat([tm.hn[k](hp[:, :L]), tm.en[k](e)], -1))
            for mm in model.moe_layers(train=True):
                mm.cur_ids = xy[:, k + 1:k + 1 + L]
            hk = tm.blocks[k](hin, model.cos[:L], model.sin[:L])
            lg = model.head(tm.fn[k](hk))
            extra.append(F.cross_entropy(lg.float().reshape(-1, V), xy[:, k + 2:k + 2 + L].reshape(-1)))
            hp = hk
        mt = sum(extra) / D
        lam = cfg.mtp_lam * w
        return ntp, ntp + lam * mt, {"mtp": round(mt.item(), 4)}

    if t == "mtp_lin":   # MTP depth 1 with a linear map instead of a transformer block, own head
        L = T - 1
        e = model.emb(xy[:, 1:1 + L])
        hk = tm.proj(torch.cat([h[:, :L], e], -1))
        lg = tm.head(tm.norm(hk))
        aux = F.cross_entropy(lg.float().reshape(-1, V), xy[:, 2:2 + L].reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    if t == "mtp_share":  # MTP depth 1 reusing the main model's last block (weight-shared)
        L = T - 1
        e = model.emb(xy[:, 1:1 + L])
        hin = tm.proj(torch.cat([tm.hn(h[:, :L]), tm.en(e)], -1))
        hk = model.blocks[-1](hin, model.cos[:L], model.sin[:L])
        lg = model.head(tm.fn(hk))
        aux = F.cross_entropy(lg.float().reshape(-1, V), xy[:, 2:2 + L].reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    # ---- target-distribution changes (replace the one-hot target)
    if t == "bigram_kd":
        M = S["big"]

        def qb(b0, b1):
            rows = torch.from_numpy(M[x[b0:b1].reshape(-1).cpu().numpy()].toarray()).float().to(logits.device)
            q = (rows + S["uni"]) / (rows.sum(-1, keepdim=True) + 1.0)
            return shape_teacher(q, cfg).view(b1 - b0, T, V)
        st = soft_term(logits, qb, B)
        return ntp, (1 - a) * ntp + a * st, {}
    if t in ("ls", "ls_uni", "bigram_kd", "self_boot", "fsmooth", "ctxbig", "logit_ema"):
        lp = logp()
        if t == "ls":
            tl = (1 - a) * nll + a * (-lp.mean(-1))
        elif t == "ls_uni":
            tl = (1 - a) * nll + a * (-(lp * S["uni"]).sum(-1))
        elif t == "bigram_kd":
            M = S["big"]
            rows = M[x.reshape(-1).cpu().numpy()].toarray()
            rows = torch.from_numpy(rows).to(logits.device)
            al = 1.0
            q = (rows + al * S["uni"]) / (rows.sum(-1, keepdim=True) + al)
            q = shape_teacher(q, cfg)
            tl = (1 - a) * nll + a * (-(lp.reshape(-1, V) * q).sum(-1)).view(B, T)
        elif t == "self_boot":
            q = lp.detach().exp()
            tl = (1 - a) * nll + a * (-(lp * q).sum(-1))
        elif t == "fsmooth":   # mix target with a bag of the following k tokens (weights gamma^j)
            k, g = int(cfg.tb or 4), (cfg.tc or 0.7)
            fut = torch.zeros(B, T, V, device=x.device)
            for j in range(1, k + 1):
                idx = torch.cat([xy[:, 1 + j:], xy[:, -1:].expand(B, j)], 1)[:, :T]
                valid = (torch.arange(T, device=x.device) + 1 + j <= T).float()
                fut.scatter_add_(2, idx.unsqueeze(-1), (g ** (j - 1) * valid).expand(B, T).unsqueeze(-1).contiguous())
            fut = fut / fut.sum(-1, keepdim=True).clamp_min(1e-9)
            tl = (1 - a) * nll + a * (-(lp * fut).sum(-1))
        elif t == "ctxbig":     # in-context bigram: tokens that followed the current token earlier in the sequence
            same = (x.unsqueeze(2) == x.unsqueeze(1))                  # [B, i, j] x_j == x_i
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
            m = (same & causal).float()                                # j < i
            q = torch.zeros(B, T, V, device=x.device)
            q.scatter_add_(2, y.unsqueeze(1).expand(B, T, T), m)       # count y_j over j
            n = q.sum(-1, keepdim=True)
            has = (n.squeeze(-1) > 0).float()
            q = q / n.clamp_min(1)
            tl = nll + a * has * ((-(lp * q).sum(-1)) - nll)
        elif t == "logit_ema":  # per-input-token running average of the model's own predictions
            R = S["rowp"]
            ids = x.reshape(-1)
            q = R[ids].view(B, T, V)
            tl = (1 - a) * nll + a * (-(lp * q).sum(-1))
            with torch.no_grad():
                u, inv = ids.unique(return_inverse=True)
                pm = torch.zeros(len(u), V, device=x.device).index_add_(0, inv, lp.detach().exp().reshape(-1, V))
                cnt = torch.bincount(inv, minlength=len(u)).float()
                R[u] = 0.9 * R[u] + 0.1 * pm / cnt[:, None]
        return ntp, tl.mean(), {}

    if t == "ema_kd":
        te = S["teacher"]
        with torch.no_grad():
            dec = cfg.tb or 0.99
            for pt, ps in zip(te.parameters(), model.parameters()):
                pt.mul_(dec).add_(ps.detach(), alpha=1 - dec)
            tq = F.softmax(te(x).float(), -1)
        lp = logp()
        tl = (1 - a) * nll + a * (-(lp * tq).sum(-1))
        return ntp, tl.mean(), {}

    # ---- per-token reweighting
    if t in ("focal", "freqw", "trunc", "hardw", "posw", "losscurr", "seqdro", "hardrare"):
        if t == "focal":
            p = (-nll).exp()
            tl = ((1 - p) ** cfg.ta * nll).mean()
        elif t == "freqw":
            wt = S["uni"][y] ** (-a)
            tl = (wt / wt.mean() * nll).mean()
        elif t in ("trunc", "losscurr"):
            q = a if t == "trunc" else cfg.ta * (1 - step / max(1, total - 1))
            n = nll.reshape(-1)
            kk = int(n.numel() * (1 - q))
            keep = torch.topk(n.detach(), kk, largest=False).indices
            tl = n[keep].mean()
        elif t == "hardw":
            wt = nll.detach().clamp_min(1e-3) ** a
            tl = (wt / wt.mean() * nll).mean()
        elif t == "posw":
            wt = (torch.arange(T, device=x.device).float() + 1) ** a
            tl = (wt / wt.mean() * nll).mean()
        elif t == "seqdro":
            ls = nll.detach().mean(1)
            wt = torch.softmax(a * ls, 0) * B
            tl = (wt[:, None] * nll).mean()
        return ntp, tl, {}

    if t == "poly1":
        p = (-nll).exp()
        return ntp, (nll + a * (1 - p)).mean(), {}
    if t == "zloss":
        z = torch.logsumexp(logits.float(), -1)
        return ntp, ntp + a * (z ** 2).mean(), {}
    if t == "confpen":
        lp = logp()
        ent = -(lp.exp() * lp).sum(-1)
        return ntp, ntp - a * ent.mean(), {"ent": round(ent.mean().item(), 3)}
    if t == "tempce":
        tau = cfg.ta
        return ntp, F.cross_entropy((logits.float() / tau).reshape(-1, V), y.reshape(-1)), {}
    if t == "margin":
        zy = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        zo = logits.scatter(-1, y.unsqueeze(-1), -1e9).max(-1).values
        return ntp, ntp + a * F.relu((cfg.tb or 1.0) - (zy - zo)).mean(), {}
    if t == "riskstd":
        return ntp, ntp + a * nll.std(), {}
    if t == "topk_ce":
        K = int(cfg.tb)
        lg = logits.float()
        top = torch.topk(lg.detach(), K, -1).indices
        msk = torch.full_like(lg, float("-inf")).scatter(-1, top, 0.0).scatter(-1, y.unsqueeze(-1), 0.0)
        sub = F.cross_entropy((lg + msk).reshape(-1, V), y.reshape(-1))
        return ntp, (1 - a) * ntp + a * sub, {}
    if t == "ctl":    # contrastive token loss: negatives = the previous w context tokens (not the target)
        wn = int(cfg.tb or 8)
        zy = logits.gather(-1, y.unsqueeze(-1))
        negs = []
        for j in range(wn):
            idx = torch.cat([x[:, :1].expand(B, j), x[:, :T - j]], 1) if j else x
            negs.append(idx)
        ni = torch.stack(negs, -1)
        zn = logits.gather(-1, ni)
        valid = (ni != y.unsqueeze(-1)).float()
        lc = torch.log1p((torch.exp((zn - zy).float().clamp(max=30)) * valid).sum(-1))
        return ntp, ntp + a * lc.mean(), {}
    if t == "noisekl":
        sig = cfg.tb or 0.1
        e = model.emb(x)
        lg2 = model.head(model.norm(_run_blocks(model, e + sig * e.std() * torch.randn_like(e))))
        kl = F.kl_div(F.log_softmax(lg2.float(), -1), F.log_softmax(logits.float(), -1).detach(), log_target=True, reduction="none").sum(-1)
        return ntp, ntp + a * kl.mean(), {}

    # ---- deep supervision
    if t == "deepsup":
        hs = c["hs"][:-1]
        aux = sum(F.cross_entropy(model.head(model.norm(hh)).float().reshape(-1, V), y.reshape(-1)) for hh in hs) / len(hs)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    # ---- extra vocab heads
    if t in ("skip2", "prev_rec", "cur_rec", "fbce"):
        lg = tm.head(tm.norm(h)).float()
        if t == "skip2":
            aux = F.cross_entropy(lg[:, :-1].reshape(-1, V), xy[:, 2:].reshape(-1))
        elif t == "prev_rec":
            aux = F.cross_entropy(lg[:, 1:].reshape(-1, V), x[:, :-1].reshape(-1))
        elif t == "cur_rec":
            aux = F.cross_entropy(lg.reshape(-1, V), x.reshape(-1))
        else:
            k = int(cfg.tb or 8)
            tgt = torch.zeros(B, T, V, device=x.device)
            for j in range(1, k + 1):
                idx = torch.cat([xy[:, j:], xy[:, -1:].expand(B, j - 1)], 1)[:, :T]
                tgt.scatter_(2, idx.unsqueeze(-1), 1.0)
            aux = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none").sum(-1).mean() / k
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    # ---- latent targets (d-dim)
    if t in ("nexth", "infonce", "embreg", "futsum"):
        z = tm.lin(tm.norm(h))
        if t == "nexth":
            tgt = model.norm(h).detach()[:, 1:]
            aux = (1 - F.cosine_similarity(z[:, :-1], tgt, dim=-1)).mean()
        elif t == "embreg":
            aux = (1 - F.cosine_similarity(z, model.emb(y), dim=-1)).mean()
        elif t == "infonce":
            zz = F.normalize(z.reshape(-1, z.shape[-1]), dim=-1)
            ee = F.normalize(model.emb(y).reshape(-1, z.shape[-1]), dim=-1)
            lg = zz @ ee.T / 0.1
            same = (y.reshape(-1, 1) == y.reshape(1, -1))
            lg = lg.masked_fill(same & ~torch.eye(lg.shape[0], dtype=torch.bool, device=x.device), float("-inf"))
            aux = F.cross_entropy(lg, torch.arange(lg.shape[0], device=x.device))
        else:
            k, g = int(cfg.tb or 4), (cfg.tc or 0.7)
            E = model.emb.weight.detach()
            acc = torch.zeros_like(h)
            for j in range(1, k + 1):
                idx = torch.cat([xy[:, j:], xy[:, -1:].expand(B, j - 1)], 1)[:, :T]
                acc = acc + g ** (j - 1) * E[idx]
            aux = (1 - F.cosine_similarity(z, acc, dim=-1)).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    if t == "copyflag":
        o = tm.lin(tm.norm(h)).squeeze(-1)
        same = (y.unsqueeze(2) == x.unsqueeze(1)) & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        lab = same.any(-1).float()
        aux = F.binary_cross_entropy_with_logits(o, lab)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "nextfreq":
        o = tm.lin(tm.norm(h)).squeeze(-1)
        lab = -torch.log(S["uni"][y]) / 10
        aux = F.mse_loss(o, lab)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "cluster":
        o = tm.lin(tm.norm(h)).float()
        aux = F.cross_entropy(o.reshape(-1, 64), S["clu"][y].reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    # ---- round 2
    if t == "lcs2":   # round 11: overlap self-teacher, 1.00x. Batch pairs (needs ovl): sequence 2k+1 starts o tokens after 2k,
        # so its positions j < T-o see short context for the tokens that sequence 2k sees at positions o+j with longer context.
        o = int(cfg.ovl * T)
        n_ = T - o
        teach = logits[0::2, o:].detach().float()            # long-context predictions of the shared tokens
        stu = logits[1::2, :n_]
        if "half" in (cfg.tm or ""):       # only the first half of the student positions (the shortest contexts)
            teach, stu = teach[:, :n_ // 2], stu[:, :n_ // 2]
        sl = soft_term(stu, lambda b0, b1: torch.softmax(teach[b0:b1] / (cfg.te or 1.0), -1), stu.shape[0])
        return ntp, ntp + a * sl, {"lcs2": round(sl.item(), 4)}
    if t == "lcs":   # round 10: long-context self-teacher. Short windows (td tokens) of the first tc*B sequences are trained
        # toward the full-context pass's own detached prediction mixed with the label: loss = ntp + tb*[(1-a)*CE + a*softCE]
        w_ = int(cfg.td or 16)
        nb = max(1, int(round((cfg.tc or 0.25) * B)))
        teach = logits[:nb].detach().float().reshape(nb * T // w_, w_, V)
        keep = [dict(m.aux) for m in model.moe_layers()]
        xs, ys = x[:nb].reshape(-1, w_), y[:nb].reshape(-1, w_)
        lg_s, nll_s = model(xs, ys)
        for m, k_ in zip(model.moe_layers(), keep):
            m.aux = k_
        def qf(b0, b1):
            q = torch.softmax(teach[b0:b1], -1)
            oh = F.one_hot(ys[b0:b1], V).float()
            return (1 - a) * oh + a * q
        sl = soft_term(lg_s, qf, xs.shape[0])
        return ntp, ntp + (cfg.tb or 1.0) * sl, {"lcs": round(sl.item(), 4)}
    if t == "hnb":   # round 9: head-neighbour target. Mass a on the nbk tokens whose head rows are nearest the true token's
        W = (model.emb.weight if "emb" in (cfg.tm or "") else model.head.weight).detach().float()
        k = int(cfg.nbk)
        if "nb" not in S or step % int(cfg.nbr) == 0:      # refresh the neighbour table (stop-gradient, no matmul in backward)
            Wn = F.normalize(W, dim=-1)
            idx = torch.empty(V, k, dtype=torch.long, device=W.device)
            sim = torch.empty(V, k, device=W.device)
            for r0 in range(0, V, 1024):
                sc = Wn[r0:r0 + 1024] @ Wn.T
                sc[torch.arange(sc.shape[0], device=W.device), torch.arange(r0, r0 + sc.shape[0], device=W.device)] = -2.0
                v_, i_ = sc.topk(k, -1)
                idx[r0:r0 + 1024], sim[r0:r0 + 1024] = i_, v_
            wt = torch.softmax(sim / cfg.nbt, -1) if "flat" not in (cfg.tm or "") else torch.full_like(sim, 1.0 / k)
            S["nb"] = (idx, wt)
        idx, wt = S["nb"]
        lp_ = F.log_softmax(logits.float(), -1)
        nbl = -(lp_.gather(-1, idx[y]) * wt[y]).sum(-1).mean()   # soft CE to the neighbour distribution, gathered
        return ntp, (1 - a) * ntp + a * nbl, {"nb": round(nbl.item(), 4)}
    if t in ("otri_x", "oquad_x"):   # round 5: one soft target per position, built only from counts and labels (no extra matmuls)
        ng = S["ng"]
        mode = cfg.tm or ""
        def qf(b0, b1):
            xb = xy[b0:b1]
            if "unif" in mode:       # ablation: plain label smoothing toward uniform
                q = torch.full((b1 - b0, T, V), 1.0 / V, device=x.device)
            elif "uni" in mode:      # ablation: label smoothing toward the online unigram (no context)
                q = (ng.u / ng.u.sum()).to(x.device).view(1, 1, V).expand(b1 - b0, T, V)
            else:
                q = shape_teacher(ng.query(xb).view(b1 - b0, T, V), cfg)
            yb = y[b0:b1]
            if "ratio" in mode:      # sharpen toward what the trigram knows beyond the bigram
                qb = ng._big(xb[:, :-1].reshape(-1).cpu()).to(q.device).view(b1 - b0, T, V)
                q = q * (q / qb.clamp_min(1e-9)) ** (cfg.tg or 0.5)
                q = q / q.sum(-1, keepdim=True)
            oh = F.one_hot(yb, V).float()
            if "dark" in mode:       # teacher mass only on the wrong tokens
                q = q * (1 - oh)
                q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
            w = torch.full((b1 - b0, T, 1), a, device=q.device)
            if "agree" in mode or "disag" in mode:
                ok = (q.argmax(-1, keepdim=True) == yb.unsqueeze(-1)).float()
                w = a * ((1.5 * ok + 0.5 * (1 - ok)) if "agree" in mode else (0.5 * ok + 1.5 * (1 - ok)))
                w = w.clamp(max=1.0)
            tgt_ = (1 - w) * oh + w * q
            if cfg.tfm:               # small mass on the token after next (label-side MTP, no head)
                nx = xy[b0:b1, 2:]
                f = torch.zeros_like(tgt_[:, :, :1]); f[:, :T - 1] = cfg.tfm
                oh2 = torch.zeros_like(tgt_); oh2[:, :T - 1] = F.one_hot(nx, V).float()
                tgt_ = (1 - f) * tgt_ + f * oh2
            if cfg.tpm:               # small mass on the current token
                tgt_ = (1 - cfg.tpm) * tgt_ + cfg.tpm * F.one_hot(x[b0:b1], V).float()
            return tgt_
        loss = soft_term(logits, qf, B)
        ng.update(xy)
        if cfg.tcp:                   # confidence penalty: subtract prediction entropy
            lp_ = F.log_softmax(logits.float(), -1)
            loss = loss - cfg.tcp * (-(lp_.exp() * lp_).sum(-1)).mean()
        if cfg.tz:
            loss = loss + cfg.tz * (torch.logsumexp(logits.float(), -1) ** 2).mean()
        return ntp, loss, {}
    if t == "wx":
        return _wx(logits, nll, xy, cfg, a)
    if t in CC_MODES:
        S["cc_frac"] = step / max(1, total - 1)
        st = soft_term(logits, lambda b0, b1: _cc_target(t, model, xy[b0:b1], cfg, b0, b1), B)
        S["ng"].update(xy)
        return ntp, st, {}
    if t in ("obig", "otri", "oskip", "ctrig"):
        ng = S["ng"]
        st = soft_term(logits, lambda b0, b1: shape_teacher(ng.query(xy[b0:b1]).view(b1 - b0, T, V), cfg), B)
        ng.update(xy)
        return ntp, (1 - a) * ntp + a * st, {}
    if t == "bigmid":     # layer-1 output through the main head distilled toward the corpus bigram; final = pure NTP
        M = S["big"]
        rows = torch.from_numpy(M[x.reshape(-1).cpu().numpy()].toarray()).to(x.device)
        q = (rows + S["uni"]) / (rows.sum(-1, keepdim=True) + 1)
        l1 = F.log_softmax(model.head(model.norm(c["hs"][0])).float(), -1).reshape(-1, V)
        aux = -(l1 * q).sum(-1).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t in ("cur_rec_sg", "cur_rec_all", "cur_rec_l1", "cur_rec_rare", "ntp2", "rand_tgt", "bagrec"):
        if t == "cur_rec_sg":
            aux = F.cross_entropy(tm.head(tm.norm(h.detach())).float().reshape(-1, V), x.reshape(-1))
        elif t == "cur_rec_all":
            hs = c["hs"]
            aux = sum(F.cross_entropy(tm.head(tm.norm(hh)).float().reshape(-1, V), x.reshape(-1)) for hh in hs) / len(hs)
        elif t == "cur_rec_l1":
            aux = F.cross_entropy(tm.head(tm.norm(c["hs"][0])).float().reshape(-1, V), x.reshape(-1))
        elif t == "cur_rec_rare":
            l = F.cross_entropy(tm.head(tm.norm(h)).float().reshape(-1, V), x.reshape(-1), reduction="none")
            r = S["rare"][x.reshape(-1)].float()
            aux = (l * r).sum() / r.sum().clamp_min(1)
        elif t == "ntp2":
            aux = F.cross_entropy(tm.head(tm.norm(h)).float().reshape(-1, V), y.reshape(-1))
        elif t == "rand_tgt":
            aux = F.cross_entropy(tm.head(tm.norm(h)).float().reshape(-1, V), torch.randint(0, V, (B * T,), device=x.device))
        else:  # bagrec: soft CE onto a uniform bag of the last k tokens (current included)
            k = int(cfg.tb or 4)
            bag = torch.zeros(B, T, V, device=x.device)
            for j in range(k):
                idx = torch.cat([x[:, :1].expand(B, j), x[:, :T - j]], 1) if j else x
                bag.scatter_add_(2, idx.unsqueeze(-1), torch.ones(B, T, 1, device=x.device))
            bag = bag / bag.sum(-1, keepdim=True)
            aux = soft_ce(tm.head(tm.norm(h)), bag).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "cur_rec_tied":   # reconstruct the current token with the input embedding matrix as classifier
        aux = F.cross_entropy(((cfg.tb or 1.0) * (model.norm(h) @ model.emb.weight.T)).float().reshape(-1, V), x.reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t in ("cur_rec_cos", "infonce2", "infonce_cur", "infonce_prev", "ctxmean"):
        z = tm.lin(tm.norm(h))
        if t == "cur_rec_cos":
            aux = (1 - F.cosine_similarity(z, model.emb(x), dim=-1)).mean()
        elif t == "ctxmean":
            E = model.emb(x).detach()
            cm = E.cumsum(1) / torch.arange(1, T + 1, device=x.device).view(1, T, 1)
            aux = (1 - F.cosine_similarity(z, cm, dim=-1)).mean()
        else:
            if t == "infonce2":
                zz, tg = z[:, :-1], xy[:, 2:]
            elif t == "infonce_cur":
                zz, tg = z, x
            else:
                zz, tg = z[:, 1:], x[:, :-1]
            aux = _nce(zz, model.emb(tg), tg, cfg.tb or 0.1)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "tokcount":
        o = tm.lin(tm.norm(h)).squeeze(-1)
        same = (x.unsqueeze(2) == x.unsqueeze(1)) & torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
        lab = torch.log1p(same.float().sum(-1)) / 2
        aux = F.mse_loss(o, lab)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "posrec":
        o = tm.lin(tm.norm(h)).float()
        aux = F.cross_entropy(o.reshape(-1, T), torch.arange(T, device=x.device).repeat(B))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "heademb":
        aux = (1 - F.cosine_similarity(model.head.weight, model.emb.weight, dim=-1)).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "embpath":
        aux = F.cross_entropy(model.head(model.norm(c["emb"])).float().reshape(-1, V), y.reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "cosce":
        sc = cfg.tb or 10.0
        lg = sc * F.normalize(model.norm(h), dim=-1) @ F.normalize(model.head.weight, dim=-1).T
        aux = F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "logitcenter":
        return ntp, ntp + a * (logits.float().mean(-1) ** 2).mean(), {}
    if t == "vicreg":
        hh = model.norm(h).reshape(-1, h.shape[-1])
        aux = F.relu(cfg.vics - hh.std(0)).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t == "mtp_lin_tied":   # t+2 from [h_i; Emb(t_{i+1})] via a linear map, through the MAIN head
        L = T - 1
        hk = tm.proj(torch.cat([h[:, :L], model.emb(xy[:, 1:1 + L])], -1))
        aux = F.cross_entropy(model.head(tm.fn(hk)).float().reshape(-1, V), xy[:, 2:2 + L].reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    # ---- round 3
    if t in ("oquad", "otri_conf", "otri_ent", "otri_logit", "ent_match", "ocache", "oind", "obig_unl"):
        ng = S["ng"]
        q = ng.query(xy).view(B, T, V)
        if t == "ocache":      # in-sequence unigram cache (tokens x_0..x_i), backoff to the online bigram
            k = cfg.te or 8.0
            cnt = torch.zeros(B, T, V, device=x.device)
            cnt.scatter_add_(2, x.unsqueeze(-1), torch.ones(B, T, 1, device=x.device))
            cnt = cnt.cumsum(1)
            q = (cnt + k * q) / (cnt.sum(-1, keepdim=True) + k)
        if t == "oind":        # in-sequence bigram (what followed earlier copies of x_i), backoff to the online bigram
            k = cfg.te or 2.0
            M = ((x.unsqueeze(2) == x.unsqueeze(1)) & torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)).float()
            cnt = M @ F.one_hot(y, V).float()
            q = (cnt + k * q) / (cnt.sum(-1, keepdim=True) + k)
        lp = logp()
        if t == "otri_logit":  # teacher log-probs as a training-only logit bias (product of experts); eval has no bias
            lb = F.log_softmax(logits.float() + a * torch.log(q + 1e-9), -1)
            tl = F.nll_loss(lb.reshape(-1, V), y.reshape(-1))
            ng.update(xy)
            return ntp, tl, {}
        ce_q = -(lp * q).sum(-1)
        if t == "otri_conf":
            ai = a * q.max(-1).values
            tl = ((1 - ai) * nll + ai * ce_q).mean()
        elif t == "otri_ent":
            ai = a * (-(q * torch.log(q + 1e-9)).sum(-1) / math.log(V))
            tl = ((1 - ai) * nll + ai * ce_q).mean()
        elif t == "ent_match":
            hp = -(lp.exp() * lp).sum(-1)
            hq = -(q * torch.log(q + 1e-9)).sum(-1)
            tl = ntp + a * ((hp - hq) ** 2).mean()
        elif t == "obig_unl":  # unlikelihood on next tokens never seen after x_i (row count >= 20)
            r = ng.rows(x.reshape(-1)).to(x.device).view(B, T, V)
            zero = (r == 0) & (r.sum(-1, keepdim=True) >= 20)
            pz = (lp.exp() * zero).sum(-1).clamp(max=1 - 1e-4)
            tl = ntp + a * (-torch.log1p(-pz)).mean()
        else:
            tl = ((1 - a) * nll + a * ce_q).mean()
        ng.update(xy)
        return ntp, tl, {}
    if t in ("ctied_all", "cur_rec_tied_sge", "prev_tied", "next_tied", "ctied_mid"):
        E = model.emb.weight
        if t == "ctied_all":
            hs = c["hs"]
            aux = sum(F.cross_entropy((model.norm(hh) @ E.T).float().reshape(-1, V), x.reshape(-1)) for hh in hs) / len(hs)
        elif t == "cur_rec_tied_sge":
            aux = F.cross_entropy((model.norm(h) @ E.detach().T).float().reshape(-1, V), x.reshape(-1))
        elif t == "prev_tied":
            aux = F.cross_entropy((model.norm(h[:, 1:]) @ E.T).float().reshape(-1, V), x[:, :-1].reshape(-1))
        elif t == "next_tied":
            aux = F.cross_entropy((model.norm(h) @ E.T).float().reshape(-1, V), y.reshape(-1))
        else:
            hh = c["hs"][len(c["hs"]) // 2 - (1 if len(c["hs"]) % 2 == 0 else 0)]
            aux = F.cross_entropy((model.norm(hh) @ E.T).float().reshape(-1, V), x.reshape(-1))
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}
    if t in ("vic_all", "vic_l1", "decor", "unif", "effrank", "emb_vic", "hnorm"):
        hn = model.norm(h).reshape(-1, h.shape[-1])
        if t == "vic_all":
            aux = sum(_vfloor(_rms(hh), cfg.vics) for hh in c["hs"]) / len(c["hs"])
        elif t == "vic_l1":
            aux = _vfloor(_rms(c["hs"][0]), cfg.vics)
        elif t == "decor":
            aux = _offcov(hn)
        elif t == "unif":
            idx = torch.randperm(hn.shape[0], device=x.device)[:256]
            z = F.normalize(hn[idx], dim=-1)
            aux = torch.log(torch.exp(-2 * torch.cdist(z, z).pow(2)).mean())
        elif t == "effrank":
            sv = torch.linalg.svdvals((hn - hn.mean(0)).float())
            aux = sv[0] ** 2 / (sv ** 2).sum()
        elif t == "emb_vic":
            aux = _vfloor(_rms(model.emb(x)), cfg.vics)
        else:
            aux = ((h.pow(2).mean(-1).sqrt() - 1) ** 2).mean()
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    if t == "embotri_u":   # embpath_otri computed once per distinct current token (cheap): CE is linear in q,
        ng = S["ng"]       # so positions sharing a current token share one student row and their averaged teacher
        ids = x.reshape(-1)
        uq, inv = torch.unique(ids, return_inverse=True)
        Qs = torch.zeros(len(uq), V, device=x.device)
        for b0 in range(0, B, 4):
            b1 = min(B, b0 + 4)
            Qs.index_add_(0, inv[b0 * T:b1 * T], ng.query(xy[b0:b1]).float())
        n = torch.bincount(inv, minlength=len(uq)).float()
        K = int(cfg.td or 0)
        sel = torch.randperm(len(uq), device=x.device)[:K] if 0 < K < len(uq) else torch.arange(len(uq), device=x.device)
        pl = model.head(model.norm(model.emb.weight[uq[sel]])).float()
        lp_ = F.log_softmax(pl, -1)
        aux = -((Qs[sel] * lp_).sum(-1)).sum() / n[sel].sum()
        ng.update(xy)
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4), "nu": len(uq)}

    # ---- round 4: the zero-layer path (embedding -> head) and current-token reconstruction variants
    if t in ("embdot", "embpath2", "embbag2", "embpath_sgE", "embpath_sgH", "embpath_obig", "embpath_otri",
             "embpath_kd", "embpath_self", "embpath_oquad", "cur_head", "sgns", "crt_sgh", "pc_tied", "futbag_tied"):
        e = c["emb"]
        if t == "embdot":       # raw dot product emb(x) . head(b), no norm
            aux = F.cross_entropy((e @ model.head.weight.T).float().reshape(-1, V), y.reshape(-1))
        elif t == "embpath2":   # the embedding path predicts the token two ahead
            aux = F.cross_entropy(model.head(model.norm(e[:, :-1])).float().reshape(-1, V), xy[:, 2:].reshape(-1))
        elif t == "embbag2":    # emb(x_t) + emb(x_{t-1}) through the head
            pe = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1)
            aux = F.cross_entropy(model.head(model.norm(e + pe)).float().reshape(-1, V), y.reshape(-1))
        elif t == "embpath_sgE":   # only the head learns from the path
            aux = F.cross_entropy(model.head(model.norm(e.detach())).float().reshape(-1, V), y.reshape(-1))
        elif t == "embpath_sgH":   # only the embedding learns from the path
            aux = F.cross_entropy((model.norm(e) @ model.head.weight.detach().T).float().reshape(-1, V), y.reshape(-1))
        elif t in ("embpath_obig", "embpath_otri", "embpath_oquad", "embpath_kd", "embpath_self"):
            pl = model.head(model.norm(e))
            if t == "embpath_kd":
                M = S["big"]
                def qf(b0, b1):
                    rows = torch.from_numpy(M[x[b0:b1].reshape(-1).cpu().numpy()].toarray()).float().to(pl.device)
                    return ((rows + S["uni"]) / (rows.sum(-1, keepdim=True) + 1.0)).view(b1 - b0, T, V)
            elif t == "embpath_self":
                lgd = logits.detach()
                def qf(b0, b1):
                    return F.softmax(lgd[b0:b1].float(), -1)
            else:
                ng = S["ng"]
                def qf(b0, b1):
                    return ng.query(xy[b0:b1]).view(b1 - b0, T, V)
            aux = soft_term(pl, qf, B)
            if t in ("embpath_obig", "embpath_otri", "embpath_oquad"):
                S["ng"].update(xy)
        elif t == "cur_head":   # current token through the main output head
            aux = F.cross_entropy(model.head(model.norm(h)).float().reshape(-1, V), x.reshape(-1))
        elif t == "sgns":       # word2vec negative sampling between embedding rows and head rows
            k = int(cfg.td or 8)
            ea = e.reshape(-1, e.shape[-1]); hp = model.head.weight[y.reshape(-1)]
            neg = model.head.weight[torch.randint(0, V, (ea.shape[0], k), device=x.device)]
            sc = cfg.tb or 1.0
            aux = -(F.logsigmoid(sc * (ea * hp).sum(-1)).mean() + F.logsigmoid(-sc * (neg @ ea.unsqueeze(-1)).squeeze(-1)).sum(-1).mean())
        elif t == "crt_sgh":    # cur_rec_tied with the trunk detached: only the embedding learns
            aux = F.cross_entropy((model.norm(h.detach()) @ model.emb.weight.T).float().reshape(-1, V), x.reshape(-1))
        elif t == "pc_tied":    # half current, half previous token through Emb^T
            lp_ = F.log_softmax((model.norm(h) @ model.emb.weight.T).float(), -1)
            prev = torch.cat([x[:, :1], x[:, :-1]], 1)
            aux = -(0.5 * lp_.gather(-1, x.unsqueeze(-1)) + 0.5 * lp_.gather(-1, prev.unsqueeze(-1))).mean()
        else:                   # futbag_tied: bag of the next k tokens through Emb^T
            k = int(cfg.td or 4)
            lp_ = F.log_softmax((model.norm(h[:, :T - k]) @ model.emb.weight.T).float(), -1)
            aux = -sum(lp_.gather(-1, xy[:, 1 + j:1 + j + T - k].unsqueeze(-1)).mean() for j in range(k)) / k
        return ntp, ntp + a * aux, {"aux": round(aux.item(), 4)}

    raise ValueError(f"unknown tgt {t}")


def _nce(z, e, ids, tau):
    zz = F.normalize(z.reshape(-1, z.shape[-1]), dim=-1)
    ee = F.normalize(e.reshape(-1, z.shape[-1]), dim=-1)
    lg = zz @ ee.T / tau
    idv = ids.reshape(-1)
    same = (idv[:, None] == idv[None, :]) & ~torch.eye(lg.shape[0], dtype=torch.bool, device=z.device)
    lg = lg.masked_fill(same, float("-inf"))
    return F.cross_entropy(lg, torch.arange(lg.shape[0], device=z.device))


# ---------------------------------------------------------------- round w1: wild targets on the CoCoNTs teacher
# tgt=wx, mode in cfg.tm. Base: CoCoNTs-style corpus trigram q, target (1 - w) * onehot + w * q', mean w = ta.
# Every mode uses something a fixed corpus table cannot: the model's own current prediction or error history,
# tokens after the label (training only), or this sequence's own tokens. Eval is unchanged next-token CE.
def _skip_csr(train, V, name, g=2):
    import scipy.sparse as sp
    f = os.path.join(CACHE, f"{name}_skip{g}.npz")
    if os.path.exists(f):
        return sp.load_npz(f)
    a, _ = _counts(train, V)
    code = a[:-g] * V + a[g:]
    u, c = np.unique(code, return_counts=True)
    M = sp.csr_matrix((c.astype(np.float32), (u // V, u % V)), shape=(V, V))
    sp.save_npz(f, M)
    return M


def _wx_setup(train, cfg, name, dev):
    ng, m, V = S["ng"], cfg.tm, cfg.vocab
    W = {"gap": 1.0}
    if m.startswith("f"):
        W["bT"] = ng.big.T.tocsr()
        W["rs"] = torch.from_numpy(np.asarray(ng.big.sum(1)).ravel()).float()
        if m == "fpost2":
            M2 = _skip_csr(train, V, name)
            W["s2T"] = M2.T.tocsr()
            W["rs2"] = torch.from_numpy(np.asarray(M2.sum(1)).ravel()).float()
    if m in ("mconf", "mconfneg", "mdebias"):
        W["C"] = torch.full((V, V), 1.0 / V, device=dev)
    if m in ("wid", "dhist", "dhistx", "dlearn"):
        W["e"] = torch.full((V,), 1.0 if m == "wid" else math.log(V), device=dev)
        W["e2"] = torch.full((V,), math.log(V), device=dev)
    S["wx"] = W


def _ema_id(e, ids, val, mom):
    u, inv = ids.unique(return_inverse=True)
    sm = torch.zeros(len(u), *val.shape[1:], device=e.device).index_add_(0, inv, val)
    c = torch.bincount(inv, minlength=len(u)).float().view(-1, *([1] * (val.dim() - 1)))
    e[u] = mom * e[u] + (1 - mom) * sm / c


def _nz(r, q):
    s = r.sum(-1, keepdim=True)
    return torch.where(s > 1e-9, r / s.clamp_min(1e-12), q)


def _fev(W, ids, col_T, rs, uni, gam):
    """P(next-next token | v) for every v, as a (len(ids), V) tensor, from a transposed count table."""
    c = torch.from_numpy(col_T[ids.numpy()].toarray())
    return ((c + uni[ids].unsqueeze(1)) / (rs + 1.0)) ** gam


def _wx(logits, nll, xy, cfg, a):
    ng, W, m = S["ng"], S["wx"], cfg.tm
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    V = cfg.vocab
    dev = logits.device

    def qf(b0, b1):
        n = b1 - b0
        xb, yb = x[b0:b1], y[b0:b1]
        q0 = ng.query(xy[b0:b1]).view(n, T, V)
        q = q0
        p = F.softmax(logits[b0:b1].float(), -1)
        py = p.gather(-1, yb.unsqueeze(-1))                      # (n, T, 1)
        qy = q0.gather(-1, yb.unsqueeze(-1))
        ell = nll[b0:b1].detach().float().unsqueeze(-1)
        lq = -torch.log(qy.clamp_min(1e-12))
        oh = F.one_hot(yb, V).float()
        s = None                                                  # weight shape; normalised to mean a
        wabs = None                                               # absolute weight (not normalised)
        # ---- teacher reshaping with the model's prediction
        if m in ("qres", "qboth"):
            q = _nz(F.relu(q0 - p), q0)
        elif m == "qup":
            q = _nz(q0 * (q0 > p), q0)
        elif m == "qgeo":
            q = _nz((q0 * p).sqrt(), q0)
        elif m == "qprod":
            q = _nz(q0 * p, q0)
        elif m == "qtopk":
            i = p.topk(16, -1).indices
            q = _nz(q0 * torch.zeros_like(p).scatter_(-1, i, 1.0), q0)
        elif m == "qdiv":
            q = _nz(q0 / p.clamp_min(1e-6).sqrt(), q0)
        elif m == "qtemp":
            q = _nz(q0 ** (1.0 / (1.0 + ell / 4)), q0)
        elif m == "lmaxent":
            q = _nz(q0 * (1 - p), q0)
        elif m == "lself":
            q = 0.5 * q0 + 0.5 * p
        elif m == "qsharp":
            q = _nz(q0 ** (1.0 + py), q0)
        # ---- future-span posterior (tokens after the label; training only)
        elif m.startswith("f"):
            uni = ng.uni
            ev = torch.ones(n, T, V)
            gam = 1.0 if m == "fpost1" else 0.5
            ids2 = xy[b0:b1, 2:].cpu().reshape(-1)
            e2 = _fev(W, ids2, W["bT"], W["rs"], uni, 1.0 if m == "fonly" else gam)
            ev[:, :T - 1] = e2.view(n, T - 1, V)
            if m == "fpost2":
                ids3 = xy[b0:b1, 3:].cpu().reshape(-1)
                ev[:, :T - 2] *= _fev(W, ids3, W["s2T"], W["rs2"], uni, 0.25).view(n, T - 2, V)
            ev = ev.to(dev)
            if m == "fonly":
                r = ev * uni.to(dev)
                r[:, T - 1] = q0[:, T - 1]
                q = _nz(r, q0)
            else:
                q = _nz(q0 * ev, q0)
            if m == "fpostw":
                s = 1 - py
            if m == "fpostg":
                wabs = a * min(1.0, max(0.0, W["gap"] / 0.5))
        # ---- this sequence's own tokens
        elif m in ("ccache", "cpost"):
            cnt = torch.zeros(n, T, V, device=dev).scatter_add_(2, xb.unsqueeze(-1), torch.ones(n, T, 1, device=dev)).cumsum(1)
            cache = cnt / cnt.sum(-1, keepdim=True)
            q = 0.7 * q0 + 0.3 * cache if m == "ccache" else _nz(q0 * (1 + 3 * cache), q0)
        elif m in ("cind", "cind2"):
            tri = torch.ones(T, T, dtype=torch.bool, device=dev).tril(-1)
            M = (xb.unsqueeze(2) == xb.unsqueeze(1)) & tri
            if m == "cind2":
                pv = torch.cat([torch.full((n, 1), -1, device=dev, dtype=xb.dtype), xb[:, :-1]], 1)
                M = M & (pv.unsqueeze(2) == pv.unsqueeze(1)) & (pv.unsqueeze(2) >= 0)
            cnt = M.float() @ oh
            q = (cnt + 2.0 * q0) / (cnt.sum(-1, keepdim=True) + 2.0)
        # ---- the model's own confusion history
        elif m in ("mconf", "mconfneg"):
            Cy = W["C"][yb]
            q = 0.5 * q0 + 0.5 * Cy if m == "mconf" else _nz(F.relu(q0 - 0.5 * Cy), q0)
            _ema_id(W["C"], yb.reshape(-1), p.reshape(-1, V), 0.9)
        elif m == "mdebias":
            R = W["C"][xb]
            q = _nz(q0 / R.clamp_min(1e-6).sqrt(), q0)
            _ema_id(W["C"], xb.reshape(-1), p.reshape(-1, V), 0.9)
        # ---- per-token teacher weight from the model's errors
        if m in ("werr", "fpostw"):
            s = 1 - py
        elif m == "weasy":
            s = py
        elif m in ("wbehind", "qboth"):
            s = (qy > py).float()
        elif m == "wbehs":
            s = torch.sigmoid(4 * (torch.log(qy.clamp_min(1e-12)) - torch.log(py.clamp_min(1e-12))))
        elif m == "wtv":
            s = 0.5 * (p - q0).abs().sum(-1, keepdim=True)
        elif m in ("wtopl", "weasyl"):
            med = ell.median()
            s = (ell > med).float() if m == "wtopl" else (ell <= med).float()
        elif m == "wwrong":
            s = (p.argmax(-1, keepdim=True) != yb.unsqueeze(-1)).float()
        elif m == "went":
            s = -(p * torch.log(p.clamp_min(1e-12))).sum(-1, keepdim=True)
        elif m == "wglob":
            wabs = a * min(1.0, max(0.0, W["gap"] / 0.5))
        elif m == "wseq":
            s = (ell.mean(1, keepdim=True) > lq.mean(1, keepdim=True)).float().expand(n, T, 1)
        elif m == "wpos":
            if "gp" not in W:
                W["gp"] = torch.ones(T, device=dev)
            wabs = a * (W["gp"] / 0.5).clamp(0, 1).view(1, T, 1)
            W["gp"] = 0.95 * W["gp"] + 0.05 * (ell - lq).mean((0, 2))
        elif m == "wid":
            wabs = a * (W["e"][yb] / 0.5).clamp(0, 1).unsqueeze(-1)
            _ema_id(W["e"], yb.reshape(-1), (ell - lq).reshape(-1), 0.9)
        elif m == "dhist":
            s = W["e"][yb].unsqueeze(-1)
            _ema_id(W["e"], yb.reshape(-1), ell.reshape(-1), 0.9)
        elif m == "dhistx":
            s = W["e"][xb].unsqueeze(-1)
            _ema_id(W["e"], xb.reshape(-1), ell.reshape(-1), 0.9)
        elif m == "dlearn":
            s = F.relu(W["e2"][yb] - W["e"][yb]).unsqueeze(-1) + 0.01
            _ema_id(W["e"], yb.reshape(-1), ell.reshape(-1), 0.7)    # fast
            _ema_id(W["e2"], yb.reshape(-1), ell.reshape(-1), 0.97)  # slow
        W["gap"] = 0.99 * W["gap"] + 0.01 * (ell - lq).mean().item()
        if wabs is not None:
            w = wabs if torch.is_tensor(wabs) else torch.full((n, T, 1), wabs, device=dev)
        elif s is not None:
            w = (a * s / s.mean().clamp_min(1e-9)).clamp(0, 1)
        else:
            w = torch.full((n, T, 1), a, device=dev)
        return (1 - w) * oh + w * q

    loss = soft_term(logits, qf, B)
    return nll.mean(), loss, {"gap": round(W["gap"], 3)}


# ---------------------------------------------------------------- round w2: add-on targets (cfg.wy, strength cfg.wyl)
# Stacks on any tgt (run on CoCoNTs + floor). None of these use n-gram / previous-token statistics.
# Families: TS token-string similarity soft targets, H lexical/byte heads, W word-structure heads, Z distance-to-
# structure heads, C corrupted inputs, K two-pass consistency, A adversarial embeddings, X mixup, M batch-marginal
# matching, E head-geometry soft targets, L proper scores, G head noise.
WY_NB = {"casevar": "nb_case", "casevar3": "nb_case", "spacevar": "nb_space", "pref3": "nb_pref3", "pref2": "nb_pref2",
         "suf3": "nb_suf3", "charsim": "nb_char"}
WY_HEAD = {"nspace": 1, "nlen": 8, "ncls": 6, "nbyte1": 256, "nbytel": 256, "nbag": 256, "lexall": 15, "nbyte2": 256,
           "nextword": None, "wordend": 1, "wordlen": 4, "distnl": 8, "distpunct": 8, "distcap": 8, "sincepunct": 8, "rtd": 1}
WY_LAM = {"casevar3": 0.3, "ckl_noise": 1.0, "ckl_tok": 1.0, "fgsm": 0.5, "fgsm4": 0.5, "fgsmkl": 1.0, "gradpen": 0.01, "embmix": 0.5,
          "hmix1": 0.5, "margkl": 1.0, "margent": 0.1, "brier": 1.0, "spher": 1.0, "esim": 0.1, "esim_in": 0.1,
          "lgnoise": 1.0, "lgnoise3": 3.0, "hdrop": 0.1, "hdrop3": 0.3, "inrand10": 0.1, "inrand3": 0.03, "inuni10": 0.1,
          "instut10": 0.1, "inmask15": 0.15, "inswap10": 0.1, "rtd": 0.3}


def _wy_modules(m, cfg, d):
    from model import RMSNorm
    k = cfg.wy
    if k in WY_HEAD or k in W3_HEAD or k in W4_HEAD or k in W5_HEAD:
        n = WY_HEAD.get(k) or W3_HEAD.get(k) or W4_HEAD.get(k) or W5_HEAD.get(k) or cfg.vocab
        m.wyn = RMSNorm(d)
        m.wyh = nn.Linear(d, d if n == -1 else n, bias=True)


def _wy_lam(cfg):
    if getattr(cfg, "wyl", 0) > 0:
        return cfg.wyl
    if cfg.wy in W3_LAM:
        return W3_LAM[cfg.wy]
    if cfg.wy in W4_LAM:
        return W4_LAM[cfg.wy]
    if cfg.wy in W5_LAM:
        return W5_LAM[cfg.wy]
    return WY_LAM.get(cfg.wy, 0.1 if cfg.wy in WY_NB else 0.3)


def _wy_setup(model, train, cfg, dev):
    k = cfg.wy
    L = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lex8192.npz"))
    W = {kk: torch.from_numpy(L[kk].astype(np.int64)).to(dev) for kk in L.files}
    if k in ("inuni10", "logadj", "logadj05", "logadjneg"):
        _, uni = _counts(train, cfg.vocab)
        W["uni"] = torch.from_numpy((uni + 1) / (uni + 1).sum()).float().to(dev)
        W["luni"] = W["uni"].log()
    W["emb_fn"], W["head_out"], W["head_in"], W["b0_fn"] = None, None, None, None
    W["pmul"] = getattr(cfg, "wyp", 0.0) or 1.0          # corruption-rate multiplier (changed variants)
    S["wy"] = W

    def emb_hook(mod, inp, out):
        f = W["emb_fn"]
        return f(out, inp[0]) if (f is not None and mod.training) else out

    def head_hook(mod, inp, out):
        f = W["head_out"]
        return f(out) if (f is not None and mod.training) else out

    def head_pre(mod, inp):
        f = W["head_in"]
        return (f(inp[0]),) if (f is not None and mod.training) else None

    def b0_hook(mod, inp, out):
        f = W["b0_fn"]
        return f(out) if (f is not None and mod.training) else out

    model.emb.register_forward_hook(emb_hook)
    model.head.register_forward_hook(head_hook)
    model.head.register_forward_pre_hook(head_pre)
    model.blocks[0].register_forward_hook(b0_hook)
    if k in W3_ATT or k in ("ldkl", "ldce"):
        _w3_hooks(model, W)
    if k in W4_ALL:
        _w4_setup(model, train, cfg, W, dev)
    if k in W5_ALL:
        _w5_setup(model, train, cfg, W, dev)


def _corrupt(idx, k, W, V):
    """returns (new ids, replaced mask) for the token-corruption modes."""
    p = {"inrand_dec": 0.2 * (1 - W.get("frac", 0.0)), "inrand10": 0.1, "inrand3": 0.03, "inuni10": 0.1, "instut10": 0.1, "inswap10": 0.1, "rtd": 0.15,
         "ckl_tok": 0.1}.get(k, 0.1)
    p = p * W.get("pmul", 1.0)
    msk = torch.rand(idx.shape, device=idx.device) < p
    msk[:, 0] = False
    if k == "inuni10":
        rep = torch.multinomial(W["uni"], idx.numel(), replacement=True).view(idx.shape)
    elif k == "instut10":
        rep = torch.cat([idx[:, :1], idx[:, :-1]], 1)
    elif k == "inswap10":
        z = torch.zeros_like(msk[:, :1])
        m = msk & ~torch.cat([z, msk[:, :-1]], 1)
        m = m & ~torch.cat([m[:, 1:], z], 1)
        nxm = torch.cat([m[:, 1:], z], 1)                       # position i-1 of a swap at i
        prv = torch.cat([idx[:, :1], idx[:, :-1]], 1)
        nxt = torch.cat([idx[:, 1:], idx[:, -1:]], 1)
        new = torch.where(m, prv, torch.where(nxm, nxt, idx))
        return new, m | nxm
    else:
        rep = torch.randint(0, V, idx.shape, device=idx.device)
    return torch.where(msk, rep, idx), msk & (rep != idx)


def _wy_pre(model, xy, cfg):
    W, k, V = S["wy"], cfg.wy, cfg.vocab
    if k in ("inrand10", "inrand3", "inrand_dec", "inuni10", "instut10", "inswap10", "rtd"):
        def f(out, idx):
            new, msk = _corrupt(idx, k, W, V)
            W["rmask"] = msk
            return model.emb.weight[new]
        W["emb_fn"] = f
    elif k == "inmask15":
        def f(out, idx):
            keep = (torch.rand(idx.shape, device=idx.device) >= 0.15 * W.get("pmul", 1.0)).float().unsqueeze(-1)
            return out * keep
        W["emb_fn"] = f
    elif k in ("lgnoise", "lgnoise3"):
        s = 1.0 if k == "lgnoise" else 3.0
        W["head_out"] = lambda o: o + s * torch.randn_like(o)
    elif k in ("hdrop", "hdrop3"):
        W["head_in"] = lambda h: F.dropout(h, 0.1 if k == "hdrop" else 0.3, True)
    elif k in ("logadj", "logadj05", "logadjneg"):
        tau = {"logadj": 1.0, "logadj05": 0.5, "logadjneg": -0.5}[k] * W.get("pmul", 1.0)
        W["head_out"] = lambda o: o + tau * W["luni"]
    if k in W3_ATT:
        W["att_on"], W["att"] = True, {}
    if k in W4_ALL:
        _w4_pre(model, xy, cfg, W, k)
    if k in W5_ALL:
        _w5_pre(model, xy, cfg, W, k)


def _second_pass(model, x, emb_fn=None, b0_fn=None):
    """Forward again with a perturbation; keeps the first pass's MoE aux losses, expert counts and ctx."""
    W = S["wy"]
    moes = model.moe_layers()
    sv = [(dict(m.aux), m.local_tokens_per_expert.clone() if hasattr(m, "local_tokens_per_expert") else None) for m in moes]
    ctx = {kk: model.ctx.get(kk) for kk in ("h", "hs", "emb")}
    W["emb_fn"], W["b0_fn"] = emb_fn, b0_fn
    lg = model(x)
    W["h2"] = model.ctx["h"]
    W["emb_fn"], W["b0_fn"] = None, None
    for m, (a, c) in zip(moes, sv):
        m.aux = a
        if c is not None:
            m.local_tokens_per_expert.copy_(c)
    model.ctx.update(ctx)
    return lg.float()


def _nb_soft(logits, y, nb):
    """mean soft CE to a uniform distribution over the string neighbours of the label (positions with none skipped)."""
    N = nb[y]                                         # (B, T, K)
    ok = N >= 0
    cnt = ok.sum(-1)
    has = cnt > 0
    if not has.any():
        return logits.sum() * 0.0
    lp = F.log_softmax(logits.float(), -1)
    g = lp.gather(-1, N.clamp_min(0)) * ok
    return -(g.sum(-1)[has] / cnt[has]).mean()


def _bucket(d):
    return torch.where(d <= 0, torch.zeros_like(d), (torch.log2(d.clamp_min(1).float()).floor().long() + 1).clamp(max=6))


def _wy_post(model, xy, cfg, tl, parts):
    W, k, V = S["wy"], cfg.wy, cfg.vocab
    W["emb_fn"], W["head_out"], W["head_in"] = None, None, None
    W["att_on"] = False
    W["tl"] = tl
    lam = _wy_lam(cfg)
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    c = model.ctx
    logits = c["lg"]
    dev = logits.device
    aux = None
    if k in WY_NB:
        aux = _nb_soft(logits, y, W[WY_NB[k]])
    elif k in WY_HEAD:
        h = c["h"]
        o = model.tmods.wyh(model.tmods.wyn(h)).float()
        ar = torch.arange(T + 1, device=dev)
        INF = 10 ** 6
        def bce(z, t, m=None):
            l = F.binary_cross_entropy_with_logits(z, t.float(), reduction="none")
            return l.mean() if m is None else (l * m).sum() / m.sum().clamp_min(1)
        def ce(z, t, m=None):
            l = F.cross_entropy(z.reshape(-1, z.shape[-1]), t.reshape(-1), reduction="none").view(t.shape)
            return l.mean() if m is None else (l * m).sum() / m.sum().clamp_min(1)
        def nxt_true(flag):                           # (B, T+1): smallest j >= i with flag[xy[:, j]]
            v = torch.where(flag[xy].bool(), ar.expand(B, -1), torch.full_like(xy, INF))
            return v.flip(1).cummin(1).values.flip(1)
        if k == "nspace":
            aux = bce(o[..., 0], W["ws"][y])
        elif k == "nlen":
            aux = ce(o, W["blen"][y])
        elif k == "ncls":
            aux = ce(o, W["cls"][y])
        elif k == "nbyte1":
            aux = ce(o, W["b1"][y])
        elif k == "nbytel":
            aux = ce(o, W["bl"][y])
        elif k == "nbag":
            aux = bce(o, W["bag"][y]) * 8
        elif k == "lexall":
            aux = bce(o[..., 0], W["ws"][y]) + ce(o[..., 1:9], W["blen"][y]) + ce(o[..., 9:15], W["cls"][y])
        elif k == "nbyte2":
            m = torch.ones(B, T, device=dev); m[:, -1] = 0
            t2 = torch.cat([xy[:, 2:], xy[:, -1:]], 1)
            aux = ce(o, W["b1"][t2], m)
        elif k in ("nextword", "wordlen"):
            nx = nxt_true(W["ws"])                         # (B, T+1)
            j = torch.cat([nx[:, 2:], torch.full_like(nx[:, :1], INF)], 1)   # first word start at >= t+2
            m = (j < INF).float()
            if k == "nextword":
                aux = ce(o, xy.gather(1, j.clamp(max=T)), m)
            else:
                aux = ce(o, (j - (ar[:T] + 1)).clamp(1, 4).sub(1) * (j < INF), m)
        elif k == "wordend":
            m = torch.ones(B, T, device=dev); m[:, -1] = 0
            t2 = torch.cat([xy[:, 2:], xy[:, -1:]], 1)
            aux = bce(o[..., 0], W["ws"][t2] | (W["cls"][t2] == 3).long(), m)
        elif k in ("distnl", "distpunct", "distcap"):
            fl = {"distnl": W["nl"], "distpunct": W["send"], "distcap": W["ws"] * (W["cls"] == 1).long()}[k]
            nx = nxt_true(fl)[:, 1:]     # first at >= t+1
            d = nx - (ar[:T] + 1)
            aux = ce(o, torch.where(nx < INF, _bucket(d), torch.full_like(d, 7)))
        elif k == "sincepunct":
            v = torch.where(W["send"][x].bool(), ar[:T].expand(B, -1), torch.full_like(x, -INF))
            last = v.cummax(1).values
            d = ar[:T] - last
            aux = ce(o, torch.where(last > -INF, _bucket(d), torch.full_like(d, 7)))
        elif k == "rtd":
            aux = bce(o[..., 0], W["rmask"])
    elif k in ("ckl_noise", "ckl_tok"):
        if k == "ckl_noise":
            f = lambda out, idx: out + 0.1 * out.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(out)
        else:
            f = lambda out, idx: model.emb.weight[_corrupt(idx, k, W, V)[0]]
        lg2 = _second_pass(model, x, f)
        lp = F.log_softmax(logits.detach().float(), -1)
        aux = (lp.exp() * (lp - F.log_softmax(lg2, -1))).sum(-1).mean()
    elif k in ("fgsm", "fgsm4", "fgsmkl", "gradpen"):
        e = c["emb"]
        if k == "gradpen":
            g, = torch.autograd.grad(c["nll"].mean(), e, create_graph=True)
            aux = (g * (B * T)).pow(2).sum(-1).mean()
        else:
            g, = torch.autograd.grad(tl, e, retain_graph=True)
            eps = 0.1 if k == "fgsm" else 0.4
            dl = (eps * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12) * e.detach().norm(dim=-1, keepdim=True)).detach()
            lg2 = _second_pass(model, x, lambda out, idx: out + dl)
            if k == "fgsmkl":
                lp = F.log_softmax(logits.detach().float(), -1)
                aux = (lp.exp() * (lp - F.log_softmax(lg2, -1))).sum(-1).mean()
            else:
                aux = F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    elif k in ("embmix", "hmix1"):
        perm = torch.randperm(B, device=dev)
        lmb = float(np.random.beta(0.4, 0.4))
        lmb = max(lmb, 1 - lmb)
        mix = lambda out, idx=None: lmb * out + (1 - lmb) * out[perm]
        lg2 = _second_pass(model, x, mix if k == "embmix" else None, mix if k == "hmix1" else None)
        aux = lmb * F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1)) + \
              (1 - lmb) * F.cross_entropy(lg2.reshape(-1, V), y[perm].reshape(-1))
    elif k in ("margkl", "margent"):
        pbar = F.softmax(logits.float(), -1).mean((0, 1))
        if k == "margkl":
            hist = torch.bincount(y.reshape(-1), minlength=V).float()
            hist = hist / hist.sum()
            nz = hist > 0
            aux = (hist[nz] * (hist[nz].log() - pbar[nz].clamp_min(1e-12).log())).sum()
        else:
            aux = (pbar * pbar.clamp_min(1e-12).log()).sum()          # minus entropy
    elif k in ("esim", "esim_in"):
        Wm = (model.head.weight if k == "esim" else model.emb.weight).detach()
        Wn = F.normalize(Wm.float(), dim=-1)
        u, inv = y.reshape(-1).unique(return_inverse=True)
        sim = Wn[u] @ Wn.T
        sim[torch.arange(len(u), device=dev), u] = -1e4
        tv, ti = sim.topk(16, -1)
        q = F.softmax(tv / 0.1, -1)                                   # (U, 16)
        lp = F.log_softmax(logits.float(), -1).reshape(-1, V)
        aux = -(lp.gather(-1, ti[inv]) * q[inv]).sum(-1).mean()
    elif k in ("brier", "spher"):
        p = F.softmax(logits.float(), -1)
        py = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        if k == "brier":
            aux = (p.pow(2).sum(-1) - 2 * py + 1).mean()
        else:
            aux = -(py / p.norm(dim=-1)).mean()
    if aux is None and k in W3_ALL:
        aux = _w3_aux(model, xy, cfg, W, k)
    if aux is None and k in W4_ALL:
        aux = _w4_aux(model, xy, cfg, W, k)
    if aux is None and k in W5_ALL:
        aux = _w5_aux(model, xy, cfg, W, k)
    if aux is not None:
        tl = tl + lam * aux
        parts["wy"] = round(aux.item(), 4)
    return tl


# ---------------------------------------------------------------- round w3: add-on targets, new mechanism families
# Same switch as w2 (cfg.wy = mode, cfg.wyl = strength, cfg.wyp = rate/strength multiplier). Families:
# N attention supervision (aux loss on attention maps recomputed from q, k), Q sequence identity (contrastive halves,
# spliced / swapped / shuffled second pass), S self-model heads (predict own loss, correctness, rank, entropy),
# R in-context recurrence distances, P logit adjustment by the unigram prior (training logits only),
# J joint next-two-token hash head, B code-structure state heads, T reverse-direction LM, U calibration penalties,
# V context distillation (short-context pass matches full-context pass), W unlikelihood, E temporal smoothness of the
# final state, K layer-drop consistency. None uses previous-token / n-gram count statistics.
W3_HEAD = {"splice": 1, "sop": 1, "chunkshuf": 1, "losspred": 1, "corrpred": 1, "nlosspred": 1, "rankpred": 8,
           "entpred": 10, "futdist": 8, "pastdist": 8, "nuniq": 9, "pairhash": 4096, "pair16k": 16384, "depth": 8,
           "quote": 1}
W3_ATT = ("attprev", "attind", "attcopy", "attsharp", "attsmooth", "attdiv", "attsink", "attlocal", "attfar")
W3_LAM = {"attprev": 0.1, "attind": 0.1, "attcopy": 0.1, "attsharp": 0.01, "attsmooth": 0.01, "attdiv": 0.1,
          "attsink": 0.3, "attlocal": 0.1, "attfar": 0.1, "halfnce": 0.1, "halfnce_last": 0.1,
          "logadj": 1.0, "logadj05": 1.0, "logadjneg": 1.0, "revlm": 0.3, "calgap": 1.0, "confsq": 1.0,
          "ctxkd_half": 1.0, "ctxkd_8": 1.0, "ul_top": 0.3, "ul_ctx": 0.1, "ul_rep": 0.3, "hsmooth": 0.1,
          "htfloor": 0.3, "ldkl": 1.0, "ldce": 0.3}
W3_ALL = set(W3_HEAD) | set(W3_ATT) | set(W3_LAM)


def _w3_hooks(model, W):
    from model import rope

    def att_hook(i):
        def f(mod, inp, out):
            if not (mod.training and W.get("att_on")):
                return None
            x, cos, sin = inp[0], inp[1], inp[2]
            B, T, D = x.shape
            q, k, _ = mod.qkv(x).view(B, T, 3, mod.h, mod.dh).permute(2, 0, 3, 1, 4)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            sc = (q @ k.transpose(-1, -2)) / mod.dh ** 0.5
            sc = sc.float().masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            W["att"][i] = sc.softmax(-1)                     # (B, H, T, T)
            return None
        return f

    def skip_hook(i):
        def f(mod, inp, out):
            return inp[0] if (mod.training and W.get("skip") == i) else None
        return f

    for i, b in enumerate(model.blocks):
        b.attn.register_forward_hook(att_hook(i))
        b.register_forward_hook(skip_hook(i))


def _kl_to(logits_t, lg2):
    lp = F.log_softmax(logits_t.detach().float(), -1)
    return (lp.exp() * (lp - F.log_softmax(lg2.float(), -1))).sum(-1).mean()


def _soft_att(att, E):
    """mean over rows with at least one target position of the CE between attention row and uniform over E."""
    cnt = E.sum(-1)
    has = cnt > 0
    if not has.any():
        return att.sum() * 0.0
    g = (torch.log(att.clamp_min(1e-9)) * E).sum(-1)
    return -(g[has] / cnt[has]).mean()


def _w3_aux(model, xy, cfg, W, k):
    V = cfg.vocab
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    c = model.ctx
    logits = c["lg"].float()
    dev = logits.device
    ar = torch.arange(T + 1, device=dev)
    INF = 10 ** 6
    tri = torch.ones(T, T, dtype=torch.bool, device=dev).tril()   # j <= t
    pm = W.get("pmul", 1.0)
    # ---- N attention supervision
    if k in W3_ATT:
        A = W["att"]
        L = len(A)
        l1 = min(1, L - 1)
        if k == "attprev":
            a = A[0][:, 0]
            return -torch.log(a[:, 1:, :].diagonal(0, 1, 2).clamp_min(1e-9)).mean()
        if k == "attind":                                   # attend to j where x[j-1] == x[t] (induction)
            prv = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
            E = ((prv.unsqueeze(1) == x.unsqueeze(2)) & tri).float()
            return _soft_att(A[l1][:, 0], E)
        if k == "attcopy":                                  # attend to j where x[j] == y[t] (pointer to the answer)
            E = ((x.unsqueeze(1) == y.unsqueeze(2)) & tri).float()
            return _soft_att(A[l1][:, -1], E)
        if k in ("attsharp", "attsmooth"):
            ent = torch.stack([-(a * torch.log(a.clamp_min(1e-9))).sum(-1)[..., 1:].mean() for a in A.values()]).mean()
            return ent if k == "attsharp" else -ent
        if k == "attdiv":
            tot = 0.0
            for a in A.values():
                f = F.normalize(a.flatten(2), dim=-1)          # (B, H, T*T)
                g = f @ f.transpose(1, 2)
                H = g.shape[1]
                tot = tot + (g.sum((1, 2)) - g.diagonal(0, 1, 2).sum(-1)).mean() / max(1, H * (H - 1))
            return tot / L
        if k == "attsink":
            return torch.stack([a[..., 1:, 0].mean() for a in A.values()]).mean()
        if k in ("attlocal", "attfar"):
            dist = (ar[:T].view(T, 1) - ar[:T].view(1, T)).clamp_min(0).float() / T
            m = torch.stack([(a * dist).sum(-1).mean() for a in A.values()]).mean()
            return m if k == "attlocal" else -m
    # ---- heads on the final state
    if k in W3_HEAD and k not in ("splice", "sop", "chunkshuf"):
        h = c["h"]
        o = model.tmods.wyh(model.tmods.wyn(h)).float()
        nll = c["nll"].detach().float()

        def bce(z, t, m=None):
            l = F.binary_cross_entropy_with_logits(z, t.float(), reduction="none")
            return l.mean() if m is None else (l * m).sum() / m.sum().clamp_min(1)

        def ce(z, t, m=None):
            l = F.cross_entropy(z.reshape(-1, z.shape[-1]), t.reshape(-1), reduction="none").view(t.shape)
            return l.mean() if m is None else (l * m).sum() / m.sum().clamp_min(1)
        last = torch.ones(B, T, device=dev); last[:, -1] = 0
        if k == "losspred":
            return F.mse_loss(o[..., 0], nll / 5)
        if k == "corrpred":
            return bce(o[..., 0], logits.argmax(-1) == y)
        if k == "nlosspred":
            return ((o[..., 0] - torch.cat([nll[:, 1:], nll[:, -1:]], 1) / 5).pow(2) * last).sum() / last.sum()
        if k == "rankpred":
            ly = logits.gather(-1, y.unsqueeze(-1))
            rk = (logits > ly).sum(-1)
            return ce(o, _bucket(rk))
        if k == "entpred":
            p = F.softmax(logits, -1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
            return ce(o, ent.floor().long().clamp(0, 9))
        if k == "futdist":                                  # distance to the next occurrence of x[t]
            E = (xy.unsqueeze(1) == x.unsqueeze(2)) & (ar.view(1, 1, -1) > ar[:T].view(1, -1, 1))
            j = torch.where(E, ar.view(1, 1, -1), torch.full_like(E, INF, dtype=torch.long)).min(-1).values
            d = j - ar[:T]
            return ce(o, torch.where(j < INF, _bucket(d), torch.full_like(d, 7)))
        if k == "pastdist":                                 # distance back to the last earlier occurrence of y[t]
            E = (x.unsqueeze(1) == y.unsqueeze(2)) & tri
            j = torch.where(E, ar[:T].view(1, 1, -1), torch.full_like(E, -INF, dtype=torch.long)).max(-1).values
            d = ar[:T] - j + 1
            return ce(o, torch.where(j > -INF, _bucket(d), torch.full_like(d, 7)))
        if k == "nuniq":                                    # how many of the next 8 tokens are new to this sequence
            fo = torch.where(xy.unsqueeze(1) == xy.unsqueeze(2), ar.view(1, 1, -1),
                             torch.full((B, T + 1, T + 1), INF, device=dev, dtype=torch.long)).min(-1).values  # first occ.
            new = (fo.unsqueeze(1) > ar[:T].view(1, -1, 1))            # (B, T, T+1): token j first seen after t
            win = (ar.view(1, -1) > ar[:T].view(-1, 1)) & (ar.view(1, -1) <= ar[:T].view(-1, 1) + 8)
            return ce(o, (new & win).sum(-1).clamp(max=8))
        if k in ("pairhash", "pair16k"):
            y2 = torch.cat([xy[:, 2:], xy[:, -1:]], 1)
            return ce(o, (y * 7919 + y2) % W3_HEAD[k], last)
        if k == "depth":
            dd = W["brk"][xy].cumsum(1)[:, 1:]
            return ce(o, dd.clamp(-3, 4) + 3)
        if k == "quote":
            return bce(o[..., 0], W["quo"][xy].cumsum(1)[:, 1:] % 2)
    # ---- Q sequence identity (second pass on an altered sequence; head on its final states)
    if k in ("halfnce", "halfnce_last"):
        h = F.normalize(c["h"].float(), dim=-1)
        if k == "halfnce":
            a, b = h[:, :T // 2].mean(1), h[:, T // 2:].mean(1)
        else:
            a, b = h[:, T // 2 - 1], h[:, -1]
        z = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / 0.1
        return F.cross_entropy(z, torch.arange(B, device=dev))
    if k in ("splice", "sop", "chunkshuf"):
        flip = torch.rand(B, device=dev) < 0.5
        if k == "splice":
            perm = torch.roll(torch.arange(B, device=dev), 1)
            alt = torch.cat([x[:, :T // 2], x[perm, T // 2:]], 1)
            start = T // 2
        elif k == "sop":
            alt = torch.cat([x[:, T // 2:], x[:, :T // 2]], 1)
            start = T // 2
        else:
            nc = T // 8
            order = torch.argsort(torch.rand(B, nc, device=dev), -1)
            idx = (order.unsqueeze(-1) * 8 + torch.arange(8, device=dev)).view(B, T)
            alt = x.gather(1, idx)
            start = 8
        x2 = torch.where(flip.unsqueeze(1), alt, x)
        _second_pass(model, x2)
        o = model.tmods.wyh(model.tmods.wyn(W["h2"])).float()[:, start:, 0]
        return F.binary_cross_entropy_with_logits(o, flip.float().unsqueeze(1).expand_as(o))
    # ---- P logit adjustment: the change is in the head hook (training logits only); nothing extra here
    if k in ("logadj", "logadj05", "logadjneg"):
        return None
    # ---- T reverse-direction LM
    if k == "revlm":
        xr = xy.flip(1)
        lg2 = _second_pass(model, xr[:, :-1])
        return F.cross_entropy(lg2.reshape(-1, V), xr[:, 1:].reshape(-1))
    # ---- U calibration
    if k in ("calgap", "confsq"):
        p = F.softmax(logits, -1)
        mp, am = p.max(-1)
        acc = (am == y).float()
        return (mp.mean() - acc.mean()).pow(2) if k == "calgap" else (mp - acc).pow(2).mean()
    # ---- V context distillation
    if k == "ctxkd_half":
        lg2 = _second_pass(model, x[:, T // 2:])
        return _kl_to(logits[:, T // 2:], lg2)
    if k == "ctxkd_8":
        lg2 = _second_pass(model, x.reshape(B * (T // 8), 8))
        return _kl_to(logits.reshape(B * (T // 8), 8, V), lg2)
    # ---- W unlikelihood
    if k in ("ul_top", "ul_ctx", "ul_rep"):
        p = F.softmax(logits, -1)
        if k == "ul_top":
            q = p.scatter(-1, y.unsqueeze(-1), 0.0).max(-1).values
            return -torch.log((1 - q).clamp_min(1e-6)).mean()
        if k == "ul_rep":
            px = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            m = (x != y).float()
            return -(torch.log((1 - px).clamp_min(1e-6)) * m).sum() / m.sum().clamp_min(1)
        seen = torch.zeros(B, T, V, device=dev).scatter_(2, x.unsqueeze(-1), 1.0).cummax(1).values
        seen = seen.scatter(2, y.unsqueeze(-1), 0.0)
        return -(torch.log((1 - p).clamp_min(1e-6)) * seen).sum(-1).mean()
    # ---- E temporal smoothness of the final state
    if k in ("hsmooth", "htfloor"):
        h = F.normalize(c["h"].float(), dim=-1)
        dd = (h[:, 1:] - h[:, :-1]).pow(2).sum(-1)
        return dd.mean() if k == "hsmooth" else F.relu(0.5 * pm - dd.sqrt()).mean()
    # ---- K layer-drop consistency (middle block skipped in a second pass)
    if k in ("ldkl", "ldce"):
        W["skip"] = len(model.blocks) // 2
        lg2 = _second_pass(model, x)
        W["skip"] = None
        return _kl_to(logits, lg2) if k == "ldkl" else F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    return None


class NGram:
    """Count teachers. obig: online bigram from the tokens already trained on (counts updated after each step,
    so the current batch never sees its own targets). otri: online trigram with Dirichlet backoff to the online
    bigram. oskip: online mixture of bigram and gap-2 skip-bigram. ctrig: corpus trigram (20M tokens) with backoff
    to the corpus bigram. Smoothing: bigram q = (c + alpha*u)/(n + alpha), trigram q = (c + kappa*q_big)/(n + kappa)."""

    def __init__(self, kind, train, cfg, name, dev):
        self.kind, self.V, self.dev = kind, cfg.vocab, dev
        self.alpha = cfg.tb or 1.0
        self.kappa = cfg.te or 2.0
        V = self.V
        if kind == "ctrig":
            a, uni = _counts(train, V)
            M = bigram_csr(train, V, name)
            self.big = M
            self.uni = torch.from_numpy((uni + 1) / (uni + 1).sum()).float()
            f = os.path.join(CACHE, f"{name}_trigram.npz")
            if os.path.exists(f):
                z = np.load(f); self.tc, self.tn = z["c"], z["n"]
            else:
                code = (a[:-2] * V + a[1:-1]) * V + a[2:]
                self.tc, self.tn = np.unique(code, return_counts=True)
                np.savez(f, c=self.tc, n=self.tn)
        else:
            self.C = torch.zeros(V, V)
            self.u = torch.ones(V)
            if kind == "oskip":
                self.C2 = torch.zeros(V, V)
            if kind in ("otri", "oquad"):
                self.tc = np.zeros(0, dtype=np.int64); self.tn = np.zeros(0, dtype=np.int64)
            if kind == "oquad":
                self.qc = np.zeros(0, dtype=np.int64); self.qn = np.zeros(0, dtype=np.int64)

    def _big(self, ids):
        if self.kind == "ctrig":
            rows = torch.from_numpy(self.big[ids.numpy()].toarray())
            u = self.uni
        else:
            rows = self.C[ids]
            u = self.u / self.u.sum()
        return (rows + self.alpha * u) / (rows.sum(-1, keepdim=True) + self.alpha)

    def _tri(self, ctx, qb, tc=None, tn=None):
        V = self.V
        if tc is None:
            tc, tn = self.tc, self.tn
        lo = np.searchsorted(tc, ctx * V, "left")
        hi = np.searchsorted(tc, ctx * V + V, "left")
        cnt = torch.zeros_like(qb)
        nrow = hi - lo
        if nrow.sum() > 0:
            r = np.repeat(np.arange(len(ctx)), nrow)
            idx = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
            cnt[torch.from_numpy(r), torch.from_numpy(tc[idx] % V)] = torch.from_numpy(tn[idx]).float()
        n = cnt.sum(-1, keepdim=True)
        return (cnt + self.kappa * qb) / (n + self.kappa)

    def query(self, xy):
        x = xy[:, :-1].cpu()
        B, T = x.shape
        ids = x.reshape(-1)
        qb = self._big(ids)
        if self.kind in ("otri", "ctrig", "oquad"):
            prev = torch.cat([torch.full((B, 1), -1), x[:, :-1]], 1).reshape(-1).numpy()
            ctx = np.where(prev >= 0, prev * self.V + ids.numpy(), -1).astype(np.int64)
            ctx = np.where(ctx < 0, 0, ctx)          # first position: unused pair id 0 -> mostly empty
            qb = self._tri(ctx, qb)
        if self.kind == "oquad":
            p2 = torch.cat([torch.full((B, 2), -1), x[:, :-2]], 1).reshape(-1).numpy()
            c3 = np.where((p2 >= 0) & (prev >= 0), (p2 * self.V + prev) * self.V + ids.numpy(), 0).astype(np.int64)
            qb = self._tri(c3, qb, self.qc, self.qn)
        if self.kind == "oskip":
            prev = torch.cat([x[:, :1], x[:, :-1]], 1).reshape(-1)
            r2 = self.C2[prev]
            q2 = (r2 + self.alpha * self.u / self.u.sum()) / (r2.sum(-1, keepdim=True) + self.alpha)
            qb = 0.7 * qb + 0.3 * q2
        return qb.to(self.dev)

    def update(self, xy):
        if self.kind == "ctrig":
            return
        xy = xy.cpu()
        x, y = xy[:, :-1].reshape(-1), xy[:, 1:].reshape(-1)
        V = self.V
        self.C.view(-1).index_add_(0, x * V + y, torch.ones(len(x)))
        self.u += torch.bincount(xy.reshape(-1), minlength=V).float()
        if self.kind == "oskip":
            x2, y2 = xy[:, :-2].reshape(-1), xy[:, 2:].reshape(-1)
            self.C2.view(-1).index_add_(0, x2 * V + y2, torch.ones(len(x2)))
        if self.kind in ("otri", "oquad"):
            a = xy.numpy().astype(np.int64)
            code = ((a[:, :-2] * V + a[:, 1:-1]) * V + a[:, 2:]).reshape(-1)
            self.tc, self.tn = _merge(self.tc, self.tn, code)
        if self.kind == "oquad":
            code = (((a[:, :-3] * V + a[:, 1:-2]) * V + a[:, 2:-1]) * V + a[:, 3:]).reshape(-1)
            self.qc, self.qn = _merge(self.qc, self.qn, code)

    def rows(self, ids):
        return self.C[ids.cpu()]


def _merge(tc, tn, code):
    # ops 2026-11-22 18:55 SGT: sorted insert instead of np.unique over the whole table every step (that grew with
    # the step count: 165 ms/step at step 2400 vs 21 ms now). Output arrays are identical (checked over 2,400 steps).
    u, c = np.unique(code, return_counts=True)
    pos = np.searchsorted(tc, u)
    hit = pos < len(tc)
    hit[hit] = tc[pos[hit]] == u[hit]
    tn = tn.copy()
    tn[pos[hit]] += c[hit]
    new = ~hit
    return np.insert(tc, pos[new], u[new]), np.insert(tn, pos[new], c[new].astype(np.int64))


def _run_blocks(model, x):
    T = x.shape[1]
    for b in model.blocks:
        x = b(x, model.cos[:T], model.sin[:T])
    return x


# ---------------------------------------------------------------- round w4: add-on losses, new mechanism families
# Same switch as w2/w3 (cfg.wy = mode, cfg.wyl = strength, cfg.wyp = rate multiplier). Families:
# RS restricted / sampled softmax (extra CE over a subset of the vocab), OC output codes (bits / ECOC of the label),
# OP own-prediction inputs (scheduled sampling as a second pass), GN per-token gradient-norm shaping,
# WP weight-space penalties (head / embedding / expert geometry, head anchored to its EMA),
# SN snapshot teachers (frozen copy refreshed every K steps), RT router-score targets, EX expert-drop second pass,
# KD attention key-drop second pass, LP latent prediction of the model's own future final state, PS output smoothness.
W4_HEAD = {"oc_bits": 13, "oc_fbits": 13, "oc_ecoc64": 64, "oc_embsign": -1,
           "lp_h1": -1, "lp_h2": -1, "lp_h4": -1, "lp_inv": -1}
W4_LAM = {"rs_top32": 0.3, "rs_rand256": 0.3, "rs_freq256": 0.3, "rs_vdrop20": 1.0, "rs_near32": 0.3,
          "oc_bits": 0.3, "oc_fbits": 0.3, "oc_ecoc64": 0.3, "oc_embsign": 0.3,
          "op_arg10": 0.5, "op_samp10": 0.5, "op_arg10kl": 1.0, "op_wrong": 0.5,
          "gn_tok": 1.0, "gn_clip": 1.0, "gn_seqinv": 1.0,
          "wp_headortho": 0.1, "wp_embunif": 0.1, "wp_rowcent": 1.0, "wp_expdiv": 0.1, "wp_embhead": 0.1,
          "wp_headema": 0.1,
          "sn_kd100": 0.5, "sn_kd25": 0.5, "sn_kd10": 0.5, "sn_kd50": 0.5, "sn_anti100": 0.3, "sn_gain100": 1.0,
          "rt_ent": 0.01, "rt_slow": 0.05, "rt_tokcons": 0.1, "rt_gateortho": 0.1,
          "ex_drop": 0.3, "ex_shared": 0.1,
          "kd_drop30": 0.5, "kd_drop30kl": 1.0, "kd_far": 0.3,
          "lp_h1": 0.3, "lp_h2": 0.3, "lp_h4": 0.3, "lp_inv": 0.3, "ps_klnb": 0.05}
W4_ALL = set(W4_HEAD) | set(W4_LAM)


def _w4_setup(model, train, cfg, W, dev):
    k, V = cfg.wy, cfg.vocab
    if k in ("rs_freq256", "oc_fbits"):
        _, uni = _counts(train, V)
        u = torch.from_numpy((uni + 1) / (uni + 1).sum()).float().to(dev)
        W["neg"] = u.pow(0.75) / u.pow(0.75).sum()
        W["frank"] = torch.empty(V, dtype=torch.long, device=dev).scatter_(
            0, torch.argsort(u, descending=True), torch.arange(V, device=dev))
    if k == "oc_ecoc64":
        g = torch.Generator().manual_seed(1234)
        W["code"] = (torch.rand(V, 64, generator=g) < 0.5).float().to(dev)
    if k == "wp_headema":
        W["hema"] = model.head.weight.detach().clone()
    if k.startswith("sn_"):
        import copy
        snap = copy.deepcopy(model)          # built before any forward: ctx holds no graph tensors yet
        for p_ in snap.parameters():
            p_.requires_grad_(False)
        W["n4"], W["snap"] = 0, snap
    if k.startswith("rt_") or k.startswith("ex_"):
        W["rt"], W["rt_on"], W["exmode"] = [], False, None
        for m in model.moe_layers():
            orig = m.route

            def r(xs, T, B, feat=None, _m=m, _o=orig):
                if W["rt_on"]:
                    W["rt"].append((_m, xs if feat is None else feat, T, B))
                probs, rmap = _o(xs, T, B, feat)
                if W["exmode"] == "drop1":
                    top = probs.argmax(-1, keepdim=True)
                    keep = probs.sum(-1, keepdim=True)
                    probs = probs.scatter(1, top, 0.0)
                    probs = probs / (probs.sum(-1, keepdim=True) + 1e-20) * keep
                    rmap = rmap.scatter(1, top, False)
                elif W["exmode"] == "shared":
                    probs, rmap = probs * 0.0, rmap & False
                return probs, rmap
            m.route = r
    if k.startswith("kd_"):
        W["kmask"] = None
        from model import rope

        def kd_hook(mod, inp, out):
            km = W.get("kmask")
            if not (mod.training and km is not None):
                return None
            x, cos, sin = inp[0], inp[1], inp[2]
            B, T, D = x.shape
            q, kk, v = mod.qkv(x).view(B, T, 3, mod.h, mod.dh).permute(2, 0, 3, 1, 4)
            q, kk = rope(q, cos, sin), rope(kk, cos, sin)
            yy = F.scaled_dot_product_attention(q, kk, v, attn_mask=km)
            return mod.o(yy.transpose(1, 2).reshape(B, T, D))
        for b in model.blocks:
            b.attn.register_forward_hook(kd_hook)


def _w4_pre(model, xy, cfg, W, k):
    if k.startswith("rt_"):
        W["rt"], W["rt_on"] = [], True


def _rs_ce(logits, y, neg):
    """CE over {label} + neg (B, T, K); negatives equal to the label are masked."""
    cand = torch.cat([y.unsqueeze(-1), neg], -1)
    z = logits.gather(-1, cand)
    z = torch.cat([z[..., :1], z[..., 1:].masked_fill(neg == y.unsqueeze(-1), float("-inf"))], -1)
    return F.cross_entropy(z.reshape(-1, z.shape[-1]), torch.zeros(z.shape[0] * z.shape[1], dtype=torch.long,
                                                                       device=z.device))


def _w4_aux(model, xy, cfg, W, k):
    V = cfg.vocab
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    c = model.ctx
    logits = c["lg"].float()
    dev = logits.device
    pm = W.get("pmul", 1.0)
    if k.startswith("rt_") or k.startswith("ex_"):
        W["rt_on"] = False
    # ---- RS restricted softmax
    if k.startswith("rs_"):
        if k == "rs_top32":
            return _rs_ce(logits, y, logits.detach().topk(32, -1).indices)
        if k == "rs_rand256":
            return _rs_ce(logits, y, torch.randint(0, V, (B, T, 256), device=dev))
        if k == "rs_freq256":
            return _rs_ce(logits, y, torch.multinomial(W["neg"], B * T * 256, replacement=True).view(B, T, 256))
        if k == "rs_near32":
            Wn = F.normalize(model.head.weight.detach().float(), dim=-1)
            uy, inv = torch.unique(y, return_inverse=True)
            nb = (Wn[uy] @ Wn.T).topk(33, -1).indices[:, 1:]
            return _rs_ce(logits, y, nb[inv])
        if k == "rs_vdrop20":
            drop = torch.rand(B, T, V, device=dev) < 0.2 * pm
            drop.scatter_(-1, y.unsqueeze(-1), False)
            z = logits.masked_fill(drop, float("-inf"))
            return F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
    # ---- OC output codes and LP latent prediction (extra head on the final state)
    if k in W4_HEAD:
        h = c["h"]
        o = model.tmods.wyh(model.tmods.wyn(h)).float()
        if k.startswith("oc_"):
            if k == "oc_bits":
                t = ((y.unsqueeze(-1) >> torch.arange(13, device=dev)) & 1).float()
            elif k == "oc_fbits":
                t = ((W["frank"][y].unsqueeze(-1) >> torch.arange(13, device=dev)) & 1).float()
            elif k == "oc_ecoc64":
                t = W["code"][y]
            else:
                t = (model.emb.weight[y].detach().float() > 0).float()
            return F.binary_cross_entropy_with_logits(o, t)
        hd = F.normalize(h.detach().float(), dim=-1)
        if k == "lp_h1":
            return (1 - F.cosine_similarity(o[:, :-1], hd[:, 1:], dim=-1)).mean()
        if k == "lp_h2":
            return (1 - F.cosine_similarity(o[:, :-2], hd[:, 2:], dim=-1)).mean()
        if k == "lp_h4":
            cs = torch.cat([torch.zeros_like(hd[:, :1]), hd.cumsum(1)], 1)       # cs[i] = sum hd[:i]
            tgt = (cs[:, 5:] - cs[:, 1:-4]) / 4                                   # mean hd[t+1..t+4], t = 0..T-5
            return (1 - F.cosine_similarity(o[:, :T - 4], tgt, dim=-1)).mean()
        if k == "lp_inv":
            return (1 - F.cosine_similarity(o[:, 1:], hd[:, :-1], dim=-1)).mean()
    # ---- OP own-prediction inputs
    if k.startswith("op_"):
        if k == "op_samp10":
            pred = torch.multinomial(F.softmax(logits.detach(), -1).reshape(-1, V), 1).view(B, T)
        else:
            pred = logits.detach().argmax(-1)                  # pred[:, i] = guess for x[:, i+1]
        prop = torch.cat([x[:, :1], pred[:, :-1]], 1)          # proposed input at position i
        msk = torch.rand(B, T, device=dev) < 0.1 * pm
        msk[:, 0] = False
        if k == "op_wrong":
            msk = (torch.rand(B, T, device=dev) < 0.3 * pm) & (prop != x)
            msk[:, 0] = False
        x2 = torch.where(msk, prop, x)
        lg2 = _second_pass(model, x, lambda out, idx: model.emb.weight[x2])
        if k == "op_arg10kl":
            return _kl_to(logits, lg2)
        return F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    # ---- GN per-token gradient-norm shaping (reweights the hard-label CE)
    if k.startswith("gn_"):
        nll = c["nll"].float()
        p = F.softmax(logits.detach(), -1)
        py = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        gn = (p.pow(2).sum(-1) - 2 * py + 1).clamp_min(1e-6).sqrt()   # ||p - onehot||
        if k == "gn_tok":
            w = 1 / gn
        elif k == "gn_clip":
            w = (0.7 / gn).clamp(max=1.0)
        else:
            w = (1 / nll.detach().mean(1, keepdim=True).clamp_min(1e-3)).expand(B, T)
        w = w / w.mean()
        return ((w - 1) * nll).mean()
    # ---- WP weight-space penalties
    if k.startswith("wp_"):
        if k == "wp_headortho":
            Wn = F.normalize(model.head.weight[torch.randint(0, V, (256,), device=dev)].float(), dim=-1)
            G = Wn @ Wn.T
            return (G - torch.eye(256, device=dev)).pow(2).mean()
        if k == "wp_embunif":
            E = F.normalize(model.emb.weight[torch.randint(0, V, (512,), device=dev)].float(), dim=-1)
            d2 = torch.cdist(E, E).pow(2)
            off = ~torch.eye(512, dtype=torch.bool, device=dev)
            return torch.log(torch.exp(-2 * d2[off]).mean())
        if k == "wp_rowcent":
            return model.head.weight.float().mean(0).pow(2).sum()
        if k == "wp_expdiv":
            tot = 0.0
            for m in model.moe_layers():
                f = F.normalize(m.w1.flatten(1).float(), dim=-1)
                G = f @ f.T
                n = G.shape[0]
                tot = tot + (G - torch.eye(n, device=dev)).pow(2).sum() / (n * (n - 1))
            return tot / len(model.moe_layers())
        if k == "wp_embhead":
            uy = torch.unique(y)
            return (1 - F.cosine_similarity(model.head.weight[uy].float(), model.emb.weight[uy].float(), dim=-1)).mean()
        if k == "wp_headema":
            hw = model.head.weight
            a = (hw.float() - W["hema"].float()).pow(2).sum()
            W["hema"].mul_(0.99).add_(hw.detach(), alpha=0.01)
            return a
    # ---- SN snapshot teachers
    if k.startswith("sn_"):
        K = {"sn_kd25": 25, "sn_kd10": 10, "sn_kd50": 50}.get(k, 100)
        if W["n4"] % K == 0:
            W["snap"].load_state_dict(model.state_dict())
        W["n4"] += 1
        with torch.no_grad():
            lgs = W["snap"](x).float()
        if k in ("sn_kd100", "sn_kd25", "sn_kd10", "sn_kd50"):
            return _kl_to(lgs, logits)
        lpm = F.log_softmax(logits, -1)
        if k == "sn_anti100":
            ts = lgs.argmax(-1)
            wrong = (ts != y).float()
            pt = lpm.gather(-1, ts.unsqueeze(-1)).squeeze(-1).exp()
            return (-torch.log((1 - pt).clamp_min(1e-6)) * wrong).mean()
        lps = F.log_softmax(lgs, -1)
        ns = -lps.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        nc = -lpm.detach().gather(-1, y.unsqueeze(-1)).squeeze(-1)
        m = (ns < nc).float()
        kl = (lps.exp() * (lps - lpm)).sum(-1)
        return (kl * m).sum() / m.sum().clamp_min(1)
    # ---- RT router-score targets
    if k.startswith("rt_"):
        tot, n = 0.0, 0
        if k == "rt_gateortho":
            for m in model.moe_layers():
                g = F.normalize(m.gate.float(), dim=-1)
                E = g.shape[0]
                tot = tot + (g @ g.T - torch.eye(E, device=dev)).pow(2).sum() / (E * (E - 1))
                n += 1
            return tot / max(1, n)
        for m, f, TT, BB in W["rt"]:
            s = torch.sigmoid(F.linear(f.float(), m.gate.float()))
            sn = s / (s.sum(-1, keepdim=True) + 1e-9)                # (T*B, E), seq-first
            if k == "rt_ent":
                tot = tot + (-(sn * torch.log(sn.clamp_min(1e-9))).sum(-1)).mean()
            elif k == "rt_slow":
                q = sn.view(TT, BB, -1)
                tot = tot + (q[1:] - q[:-1]).abs().sum(-1).mean()
            else:                                                     # rt_tokcons
                ids = x.transpose(0, 1).reshape(-1)
                cnt = torch.zeros(V, device=dev).index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
                mu = torch.zeros(V, sn.shape[1], device=dev).index_add_(0, ids, sn) / cnt.clamp_min(1).unsqueeze(-1)
                tot = tot + (sn - mu[ids].detach()).pow(2).sum(-1).mean()
            n += 1
        W["rt"] = []
        return tot / max(1, n) if n else logits.sum() * 0.0
    # ---- EX expert drop second pass
    if k.startswith("ex_"):
        W["exmode"] = "drop1" if k == "ex_drop" else "shared"
        try:
            lg2 = _second_pass(model, x)
        finally:
            W["exmode"] = None
        return F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    # ---- KD attention key-drop second pass
    if k.startswith("kd_"):
        ar = torch.arange(T, device=dev)
        causal = ar.view(T, 1) >= ar.view(1, T)
        eye = ar.view(T, 1) == ar.view(1, T)
        if k == "kd_far":
            d = ar.view(T, 1) - ar.view(1, T)
            km = (causal & ~((d == 1) | (d == 2))).view(1, 1, T, T)
        else:
            keep = torch.rand(B, 1, 1, T, device=dev) >= 0.3 * pm
            km = causal.view(1, 1, T, T) & (keep | eye.view(1, 1, T, T))
        W["kmask"] = km
        try:
            lg2 = _second_pass(model, x)
        finally:
            W["kmask"] = None
        if k == "kd_drop30kl":
            return _kl_to(logits, lg2)
        return F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    # ---- PS output smoothness
    if k == "ps_klnb":
        lp = F.log_softmax(logits, -1)
        t = lp[:, :-1].detach()
        return (t.exp() * (t - lp[:, 1:])).sum(-1).mean()
    return None


# ---------------------------------------------------------------- round w5: add-on losses, new mechanism families
# Same switch as w2-w4 (cfg.wy = mode, cfg.wyl = strength). Families:
# N* attention lead (w3 attprev follow-ups), FL flooding / label relaxation, L* Fenchel-Young and robust losses
# (entmax, GCE), FM fill-in-the-middle second pass, LR listwise ranking of the corpus teacher's top 8,
# SL sequence-likelihood calibration (real sequence beats a corrupted one), VB variational information bottleneck on
# the head input, PC predictive coding between blocks, OT expected head/embedding distance to the label,
# WN weight-noise second pass, AM additive label margin, KN kNN teacher from a memory of the model's own states,
# MR Matryoshka (nested hidden dims, nested vocabulary), EB expert-balanced token weights, TK top-k hinge,
# LN LogitNorm. None uses previous-token / n-gram count statistics as a new signal.
W5_HEAD = {"vb_3": -1, "vb_2": -1, "pc_next": -1, "pc_res": -1}
W5_ATT = ("at_prev_l1", "at_prev2", "at_prev_both", "at_prev_dec", "at_prev_soft", "at_prev_hinge")
W5_LAM = {"at_prev_l1": 0.1, "at_prev2": 0.1, "at_prev_both": 0.1, "at_prev_dec": 0.1, "at_prev_soft": 0.1,
          "at_prev_hinge": 0.1,
          "fl_glob": 1.0, "fl_tok": 1.0, "lr_relax": 1.0,
          "fy_sparse": 0.3, "fy_ent15": 0.3, "fy_gce7": 1.0, "fy_gce3": 1.0,
          "fm_psm": 0.3, "fm_spm": 0.3,
          "lr_listmle8": 0.1, "lr_listnet8": 0.3,
          "sl_swap": 1.0, "sl_rand": 1.0,
          "vb_3": 1e-3, "vb_2": 1e-2,
          "pc_next": 0.1, "pc_res": 0.1,
          "ot_head": 1.0, "ot_emb": 1.0,
          "wn_ce": 0.5, "wn_kl": 1.0,
          "am_05": 0.5, "am_10": 0.5,
          "kn_16": 0.2, "kn_64": 0.2,
          "mr_dims": 0.3, "mr_vocab": 0.3,
          "eb_cnt": 1.0, "eb_conf": 1.0,
          "tk_5": 0.1, "tk_20": 0.1,
          "ln_norm": 0.3}
W5_ALL = set(W5_HEAD) | set(W5_LAM)


def _w5_setup(model, train, cfg, W, dev):
    k, V = cfg.wy, cfg.vocab
    if k in W5_ATT:
        _w3_hooks(model, W)
    if k == "mr_vocab":
        _, uni = _counts(train, V)
        u = torch.from_numpy((uni + 1) / (uni + 1).sum()).float().to(dev)
        W["frank"] = torch.empty(V, dtype=torch.long, device=dev).scatter_(
            0, torch.argsort(u, descending=True), torch.arange(V, device=dev))
        W["topv"] = torch.argsort(u, descending=True)[:1024]
    if k.startswith("vb_"):
        h = model.tmods.wyh
        nn.init.zeros_(h.weight)
        nn.init.constant_(h.bias, -3.0)                    # sigma about 0.05 at start
    if k.startswith("kn_"):
        n = 8192 if k == "kn_16" else 32768
        W["kb_k"] = torch.zeros(n, cfg.d_model, device=dev)
        W["kb_y"] = torch.zeros(n, dtype=torch.long, device=dev)
        W["kb_n"], W["kb_p"] = 0, 0
    if k.startswith("eb_"):
        W["eb_on"], W["eb"] = False, None
        m0 = model.moe_layers()[0]
        orig = m0.route

        def r(xs, T, B, feat=None, _o=orig):
            probs, rmap = _o(xs, T, B, feat)
            if W["eb_on"] and W["eb"] is None:
                W["eb"] = (probs.detach().float(), T, B)
            return probs, rmap
        m0.route = r
    W["tlema"] = None


def _w5_pre(model, xy, cfg, W, k):
    if k in W5_ATT:
        W["att_on"], W["att"] = True, {}
    if k.startswith("eb_"):
        W["eb_on"], W["eb"] = True, None
    if k.startswith("vb_"):
        def f(hn):
            ls = model.tmods.wyh(hn).clamp(-8, 2)
            W["vb"] = (hn, ls)
            return hn + torch.exp(ls) * torch.randn_like(hn)
        W["head_in"] = f


def _entmax(z, alpha, it=30):
    """alpha-entmax by bisection over the last dim (alpha > 1)."""
    zs = (alpha - 1) * z
    mx = zs.max(-1, keepdim=True).values
    lo, hi = mx - 1.0, mx - (1.0 / z.shape[-1]) ** (alpha - 1)
    for _ in range(it):
        tau = (lo + hi) / 2
        s = (zs - tau).clamp_min(0).pow(1 / (alpha - 1)).sum(-1, keepdim=True)
        lo = torch.where(s >= 1, tau, lo)
        hi = torch.where(s < 1, tau, hi)
    p = (zs - lo).clamp_min(0).pow(1 / (alpha - 1))
    return p / p.sum(-1, keepdim=True)


def _second_pass_params(model, x, noise):
    """forward with every weight matrix perturbed by noise * its RMS * N(0,1); keeps MoE aux, counts and ctx."""
    from torch.func import functional_call
    moes = model.moe_layers()
    sv = [(dict(m.aux), m.local_tokens_per_expert.clone() if hasattr(m, "local_tokens_per_expert") else None) for m in moes]
    ctx = {kk: model.ctx.get(kk) for kk in ("h", "hs", "emb")}
    P = {}
    for n_, p_ in model.named_parameters():
        if p_.dim() >= 2 and not n_.startswith("tmods"):
            P[n_] = p_ + noise * p_.detach().float().pow(2).mean().sqrt().to(p_.dtype) * torch.randn_like(p_)
        else:
            P[n_] = p_
    lg = functional_call(model, P, (x,), strict=False)
    for m, (a, c) in zip(moes, sv):
        m.aux = a
        if c is not None:
            m.local_tokens_per_expert.copy_(c)
    model.ctx.update(ctx)
    return lg.float()


def _fim(xy, spm):
    """per row: random a < b; PSM order [0:a] [b:] [a:b] (SPM: [b:] [0:a] [a:b]). returns new xy and a label mask
    that drops the two labels across a join."""
    B, N = xy.shape
    dev = xy.device
    a = torch.randint(1, N - 2, (B,), device=dev)
    b = a + 1 + (torch.rand(B, device=dev) * (N - 1 - a).float()).long().clamp(max=N - 2 - a)
    ar = torch.arange(N, device=dev).unsqueeze(0)
    la, lb = a.unsqueeze(1), b.unsqueeze(1)
    n_suf = N - lb
    if not spm:                               # [0:a] [b:N] [a:b]
        src = torch.where(ar < la, ar, torch.where(ar < la + n_suf, ar - la + lb, ar - la - n_suf + la))
        j1, j2 = la, la + n_suf               # first positions of the 2nd and 3rd pieces
    else:                                     # [b:N] [0:a] [a:b]
        src = torch.where(ar < n_suf, ar + lb, torch.where(ar < n_suf + la, ar - n_suf, ar - n_suf))
        j1, j2 = n_suf, n_suf + la
    new = xy.gather(1, src)
    lab = torch.ones(B, N - 1, device=dev)
    for j in (j1, j2):                         # label at position j-1 predicts the first token of a new piece
        lab.scatter_(1, (j - 1).clamp(0, N - 2), 0.0)
    return new, lab


def _w5_aux(model, xy, cfg, W, k):
    V = cfg.vocab
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    c = model.ctx
    logits = c["lg"].float()
    dev = logits.device
    if k.startswith("eb_"):
        W["eb_on"] = False
    nll = c["nll"].float()
    # ---- N* attention lead
    if k in W5_ATT:
        A = W["att"]
        L = len(A)
        l1 = min(1, L - 1)
        a0 = A[0][:, 0]
        prev = a0[:, 1:, :].diagonal(0, 1, 2).clamp_min(1e-9)          # a[t, t-1], t >= 1
        if k == "at_prev_l1":
            return -torch.log(A[l1][:, 0][:, 1:, :].diagonal(0, 1, 2).clamp_min(1e-9)).mean()
        if k == "at_prev2":
            return -torch.log(A[0][:, 1][:, 2:, :].diagonal(0, 1, 2).clamp_min(1e-9)).mean()
        if k == "at_prev_both":
            tri = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            prv = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
            E = ((prv.unsqueeze(1) == x.unsqueeze(2)) & tri).float()
            return -torch.log(prev).mean() + _soft_att(A[l1][:, 0], E)
        if k == "at_prev_dec":
            return -torch.log(prev).mean() * max(0.0, 1.0 - W.get("frac", 0.0))
        if k == "at_prev_soft":
            selfa = a0[:, 1:, 1:].diagonal(0, 1, 2).clamp_min(1e-9)
            return -(0.7 * torch.log(prev) + 0.3 * torch.log(selfa)).mean()
        if k == "at_prev_hinge":
            return F.relu(np.log(0.5) - torch.log(prev)).mean()
    # ---- FL flooding / label relaxation
    if k == "fl_glob":
        tl = W["tl"]
        b = tl.detach() if W["tlema"] is None else 0.97 * W["tlema"]
        W["tlema"] = tl.detach() if W["tlema"] is None else 0.98 * W["tlema"] + 0.02 * tl.detach()
        return (tl - b).abs() + b - tl
    if k == "fl_tok":
        b = 0.5
        return ((nll - b).abs() + b - nll).mean()
    if k == "lr_relax":
        py = (-nll).exp().detach()
        return -(1 - cfg.ta) * (nll * (py > 0.9).float()).mean()
    # ---- L* Fenchel-Young / robust losses (FY: gradient p_alpha - onehot, value shown is <p - e_y, z>)
    if k in ("fy_sparse", "fy_ent15"):
        with torch.no_grad():
            p = _entmax(logits, 2.0 if k == "fy_sparse" else 1.5)
            g = p - F.one_hot(y, V).float()
        return (g * logits).sum(-1).mean()
    if k in ("fy_gce7", "fy_gce3"):
        q = 0.7 if k == "fy_gce7" else 0.3
        py = (-nll).exp()
        return ((1 - py.clamp_min(1e-9).pow(q)) / q).mean()
    # ---- FM fill in the middle
    if k in ("fm_psm", "fm_spm"):
        xy2, lab = _fim(xy, k == "fm_spm")
        lg2 = _second_pass(model, xy2[:, :-1])
        l = F.cross_entropy(lg2.reshape(-1, V), xy2[:, 1:].reshape(-1), reduction="none").view(B, T)
        return (l * lab).sum() / lab.sum()
    # ---- LR listwise ranking of the corpus teacher's top 8
    if k in ("lr_listmle8", "lr_listnet8"):
        with torch.no_grad():
            q = shape_teacher(S["ng"].query(xy).view(B, T, V), cfg)
            qv, qi = q.topk(8, -1)
        z8 = logits.gather(-1, qi)
        if k == "lr_listmle8":
            return (torch.logcumsumexp(z8.flip(-1), -1).flip(-1) - z8).mean()
        qn = qv / qv.sum(-1, keepdim=True)
        return -(qn * F.log_softmax(z8, -1)).sum(-1).mean()
    # ---- SL sequence-likelihood calibration
    if k in ("sl_swap", "sl_rand"):
        if k == "sl_swap":
            xy2, _ = _corrupt(xy, "inswap10", W, V)
            m = 0.1
        else:
            xy2, _ = _corrupt(xy, "inrand10", W, V)
            m = 0.2
        lg2 = _second_pass(model, xy2[:, :-1])
        n2 = F.cross_entropy(lg2.reshape(-1, V), xy2[:, 1:].reshape(-1), reduction="none").view(B, T)
        return F.relu(m - (n2.mean(1) - nll.mean(1))).mean()
    # ---- VB variational information bottleneck on the head input
    if k.startswith("vb_"):
        mu, ls = W["vb"]
        mu = mu.float()
        ls = ls.float()
        return 0.5 * (mu.pow(2) + torch.exp(2 * ls) - 1 - 2 * ls).sum(-1).mean()
    # ---- PC predictive coding between blocks
    if k in ("pc_next", "pc_res"):
        hs = c["hs"]
        tot = 0.0
        for i in range(len(hs) - 1):
            o = model.tmods.wyh(model.tmods.wyn(hs[i])).float()
            t = hs[i + 1].detach().float() if k == "pc_next" else (hs[i + 1] - hs[i]).detach().float()
            tot = tot + (1 - F.cosine_similarity(o, t, dim=-1)).mean()
        return tot / max(1, len(hs) - 1)
    # ---- OT expected distance between the prediction and the label under a head / embedding geometry
    if k in ("ot_head", "ot_emb"):
        Wm = model.head.weight if k == "ot_head" else model.emb.weight
        Wn = F.normalize(Wm.detach().float(), dim=-1)
        pbar = F.softmax(logits, -1) @ Wn
        return (1 - (pbar * Wn[y]).sum(-1)).mean()
    # ---- WN weight-noise second pass
    if k in ("wn_ce", "wn_kl"):
        lg2 = _second_pass_params(model, x, 0.05)
        if k == "wn_kl":
            return _kl_to(logits, lg2)
        return F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
    # ---- AM additive margin on the label logit
    if k in ("am_05", "am_10"):
        m = 0.5 if k == "am_05" else 1.0
        z = logits - m * F.one_hot(y, V).float()
        return F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
    # ---- KN kNN teacher from a memory of the model's own final states
    if k.startswith("kn_"):
        K = 16 if k == "kn_16" else 64
        hn = F.normalize(model.norm(c["h"]).detach().float().reshape(-1, cfg.d_model), dim=-1)
        out = None
        n = W["kb_n"]
        if n >= K * 4:
            sim = hn @ W["kb_k"][:n].T / 0.1
            v, i = sim.topk(K, -1)
            w = F.softmax(v, -1)
            q = torch.zeros(B * T, V, device=dev).scatter_add_(1, W["kb_y"][:n][i], w)
            out = -(q * F.log_softmax(logits.reshape(-1, V), -1)).sum(-1).mean()
        N = W["kb_k"].shape[0]
        idx = (W["kb_p"] + torch.arange(B * T, device=dev)) % N
        W["kb_k"][idx] = hn
        W["kb_y"][idx] = y.reshape(-1)
        W["kb_p"] = (W["kb_p"] + B * T) % N
        W["kb_n"] = min(N, n + B * T)
        return out if out is not None else logits.sum() * 0.0
    # ---- MR Matryoshka
    if k == "mr_dims":
        hn = model.norm(c["h"]).float()
        Wh = model.head.weight.float()
        d = hn.shape[-1]
        tot = 0.0
        for kk in (16, 32):
            z = (hn[..., :kk] @ Wh[:, :kk].T) * (d / kk) ** 0.5
            if model.head.bias is not None:
                z = z + model.head.bias.float()
            tot = tot + F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
        return tot / 2
    if k == "mr_vocab":
        top = W["topv"]
        zt = logits[..., top]
        rest = logits.clone()
        rest[..., top] = float("-inf")
        z = torch.cat([zt, torch.logsumexp(rest, -1, keepdim=True)], -1)
        r = W["frank"][y]
        t = torch.where(r < 1024, r, torch.full_like(r, 1024))
        # map label to its column in zt: zt columns are in frequency order, so the column is the rank
        return F.cross_entropy(z.reshape(-1, 1025), t.reshape(-1))
    # ---- EB expert-balanced token weights on the hard-label CE (layer-0 router)
    if k.startswith("eb_"):
        if W["eb"] is None:
            return logits.sum() * 0.0
        probs, TT, BB = W["eb"]
        W["eb"] = None
        if k == "eb_cnt":
            top = probs.argmax(-1)
            cnt = torch.bincount(top, minlength=probs.shape[1]).float()
            w = cnt[top].clamp_min(1).pow(-0.5)
        else:
            w = 1.0 / (probs.max(-1).values / probs.sum(-1).clamp_min(1e-9)).clamp_min(1e-3)
        w = w.view(TT, BB).transpose(0, 1)
        w = w / w.mean()
        return ((w - 1) * nll).mean()
    # ---- TK top-k hinge: the label only has to beat the k-th best other token
    if k in ("tk_5", "tk_20"):
        kk = 5 if k == "tk_5" else 20
        zy = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        zo = logits.scatter(-1, y.unsqueeze(-1), float("-inf"))
        kth = zo.topk(kk, -1).values[..., -1]
        return F.relu(1 + kth - zy).mean()
    # ---- LN LogitNorm
    if k == "ln_norm":
        zn = logits / (logits.norm(dim=-1, keepdim=True) * 0.04 + 1e-6)
        return F.cross_entropy(zn.reshape(-1, V), y.reshape(-1))
    return None


# ---------------------------------------------------------------- round m7 (O36): CoCoNTs failure-aimed teachers
# Every cc_ mode builds one full target per position, Q = (1 - a_t) onehot(y) + a_t q', where q' is the corpus trigram
# teacher (CoCoNTs, arXiv 2407.02819) after one change, and trains soft CE to Q. With a_t = ta and q' = q this equals
# CoCoNTs exactly (cc_null). Measured failures (Mac/GPU XS 600, FineWeb, CoCoNTs vs NTP, nats): rare targets stay at
# 8.25 (31% of tokens); after frequent inputs the gain is -0.30 vs -0.57 after rare inputs; on repeated targets -0.22
# vs -0.45 on new ones.
T2_MODES = ("cc_agrade", "cc_rg3", "cc_qgate", "cc_lossgate", "cc_cntgate", "cc_inrgate", "cc_rgsched", "cc_rgtemp",
            "cc_rgls", "cc_rgmodel", "cc_repup", "cc_newup", "cc_finup", "cc_findn", "cc_freqlow", "cc_rgshift",
            "cc_mixbig", "cc_nogold", "cc_hnball")
T2_COMBOS = ("cc_rareboost_rg", "cc_kn_rg", "cc_hnb_rg", "cc_indtri_rg")
CC_MODES = ("cc_null", "cc_kn", "cc_rareboost", "cc_raregate", "cc_flatuni", "cc_hnb", "cc_quad", "cc_sharpf",
            "cc_skipf", "cc_cache", "cc_cacheu", "cc_ind", "cc_repgate", "cc_indtri", "cc_kn_ind",
            "cc_freqgate", "cc_randgate", "cc_rarehard") + T2_MODES + T2_COMBOS


def _cc_setup(train, cfg, name, dev):
    V = cfg.vocab
    t = cfg.tgt
    a, uni = _counts(train, V)
    rk = np.empty(V, dtype=np.int64); rk[np.argsort(-uni)] = np.arange(V)
    S["cc_rank"] = torch.from_numpy(rk).to(dev)
    S["cc_uni"] = torch.from_numpy((uni + 1) / (uni + 1).sum()).float()
    ng = S["ng"]
    if t in ("cc_kn", "cc_kn_ind", "cc_kn_rg"):
        M = ng.big
        if hasattr(M, "tocsc"):
            cont = np.asarray((M > 0).sum(0)).ravel().astype(np.float64)
            nfol = np.asarray((M > 0).sum(1)).ravel().astype(np.float64)
        else:
            cont = np.bincount(M.indices, minlength=V).astype(np.float64)
            nfol = np.diff(M.indptr).astype(np.float64)
        S["cc_cont"] = torch.from_numpy((cont + 0.5) / (cont + 0.5).sum()).float()
        S["cc_nfol"] = torch.from_numpy(nfol).float()
    if t == "cc_quad":
        f = os.path.join(CACHE, f"{name}_quadgram.npz")
        if os.path.exists(f):
            z = np.load(f); S["cc_qc"], S["cc_qn"] = z["c"], z["n"]
        else:
            code = ((a[:-3] * V + a[1:-2]) * V + a[2:-1]) * V + a[3:]
            S["cc_qc"], S["cc_qn"] = np.unique(code, return_counts=True)
            np.savez(f, c=S["cc_qc"], n=S["cc_qn"])
    if t == "cc_skipf":
        f = os.path.join(CACHE, f"{name}_skipgram.npz")
        if os.path.exists(f):
            z = np.load(f); sc, sn = z["c"], z["n"]
        else:
            sc, sn = np.unique(a[:-2] * V + a[2:], return_counts=True)
            np.savez(f, c=sc, n=sn)
        S["cc_sc"], S["cc_sn"] = sc, sn


def _cc_kn(ng, xy, D=0.75):
    """Interpolated Kneser-Ney style teacher: absolute discount D at trigram and bigram level, bigram backs off to the
    continuation unigram (how many distinct tokens precede v), not the raw unigram."""
    V = ng.V
    x = xy[:, :-1].cpu(); B, T = x.shape
    ids = x.reshape(-1)
    rows = torch.from_numpy(ng.big[ids.numpy()].toarray())
    n = rows.sum(-1, keepdim=True)
    cont = S["cc_cont"]
    nf = S["cc_nfol"][ids].unsqueeze(-1)
    qb = torch.where(n > 0, (rows - D).clamp_min(0) / n.clamp_min(1) + D * nf / n.clamp_min(1) * cont, cont.expand_as(rows))
    prev = torch.cat([torch.full((B, 1), -1), x[:, :-1]], 1).reshape(-1).numpy()
    ctx = np.where(prev >= 0, prev * V + ids.numpy(), 0).astype(np.int64)
    lo = np.searchsorted(ng.tc, ctx * V, "left"); hi = np.searchsorted(ng.tc, ctx * V + V, "left")
    cnt = torch.zeros_like(qb)
    nrow = hi - lo
    if nrow.sum() > 0:
        r = np.repeat(np.arange(len(ctx)), nrow)
        idx = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        cnt[torch.from_numpy(r), torch.from_numpy(ng.tc[idx] % V)] = torch.from_numpy(ng.tn[idx]).float()
    nt = cnt.sum(-1, keepdim=True)
    k = torch.from_numpy(nrow.astype(np.float32)).unsqueeze(-1)
    q = torch.where(nt > 0, (cnt - D).clamp_min(0) / nt.clamp_min(1) + D * k / nt.clamp_min(1) * qb, qb)
    return q / q.sum(-1, keepdim=True)


def _cc_ctx_follow(x, order):
    """In-context followers: for each position t, counts of the tokens that followed earlier occurrences of the last
    `order` tokens (order 1: x_t, order 2: x_{t-1} x_t). Returns (B, T, V) counts using only tokens up to t."""
    B, T = x.shape
    m = x.unsqueeze(1) == x.unsqueeze(2)                   # m[b, t, s]: x_s == x_t
    if order == 2:
        pv = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
        m = m & (pv.unsqueeze(1) == pv.unsqueeze(2)) & (pv.unsqueeze(2) >= 0)
    m = m & torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)   # s < t, so x_{s+1} is at most x_t
    oh = F.one_hot(x[:, 1:], S["cc_V"]).float()          # followers x_{s+1}, s = 0..T-2
    return torch.bmm(m[:, :, :T - 1].float(), oh)


def _cc_target(t, model, xy, cfg, b0=0, b1=None):
    t0 = t
    rg = t.endswith("_rg")                  # combination: the rare-target gate (0.8 where rank >= 1024) on top
    if rg:
        t = t[:-3]
    ng = S["ng"]; V = cfg.vocab
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    dev = xy.device
    a = cfg.ta
    lam = 0.0 if rg else cfg.wyl
    if t == "cc_mixbig":
        q = ng._big(x.reshape(-1).cpu()).to(dev)
        q = 0.5 * q + 0.5 * ng.query(xy)
    elif t in ("cc_kn", "cc_kn_ind"):
        q = _cc_kn(ng, xy).to(dev)
    else:
        q = ng.query(xy)
    if t == "cc_quad":
        xc = x.cpu(); ids = xc.reshape(-1).numpy()
        p1 = torch.cat([torch.full((B, 1), -1), xc[:, :-1]], 1).reshape(-1).numpy()
        p2 = torch.cat([torch.full((B, 2), -1), xc[:, :-2]], 1).reshape(-1).numpy()
        c3 = np.where((p2 >= 0) & (p1 >= 0), (p2 * V + p1) * V + ids, 0).astype(np.int64)
        q = ng._tri(c3, q.cpu(), S["cc_qc"], S["cc_qn"]).to(dev)
    q = shape_teacher(q.view(B, T, V), cfg)
    rk = S["cc_rank"]
    at = torch.full((B, T, 1), a, device=dev)
    if t == "cc_rareboost":
        w = S["cc_uni"].to(dev) ** -(cfg.wyp or 0.25)
        q = q * w; q = q / q.sum(-1, keepdim=True)
    elif t == "cc_raregate":      # wyp: rank threshold (default 1024); wyl: weight on rare-target positions
        thr = int(cfg.wyp) if cfg.wyp else 1024
        at = torch.where((rk[y] >= thr).unsqueeze(-1), torch.full_like(at, lam or 0.8), at)
    elif t == "cc_freqgate":      # mirror: the changed weight sits on frequent targets (rank < wyp, default 256)
        thr = int(cfg.wyp) if cfg.wyp else 256
        at = torch.where((rk[y] < thr).unsqueeze(-1), torch.full_like(at, lam or 0.8), at)
    elif t == "cc_randgate":      # ablation: same weight on a random wyp share of positions (default 0.31 = rare share)
        at = torch.where((torch.rand(B, T, 1, device=dev) < (cfg.wyp or 0.31)), torch.full_like(at, lam or 0.8), at)
    elif t == "cc_rarehard":      # no teacher: hard-label weight (1 - wyl) on rare targets, CE renormalised by nothing
        thr = int(cfg.wyp) if cfg.wyp else 1024
        at = torch.where((rk[y] >= thr).unsqueeze(-1), torch.full_like(at, lam or 0.8), torch.zeros_like(at))
        q = torch.zeros_like(q)
    elif t == "cc_flatuni":
        f = S["cc_uni"].to(dev) ** 0.5; f = f / f.sum()
        q = (1 - (lam or 0.1)) * q + (lam or 0.1) * f
    elif t == "cc_hnb":
        W = model.head.weight.detach().float() if hasattr(model.head, "weight") else model.emb.weight.detach().float()
        Wn = F.normalize(W, dim=-1)
        yy = y.reshape(-1)
        sim = Wn[yy] @ Wn.t()
        sim[torch.arange(len(yy), device=dev), yy] = -2
        nb = sim.topk(8, -1).indices
        rare = (rk[yy] >= 256).float().unsqueeze(-1)
        add = torch.zeros(B * T, V, device=dev).scatter_(-1, nb, 1.0 / 8)
        mass = (lam or 0.1) * rare
        q = ((1 - mass) * q.view(B * T, V) + mass * add).view(B, T, V)
    elif t == "cc_sharpf":
        fr = (rk[x] < 256).unsqueeze(-1)
        qs = q ** (1.0 / 0.7); qs = qs / qs.sum(-1, keepdim=True)
        q = torch.where(fr, qs, q)
    elif t == "cc_skipf":
        xc = x.cpu()
        p1 = torch.cat([xc[:, :1], xc[:, :-1]], 1).reshape(-1).numpy().astype(np.int64)
        lo = np.searchsorted(S["cc_sc"], p1 * V, "left"); hi = np.searchsorted(S["cc_sc"], p1 * V + V, "left")
        cnt = torch.zeros(B * T, V)
        nrow = hi - lo
        if nrow.sum() > 0:
            r = np.repeat(np.arange(len(p1)), nrow)
            idx = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
            cnt[torch.from_numpy(r), torch.from_numpy(S["cc_sc"][idx] % V)] = torch.from_numpy(S["cc_sn"][idx]).float()
        qsk = ((cnt + S["cc_uni"]) / (cnt.sum(-1, keepdim=True) + 1)).to(dev).view(B, T, V)
        fr = (rk[x] < 256).unsqueeze(-1).float() * (lam or 0.3)
        q = (1 - fr) * q + fr * qsk
    elif t in ("cc_cache", "cc_cacheu"):
        ar = torch.arange(T, device=dev)
        g = 0.9 if t == "cc_cache" else 1.0
        Wd = (g ** (ar.unsqueeze(1) - ar.unsqueeze(0)).clamp_min(0).float()) * torch.ones(T, T, device=dev).tril()
        Wd = Wd / Wd.sum(-1, keepdim=True)
        cache = torch.einsum("ts,bsv->btv", Wd, F.one_hot(x, V).float())
        l_ = lam or (0.3 if t == "cc_cache" else 0.2)
        q = (1 - l_) * q + l_ * cache
    elif t in ("cc_ind", "cc_indtri", "cc_kn_ind"):
        S["cc_V"] = V
        Fc = _cc_ctx_follow(x, 2 if t == "cc_indtri" else 1)
        n = Fc.sum(-1, keepdim=True)
        hit = (n > 0).float() * (lam or (0.7 if t == "cc_indtri" else 0.5))
        q = (1 - hit) * q + hit * Fc / n.clamp_min(1)
    elif t == "cc_repgate":
        rep = ((x.unsqueeze(1) == y.unsqueeze(2)) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()).any(-1)
        at = torch.where(rep.unsqueeze(-1), torch.full_like(at, lam or 0.2), at)
    at, q = _cc_t2(t, model, xy, cfg, b0, b1, q, at, lam, rk)
    if rg:
        at = torch.where((rk[y] >= 1024).unsqueeze(-1), torch.full_like(at, cfg.wyl or 0.8), at)
    oh = F.one_hot(y, V).float()
    return (1 - at) * oh + at * q


def _cc_t2(t, model, xy, cfg, b0, b1, q, at, lam, rk):
    """Round t2: per-position soft weight a_t set by one measured signal, or the extra soft mass on rare-target
    positions sent somewhere other than the teacher. All aimed at the m7 failures (rare targets, frequent inputs,
    repeats) and at testing rule R1 (rare-target gate)."""
    x, y = xy[:, :-1], xy[:, 1:]
    B, T, V = q.shape
    dev = q.device
    a = cfg.ta
    hi = lam or 0.8
    rare = (rk[y] >= 1024).unsqueeze(-1)
    full = lambda v: torch.full_like(at, v)
    if t == "cc_agrade":            # smooth: a_t = 0.5 + 0.4 * (rank / V) ** 0.5
        at = (a + 0.4 * (rk[y].float() / V) ** 0.5).unsqueeze(-1)
    elif t == "cc_rg3":             # three levels: 0.8 rare, 0.65 middle (256-1023), 0.5 frequent
        at = torch.where(rare, full(0.8), torch.where((rk[y] >= 256).unsqueeze(-1), full(0.65), at))
    elif t == "cc_qgate":           # gate by the teacher, not the label: 0.8 where q(gold) < 0.01
        qy = q.gather(-1, y.unsqueeze(-1))
        at = torch.where(qy < 0.01, full(hi), at)
    elif t == "cc_lossgate":        # gate by the model: 0.8 on the 31% of positions with the highest current loss
        nll = model.ctx["nll"][b0:b1].detach().view(B, T)
        thr = torch.quantile(nll.float().reshape(-1), 0.69)
        at = torch.where((nll >= thr).unsqueeze(-1), full(hi), at)
    elif t == "cc_cntgate":         # 0.8 where the trigram context was seen >= 100 times (teacher well counted)
        ng = S["ng"]
        if "cc_tncum" not in S:
            S["cc_tncum"] = np.concatenate([[0], np.cumsum(ng.tn)])
        xc = x.cpu(); ids = xc.reshape(-1).numpy()
        prev = torch.cat([torch.full((B, 1), -1), xc[:, :-1]], 1).reshape(-1).numpy()
        ctx = np.where(prev >= 0, prev * V + ids, 0).astype(np.int64)
        lo = np.searchsorted(ng.tc, ctx * V, "left"); hh = np.searchsorted(ng.tc, ctx * V + V, "left")
        n = torch.from_numpy(S["cc_tncum"][hh] - S["cc_tncum"][lo]).to(dev).view(B, T, 1)
        at = torch.where(n >= 100, full(hi), at)
    elif t == "cc_inrgate":         # input-rarity control: 0.8 where the INPUT token is rare
        at = torch.where((rk[x] >= 1024).unsqueeze(-1), full(hi), at)
    elif t == "cc_rgsched":         # rare gate ramped 0.5 -> 0.9 over training
        at = torch.where(rare, full(a + 0.4 * S.get("cc_frac", 0.0)), at)
    elif t == "cc_rgtemp":          # rare positions keep a = 0.5 but the teacher is flattened (temperature 2)
        qf = q ** 0.5; qf = qf / qf.sum(-1, keepdim=True)
        q = torch.where(rare, qf, q)
    elif t == "cc_rgls":            # rare positions: the extra 0.3 goes to uniform, not to the teacher
        u = torch.full_like(q, 1.0 / V)
        q = torch.where(rare, (a * q + 0.3 * u) / (a + 0.3), q)
        at = torch.where(rare, full(a + 0.3), at)
    elif t == "cc_rgmodel":         # rare positions: the extra 0.3 goes to the model's own detached prediction
        p = F.softmax(model.ctx["lg"][b0:b1].detach().float(), -1)
        q = torch.where(rare, (a * q + 0.3 * p) / (a + 0.3), q)
        at = torch.where(rare, full(a + 0.3), at)
    elif t in ("cc_repup", "cc_newup"):
        rep = ((x.unsqueeze(1) == y.unsqueeze(2)) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()).any(-1)
        m = rep if t == "cc_repup" else ~rep
        at = torch.where(m.unsqueeze(-1), full(hi), at)
    elif t in ("cc_finup", "cc_findn"):
        fr = (rk[x] < 256).unsqueeze(-1)
        at = torch.where(fr, full(hi if t == "cc_finup" else (lam or 0.2)), at)
    elif t == "cc_freqlow":         # other side of R1: more hard label (a = 0.3) on frequent targets
        at = torch.where((rk[y] < 256).unsqueeze(-1), full(lam or 0.3), at)
    elif t == "cc_rgshift":         # rare 0.9, others 0.4: same mean weight as plain 0.5 shifted toward rare
        at = torch.where(rare, full(0.9), full(0.4))
    elif t == "cc_nogold":          # teacher with the gold token removed (only the alternatives)
        q = q.scatter(-1, y.unsqueeze(-1), 0.0)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    elif t == "cc_hnball":          # head-neighbour mass on every target, not only rare ones
        W = model.head.weight.detach().float() if hasattr(model.head, "weight") else model.emb.weight.detach().float()
        Wn = F.normalize(W, dim=-1)
        yy = y.reshape(-1)
        sim = Wn[yy] @ Wn.t()
        sim[torch.arange(len(yy), device=dev), yy] = -2
        nb = sim.topk(8, -1).indices
        add = torch.zeros(B * T, V, device=dev).scatter_(-1, nb, 1.0 / 8)
        q = (0.9 * q.view(B * T, V) + 0.1 * add).view(B, T, V)
    return at, q
