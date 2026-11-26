"""Rare-row rules for the embedding and output rows (learning search, round 10, 2026-11-22).

Off by default (cfg.erule == "none"): train.py then never imports this file and the embedding and head stay on the
default scalar optimizer. With erule set, train.py takes emb.weight and head.weight out of the default scalar
optimizer and EmbOpt updates them row by row with the same betas (0.9, 0.95), eps 1e-8 and lr schedule.

A row is "present" on a step when its token id is in the batch: input ids for emb rows, target ids for head rows.
Absent emb rows get a zero gradient; head rows always get a gradient (the softmax pushes every logit down).
Under the default scalar optimizer an absent row keeps moving: m decays 0.9^k, sqrt(v) about 0.975^k.

cfg.erule = "<rule>" applies to emb and head; "e:<rule>" emb only, "h:<rule>" head only.
cfg.ea, cfg.eb: the rule's free knobs.
"""
import math
import torch

B1, B2, EPS = 0.9, 0.95, 1e-8


def _ns(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class EmbOpt:
    def __init__(self, emb, head, cfg):
        r = cfg.erule
        self.cfg, self.t = cfg, 0
        side, name = (r.split(":", 1) if ":" in r else ("eh", r))
        self.params = []
        if "e" in side:
            self.params.append(("emb", emb, name))
        else:
            self.params.append(("emb", emb, "dense"))
        if "h" in side:
            self.params.append(("head", head, name))
        else:
            self.params.append(("head", head, "dense"))
        self.state = {}
        for k, p, _ in self.params:
            V = p.shape[0]
            self.state[k] = dict(m=torch.zeros_like(p), v=torch.zeros_like(p), n=0,
                                 last=torch.full((V,), -1, dtype=torch.long, device=p.device),
                                 cnt=torch.zeros(V, device=p.device), nrow=torch.zeros(V, device=p.device))
        self.lr = cfg.lr
        self.eps = float(getattr(cfg, "eeps", EPS))
        self.present = {}
        self.graw, self.urms = {}, {}
        self.W = {k: p for k, p, _ in self.params}

    def set_batch(self, xy):
        V = self.cfg.vocab
        self.present["emb"] = torch.bincount(xy[:, :-1].reshape(-1), minlength=V) > 0
        self.present["head"] = torch.bincount(xy[:, 1:].reshape(-1), minlength=V) > 0

    def zero_grad(self):
        for _, p, _ in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        self.graw = {k: (p.grad.float().clone() if p.grad is not None else None) for k, p, _ in self.params}
        if self.cfg.erule == "tied":         # weight tying (Press & Wolf 2017) as a baseline: head := emb at step 1,
            E, H = self.W["emb"], self.W["head"]  # then both take the summed gradient with one shared state
            if self.t == 1:
                H.copy_(E)
            G = self.graw["emb"] + self.graw["head"]
            st = self.state["emb"]
            self.key = "emb"
            self._apply("dense", E, G, st, self.present["emb"] | self.present["head"])
            H.copy_(E)
            return
        if self.cfg.erule == "tiedcoupled":  # gate-0 audit (2026-11-23): Coupled Adam (arXiv 2502.08441) on a tied table,
            E, H = self.W["emb"], self.W["head"]  # the paper's setting; same as "tied" but the shared state uses "coupled"
            if self.t == 1:
                H.copy_(E)
            G = self.graw["emb"] + self.graw["head"]
            st = self.state["emb"]
            self.key = "emb"
            self._apply("coupled", E, G, st, self.present["emb"] | self.present["head"])
            H.copy_(E)
            return
        for k, p, rule in self.params:
            self.key = k
            if p.grad is None:
                continue
            self._apply(rule, p, p.grad.float(), self.state[k], self.present[k])

    def _apply(self, rule, W, G, st, P, ):
        cfg, t, lr = self.cfg, self.t, self.lr
        a, b = cfg.ea, cfg.eb
        m, v = st["m"], st["v"]
        Pf = P.float()[:, None]
        gap = (t - 1 - st["last"]).clamp(min=0)          # steps since last present (t-1 if never)
        gap = torch.where(st["last"] < 0, torch.zeros_like(gap), gap)
        st["cnt"] += P.float()
        st["n"] += 1
        n = st["n"]
        bc1, bc2 = 1 - B1 ** n, 1 - B2 ** n
        freq = st["cnt"] / t                              # fraction of steps the row was present
        U = None

        def dense():
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            return (m / bc1) / ((v / bc2).sqrt() + self.eps)

        def lazy(decay=False, rowbc=False):
            if decay:   # catch the state up for the steps it missed, as the dense optimizer would have decayed it
                g = gap.float()[:, None]
                m.mul_(torch.where(P[:, None], B1 ** g, torch.ones_like(g)))
                v.mul_(torch.where(P[:, None], B2 ** g, torch.ones_like(g)))
            m.copy_(torch.where(P[:, None], B1 * m + (1 - B1) * G, m))
            v.copy_(torch.where(P[:, None], B2 * v + (1 - B2) * G * G, v))
            st["nrow"] += P.float()
            if rowbc:
                nr = st["nrow"].clamp(min=1)[:, None]
                u = (m / (1 - B1 ** nr)) / ((v / (1 - B2 ** nr)).sqrt() + self.eps)
            else:
                u = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            return u * Pf

        if rule == "dense":
            U = dense()
        elif rule == "lazy":                 # LazyAdam / SparseAdam: absent rows keep state and do not move
            U = lazy()
        elif rule == "lazy_decay":           # lazy, state decayed by the missed steps on reappearance
            U = lazy(decay=True)
        elif rule == "lazy_rowbc":           # lazy with per-row bias correction (rare rows take full-size steps)
            U = lazy(rowbc=True)
        elif rule == "lazy_decay_rowbc":
            U = lazy(decay=True, rowbc=True)
        elif rule == "freeze":               # dense state, but only present rows move
            U = dense() * Pf
        elif rule == "stopk":                # absent rows coast for at most a steps, then stop
            U = dense()
            U = U * (P | (gap < a)).float()[:, None]
        elif rule == "damp":                 # absent rows move with weight a
            U = dense()
            U = U * (Pf + (1 - Pf) * a)
        elif rule == "reset_m":              # a row back after more than a steps starts its momentum from zero
            stale = P & (gap > a)
            m.mul_((~stale).float()[:, None])
            U = dense()
        elif rule == "reset_mv":             # back after more than a steps: momentum and second moment from zero
            stale = P & (gap > a)
            m.mul_((~stale).float()[:, None])
            v.copy_(torch.where(stale[:, None], G * G, v))
            U = dense()
        elif rule == "mfast":                # absent rows' momentum decays a times faster per step
            m.mul_(torch.where(P[:, None], torch.ones_like(Pf), torch.full_like(Pf, a)))
            U = dense()
        elif rule == "coast":                # absent rows: momentum frozen (no decay), keeps moving at full speed
            m.copy_(torch.where(P[:, None], B1 * m + (1 - B1) * G, m))
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "lazyv":                # v updated only on present steps; m dense
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.copy_(torch.where(P[:, None], B2 * v + (1 - B2) * G * G, v))
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "borrow":               # stale rows mix a of the present rows' mean momentum into theirs
            U0 = dense()
            if P.any():
                mm = m[P].mean(0, keepdim=True)
                m.copy_(torch.where(P[:, None], m, (1 - a) * m + a * mm))
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "borrow_back":          # a row coming back after a gap gets a of the present mean momentum
            back = P & (gap > 0)
            if P.any():
                mm = m[P & ~back].mean(0, keepdim=True) if (P & ~back).any() else m[P].mean(0, keepdim=True)
                m.copy_(torch.where(back[:, None], (1 - a) * m + a * mm, m))
            U = dense()
        elif rule == "freq_lr":              # per-row lr x (freq / mean freq)^-a, clipped to [1/4, 4]
            U = dense()
            f = freq.clamp(min=1.0 / t)
            s = (f / f.mean()).pow(-a).clamp(0.25, 4.0)
            U = U * s[:, None]
        elif rule == "cnt_lr":               # per-row lr x ((cnt + 1) / mean)^-a, rows seen often slow down
            U = dense()
            c = st["cnt"] + 1
            U = U * (c / c.mean()).pow(-a).clamp(0.25, 4.0)[:, None]
        elif rule == "catchup":              # a row back after k steps takes a step x min(1+k, a)^0.5
            U = dense()
            s = torch.where(P, (1 + gap.float()).clamp(max=a).sqrt(), torch.ones_like(freq))
            U = U * s[:, None]
        elif rule == "lazy_catchup":         # lazy, and a row back after k steps steps x min(1+k, a)^0.5
            U = lazy(decay=True)
            U = U * (1 + gap.float()).clamp(max=a).sqrt()[:, None]
        elif rule == "first_boost":          # the first step a row is ever present has lr x a
            first = P & (st["cnt"] == 1)
            U = dense()
            U = U * torch.where(first, torch.full_like(freq, a), torch.ones_like(freq))[:, None]
        elif rule == "pull_mean":            # absent rows drift toward the mean row at rate a (times lr)
            U = dense()
            mean = W[P].mean(0, keepdim=True) if P.any() else W.mean(0, keepdim=True)
            U = U + a * (1 - Pf) * (W - mean) / (W - mean).pow(2).mean().sqrt().clamp(min=1e-8)
        elif rule == "rownorm":              # each row's update scaled to unit RMS
            U = dense()
            U = U / U.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8) * a
        elif rule == "rowv":                 # one second moment per row (mean over the row), Adafactor-like
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).add_((G * G).mean(1, keepdim=True).expand_as(v), alpha=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "vfloor":               # second moment floored at a x the mean over all rows
            U0 = dense()
            vf = torch.maximum(v, a * v.mean())
            U = (m / bc1) / ((vf / bc2).sqrt() + self.eps)
        elif rule == "vglob_rare":           # rows present on fewer than a of steps use the global mean v
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            rare = (freq < a)[:, None]
            vv = torch.where(rare, v.mean().expand_as(v), v)
            U = (m / bc1) / ((vv / bc2).sqrt() + self.eps)
        elif rule == "nomom_rare":           # rows present on fewer than a of steps take no momentum
            U0 = dense()
            rare = (freq < a)[:, None]
            U = torch.where(rare, G / ((v / bc2).sqrt() + self.eps), U0)
        elif rule == "sign_rare":            # rows present on fewer than a of steps take sign(m) x b
            U0 = dense()
            rare = (freq < a)[:, None]
            U = torch.where(rare, torch.sign(m) * b, U0)
        elif rule == "wd_absent":            # absent rows shrink by lr x a per step
            U = dense() + a * (1 - Pf) * W
        elif rule == "muon":                 # Newton-Schulz orthogonalized momentum, RMS-matched to the dense step
            m.mul_(B1).add_(G, alpha=1 - B1)
            O = _ns(m)
            U = O * (0.2 * math.sqrt(max(O.shape)) * a)
        elif rule == "muon_present":         # orthogonalize only the present rows' momentum
            m.mul_(B1).add_(G, alpha=1 - B1)
            U = torch.zeros_like(m)
            if P.any():
                O = _ns(m[P])
                U[P] = O * (0.2 * math.sqrt(max(O.shape)) * a)
        elif rule == "decay_absent_v":       # absent rows' second moment does not decay (their steps shrink as m^k)
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.copy_(torch.where(P[:, None], B2 * v + (1 - B2) * G * G, torch.maximum(v, B2 * v + (1 - B2) * G * G)))
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "eps":                  # the default scalar optimizer with eps = a (control for the v floor)
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + a)
        elif rule == "vfloor_row":           # floor at a x the mean of the row's own history-free tensor mean, per column
            U0 = dense()
            vf = torch.maximum(v, a * v.mean(0, keepdim=True))
            U = (m / bc1) / ((vf / bc2).sqrt() + self.eps)
        elif rule == "coupled":              # Coupled Adam (arXiv 2502.08441, Eq. 24): v averaged over the vocabulary
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / bc1) / ((v.mean(0, keepdim=True) / bc2).sqrt() + self.eps)
        elif rule == "rare_lr":              # rows present on fewer than b of steps: lr x a
            U = dense()
            U = U * torch.where(freq < b, torch.full_like(freq, a), torch.ones_like(freq))[:, None]
        # ---- round 11 (2026-11-23): mechanisms outside the rare-row family ----
        elif rule in ("xgrad", "xv", "xmom"):  # input-output coupling: the same token's other row lends signal
            ok = "head" if self.key == "emb" else "emb"
            if rule == "xgrad":
                G = G + a * self.graw[ok]
                U = dense()
            elif rule == "xv":
                U0 = dense()
                vv = (1 - a) * v + a * self.state[ok]["v"]
                U = (m / bc1) / ((vv / bc2).sqrt() + self.eps)
            else:
                U0 = dense()
                mm = m + a * self.state[ok]["m"]
                U = (mm / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "radial":               # per row: the part of the step along the row's own direction x a
            U = dense()
            r = W.float() / (W.float().norm(dim=1, keepdim=True) + 1e-12)
            c = (U * r).sum(1, keepdim=True)
            U = U - c * r + a * c * r
        elif rule == "trad":                 # whole tensor: the part of the step along W itself x a
            U = dense()
            Wf = W.float()
            c = (U * Wf).sum() / (Wf * Wf).sum().clamp(min=1e-12)
            U = U - c * Wf + a * c * Wf
        elif rule == "warm":                 # extra warmup: lr x min(1, t / a)
            U = dense() * min(1.0, t / a)
        elif rule == "cool":                 # lr x linear decay from 1 to b over the run
            U = dense() * (1 - (1 - b) * min(1.0, t / max(1, cfg.T)))
        elif rule == "hot":                  # lr x (1 + (a - 1)(1 - t/T)): a times faster early
            U = dense() * (1 + (a - 1) * (1 - min(1.0, t / max(1, cfg.T))))
        elif rule in ("dwhite", "outmetric"):  # precondition the d_model side; dwhite: gradient covariance, outmetric: head Gram
            U0 = dense()
            if rule == "dwhite":
                C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                C.mul_(B2).add_(G.T @ G / G.shape[0], alpha=1 - B2)
                M = C / bc2
            else:
                Hh = self.W["head"].float()
                M = Hh.T @ Hh / Hh.shape[0]
            M = M / (M.diagonal().mean() + 1e-12)
            e, Q = torch.linalg.eigh(M + a * torch.eye(M.shape[0], device=M.device))
            P_ = Q @ torch.diag(e.clamp(min=1e-6) ** -0.5) @ Q.T
            U = U0 @ P_
            U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule in ("null", "nullw"):      # softmax ignores a row added to every head row: drop a of the step's mean row
            U = dense()
            U = U - a * U.mean(0, keepdim=True)
        elif rule in ("maxnorm", "eqnorm"):  # row-norm control after the step (applied below)
            U = dense()
        elif rule == "rowclip":              # each row's gradient clipped to a x the median nonzero row norm
            rn = G.norm(dim=1)
            nz = rn[rn > 0]
            if nz.numel():
                cap = a * nz.median()
                G = G * (cap / rn.clamp(min=1e-12)).clamp(max=1.0)[:, None]
            U = dense()
        elif rule == "balance":              # step RMS matched to the other tensor's last step RMS
            U = dense()
            ok = "head" if self.key == "emb" else "emb"
            if ok in self.urms:
                U = U * (self.urms[ok] / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule == "epst":                 # eps moves geometrically from a to b over the run
            f = min(1.0, t / max(1, cfg.T))
            ep = a * (b / a) ** f
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + ep)
        # ---- round 12 (2026-11-23) ----
        elif rule in ("dwhitep25", "uwhite", "dwhite10", "dwvfc", "dwxmom"):  # d_model-side whitening follow-ups
            if rule == "dwvfc":
                U0 = dense()
                vf = torch.maximum(v, 0.1 * v.mean(0, keepdim=True))
                U0 = (m / bc1) / ((vf / bc2).sqrt() + self.eps)
            elif rule == "dwxmom":
                U0 = dense()
                ok = "head" if self.key == "emb" else "emb"
                U0 = ((m + 0.1 * self.state[ok]["m"]) / bc1) / ((v / bc2).sqrt() + self.eps)
            else:
                U0 = dense()
            src = U0 if rule == "uwhite" else G
            C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
            C.mul_(B2).add_(src.T @ src / G.shape[0], alpha=1 - B2)
            if rule != "dwhite10" or "P" not in st or t % 10 == 1:
                M = C / bc2
                M = M / (M.diagonal().mean() + 1e-12)
                e, Q = torch.linalg.eigh(M + a * torch.eye(M.shape[0], device=M.device))
                pw = -0.25 if rule == "dwhitep25" else -0.5
                st["P"] = Q @ torch.diag(e.clamp(min=1e-6) ** pw) @ Q.T
            U = U0 @ st["P"]
            U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule in ("shampoo1", "soap1", "soapx", "xsoap"):  # published one-sided baselines: Shampoo (Gupta 2018) right factor, SOAP (Vyas 2024) one side; soapx: SOAP + a x the other row's momentum (L-A)
            U0 = dense()
            mx = m
            if rule == "soapx":
                ok = "head" if self.key == "emb" else "emb"
                mx = m + a * self.state[ok]["m"]
            C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
            C.mul_(B2).add_(G.T @ G / G.shape[0], alpha=1 - B2)
            M = C / bc2
            e, Q = torch.linalg.eigh(M + 1e-6 * M.diagonal().mean() * torch.eye(M.shape[0], device=M.device))
            if rule == "shampoo1":           # momentum x C^-1/2, grafted to the Adam step's RMS
                U = (m / bc1) @ (Q @ torch.diag(e.clamp(min=1e-12) ** -0.5) @ Q.T)
                U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            else:                            # Adam in the eigenbasis of C
                vq = st.setdefault("vq", torch.zeros_like(G))
                Gq = G @ Q
                vq.mul_(B2).addcmul_(Gq, Gq, value=1 - B2)
                U = (((mx @ Q) / bc1) / ((vq / bc2).sqrt() + self.eps)) @ Q.T
                if rule == "xsoap":          # L-A: SOAP + a x the other row's Adam-normalized momentum
                    ok = "head" if self.key == "emb" else "emb"
                    so = self.state[ok]
                    no = max(1, so["n"])
                    U = U + a * (so["m"] / (1 - B1 ** no)) / ((so["v"] / (1 - B2 ** no)).sqrt() + self.eps)
        elif rule == "period":               # gradients summed over a steps, one step every a steps, lr x b
            acc = st.setdefault("acc", torch.zeros_like(G))
            acc.add_(G)
            if t % int(a) != 0:
                U = torch.zeros_like(G)
            else:
                G = acc / a
                acc.zero_()
                U = dense() * b
        elif rule == "restart":              # SGDR-style warm restarts: a cosine cycles on top of the run's schedule
            f = (a * t / max(1, cfg.T)) % 1.0
            U = dense() * (b + (1 - b) * 0.5 * (1 + math.cos(math.pi * f)))
        elif rule in ("lion", "lionmag"):    # Lion (Chen 2023): sign(0.9 m + 0.1 g), m with beta 0.99; lionmag: sign(m) x Adam step RMS
            if rule == "lion":
                c = B1 * m + (1 - B1) * G
                U = a * torch.sign(c)
                m.mul_(0.99).add_(G, alpha=0.01)
            else:
                U0 = dense()
                U = torch.sign(m) * U0.pow(2).mean().sqrt()
        elif rule == "sfree":                # schedule-free (Defazio 2024): Adam step on z, x = running average, weights y
            U = dense()
            z = st.setdefault("z", W.float().clone())
            x = st.setdefault("x", W.float().clone())
            z.sub_(lr * U)
            x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            yv = x if t >= cfg.T else (1 - a) * z + a * x
            st["last"] = torch.where(P, torch.full_like(st["last"], t - 1), st["last"])
            self.urms[self.key] = float(U.pow(2).mean().sqrt())
            W.copy_(yv.to(W.dtype))
            return
        elif rule == "nest":                 # NAdam-style Nesterov momentum
            U0 = dense()
            U = ((B1 * m + (1 - B1) * G) / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "factv":                # Adafactor factored second moment: row x column / total
            m.mul_(B1).add_(G, alpha=1 - B1)
            r_ = st.setdefault("vr", torch.zeros(G.shape[0], device=G.device))
            c_ = st.setdefault("vc", torch.zeros(G.shape[1], device=G.device))
            G2 = G * G + 1e-30
            r_.mul_(B2).add_(G2.mean(1), alpha=1 - B2)
            c_.mul_(B2).add_(G2.mean(0), alpha=1 - B2)
            vh = r_[:, None] * c_[None, :] / r_.mean().clamp(min=1e-30)
            U = (m / bc1) / ((vh / bc2).sqrt() + self.eps)
        elif rule == "wsd":                  # warmup-stable-decay: flat lr, then linear decay over the last b of the run
            T = max(1, cfg.T)
            w = min(1.0, t / max(1, cfg.warmup))
            d = 1.0 if t < (1 - b) * T else 0.1 + 0.9 * max(0.0, (T - t) / (b * T))
            U = dense() * (cfg.lr * w * d / max(lr, 1e-12))
        elif rule == "belief":               # AdaBelief: second moment of (g - m)
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G - m, G - m, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule in ("b2t", "b2ramp"):      # beta2 schedules: Adafactor 1 - t^-0.8, or a ramp 0.9 -> 0.99
            b2 = (1 - t ** -0.8) if rule == "b2t" else 0.9 + 0.09 * min(1.0, t / max(1, cfg.T))
            st["p2"] = st.get("p2", 1.0) * b2
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(b2).addcmul_(G, G, value=1 - b2)
            U = (m / bc1) / ((v / max(1e-12, 1 - st["p2"]) if rule == "b2ramp" else v).sqrt() + self.eps)
        elif rule == "msqv":                 # denominator from the momentum's own second moment
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(m, m, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "uclip":                # Adafactor update clipping: step RMS capped at a
            U = dense()
            U = U / max(1.0, float(U.pow(2).mean().sqrt()) / a)
        elif rule == "vlag":                 # denominator from last step's second moment (AdaShift-like decorrelation)
            vp = v.clone()
            bp = 1 - B2 ** max(1, n - 1)
            U0 = dense()
            U = U0 if n == 1 else (m / bc1) / ((vp / bp).sqrt() + self.eps)
        # ---- round 13 (2026-11-23) ----
        elif rule == "cautious":             # Cautious Adam (Liang 2024): drop step parts whose sign disagrees with g, rescale
            U = dense()
            mk = (U * G > 0).float()
            U = U * mk * (mk.numel() / (mk.sum() + 1.0))
        elif rule == "cautrow":              # idea: cautious mask per row (row kept only if its step agrees with its gradient)
            U = dense()
            mk = ((U * G).sum(1, keepdim=True) > 0).float()
            U = U * mk * (mk.numel() / (mk.sum() + 1.0))
        elif rule == "ademamix":             # AdEMAMix (Pagliardini 2024): + alpha x slow momentum (beta 0.999), alpha warmed up
            U0 = dense()
            m2 = st.setdefault("m2", torch.zeros_like(G))
            m2.mul_(0.999).add_(G, alpha=0.001)
            al = a * min(1.0, t / max(1, cfg.T))
            U = (m / bc1 + al * m2) / ((v / bc2).sqrt() + self.eps)
        elif rule == "adan":                 # Adan (Xie 2022): momentum of g and of g - g_prev, denominator of the corrected g
            gp = st.setdefault("gp", G.clone())
            dg = G - gp
            st["gp"] = G.clone()
            d_ = st.setdefault("dm", torch.zeros_like(G))
            m.mul_(0.98).add_(G, alpha=0.02)
            d_.mul_(0.92).add_(dg, alpha=0.08)
            q = G + 0.92 * dg
            v.mul_(0.99).addcmul_(q, q, value=0.01)
            U = (m / (1 - 0.98 ** n) + 0.92 * d_ / (1 - 0.92 ** n)) / ((v / (1 - 0.99 ** n)).sqrt() + self.eps)
        elif rule == "mars":                 # MARS (Yuan 2024): variance-reduced c = g + a b1/(1-b1) (g - g_prev), norm-capped, then Adam
            gp = st.setdefault("gp", G.clone())
            c = G + a * (B1 / (1 - B1)) * (G - gp)
            st["gp"] = G.clone()
            cn = float(c.norm())
            gn = float(G.norm())
            if cn > gn > 0:
                c = c * (gn / cn)
            G = c
            U = dense()
        elif rule in ("lamb", "rowtrust"):   # LAMB (You 2019) trust ratio |W|/|U|; lamb per tensor (1 at step 1), rowtrust per row / tensor mean
            U = dense()
            Wf = W.float()
            if rule == "lamb":
                r = float(Wf.norm()) / (float(U.norm()) + 1e-12)
                r0 = st.setdefault("r0", r)
                U = U * min(10.0, r / max(r0, 1e-12))
            else:
                r = Wf.norm(dim=1) / (U.norm(dim=1) + 1e-12)
                r = torch.where(U.norm(dim=1) > 0, r, torch.zeros_like(r))
                rm = r[r > 0].mean() if (r > 0).any() else torch.tensor(1.0)
                U = U * (r / (rm + 1e-12)).clamp(max=a)[:, None]
        elif rule == "spam":                 # SPAM (Huang 2025): momentum reset every a steps + spike clip at g^2 > 5000 v
            if t > 1 and int(a) > 0 and (t - 1) % int(a) == 0:
                m.zero_(); v.zero_(); st["n"] = 1; n = 1
                bc1, bc2 = 1 - B1, 1 - B2
            if n > 1:
                th = (5000 * v).sqrt()
                G = torch.where(G.abs() > th, torch.sign(G) * th, G)
            U = dense()
        elif rule == "galore":               # GaLore (Zhao 2024): Adam in a rank-a subspace of the d_model side, refreshed every 50 steps
            r_ = int(a)
            if "Pg" not in st or t % 50 == 1:
                _, _, Vh = torch.linalg.svd(G, full_matrices=False)
                st["Pg"] = Vh[:r_].T.contiguous()
                st.setdefault("gm", torch.zeros(G.shape[0], r_, device=G.device))
                st.setdefault("gv", torch.zeros(G.shape[0], r_, device=G.device))
            Pg = st["Pg"]
            Gl = G @ Pg
            st["gm"].mul_(B1).add_(Gl, alpha=1 - B1)
            st["gv"].mul_(B2).addcmul_(Gl, Gl, value=1 - B2)
            Ul = (st["gm"] / bc1) / ((st["gv"] / bc2).sqrt() + self.eps)
            U = (Ul @ Pg.T) * b
        elif rule == "grokfast":             # Grokfast (Lee 2024): g + a x EMA_0.98(g), slow gradient part amplified
            ge = st.setdefault("ge", torch.zeros_like(G))
            ge.mul_(0.98).add_(G, alpha=0.02)
            G = G + a * ge
            U = dense()
        elif rule == "atan2":                # Adam-atan2 (Everett 2024): eps-free 1.2732 atan2(m, sqrt v)
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = 1.2732 * torch.atan2(m / bc1, (v / bc2).sqrt())
        elif rule == "gc":                   # gradient centralization (Yong 2020): each row's gradient minus its d_model mean
            G = G - G.mean(1, keepdim=True)
            U = dense()
        elif rule == "graft":                # grafting (Agarwal 2020): momentum direction, Adam step size per tensor
            U0 = dense()
            mh = m / bc1
            U = mh * (U0.norm() / (mh.norm() + 1e-12))
        elif rule == "graftrow":             # idea: grafting per row (each row: momentum direction, its Adam row norm)
            U0 = dense()
            mh = m / bc1
            U = mh * (U0.norm(dim=1, keepdim=True) / (mh.norm(dim=1, keepdim=True) + 1e-12))
        elif rule == "gnoise":               # gradient noise (Neelakantan 2015): + N(0, (a RMS(g))^2 / (1+t)^0.55)
            s = a * float(G.pow(2).mean().sqrt()) / (1 + t) ** 0.275
            G = G + s * torch.randn_like(G)
            U = dense()
        elif rule == "xstep":                # L-A follow-up: borrow the other row's Adam-normalized momentum (no small-v blow-up)
            U0 = dense()
            ok = "head" if self.key == "emb" else "emb"
            so = self.state[ok]
            no = max(1, so["n"])
            Uo = (so["m"] / (1 - B1 ** no)) / ((so["v"] / (1 - B2 ** no)).sqrt() + self.eps)
            U = U0 + a * Uo
        elif rule in ("xmomsg", "xmomd", "xmomf"):  # L-A follow-ups: sign-agreeing borrow, decaying borrow, rare-row borrow
            U0 = dense()
            ok = "head" if self.key == "emb" else "emb"
            mo = self.state[ok]["m"]
            if rule == "xmomsg":
                mm = m + a * mo * (mo * m > 0).float()
            elif rule == "xmomd":
                mm = m + a * max(0.0, 1 - t / max(1, cfg.T)) * mo
            else:
                mm = m + a * (1 - freq).clamp(min=0)[:, None] * mo
            U = (mm / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "adagrad":              # AdaGrad (Duchi 2011): g^2 summed without decay, no momentum; a = step scale
            v.add_(G * G)
            U = a * G / (v.sqrt() + self.eps)
        elif rule == "rowadagrad":           # row-wise AdaGrad (FBGEMM / DLRM embeddings): one summed mean g^2 per row
            vr = st.setdefault("vr", torch.zeros(G.shape[0], 1, device=G.device))
            vr.add_(G.pow(2).mean(1, keepdim=True))
            U = a * G / (vr.sqrt() + self.eps)
        elif rule == "amsgrad":              # AMSGrad (Reddi 2018): running max of v
            dense()
            vm = st.setdefault("vmax", torch.zeros_like(G))
            torch.maximum(vm, v, out=vm)
            U = (m / bc1) / ((vm / bc2).sqrt() + self.eps)
        elif rule == "yogi":                 # Yogi (Zaheer 2018): additive v update
            m.mul_(B1).add_(G, alpha=1 - B1)
            g2 = G * G
            v.sub_((1 - B2) * torch.sign(v - g2) * g2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "radam":                # RAdam (Liu 2020): variance rectification, momentum-only steps early
            dense()
            ri = 2 / (1 - B2) - 1
            rt = ri - 2 * n * B2 ** n / (1 - B2 ** n)
            if rt > 4:
                r = math.sqrt((rt - 4) * (rt - 2) * ri / ((ri - 4) * (ri - 2) * rt))
                U = r * (m / bc1) / ((v / bc2).sqrt() + self.eps)
            else:
                U = m / bc1
        elif rule == "adamax":               # Adamax (Kingma 2015): infinity-norm second moment
            m.mul_(B1).add_(G, alpha=1 - B1)
            torch.maximum(v * B2, G.abs(), out=v)
            U = (m / bc1) / (v + self.eps)
        elif rule == "qhadam":               # QHAdam (Ma 2019): nu1 = a mix of momentum and raw gradient, nu2 = 1
            dense()
            U = ((1 - a) * G + a * m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "padam":                # Padam (Chen 2018): v^a partial adaptivity, grafted to the Adam step RMS
            U0 = dense()
            U = (m / bc1) / ((v / bc2).pow(a) + self.eps ** (2 * a))
            U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule == "sm3":                  # SM3-II (Anil 2019): row/column max cover of summed g^2, momentum on the scaled step
            r_ = st.setdefault("r3", torch.zeros(G.shape[0], device=G.device))
            c_ = st.setdefault("c3", torch.zeros(G.shape[1], device=G.device))
            nu = torch.minimum(r_[:, None], c_[None, :]) + G * G
            r_.copy_(nu.max(1).values)
            c_.copy_(nu.max(0).values)
            m.mul_(B1).add_(G / (nu.sqrt() + self.eps), alpha=1 - B1)
            U = a * m / bc1
        elif rule in ("rprop", "flipreset"):  # Rprop (Riedmiller 1993) sign steps with per-element sizes; flipreset: m := g where sign flips
            gp = st.setdefault("gp", torch.zeros_like(G))
            if rule == "rprop":
                s = st.setdefault("s", torch.ones_like(G))
                agree = gp * G
                s.mul_(torch.where(agree > 0, torch.full_like(s, 1.2), torch.where(agree < 0, torch.full_like(s, 0.5), torch.ones_like(s))))
                s.clamp_(1e-2, 3.0)
                U = torch.sign(G) * s
            else:
                m.copy_(torch.where(m * G < 0, G, m))
                U = dense()
            gp.copy_(G)
        elif rule == "madam":                # Madam (Bernstein 2020): multiplicative step W exp(-a lr sign(W) g/sqrt v)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            gn = G / ((v / bc2).sqrt() + self.eps)
            Wf = W.float()
            U = Wf * (1 - torch.exp(-a * lr * torch.sign(Wf) * gn)) / lr
        elif rule == "zclip":                # ZClip (Kumar 2025): clip the gradient norm at its EMA mean + 2.5 std
            gnrm = float(G.norm())
            mu, var = st.get("zmu"), st.get("zvar")
            if mu is None:
                st["zmu"], st["zvar"] = gnrm, 0.0
            else:
                cap = mu + 2.5 * math.sqrt(var)
                if t > 5 and gnrm > cap:
                    G = G * (cap / gnrm)
                    gnrm = cap
                d_ = gnrm - mu
                st["zmu"] = mu + 0.03 * d_
                st["zvar"] = 0.97 * (var + 0.03 * d_ * d_)
            U = dense()
        elif rule in ("radrot", "hadarot"):  # idea: Adam in a fixed random (radrot) or Hadamard (hadarot) basis of d_model
            if "Qr" not in st:
                d = G.shape[1]
                if rule == "hadarot" and d & (d - 1) == 0:
                    H_ = torch.ones(1, 1)
                    while H_.shape[0] < d:
                        H_ = torch.cat([torch.cat([H_, H_], 1), torch.cat([H_, -H_], 1)], 0)
                    st["Qr"] = (H_ / math.sqrt(d)).to(G.device)
                else:
                    gen = torch.Generator().manual_seed(0)
                    st["Qr"] = torch.linalg.qr(torch.randn(d, d, generator=gen))[0].to(G.device)
            Q = st["Qr"]
            Gq = G @ Q
            m.mul_(B1).add_(Gq, alpha=1 - B1)
            v.mul_(B2).addcmul_(Gq, Gq, value=1 - B2)
            U = ((m / bc1) / ((v / bc2).sqrt() + self.eps)) @ Q.T
        elif rule == "signmom":              # idea: momentum of gradient signs, step size = sign agreement over time
            m.mul_(B1).add_(torch.sign(G), alpha=1 - B1)
            U = m / bc1
        elif rule == "median3":              # idea: element-wise median of the last 3 gradients feeds Adam
            g1, g2 = st.get("g1"), st.get("g2")
            G0 = G
            if g1 is not None and g2 is not None:
                G = torch.stack([G0, g1, g2]).median(0).values
            st["g2"], st["g1"] = g1, G0.clone()
            U = dense()
        elif rule == "topkrow":              # idea: only the top a fraction of rows by gradient norm update; rest kept as error feedback
            e_ = st.setdefault("err", torch.zeros_like(G))
            Ge = G + e_
            k = max(1, int(a * G.shape[0]))
            idx = Ge.norm(dim=1).topk(k).indices
            mk = torch.zeros(G.shape[0], 1, device=G.device)
            mk[idx] = 1.0
            G = Ge * mk
            e_.copy_(Ge * (1 - mk))
            U = dense()
        elif rule == "udrop":                # idea: dropout on the update (a = drop rate), rescaled
            U = dense()
            U = U * (torch.rand_like(U) > a).float() / (1 - a)
        elif rule == "bbdiag":               # idea: diagonal secant curvature |dg|/|dw| replaces sqrt v, grafted to the Adam RMS
            U0 = dense()
            gp, wp = st.get("bgp"), st.get("bwp")
            hv = st.setdefault("hv", torch.zeros_like(G))
            if gp is not None:
                hd = (G - gp).abs() / ((W.float() - wp).abs() + 1e-12)
                hv.mul_(B2).addcmul_(hd, hd, value=1 - B2)
                hs = hv.sqrt()
                U = (m / bc1) / (hs + 0.1 * hs.mean() + 1e-12)
                U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            else:
                U = U0
            st["bgp"], st["bwp"] = G.clone(), W.float().clone()
        elif rule == "orthprev":             # idea: remove a x the part of the step along the previous step
            U = dense()
            up = st.get("up")
            if up is not None:
                U = U - a * ((U * up).sum() / (up.pow(2).sum() + 1e-12)) * up
            st["up"] = U.clone()
        elif rule == "l2init":               # idea: decoupled pull toward the initial rows, strength a
            w0 = st.setdefault("w0", W.float().clone())
            U = dense() + a * (W.float() - w0)
        elif rule in ("deflate", "gdeflate"):  # idea: remove a x the top singular direction of the step (deflate) or gradient (gdeflate)
            X = dense() if rule == "deflate" else G
            q = st.setdefault("pv", torch.randn(X.shape[1], device=X.device))
            for _ in range(3):
                q = X.T @ (X @ q)
                q = q / (q.norm() + 1e-12)
            st["pv"] = q
            X = X - a * (X @ q)[:, None] * q[None, :]
            if rule == "gdeflate":
                G = X
                U = dense()
            else:
                U = X
        elif rule == "phase":                # idea: each d_model column gets lr x (1 + a cos(2 pi t / 50 + 2 pi j / d))
            U = dense()
            d = U.shape[1]
            ph = torch.arange(d, device=U.device).float() * (2 * math.pi / d)
            U = U * (1 + a * torch.cos(2 * math.pi * t / 50 + ph))[None, :]
        elif rule == "l1v":                  # idea: first absolute moment as the normalizer instead of sqrt of the second
            m.mul_(B1).add_(G, alpha=1 - B1)
            v.mul_(B2).add_(G.abs(), alpha=1 - B2)
            U = (m / bc1) / (v / bc2 + self.eps)
        elif rule == "snr":                  # idea: row step x (row signal-to-noise / mean)^a, clamped to [0.2, 5]
            U = dense()
            s_ = (m / bc1).pow(2).sum(1) / ((v / bc2).sum(1) + 1e-30)
            U = U * ((s_ / (s_.mean() + 1e-30)).pow(a).clamp(0.2, 5.0))[:, None]
        elif rule == "aligndamp":            # idea: row step x (1 - a cos(step row, weight row)): damp steps that shrink a row
            U = dense()
            Wf = W.float()
            c_ = (U * Wf).sum(1) / (U.norm(dim=1) * Wf.norm(dim=1) + 1e-12)
            U = U * (1 - a * c_)[:, None]
        elif rule in ("sinkhorn", "colnorm"):  # idea: balance the step's row and column RMS (sinkhorn, 2 rounds) or columns only
            U0 = dense()
            U = U0
            for _ in range(2 if rule == "sinkhorn" else 1):
                if rule == "sinkhorn":
                    rr = U.pow(2).mean(1, keepdim=True).sqrt()
                    U = U / (rr / (rr.mean() + 1e-12) + 1e-3)
                cc = U.pow(2).mean(0, keepdim=True).sqrt()
                U = U / (cc / (cc.mean() + 1e-12) + 1e-3)
            U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule == "nobc":                 # idea: Adam without bias correction
            dense()
            U = m / (v.sqrt() + self.eps)
        elif rule == "vlag":                 # idea (AdaShift-like): divide by last step's v, so the current g is not in its own normalizer
            vo, no_ = v.clone(), max(1, n - 1)
            dense()
            if n > 1:
                U = (m / bc1) / ((vo / (1 - B2 ** no_)).sqrt() + self.eps)
            else:
                U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule in ("xstepgn", "uwx"):     # L-A follow-ups: xstep + gradient noise 0.1 (xstepgn); xstep then uwhite with ridge b (uwx)
            if rule == "xstepgn":
                G = G + 0.1 * float(G.pow(2).mean().sqrt()) / (1 + t) ** 0.275 * torch.randn_like(G)
            U0 = dense()
            ok = "head" if self.key == "emb" else "emb"
            so = self.state[ok]
            no = max(1, so["n"])
            U0 = U0 + a * (so["m"] / (1 - B1 ** no)) / ((so["v"] / (1 - B2 ** no)).sqrt() + self.eps)
            if rule == "uwx":
                C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                C.mul_(B2).add_(U0.T @ U0 / G.shape[0], alpha=1 - B2)
                M = C / bc2
                M = M / (M.diagonal().mean() + 1e-12)
                e, Q = torch.linalg.eigh(M + b * torch.eye(M.shape[0], device=M.device))
                U = U0 @ (Q @ torch.diag(e.clamp(min=1e-6) ** -0.5) @ Q.T)
                U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            else:
                U = U0
        elif rule == "fixrms":               # idea: Adam direction, step RMS fixed at a every step (no natural step decay)
            U = dense()
            U = U * (a / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule == "abeta":                # idea: per-row momentum coefficient = 0.9 x max(0, cos(g row, m row)); restart where they disagree
            nm = m.norm(dim=1) * G.norm(dim=1)
            cs = ((m * G).sum(1) / (nm + 1e-30)).clamp(min=0)
            bt = (B1 * cs)[:, None]
            m.mul_(bt).add_((1 - bt) * G)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = m / ((v / bc2).sqrt() + self.eps)
        elif rule == "antipgd":              # Anti-PGD (Orvieto 2022): add xi_t - xi_{t-1}, xi ~ N(0, (a RMS g)^2)
            xi = a * float(G.pow(2).mean().sqrt()) * torch.randn_like(G)
            xp = st.get("xi")
            G = G + xi - (xp if xp is not None else 0)
            st["xi"] = xi
            U = dense()
        elif rule == "iso":                  # idea: step + a x (W (C/mean diag - I)) grafted to the step RMS: pushes the rows' d_model covariance to isotropy
            U = dense()
            Wf = W.float()
            Cw = Wf.T @ Wf / Wf.shape[0]
            Cw = Cw / (Cw.diagonal().mean() + 1e-12)
            Rg = Wf @ (Cw - torch.eye(Cw.shape[0], device=Cw.device))
            U = U + a * Rg * (U.pow(2).mean().sqrt() / (Rg.pow(2).mean().sqrt() + 1e-12))
        elif rule in ("mrestart", "rowrestart"):  # adaptive restart (O'Donoghue 2015): zero m (whole tensor / per row) when it opposes g
            if rule == "mrestart":
                if float((m * G).sum()) < 0:
                    m.zero_()
            else:
                m.mul_(((m * G).sum(1, keepdim=True) >= 0).float())
            U = dense()
        elif rule == "hball":                # idea: heavy-ball momentum a on the Adam step itself
            U = dense()
            hb = st.get("hbu")
            if hb is not None:
                U = U + a * hb
            st["hbu"] = U.clone()
        elif rule == "uema":                 # idea: EMA (decay a) of the Adam step, bias-corrected
            ue = st.setdefault("ue", torch.zeros_like(G))
            ue.mul_(a).add_(dense(), alpha=1 - a)
            U = ue / (1 - a ** n)
        elif rule == "rowgn":                # idea: every nonzero row's gradient rescaled to the tensor's RMS row norm before Adam
            rn = G.norm(dim=1, keepdim=True)
            nz = rn > 0
            if nz.any():
                G = torch.where(nz, G * (rn[nz].pow(2).mean().sqrt() / rn.clamp(min=1e-30)), G)
            U = dense()
        elif rule in ("sclip", "svpow"):     # idea: shape the step's singular values: clip at a x median (sclip) or S^a (svpow), grafted to the step RMS
            U0 = dense()
            Qs, S, Vh = torch.linalg.svd(U0, full_matrices=False)
            S2 = torch.minimum(S, a * S.median()) if rule == "sclip" else S.clamp(min=1e-12).pow(a)
            U = (Qs * S2[None, :]) @ Vh
            U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
        elif rule == "roweps":               # idea: per-row eps = a x the row's mean sqrt v
            dense()
            sv = (v / bc2).sqrt()
            U = (m / bc1) / (sv + a * sv.mean(1, keepdim=True) + self.eps)
        elif rule == "b1ramp":               # idea: beta1 ramps from a to 0.9 over the first half of training
            b1 = a + (B1 - a) * min(1.0, t / max(1, cfg.T / 2))
            st["pb1"] = st.get("pb1", 1.0) * b1
            m.mul_(b1).add_(G, alpha=1 - b1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / (1 - st["pb1"])) / ((v / bc2).sqrt() + self.eps)
        elif rule == "split":                # L-A x L-D: head = update whitening with ridge a, emb = borrow b x the head's momentum
            U0 = dense()
            if self.key == "head":
                C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                C.mul_(B2).add_(U0.T @ U0 / G.shape[0], alpha=1 - B2)
                M = C / bc2
                M = M / (M.diagonal().mean() + 1e-12)
                e, Q = torch.linalg.eigh(M + a * torch.eye(M.shape[0], device=M.device))
                U = U0 @ (Q @ torch.diag(e.clamp(min=1e-6) ** -0.5) @ Q.T)
                U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            else:
                U = ((m + b * self.state["head"]["m"]) / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "laprop":               # LaProp (Ziyin 2020): momentum of the v-normalized gradient
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            m.mul_(B1).add_(G / ((v / bc2).sqrt() + self.eps), alpha=1 - B1)
            U = m / bc1
        elif rule == "adabound":             # AdaBound (Luo 2019): per-coordinate step size clipped to bounds closing on the median at speed a
            dense()
            eta = 1.0 / ((v / bc2).sqrt() + self.eps)
            c = eta.median()
            eta = torch.minimum(torch.maximum(eta, c * (1 - 1 / (a * t + 1))), c * (1 + 1 / (a * t)))
            U = (m / bc1) * eta
        elif rule == "grams":                # Grams (Cao 2024): Adam step magnitude with the current gradient's sign
            U = dense().abs() * torch.sign(G)
        elif rule == "normuon":              # NorMuon (2025): Newton-Schulz momentum, per-row second-moment normalized, grafted to a x the dense step RMS
            Ud = dense()
            O = _ns(m)
            vr = st.setdefault("vr", torch.zeros(O.shape[0], 1, device=O.device))
            vr.mul_(B2).add_(O.pow(2).mean(1, keepdim=True), alpha=1 - B2)
            O = O / ((vr / bc2).sqrt() + 1e-8)
            U = O * (a * Ud.pow(2).mean().sqrt() / (O.pow(2).mean().sqrt() + 1e-12))
        elif rule in ("hyper", "rowhyper"):  # hypergradient descent (Baydin 2018): lr multiplier x (1 + a cos(g_t, U_t-1)), clamped to [0.25, 4]
            Ud = dense()
            pu = st.get("pu")
            if rule == "hyper":
                s_ = st.get("hs", 1.0)
                if pu is not None:
                    c = float((G * pu).sum() / (G.norm() * pu.norm() + 1e-12))
                    s_ = min(4.0, max(0.25, s_ * (1 + a * c)))
            else:
                s_ = st.get("hs", torch.ones(G.shape[0], 1, device=G.device))
                if pu is not None:
                    c = (G * pu).sum(1, keepdim=True) / (G.norm(dim=1, keepdim=True) * pu.norm(dim=1, keepdim=True) + 1e-12)
                    s_ = (s_ * (1 + a * c)).clamp(0.25, 4.0)
            st["hs"], st["pu"] = s_, Ud.clone()
            U = Ud * s_
        elif rule == "sophiaclip":           # Sophia-style (Liu 2023): m / (a sqrt(h)), h = g^2 EMA refreshed every 10 steps, clipped elementwise to [-1, 1]
            m.mul_(B1).add_(G, alpha=1 - B1)
            h = st.setdefault("h", G * G)
            if t % 10 == 0:
                h.mul_(0.9).addcmul_(G, G, value=0.1)
            U = ((m / bc1) / (a * h.sqrt() + self.eps)).clamp(-1.0, 1.0)
        elif rule == "agc":                  # adaptive gradient clipping (Brock 2021): each row's gradient clipped to a x its weight-row norm, then Adam
            gn = G.norm(dim=1, keepdim=True)
            wn = W.float().norm(dim=1, keepdim=True).clamp(min=1e-3)
            G = G * torch.clamp(a * wn / (gn + 1e-12), max=1.0)
            U = dense()
        elif rule == "diffgrad":             # diffGrad (Dubey 2019), scale-free: step x sigmoid(a |g_t - g_t-1| / sqrt(v))
            pg = st.get("pg")
            Ud = dense()
            xi = torch.sigmoid(a * (G - pg).abs() / ((v / bc2).sqrt() + self.eps)) if pg is not None else 1.0
            st["pg"] = G.clone()
            U = Ud * xi
        elif rule == "pnm":                  # positive-negative momentum (Xie 2021): alternate beta^2 buffers, step ((1+a) m_cur - a m_other) / sqrt((1+a)^2 + a^2), Adam-normalized
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            m2 = st.setdefault("m2", torch.zeros_like(m))
            cur, oth = (m, m2) if t % 2 else (m2, m)
            cur.mul_(B1 ** 2).add_(G, alpha=1 - B1 ** 2)
            nc, no = (t + 1) // 2 if t % 2 else t // 2, t // 2 if t % 2 else (t + 1) // 2
            mc = cur / (1 - B1 ** (2 * nc))
            mo = oth / (1 - B1 ** (2 * no)) if no > 0 else mc
            U = (((1 + a) * mc - a * mo) / math.sqrt((1 + a) ** 2 + a ** 2)) / ((v / bc2).sqrt() + self.eps)
        elif rule == "aggmo":                # AggMo (Lucas 2019): mean of momentum buffers with beta 0, 0.9 and a, Adam-normalized
            dense()
            m3 = st.setdefault("m3", torch.zeros_like(m))
            m3.mul_(a).add_(G, alpha=1 - a)
            mm = (G + m / bc1 + m3 / (1 - a ** n)) / 3
            U = mm / ((v / bc2).sqrt() + self.eps)
        elif rule == "adanorm":              # AdaNorm (Dubey 2023): gradient scaled up to the EMA a of past gradient norms before the first moment
            gn = float(G.norm())
            e = a * st.get("en", gn) + (1 - a) * gn
            st["en"] = e
            m.mul_(B1).add_(G * max(1.0, e / (gn + 1e-12)), alpha=1 - B1)
            v.mul_(B2).addcmul_(G, G, value=1 - B2)
            U = (m / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "splitc":               # L-CX x L-A: head = singular-value clip at a x median, emb = borrow b x the head's momentum
            U0 = dense()
            if self.key == "head":
                Qs, S, Vh = torch.linalg.svd(U0, full_matrices=False)
                U = (Qs * torch.minimum(S, a * S.median())[None, :]) @ Vh
                U = U * (U0.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            else:
                U = ((m + b * self.state["head"]["m"]) / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "wbasis":               # idea (r17): Adam in the right-singular basis of the weight itself, basis refreshed every a steps, momentum rotated along
            Q = st.get("Qw")
            if Q is None or t % int(a) == 1:
                _, _, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                Qn = Vh.T
                if Q is not None:
                    st["mr"] = st["mr"] @ (Q.T @ Qn)
                Q = st["Qw"] = Qn
            Gr = G @ Q
            mr = st.setdefault("mr", torch.zeros_like(Gr))
            vr = st.setdefault("vrr", torch.zeros_like(Gr))
            mr.mul_(B1).add_(Gr, alpha=1 - B1)
            vr.mul_(B2).addcmul_(Gr, Gr, value=1 - B2)
            U = ((mr / bc1) / ((vr / bc2).sqrt() + self.eps)) @ Q.T
        elif rule == "softtie":              # idea (r17): soft tying, each row pulled toward the same token's row in the other table with strength a
            ok = "head" if self.key == "emb" else "emb"
            U = dense() + a * (W.float() - self.W[ok].float())
        elif rule == "wspread":              # idea (r17): flatten the weight's own spectrum: minus a x W (W^T W / mean eig - I), relative to the dense step
            Ud = dense()
            Wf = W.float()
            C = Wf.T @ Wf
            C = C / (C.diagonal().mean() + 1e-12)
            R = Wf @ (C - torch.eye(C.shape[0], device=C.device))
            U = Ud + a * R * (Ud.pow(2).mean().sqrt() / (Wf.pow(2).mean().sqrt() + 1e-12))
        elif rule == "lrmom":                # idea (r17): low-rank momentum (top a singular directions of m) plus the raw gradient outside them, Adam-normalized
            dense()
            _, _, Vh = torch.linalg.svd(m, full_matrices=False)
            Pk = Vh[:int(a)].T
            mm = (m @ Pk) @ Pk.T / bc1 + (G - (G @ Pk) @ Pk.T)
            U = mm / ((v / bc2).sqrt() + self.eps)
        elif rule == "gscale":               # idea (r17): Adam step x (row gradient norm / mean present-row norm)^a, clamped to [0.25, 4]
            Ud = dense()
            gn = G.norm(dim=1, keepdim=True)
            mu = gn[P].mean() if P.any() else gn.mean()
            U = Ud * ((gn / (mu + 1e-12)).clamp(min=1e-6) ** a).clamp(0.25, 4.0)
        elif rule == "accum":                # idea (r17): the table accumulates a steps of gradient and takes one Adam step on their mean, a x as large
            k = int(a)
            acc = st.setdefault("acc", torch.zeros_like(G))
            acc.add_(G)
            if t % k:
                U = torch.zeros_like(G)
            else:
                Ga = acc / k
                acc.zero_()
                st["na"] = st.get("na", 0) + 1
                m.mul_(B1).add_(Ga, alpha=1 - B1)
                v.mul_(B2).addcmul_(Ga, Ga, value=1 - B2)
                U = k * (m / (1 - B1 ** st["na"])) / ((v / (1 - B2 ** st["na"])).sqrt() + self.eps)
        elif rule == "hproj":                # idea (r17): the step loses its projection onto the other table's top a right-singular directions
            ok = "head" if self.key == "emb" else "emb"
            Ud = dense()
            _, _, Vh = torch.linalg.svd(self.W[ok].float(), full_matrices=False)
            Pk = Vh[:int(a)].T
            U = Ud - (Ud @ Pk) @ Pk.T
        elif rule == "crossdamp":            # idea (r17): a token's emb and head steps are damped by 1/(1 + a max(0, cos)) when they point the same way
            ok = "head" if self.key == "emb" else "emb"
            Ud = dense()
            Uo = self.state[ok].get("Ucd")
            if Uo is not None:
                c = (Ud * Uo).sum(1, keepdim=True) / (Ud.norm(dim=1, keepdim=True) * Uo.norm(dim=1, keepdim=True) + 1e-12)
                U = Ud / (1 + a * c.clamp(min=0))
            else:
                U = Ud
            st["Ucd"] = Ud
        elif rule == "uq2ef":                # idea (r17): ternary step (-s, 0, +s per row, s = row RMS, threshold a x s) with error feedback
            Ud = dense()
            X = Ud + st.setdefault("ef", torch.zeros_like(Ud))
            sr = X.pow(2).mean(1, keepdim=True).sqrt()
            U = torch.sign(X) * (X.abs() > a * sr).float() * sr
            st["ef"] = X - U
        elif rule == "knnmom":               # idea (r17): momentum mixed with the mean momentum of each row's 4 nearest rows (cosine in the weight, refreshed every 20 steps), weight a
            dense()
            if "nb" not in st or t % 20 == 1:
                Wn = W.float() / (W.float().norm(dim=1, keepdim=True) + 1e-12)
                idx = []
                for i in range(0, Wn.shape[0], 1024):
                    Sm = Wn[i:i + 1024] @ Wn.T
                    r_ = torch.arange(Sm.shape[0], device=Sm.device)
                    Sm[r_, r_ + i] = -2.0
                    idx.append(Sm.topk(4, dim=1).indices)
                st["nb"] = torch.cat(idx)
            mm = (m + a * m[st["nb"]].mean(1)) / (1 + a)
            U = (mm / bc1) / ((v / bc2).sqrt() + self.eps)
        elif rule == "specdecay":            # idea (r17): spectral-norm decay: the top singular direction of W (power iteration) is shrunk by a x the dense step norm
            Ud = dense()
            Wf = W.float()
            q = st.get("pv")
            if q is None:
                q = torch.randn(Wf.shape[1], device=Wf.device)
            for _ in range(2):
                u = Wf @ q
                u = u / (u.norm() + 1e-12)
                q = Wf.T @ u
                q = q / (q.norm() + 1e-12)
            st["pv"] = q
            U = Ud + a * Ud.norm() * torch.outer(u, q)
        elif rule == "initball":             # idea (r17): weight-space trust region: each row stays within a x the median init row norm of its init row
            Ud = dense()
            W0 = st.setdefault("w0", W.float().clone())
            r0 = st.setdefault("r0", float(W0.norm(dim=1).median()))
            if lr > 0:
                dlt = W.float() - lr * Ud - W0
                dn = dlt.norm(dim=1, keepdim=True)
                dlt = dlt * (a * r0 / (dn + 1e-12)).clamp(max=1.0)
                U = (W.float() - (W0 + dlt)) / lr
            else:
                U = Ud
        elif rule == "bucketv":              # idea (r17): second moment pooled per coordinate inside a log-frequency buckets of rows (between Adam and Coupled Adam)
            dense()
            k = int(a)
            bkt = (torch.log1p(st["cnt"]) / (math.log1p(t) + 1e-12) * (k - 1)).round().long().clamp(0, k - 1)
            vs = torch.zeros(k, v.shape[1], device=v.device).index_add_(0, bkt, v)
            nb = torch.bincount(bkt, minlength=k).clamp(min=1)[:, None].float()
            U = (m / bc1) / (((vs / nb)[bkt] / bc2).sqrt() + self.eps)
        elif rule == "blockmuon":            # idea (r17): Newton-Schulz orthogonalization on blocks of a consecutive rows of the momentum, grafted to the dense step RMS
            Ud = dense()
            k = int(a)
            O = torch.cat([_ns(m[i:i + k]) for i in range(0, m.shape[0], k)])
            U = O * (Ud.pow(2).mean().sqrt() / (O.pow(2).mean().sqrt() + 1e-12))
        elif rule == "rda":                  # idea (r17): AdaGrad dual averaging (Duchi 2011, DA form): W = W0 - a lr (sum g) / sqrt(sum g^2)
            Sg = st.setdefault("Sg", torch.zeros_like(G))
            Qg = st.setdefault("Qg", torch.zeros_like(G))
            Sg.add_(G)
            Qg.addcmul_(G, G)
            W0 = st.setdefault("w0", W.float().clone())
            U = (W.float() - (W0 - lr * a * Sg / (Qg.sqrt() + self.eps))) / lr if lr > 0 else torch.zeros_like(G)
        elif rule in ("sgn", "sgnF", "sgnE", "sflr", "shuw", "srare", "ssvp", "sxm", "selazy", "secatch", "sgnxm", "shuwxm"):
            # round 18 (2026-11-23): SOAP (Vyas 2024, one side, as soap1) plus one fix aimed at a measured SOAP failure at XS 600:
            # F1 code frequent targets (SOAP tgtF 1.433 vs gradient noise 1.416), F2 FineWeb rare targets (tgtR 8.249 vs 8.211
            # with head whitening), F3 rare inputs (code inR 4.066 vs 3.978 with the emb momentum borrow).
            hd, em = self.key == "head", self.key == "emb"
            fq = (freq > (b if rule in ("sgnF", "sflr") else 0.5))[:, None].float()
            if (rule in ("sgn",)) or (rule == "sgnE" and em) or (rule in ("sgnF", "sgnxm") and hd):
                s_ = a * float(G.pow(2).mean().sqrt()) / (1 + t) ** 0.275
                nz = s_ * torch.randn_like(G)
                G = G + (nz * fq if rule in ("sgnF", "sgnxm") else nz)
            U0 = dense()
            mx = m
            if em and rule in ("sxm", "sgnxm", "shuwxm"):
                mx = m + (b if rule != "sgnxm" else 0.2) * self.state["head"]["m"]
            C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
            C.mul_(B2).add_(G.T @ G / G.shape[0], alpha=1 - B2)
            M = C / bc2
            e, Q = torch.linalg.eigh(M + 1e-6 * M.diagonal().mean() * torch.eye(M.shape[0], device=M.device))
            vq = st.setdefault("vq", torch.zeros_like(G))
            Gq = G @ Q
            vq.mul_(B2).addcmul_(Gq, Gq, value=1 - B2)
            U = (((mx @ Q) / bc1) / ((vq / bc2).sqrt() + self.eps)) @ Q.T
            Us = U
            if rule == "sflr" and hd:
                U = U * (1 - fq + a * fq)
            elif rule == "srare" and hd:
                rr = (freq < b)[:, None].float()
                U = U * (1 - rr + a * rr)
            elif rule == "ssvp":
                Qs, S, Vh = torch.linalg.svd(U, full_matrices=False)
                U = (Qs * S.clamp(min=1e-12).pow(a)[None, :]) @ Vh
                U = U * (Us.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            elif rule in ("shuw", "shuwxm") and hd:
                Cu = st.setdefault("Cu", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                Cu.mul_(B2).add_(U.T @ U / G.shape[0], alpha=1 - B2)
                Mu = Cu / bc2
                Mu = Mu / (Mu.diagonal().mean() + 1e-12)
                eu, Qu = torch.linalg.eigh(Mu + a * torch.eye(Mu.shape[0], device=Mu.device))
                U = U @ (Qu @ torch.diag(eu.clamp(min=1e-6) ** -0.5) @ Qu.T)
                U = U * (Us.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            elif rule == "selazy" and em:
                U = U * Pf
            elif rule == "secatch" and em:
                U = U * torch.where(P, (1 + gap.float()).clamp(max=a).sqrt(), torch.ones_like(gap, dtype=torch.float))[:, None]
        elif rule == "sc" or rule.startswith("sc_"):
            # round 20 (2026-11-23): one rule, fixes as flags joined by "_" (no ":" since that picks the table).
            # base per table = SOAP (one side, as soap1); eadam / hadam switch that table to Adam (F4: SOAP on the emb hurt code).
            # numbers with a leading 0 are decimals (03 = 0.3, 005 = 0.05). huwN head update whitening ridge N (F2); xmN emb momentum
            # + N x head momentum (F3);
            # rareA[bB] head rows present on < B/100 of steps (default 5) step x A, A "15" = 1.5 (F2); catchA emb row back after
            # k steps: step x min(1+k, A)^0.5 (F3); gnEN emb gradient noise N (F3); gnFN noise on head rows present > half (F1).
            import re as _re
            fl = rule.split("_")[1:]
            hd, em = self.key == "head", self.key == "emb"
            def _fv(p, s):
                for f in fl:
                    mm = _re.fullmatch(p, f)
                    if mm: return mm
                return None
            def _d(s): return float("0." + s[1:]) if s.startswith("0") else float(s)
            gE, gF = _fv(r"gnE(\d+)", fl), _fv(r"gnF(\d+)", fl)
            if (em and gE) or (hd and gF):
                c_ = _d((gE or gF).group(1))
                nz = c_ * float(G.pow(2).mean().sqrt()) / (1 + t) ** 0.275 * torch.randn_like(G)
                G = G + (nz if em else nz * (freq > 0.5)[:, None].float())
            U0 = dense()
            mx = m
            xm = _fv(r"xm(\d+)", fl)
            if em and xm:
                mx = m + _d(xm.group(1)) * self.state["head"]["m"]
            fqr = (freq > 0.5)[:, None].float()
            fe_ = _fv(r"(h|e)feps(\d+)", fl)
            epr = self.eps
            if fe_ and ((fe_.group(1) == "h" and hd) or (fe_.group(1) == "e" and em)):
                epr = self.eps * (1 + (float(fe_.group(2)) - 1) * fqr)
            if (em and "eadam" in fl) or (hd and "hadam" in fl):
                U = (mx / bc1) / ((v / bc2).sqrt() + epr)
            else:
                C = st.setdefault("C", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                C.mul_(B2).add_(G.T @ G / G.shape[0], alpha=1 - B2)
                M = C / bc2
                e, Q = torch.linalg.eigh(M + 1e-6 * M.diagonal().mean() * torch.eye(M.shape[0], device=M.device))
                vq = st.setdefault("vq", torch.zeros_like(G))
                Gq = G @ Q
                vq.mul_(B2).addcmul_(Gq, Gq, value=1 - B2)
                U = (((mx @ Q) / bc1) / ((vq / bc2).sqrt() + epr)) @ Q.T
                if hd and ("hfsoap" in fl or "hrsoap" in fl):   # SOAP on frequent head rows only (hfsoap) or rare only (hrsoap), Adam elsewhere
                    Ua = (mx / bc1) / ((v / bc2).sqrt() + epr)
                    w_ = fqr if "hfsoap" in fl else 1 - fqr
                    U = w_ * U + (1 - w_) * Ua
            Us = U
            hw = _fv(r"huw(\d+)", fl)
            if hd and hw:
                Cu = st.setdefault("Cu", torch.zeros(G.shape[1], G.shape[1], device=G.device))
                Cu.mul_(B2).add_(U.T @ U / G.shape[0], alpha=1 - B2)
                Mu = Cu / bc2
                Mu = Mu / (Mu.diagonal().mean() + 1e-12)
                eu, Qu = torch.linalg.eigh(Mu + _d(hw.group(1)) * torch.eye(Mu.shape[0], device=Mu.device))
                U = U @ (Qu @ torch.diag(eu.clamp(min=1e-6) ** -0.5) @ Qu.T)
                U = U * (Us.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            rr_ = _fv(r"rare(\d+)(?:b(\d+))?", fl)
            if hd and rr_:
                ra = float(rr_.group(1)); ra = ra / 10 if ra >= 10 else ra
                rb = float(rr_.group(2)) / 100 if rr_.group(2) else 0.05
                rr = (freq < rb)[:, None].float()
                U = U * (1 - rr + ra * rr)
            ca = _fv(r"catch(\d+)", fl)
            if em and ca:
                U = U * torch.where(P, (1 + gap.float()).clamp(max=float(ca.group(1))).sqrt(), torch.ones_like(gap, dtype=torch.float))[:, None]
            # round 21 (2026-11-23): frequency-dependent step sizes. At XS 1200 code, halving SOAP's table lr gave frequent targets
            # -0.010 nats and rare +0.016, so frequent rows may want a smaller step than rare rows. Flags (h = head, e = emb):
            # hfpP / hfpmP: step x freq^-P (m = negative), table RMS kept; hcdC: step x (1 + present-count / C)^-0.5;
            # hmXY: step x X.Y; hflateXY[bNN]: after half of training, rows present on > half (or NN/100) of steps x X.Y;
            # hlateXY: whole table x X.Y after half; hcen / hcenF: remove the step's mean over all / frequent head rows.
            def _m(s): return float(s[0] + "." + s[1:])
            tb = "h" if hd else "e"
            Ur = U
            fp = _fv(tb + r"fp(m?)(\d+)", fl)
            if fp:
                p_ = _d(fp.group(2)) * (-1 if fp.group(1) else 1)
                U = U * freq.clamp(min=0.01).pow(-p_)[:, None]
                U = U * (Ur.pow(2).mean().sqrt() / (U.pow(2).mean().sqrt() + 1e-12))
            cd = _fv(tb + r"cd(\d+)", fl)
            if cd:
                U = U * (1 + st["cnt"] / float(cd.group(1))).pow(-0.5)[:, None]
            hm_ = _fv(tb + r"m(\d\d+)", fl)
            if hm_:
                U = U * _m(hm_.group(1))
            fl_ = _fv(tb + r"flate(\d\d+)(?:b(\d+))?", fl)
            if fl_ and t > cfg.T / 2:
                thr = float(fl_.group(2)) / 100 if fl_.group(2) else 0.5
                fz = (freq > thr)[:, None].float()
                U = U * (1 - fz + _m(fl_.group(1)) * fz)
            la = _fv(tb + r"late(\d\d+)", fl)
            if la and t > cfg.T / 2:
                U = U * _m(la.group(1))
            if hd and "hcen" in fl:
                U = U - U.mean(0, keepdim=True)
            if hd and "hcenF" in fl:
                U = U - fqr * (U * fqr).sum(0, keepdim=True) / fqr.sum().clamp(min=1)
        elif rule in ("lookahead", "ema"):   # handled after the step (below)
            U = dense()
        else:
            raise ValueError(f"unknown erule {rule}")
        st["last"] = torch.where(P, torch.full_like(st["last"], t - 1), st["last"])
        self.urms[self.key] = float(U.pow(2).mean().sqrt())
        W.sub_(lr * U.to(W.dtype))
        if rule == "lookahead":              # Lookahead (Zhang 2019): every a steps slow += b (fast - slow), fast := slow
            sl = st.setdefault("slow", W.float().clone())
            if t % int(a) == 0:
                sl.add_(W.float() - sl, alpha=b)
                W.copy_(sl.to(W.dtype))
        elif rule == "ema":                  # weight EMA (Polyak averaging, decay a), swapped in for the last step / eval
            ew = st.setdefault("ew", W.float().clone())
            ew.mul_(a).add_(W.float(), alpha=1 - a)
            if t >= cfg.T:
                W.copy_(ew.to(W.dtype))
        if rule.startswith("sc_"):              # round 21 weight averaging: hfemaNN = EMA (decay 0.NN) of frequent head rows swapped in at
            # the last step; bemaNN = EMA of every row of both tables
            import re as _re
            for f_ in rule.split("_")[1:]:
                mm = _re.fullmatch(r"(hf|b)ema(\d+)", f_)
                if mm and (mm.group(1) == "b" or self.key == "head"):
                    dec = float("0." + mm.group(2)[1:])
                    ew = st.setdefault("ew", W.float().clone())
                    ew.mul_(dec).add_(W.float(), alpha=1 - dec)
                    if t >= cfg.T:
                        if mm.group(1) == "b":
                            W.copy_(ew.to(W.dtype))
                        else:
                            fz = (st["cnt"] / t > 0.5)[:, None]
                            W.copy_(torch.where(fz, ew, W.float()).to(W.dtype))
        if rule == "nullw":
            W.sub_(W.mean(0, keepdim=True))
        elif rule == "maxnorm":
            rn = W.norm(dim=1)
            cap = a * rn.median()
            W.mul_((cap / rn.clamp(min=1e-12)).clamp(max=1.0)[:, None].to(W.dtype))
        elif rule == "eqnorm":
            rn = W.norm(dim=1)
            W.mul_(((1 - b) + b * rn.mean() / rn.clamp(min=1e-12))[:, None].to(W.dtype))
