"""Training-data order and batch schedule (search 7, data).

build_plan(train, cfg, steps) returns a list of (starts, L) pairs, one per optimizer step: the step trains on
the windows train[s : s + L + 1] for s in starts. Every arm trains on exactly steps * batch * seq_len target
tokens. With all d_* fields at their defaults the plan is identical to the bed's loop (same rng calls).

Pipeline:
  1. base windows: steps * batch windows of seq_len + 1 tokens, drawn with the bed's rng.
  2. anneal (d_anneal > 0): the last d_anneal share of windows comes from a quality slice. d_amode="new" draws
     fresh candidates and keeps the top d_atop share by key d_akey; d_amode="move" moves the run's own top
     windows to the end (pure reorder, no new data).
  3. order (d_order): shuffle (bed), sort (curriculum by key d_key, direction d_dir, noise d_noise, only the
     first d_cfrac of the run is ordered), group (batches of similar key, batch order shuffled), strat (each
     batch spans all key quantiles), plus special orders listed in ORDERS.
  4. batch-size schedule: windows per step follow a multiplier of batch: ramp from d_bs0 to 1 over the first
     d_bsf share of tokens; optional end phase: last d_bsef share at multiplier d_bse.
  5. sequence-length schedule: each window is cut into seq_len / L chunks (same tokens, shorter context);
     L goes from d_seq0 to d_seq1 (log-linear) over the first d_seqf share of steps.
"""
import numpy as np

DEFAULTS = dict(d_order="shuffle", d_key="uni", d_dir=1.0, d_noise=0.0, d_cfrac=1.0,
                d_anneal=0.0, d_amode="new", d_akey="doclen", d_atop=0.3, d_adir=1.0,
                d_bs0=1.0, d_bsf=0.0, d_bse=1.0, d_bsef=0.0,
                d_seq0=0, d_seq1=0, d_seqf=0.0, d_seqe=0, d_seqef=0.0, d_seed=7, d_k=4, d_x=0.0,
                d_rep=1, d_repf=0.0, d_repg=1, d_rep_end=False, d_align=0.0, d_alignf=1.0,
                d_online=0.0, d_onf=1.0, d_onk=4,
                d_seqmode="chop", d_lfrac=0.0, d_bspow=1.0, d_wsd=0.0, d_seqpow=1.0,
                d_lend=0, d_lendf=0.0, d_lcyc=0, d_lkey="", d_ldir=1.0, d_lpool=2, d_lkf=0.0, d_lkey2="", d_lw2=0.0, d_lalign=0.0, d_lrep=1, d_ldisj=0,
                d_lperm=0, d_cont=0.0, d_contk=0, d_xsrc="", d_xp=0.0, d_xpos="head", d_xoff=0, d_xlen=0,
                d_loc=0.0, d_locg=1, d_locpos="head", d_dsp=0.0, d_dgap=16,
                d_cov=0.0, d_covk=4, d_covpos="head", d_covdir=1.0, d_xs=0.0, d_xsk=4, d_xspos="head", d_xsdir=1.0,
                d_ovl=0.0, d_ovs=0.5, d_ovlpos="head",
                d_on2="", d_on2f=0.25, d_on2pos="head", d_on2k=4, d_on2dir=1.0, d_onr=1.0, d_onp=6, d_fbuf=512,
                d_onstep=8, d_emad=0.9, d_lagg=8, d_splq0=0.25,
                d_qr=0.0, d_qrpos="all", d_qrmode="gold", d_rcs=0.0, d_rcsk=4, d_rcspos="all", d_rcsdir=1.0,
                d_pool=0.0, d_poolp=64, d_poolmode="rand", d_poolkey="", d_pooldir=1.0, d_poolmix=1.0, d_pooldec=0, d_poolsh=0, d_poolpos="head", d_pooljit=0.0, d_poolgrow=0.0, d_poolret=0.0, d_poolc=8, d_poolk=0.0, d_poolm=0.3, d_reg=0.0, d_reg0=0.01, d_echo=0.0, d_echos=0.5, d_echopos="head",
                d_gz=0.0, d_gzk=4, d_gzpos="head", d_gzdir=1.0, d_bg=0.0, d_bgk=4, d_bgpos="head", d_bgdir=1.0,
                d_spc=0.0, d_spcs=0.25, d_spcpos="head", d_rareR=1024, d_dup=0.0, d_dups=0.25, d_duppos="head", d_rbR=64, d_rbtau=0.02, d_rbsel="prog",
                d_shs=0.05, d_on2tau=0.0, d_lagema=0.0, d_on2k1=0, d_ptrim=0.0,
                d_clu=0.0, d_clupos="head", d_cluC=8, d_cluk=8, d_clumode="pure", d_clublk=4, d_dd=0.0, d_ddpos="head", d_ddn=16, d_ddt=0.25,
                d_gns=0.0, d_gnsmin=2, d_gnsmax=32, d_gnsema=0.9, d_gns0=10)


def base_starts(train, cfg, steps):
    rng = np.random.default_rng(cfg.seed)
    return np.concatenate([rng.integers(0, len(train) - cfg.seq_len - 1, cfg.batch) for _ in range(steps)])


class Keys:
    """per-window statistics; counts come from the run's own base windows"""

    def __init__(self, train, cfg, base):
        self.train, self.L, self.V = train, cfg.seq_len, cfg.vocab
        self.base = base
        X = self.tokens(base)
        self.uni = np.bincount(X.ravel(), minlength=self.V).astype(np.float64) + 0.5
        self.luni = np.log(self.uni / self.uni.sum())
        pairs = X[:, :-1].astype(np.int64) * self.V + X[:, 1:]
        self.pk, self.pc = np.unique(pairs, return_counts=True)
        self.eot = None

    def tokens(self, starts):
        return np.asarray(self.train[(np.asarray(starts)[:, None] + np.arange(self.L + 1))]).astype(np.int64)

    def bigram_lp(self, X):
        pairs = X[:, :-1] * self.V + X[:, 1:]
        i = np.searchsorted(self.pk, pairs)
        i = np.minimum(i, len(self.pk) - 1)
        c = np.where(self.pk[i] == pairs, self.pc[i], 0).astype(np.float64)
        prev = self.uni[X[:, :-1]] - 0.5
        return np.log((c + 0.1) / (prev + 0.1 * self.V))

    def docs(self):
        if self.eot is None:
            self.eot = np.flatnonzero(np.asarray(self.train) == 0)
        return self.eot

    def __call__(self, name, starts):
        X = self.tokens(starts)
        if name == "uni":            # mean unigram surprisal (rare-token rich = hard)
            return -self.luni[X].mean(1)
        if name == "bi":             # mean bigram surprisal under the run's counts
            return -self.bigram_lp(X).mean(1)
        if name == "uniq":           # distinct-token share (low = repetitive)
            return np.array([len(np.unique(r)) for r in X]) / X.shape[1]
        if name == "eot":            # document boundaries inside the window
            return (X == 0).sum(1).astype(np.float64)
        if name == "start":          # position in the corpus (documents read in file order)
            return np.asarray(starts).astype(np.float64)
        if name == "proj":           # 1-D random projection of the window's bag of words (topic order)
            g = np.random.default_rng(5).standard_normal(self.V) / np.sqrt(self.uni)
            return g[X].mean(1)
        if name == "rand":
            return np.random.default_rng(99).random(len(X))
        if name in ("doclen", "docpos"):
            e = self.docs()
            mid = np.asarray(starts) + self.L // 2
            j = np.searchsorted(e, mid, side="right") - 1
            lo = np.where(j >= 0, e[np.maximum(j, 0)], 0)
            hi = np.where(j + 1 < len(e), e[np.minimum(j + 1, len(e) - 1)], len(self.train))
            if name == "doclen":
                return np.log((hi - lo).astype(np.float64))
            return np.log1p((np.asarray(starts) - lo).clip(0).astype(np.float64))
        if name == "novel":          # first-seen share: rarest tokens under counts (for greedy order)
            return -self.luni[X].max(1)
        if name == "gap":            # bigram surprisal minus unigram surprisal: how much context helps
            return (-self.bigram_lp(X)).mean(1) - (-self.luni[X[:, 1:]]).mean(1)
        if name == "rbi":            # share of bigrams that already occurred earlier in the window (copy chances)
            out = np.empty(len(X))
            for i, r in enumerate(X):
                pr = r[:-1] * self.V + r[1:]
                _, first = np.unique(pr, return_index=True)
                out[i] = 1.0 - len(first) / len(pr)
            return out
        if name == "skip":           # skip-bigram surprisal (token t-2 -> t) minus bigram surprisal: long-range help
            if not hasattr(self, "pk2"):
                Xb = self.tokens(self.base)
                pr = Xb[:, :-2] * self.V + Xb[:, 2:]
                self.pk2, self.pc2 = np.unique(pr, return_counts=True)
            pr = X[:, :-2] * self.V + X[:, 2:]
            i = np.minimum(np.searchsorted(self.pk2, pr), len(self.pk2) - 1)
            cnt = np.where(self.pk2[i] == pr, self.pc2[i], 0).astype(np.float64)
            lp2 = np.log((cnt + 0.1) / (self.uni[X[:, :-2]] - 0.5 + 0.1 * self.V))
            return (-lp2).mean(1) - (-self.bigram_lp(X)[:, 1:]).mean(1)
        if name == "rdist":          # mean log distance to the previous copy of the same token (none = window length)
            out = np.empty(len(X)); n = X.shape[1]
            for i, r in enumerate(X):
                last, d = {}, np.full(n, float(n))
                for j, t in enumerate(r.tolist()):
                    if t in last:
                        d[j] = j - last[t]
                    last[t] = j
                out[i] = np.log(d).mean()
            return out
        if name == "rare":           # share of tokens outside the 1024 most frequent
            if not hasattr(self, "top"):
                self.top = np.zeros(self.V, bool); self.top[np.argsort(-self.uni)[:1024]] = True
            return (~self.top[X]).mean(1)
        if name == "sstd":           # spread of bigram surprisal inside the window (bursty difficulty)
            return (-self.bigram_lp(X)).std(1)
        if name == "bpt":            # UTF-8 bytes per token (long-token windows carry more bytes per prediction)
            if not hasattr(self, "tb"):
                from score import token_bytes
                self.tb = token_bytes().numpy().astype(np.float64)
                if len(self.tb) < self.V:
                    self.tb = np.concatenate([self.tb, np.ones(self.V - len(self.tb))])
            return self.tb[X].mean(1)
        if name == "kl":             # KL(window unigram || corpus unigram): low = representative window
            out = np.empty(len(X))
            for i, r in enumerate(X):
                t, cnt = np.unique(r, return_counts=True)
                q = cnt / cnt.sum()
                out[i] = (q * (np.log(q) - self.luni[t])).sum()
            return out
        if name == "zip":            # zlib size / raw size of the token bytes (high = little repeated structure)
            import zlib
            return np.array([len(zlib.compress(r.astype(np.uint16).tobytes(), 6)) for r in X]) / (2.0 * X.shape[1])
        if name == "ent":            # entropy of the window's own token distribution
            out = np.empty(len(X))
            for i, r in enumerate(X):
                _, cnt = np.unique(r, return_counts=True)
                q = cnt / cnt.sum()
                out[i] = -(q * np.log(q)).sum()
            return out
        if name == "maxrun":         # longest run of one repeated token
            out = np.empty(len(X))
            for i, r in enumerate(X):
                ch = np.flatnonzero(np.diff(r) != 0)
                edges = np.concatenate([[-1], ch, [len(r) - 1]])
                out[i] = np.diff(edges).max()
            return out
        if name == "idmean":         # mean token id (tokenizer merge rank, no corpus counts)
            return X.mean(1).astype(np.float64)
        if name == "idstd":
            return X.std(1).astype(np.float64)
        if name == "sim":            # Jaccard overlap of the token sets of the two window halves (topic consistency)
            h = X.shape[1] // 2
            out = np.empty(len(X))
            for i, r in enumerate(X):
                a, b = set(r[:h].tolist()), set(r[h:].tolist())
                out[i] = len(a & b) / max(1, len(a | b))
            return out
        raise ValueError(name)


def z(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def order_windows(W, keys, c, rng):
    n, o = len(W), c.d_order
    if o == "shuffle" or n == 0:
        return W
    k = z(keys(c.d_key, W)) * c.d_dir
    if o == "sort":                   # curriculum: first d_cfrac of the run ordered by key
        m = int(round(n * c.d_cfrac))
        sel = rng.permutation(n)
        a, b = sel[:m], sel[m:]
        sc = k[a] + c.d_noise * rng.standard_normal(m)
        return np.concatenate([W[a[np.argsort(sc, kind="stable")]], W[b]])
    if o == "group":                  # homogeneous batches, batch order random
        B = c.batch
        idx = np.argsort(k, kind="stable")
        chunks = [idx[i:i + B] for i in range(0, n, B)]
        return np.concatenate([W[chunks[i]] for i in rng.permutation(len(chunks))])
    if o == "strat":                  # each batch spans all quantiles of the key
        B = c.batch
        idx = np.argsort(k, kind="stable")
        steps = n // B
        q = idx[:steps * B].reshape(B, steps)          # row r = quantile band r
        q = np.stack([rng.permutation(r) for r in q])   # random step inside each band
        return np.concatenate([W[q[:, s]] for s in range(steps)] + [W[idx[steps * B:]]])
    if o == "cyc":                    # d_k cycles of easy->hard (sorted inside each of d_k random parts)
        parts = np.array_split(rng.permutation(n), int(c.d_k))
        return np.concatenate([W[p[np.argsort(k[p], kind="stable")]] for p in parts])
    if o == "vee":                    # easy->hard->easy (hard in the middle)
        idx = np.argsort(k, kind="stable")
        return np.concatenate([W[idx[0::2]], W[idx[1::2][::-1]]])
    if o == "novelty":                # greedy: next batch = windows with most not-yet-seen token types
        B = c.batch
        X = keys.tokens(W)
        seen = np.zeros(keys.V, bool)
        left = np.ones(n, bool)
        out = []
        sub = int(c.d_k) * B * 4      # candidate pool per step for speed
        while left.any():
            cand = np.flatnonzero(left)
            if len(cand) > sub:
                cand = rng.choice(cand, sub, replace=False)
            score = np.array([(~seen[np.unique(X[i])]).sum() for i in cand]) * c.d_dir
            pick = cand[np.argsort(-score, kind="stable")[:B]]
            out.append(pick)
            left[pick] = False
            seen[X[pick].ravel()] = True
        return W[np.concatenate(out)]
    if o == "cover":                  # each batch greedily maximizes distinct token types (diverse batches)
        B = c.batch
        X = keys.tokens(W)
        left = np.ones(n, bool)
        out = []
        while left.any():
            cand = np.flatnonzero(left)
            if len(cand) > int(c.d_k) * B * 4:
                cand = rng.choice(cand, int(c.d_k) * B * 4, replace=False)
            have = np.zeros(keys.V, bool)
            pick = []
            for _ in range(min(B, len(cand))):
                sc = np.array([(~have[X[i]]).sum() for i in cand]) * c.d_dir
                sc[np.isin(cand, pick)] = -10**9
                j = cand[int(np.argmax(sc))]
                pick.append(j)
                have[X[j]] = True
            out.append(np.array(pick))
            left[pick] = False
        return W[np.concatenate(out)]
    raise ValueError(o)


def align(W, keys, c, rng):
    """snap a share d_align of windows (within the first d_alignf of the run) to the start of their document"""
    if c.d_align <= 0:
        return W
    e = keys.docs()
    n = len(W)
    m = int(round(n * c.d_alignf))
    j = np.searchsorted(e, W, side="right") - 1
    snapped = np.where(j >= 0, e[np.maximum(j, 0)], W)
    hit = (rng.random(n) < c.d_align) & (np.arange(n) < m)
    return np.where(hit, snapped, W)


def anneal(W, keys, c, train, rng):
    n = len(W)
    m = int(round(n * c.d_anneal))
    if m == 0:
        return W
    if c.d_amode == "move":           # pure reorder: the run's own top-key windows go last
        k = keys(c.d_akey, W) * c.d_adir
        idx = np.argsort(k, kind="stable")
        head, tail = idx[:n - m], idx[n - m:]
        return np.concatenate([W[rng.permutation(head)], W[rng.permutation(tail)]])
    pool = rng.integers(0, len(train) - c.seq_len - 1, int(np.ceil(m / c.d_atop)) + 1)
    k = keys(c.d_akey, pool) * c.d_adir
    top = pool[np.argsort(-k, kind="stable")[:m]]
    return np.concatenate([W[:n - m], rng.permutation(top)])


def bs_mult(c, p):
    mult = c.d_bs0 + (1 - c.d_bs0) * min(1.0, p / c.d_bsf) ** c.d_bspow if c.d_bsf > 0 else c.d_bs0
    if c.d_bsef > 0 and p >= 1 - c.d_bsef:
        mult = c.d_bse
    return mult


def li_plan(train, c, steps):
    """Li, Zhang & He 2022 sequence-length warmup: fixed sequences per step (batch), length grows linearly from
    d_seq0 to seq_len over the first d_seqf share of tokens (multiples of 8), fresh random windows of that length,
    steps continue until the bed's token budget is used (early steps carry fewer tokens)."""
    rng = np.random.default_rng(c.seed)
    B, L, total = c.batch, c.seq_len, steps * c.batch * c.seq_len
    keys = Keys(train, c, base_starts(train, c, steps)) if c.d_lkey else None
    krng = np.random.default_rng(c.d_seed)
    plan, used = [], 0
    while used < total:
        p = used / total
        warm = c.d_seqf > 0 and p < c.d_seqf
        Ls = c.d_seq0 + (L - c.d_seq0) * (p / c.d_seqf) ** c.d_seqpow if warm else L
        if warm and c.d_lcyc and len(plan) % 2 == 1:   # length cycling: every other warmup step is full length
            Ls = L
        if c.d_lendf > 0 and p >= 1 - c.d_lendf:        # short-length end phase, sequences per step fixed
            Ls = c.d_lend
        Ls = int(min(L, max(min(8, Ls), 8 * round(Ls / 8))))
        b = min(max(1, int(round(B * bs_mult(c, p)))), (total - used) // Ls)   # optional batch knobs (d_bs*)
        if b == 0:
            Ls, b = total - used, 1
        if keys is not None and (Ls < L or p < c.d_lkf):  # windows chosen by key from d_lpool x candidates
            cand = krng.integers(0, len(train) - L - 1, b * int(c.d_lpool))   # (short ones, or all before d_lkf)
            sc = z(keys(c.d_lkey, cand)) * c.d_ldir
            if c.d_lkey2:                                 # optional second key, signed weight d_lw2
                sc = sc + c.d_lw2 * z(keys(c.d_lkey2, cand))
            st = cand[np.argsort(-sc, kind="stable")[:b]]
        elif c.d_ldisj:                                   # stratified starts: one window per corpus stratum
            N = len(train) - L - 1
            st = ((np.arange(b) + rng.random(b)) * (N / b)).astype(np.int64)
            st = st[rng.permutation(b)]
        else:
            st = rng.integers(0, len(train) - Ls - 1, b)
        if c.d_lalign > 0 and Ls < L:                     # snap a share of short windows to their document start
            e = keys.docs() if keys is not None else np.flatnonzero(np.asarray(train) == 0)
            j = np.searchsorted(e, st, side="right") - 1
            snap = np.where(j >= 0, e[np.maximum(j, 0)], st)
            st = np.where(krng.random(len(st)) < c.d_lalign, snap, st)
        if c.d_lrep > 1 and Ls < L and plan and plan[-1][1] == Ls and len(plan[-1][0]) == b and len(plan) % int(c.d_lrep):
            st = plan[-1][0]                              # reuse the previous step's short windows (d_lrep passes)
        plan.append((st, Ls))
        used += b * Ls
    return plan


def extra_tokens(cfg, train, datasets, data):
    """idea path (d_xsrc): token source read after the corpus. "other" = the other dataset's train.bin;
    copy / rev / sort / dyck / rand / mix = procedural sequences over the corpus's 1024 most frequent ids, blocks split by 0"""
    import os
    if cfg.d_xsrc == "other":
        o = "pystd" if data.startswith("fineweb") else "fineweb"
        return np.memmap(os.path.join(datasets[o], "train.bin"), dtype=np.uint16, mode="r")
    r = np.random.default_rng(cfg.d_seed + 101)
    src = cfg.d_xsrc
    if src.rstrip("0123456789") in ("shuf", "splice", "noise", "realcopy", "revself"):
        kind = src.rstrip("0123456789")
        k = int(src[len(kind):] or 0)
        n = 400_000
        o = int(r.integers(0, len(train) - n - 1))
        seg = np.asarray(train[o:o + n]).astype(np.int64)
        if kind == "revself":                        # reversed text
            return seg[::-1].astype(np.uint16).copy()
        if kind == "shuf":                           # token order destroyed inside blocks of k tokens
            blk = seg[: n // k * k].reshape(-1, k)
            return r.permuted(blk, axis=1).ravel().astype(np.uint16)
        if kind == "splice":                         # chunks of k tokens from random corpus places, butted together
            st = r.integers(0, len(train) - k - 1, n // k)
            return np.asarray(train[(st[:, None] + np.arange(k))]).ravel().astype(np.uint16)
        if kind == "noise":                          # k percent of tokens replaced by tokens drawn from the corpus unigram
            m = r.random(n) < k / 100.0
            seg[m] = seg[r.integers(0, n, int(m.sum()))]
            return seg.astype(np.uint16)
        out, t = [], 0                               # realcopy: a real-text chunk of 8-24 tokens, then the same chunk again
        while t < n:
            m = int(r.integers(8, 25)); a = int(r.integers(0, len(train) - m - 1))
            q = np.asarray(train[a:a + m]); out += [q, q, [0]]; t += 2 * m + 1
        return np.concatenate(out).astype(np.uint16)
    smp = np.asarray(train[:2_000_000])
    pool = np.argsort(-np.bincount(smp, minlength=cfg.vocab))[:1024]
    pool = pool[pool != 0]
    out, n = [], 0
    kinds = ["copy", "rev", "sort", "dyck"]
    while n < 400_000:
        kind = cfg.d_xsrc if cfg.d_xsrc != "mix" else kinds[r.integers(4)]
        m = int(r.integers(8, 25))
        q = pool[r.integers(0, len(pool), m)]
        if kind == "copy":
            blk = np.concatenate([q, q])
        elif kind == "rev":
            blk = np.concatenate([q, q[::-1]])
        elif kind == "sort":
            blk = np.concatenate([q, np.sort(q)])
        elif kind == "dyck":                         # balanced brackets, 16 pair types, depth <= 8
            st, b = [], []
            for _ in range(2 * m):
                if st and (len(st) >= 8 or r.random() < 0.5):
                    b.append(pool[16 + st.pop()])
                else:
                    k = int(r.integers(16)); st.append(k); b.append(pool[k])
            b += [pool[16 + k] for k in reversed(st)]
            blk = np.array(b)
        elif kind == "ar":                           # associative recall: m key-value pairs, then queried pairs again
            kk = pool[r.integers(0, 512, m)]; vv = pool[512 + r.integers(0, len(pool) - 512, m)]
            qi = r.integers(0, m, max(4, m // 2))
            blk = np.concatenate([np.stack([kk, vv], 1).ravel(), np.stack([kk[qi], vv[qi]], 1).ravel()])
        elif kind == "cnt":                          # arithmetic progressions over frequency ranks
            a0, dd = int(r.integers(len(pool))), int(r.integers(1, 8))
            blk = pool[(a0 + dd * np.arange(2 * m)) % len(pool)]
        else:                                        # rand: control, no structure
            blk = pool[r.integers(0, len(pool), 2 * m)]
        out.append(blk); out.append([0]); n += len(blk) + 1
    return np.concatenate(out).astype(np.uint16)


def post_plan(plan, train, c):
    """idea-path passes over a finished plan; tokens per step unchanged. All off at defaults."""
    if c.d_lperm and plan:                           # reorder the short-length steps: -1 reversed, 1 shuffled
        idx = [i for i, (st, L) in enumerate(plan) if L < c.seq_len]
        sub = [plan[i] for i in idx]
        if c.d_lperm < 0:
            sub = sub[::-1]
        else:
            sub = [sub[j] for j in np.random.default_rng(c.d_seed + 11).permutation(len(sub))]
        plan = list(plan)
        for i, e in zip(idx, sub):
            plan[i] = e
    if c.d_cont > 0 and plan:                        # streaming: a share of rows reads on from the previous step's row
        out = [plan[0]]
        for t in range(1, len(plan)):
            st, L = plan[t]
            pst, pL = out[-1]
            if c.d_contk and t % int(c.d_contk) == 0:
                out.append((st, L)); continue
            k = min(len(st), len(pst), int(round(c.d_cont * len(st))))
            st = np.array(st, copy=True)
            nxt = np.asarray(pst[:k]) + pL
            st[:k] = np.where(nxt < len(train) - L - 1, nxt, st[:k])
            out.append((st, L))
        plan = out
    if c.d_xsrc and c.d_xp > 0 and plan:             # extra source (synthetic / other domain) at head, tail or mixed in
        r = np.random.default_rng(c.d_seed + 23)
        tot = sum(len(st) * L for st, L in plan)
        cum = np.cumsum([len(st) * L for st, L in plan]) / tot
        out = []
        for t, (st, L) in enumerate(plan):
            p0 = cum[t] - len(st) * L / tot
            if c.d_xpos == "mix":
                k = int(r.binomial(len(st), c.d_xp))
            elif c.d_xpos == "tail":
                k = len(st) if p0 >= 1 - c.d_xp - 1e-12 else 0
            else:
                k = len(st) if p0 < c.d_xp - 1e-12 else 0
            if k:
                st = np.array(st, copy=True)
                st[:k] = c.d_xoff + r.integers(0, c.d_xlen - L - 1, k)
            out.append((st, L))
        plan = out
    if c.d_loc > 0 and plan:                         # in-step contiguity: rows of a step are consecutive chunks of d_locg regions
        r = np.random.default_rng(c.d_seed + 31)
        tot = sum(len(st) * L for st, L in plan)
        cum = np.cumsum([len(st) * L for st, L in plan]) / tot
        out = []
        for t, (st, L) in enumerate(plan):
            p0 = cum[t] - len(st) * L / tot
            on = (p0 >= 1 - c.d_loc - 1e-12) if c.d_locpos == "tail" else (p0 < c.d_loc - 1e-12)
            b, g = len(st), max(1, min(int(c.d_locg), len(st)))
            if on and b > 1:
                per = -(-b // g)
                s0 = r.integers(0, len(train) - per * L - L - 2, g)
                st = (s0[:, None] + L * np.arange(per)[None, :]).ravel()[:b]
                st = st[r.permutation(b)]
            out.append((st, L))
        plan = out
    if c.d_dsp > 0 and plan:                         # document revisits: a share of rows reads another part of a document seen d_dgap steps ago
        r = np.random.default_rng(c.d_seed + 41)
        e = np.flatnonzero(np.asarray(train) == 0)
        g = int(c.d_dgap)
        out = list(plan[:g])
        for t in range(g, len(plan)):
            st, L = plan[t]
            pst = out[t - g][0]
            k = min(len(st), len(pst), int(round(c.d_dsp * len(st))))
            st = np.array(st, copy=True)
            j = np.searchsorted(e, np.asarray(pst[:k]), side="right") - 1
            lo = np.where(j >= 0, e[np.maximum(j, 0)], 0)
            hi = np.where(j + 1 < len(e), e[np.minimum(j + 1, len(e) - 1)], len(train) - 1)
            room = hi - lo - L - 1
            new = lo + (r.random(k) * np.maximum(room, 1)).astype(np.int64)
            st[:k] = np.where(room > 0, new, st[:k])
            out.append((st, L))
        plan = out
    if c.d_cov > 0 and plan:                         # batch token coverage: greedy set cover over a d_covk x pool (dir -1: least cover)
        r = np.random.default_rng(c.d_seed + 51)
        on = phase_mask(plan, c.d_cov, c.d_covpos)
        out = []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_covk) - 1))])
                X = np.asarray(train[cand[:, None] + np.arange(L + 1)])
                sets = [set(x.tolist()) for x in X]
                first = int(r.integers(len(cand)))
                chosen, seen, left = [first], set(sets[first]), set(range(len(cand))) - {first}
                while len(chosen) < b:
                    lst = sorted(left)
                    gain = np.array([len(sets[i] - seen) for i in lst], dtype=np.float64) * c.d_covdir
                    j = lst[int(np.argmax(gain + 1e-6 * r.random(len(lst))))]
                    chosen.append(j); seen |= sets[j]; left.discard(j)
                st = cand[np.array(chosen)]
            out.append((st, L))
        plan = out
    if c.d_xs > 0 and plan:                          # step-to-step dissimilarity: rows share few tokens with the previous step (dir -1: many)
        r = np.random.default_rng(c.d_seed + 61)
        on = phase_mask(plan, c.d_xs, c.d_xspos)
        out = [plan[0]]
        for t in range(1, len(plan)):
            st, L = plan[t]
            if on[t]:
                pst, pL = out[-1]
                prev = np.unique(np.asarray(train[np.asarray(pst)[:, None] + np.arange(pL + 1)]))
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_xsk) - 1))])
                X = np.asarray(train[cand[:, None] + np.arange(L + 1)])
                ov = np.isin(X, prev).mean(1) * c.d_xsdir
                st = cand[np.argsort(ov + 1e-9 * r.random(len(ov)), kind="stable")[:b]]
            out.append((st, L))
        plan = out
    if c.d_ovl > 0 and plan:                         # overlapping windows: odd rows start d_ovs x L after the even row before them
        on = phase_mask(plan, c.d_ovl, c.d_ovlpos)
        out = []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                st = np.array(st, copy=True)
                sh = max(1, int(round(c.d_ovs * L)))
                ev = st[0:len(st) - 1:2]
                ok = ev + sh < len(train) - L - 1
                st[1::2][:len(ev)] = np.where(ok, ev + sh, st[1::2][:len(ev)])
            out.append((st, L))
        plan = out
    if c.d_qr > 0 and plan:                          # quasi-random starts: golden-ratio sequence or one row per corpus stratum
        r = np.random.default_rng(c.d_seed + 81)
        on = phase_mask(plan, c.d_qr, c.d_qrpos)
        u, phi, out = float(r.random()), 0.6180339887498949, []
        for t, (st, L) in enumerate(plan):
            if on[t]:
                b, hi = len(st), len(train) - L - 1
                if c.d_qrmode == "strat":
                    st = ((np.arange(b) + r.random(b)) / b * hi).astype(np.int64)[r.permutation(b)]
                else:
                    uu = (u + phi * np.arange(1, b + 1)) % 1.0
                    u = float(uu[-1])
                    st = (uu * hi).astype(np.int64)
            out.append((st, L))
        plan = out
    if c.d_rcs > 0 and plan:                         # representative batch: greedy rows whose joint token histogram is closest (L1) to the corpus
        r = np.random.default_rng(c.d_seed + 83)
        on = phase_mask(plan, c.d_rcs, c.d_rcspos)
        smp = np.asarray(train[::16]).astype(np.int64)
        pc = np.bincount(smp, minlength=int(max(8192, smp.max() + 1))).astype(np.float64)
        pc /= pc.sum()
        out = []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_rcsk) - 1))])
                X = np.asarray(train[cand[:, None] + np.arange(L + 1)]).astype(np.int64)
                ids, inv = np.unique(X, return_inverse=True)
                inv = inv.reshape(X.shape)
                H = np.zeros((len(cand), len(ids)))
                np.add.at(H, (np.repeat(np.arange(len(cand)), X.shape[1]), inv.ravel()), 1.0)
                p, h, chosen, left = pc[ids], np.zeros(len(ids)), [], np.ones(len(cand), bool)
                for k in range(b):
                    q = (h[None, :] + H) / ((k + 1) * X.shape[1])
                    d = np.abs(q - p[None, :]).sum(1) * c.d_rcsdir + 1e-9 * r.random(len(cand))
                    d[~left] = np.inf
                    j = int(np.argmin(d)); chosen.append(j); left[j] = False; h += H[j]
                st = cand[np.array(chosen)]
            out.append((st, L))
        plan = out
    if c.d_pool > 0 and plan:                        # small fixed pool: pool steps draw rows from d_poolp windows sampled once
        r = np.random.default_rng(c.d_seed + 91)
        if c.d_poolpos == "mid":                     # pool phase starts at token share d_poolm
            tot_ = sum(len(st) * L for st, L in plan)
            p0_ = np.cumsum([0] + [len(st) * L for st, L in plan[:-1]]) / tot_
            on = (p0_ >= c.d_poolm - 1e-12) & (p0_ < c.d_poolm + c.d_pool - 1e-12)
        else:
            on = phase_mask(plan, c.d_pool, c.d_poolpos)
        if c.d_poolret > 0:
            on = on | phase_mask(plan, c.d_poolret, "tail")
        idx = np.nonzero(on)[0]; npos = {int(t): i for i, t in enumerate(idx)}
        Lmax = max(L for _, L in plan)
        P = int(c.d_poolp)
        nrow = {}
        for t in idx:
            nrow[plan[t][1]] = nrow.get(plan[t][1], 0) + len(plan[t][0])
        uni = None
        if c.d_poolkey == "uni":
            cnt = np.bincount(np.asarray(train[::16]), minlength=65536).astype(np.float64) + 1.0
            uni = -np.log(cnt / cnt.sum())
        def pick(L, n):
            nc = n * int(c.d_poolc) if c.d_poolkey else n
            cand = r.integers(0, len(train) - Lmax - 1, nc) if c.d_poolsh else r.integers(0, len(train) - L - 1, nc)
            if not c.d_poolkey:
                return cand
            W = np.stack([np.asarray(train[a:a + L]) for a in cand])
            if c.d_poolkey == "uni":
                sc = uni[W].mean(1) * c.d_pooldir           # dir 1: lowest surprisal (easy) windows
                return cand[np.argsort(sc)[:n]]
            seen, ch, left = set(), [], np.ones(nc, bool)  # div: greedy max new distinct tokens
            sets = [set(w.tolist()) for w in W]
            for _ in range(n):
                g = np.array([len(sets[j] - seen) if left[j] else -1 for j in range(nc)]) * c.d_pooldir
                g = np.where(left, g, -np.inf); j = int(np.argmax(g)); ch.append(j); left[j] = False; seen |= sets[j]
            return cand[np.array(ch)]
        pools, shared, out = {}, None, []
        for t, (st, L) in enumerate(plan):
            if on[t]:
                frac = npos[t] / max(1, len(idx) - 1)
                if c.d_poolsh:
                    if shared is None:
                        shared = pick(Lmax, P)
                    base = shared
                else:
                    if L not in pools:                  # d_poolk: pool size per length = rows at that length / k (exact repeats)
                        pools[L] = pick(L, max(1, int(np.ceil(nrow[L] / c.d_poolk))) if c.d_poolk > 0 else P)
                    base = pools[L]
                if c.d_poolgrow > 0:                    # pool grows geometrically from P to P*grow over the pool span
                    n = int(round(P * c.d_poolgrow ** frac))
                    key = ("g", L if not c.d_poolsh else 0)
                    if key not in pools:
                        pools[key] = base
                    while len(pools[key]) < n:
                        pools[key] = np.concatenate([pools[key], pick(L if not c.d_poolsh else Lmax, min(P, n - len(pools[key])))])
                    base = pools[key][:max(n, 1)]
                b = len(st)
                if c.d_poolmode == "epoch":             # visit pool windows in permuted passes
                    ek = ("e", L, len(base))
                    q = pools.get(ek, np.array([], dtype=np.int64))
                    while len(q) < b:
                        q = np.concatenate([q, r.permutation(len(base))])
                    rows, pools[ek] = base[q[:b]], q[b:]
                else:
                    rows = base[r.integers(0, len(base), b)]
                if c.d_pooljit > 0:
                    rows = rows + r.integers(0, max(1, int(c.d_pooljit * L)) + 1, b)
                    rows = np.minimum(rows, len(train) - L - 1)
                m = c.d_poolmix * ((1 - frac) if c.d_pooldec else 1.0)
                k = int(round(m * b))
                if k >= b:
                    st = rows
                elif k > 0:
                    st = st.copy(); sel = r.permutation(b)[:k]; st[sel] = rows[sel]
            out.append((st, L))
        plan = out
    if c.d_reg > 0 and plan:                         # growing corpus region: head rows come from a region that widens from d_reg0 to all
        r = np.random.default_rng(c.d_seed + 93)
        tot = sum(len(st) * L for st, L in plan)
        cum = np.cumsum([len(st) * L for st, L in plan]) / tot
        o, out = int(r.integers(0, len(train))), []
        for t, (st, L) in enumerate(plan):
            p0 = cum[t] - len(st) * L / tot
            if p0 < c.d_reg - 1e-12:
                hi = len(train) - L - 1
                w = max(L + 1, int(hi * (c.d_reg0 + (1 - c.d_reg0) * p0 / c.d_reg)))
                st = (o + r.integers(0, w, len(st))) % hi
            out.append((st, L))
        plan = out
    if c.d_echo > 0 and plan:                        # shifted echo: every second step re-reads the step before, shifted d_echos x L
        on = phase_mask(plan, c.d_echo, c.d_echopos)
        out, par = [], 0
        for t, (st, L) in enumerate(plan):
            if on[t]:
                if par and out and out[-1][1] == L:
                    pst = np.asarray(out[-1][0]); sh = int(round(c.d_echos * L))
                    st = np.where(pst + sh < len(train) - L - 1, pst + sh, pst)
                par ^= 1
            out.append((st, L))
        plan = out
    if c.d_gz > 0 and plan:                          # compressibility: pick rows by zlib ratio from a d_gzk x pool (dir +1: least compressible)
        import zlib
        r = np.random.default_rng(c.d_seed + 95)
        on, out = phase_mask(plan, c.d_gz, c.d_gzpos), []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_gzk) - 1))])
                sc = np.array([len(zlib.compress(np.asarray(train[s0:s0 + L + 1]).astype(np.uint16).tobytes(), 6)) for s0 in cand], float)
                sc = sc * c.d_gzdir + 1e-9 * r.random(len(cand))
                st = cand[np.argsort(-sc, kind="stable")[:b]]
            out.append((st, L))
        plan = out
    if c.d_bg > 0 and plan:                          # hashed-bigram surprise from a corpus sample: pick rows from a d_bgk x pool (dir +1: most surprising)
        r = np.random.default_rng(c.d_seed + 97)
        H = 1 << 22
        cb, cu = np.zeros(H), np.zeros(65536)
        for o in r.integers(0, len(train) - 65537, 64):
            x = np.asarray(train[o:o + 65537]).astype(np.int64)
            np.add.at(cb, (x[:-1] * 8209 + x[1:]) % H, 1.0); np.add.at(cu, x[:-1], 1.0)
        on, out = phase_mask(plan, c.d_bg, c.d_bgpos), []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_bgk) - 1))])
                X = np.stack([train[s0:s0 + L + 1] for s0 in cand]).astype(np.int64)
                pb = (cb[(X[:, :-1] * 8209 + X[:, 1:]) % H] + 0.1) / (cu[X[:, :-1]] + 0.1 * 8192)
                sc = (-np.log(pb)).mean(1) * c.d_bgdir + 1e-9 * r.random(len(cand))
                st = cand[np.argsort(-sc, kind="stable")[:b]]
            out.append((st, L))
        plan = out
    if c.d_spc > 0 and plan:                         # spaced repetition: a d_spcs share of rows re-read the rows of steps t-1, t-3, t-7
        on, out = phase_mask(plan, c.d_spc, c.d_spcpos), []
        for t, (st, L) in enumerate(plan):
            if on[t] and t > 0:
                st = np.array(st, copy=True); k = int(round(c.d_spcs * len(st)))
                for j in range(k):
                    iv = (1, 3, 7)[j % 3]
                    if t - iv >= 0:
                        ps = out[t - iv][0]; v = int(ps[j % len(ps)])
                        if v + L + 1 < len(train):
                            st[len(st) - 1 - j] = v
            out.append((st, L))
        plan = out
    if c.d_dup > 0 and plan:                         # in-step duplicates: the last d_dups share of rows repeat the step's first rows
        on, out = phase_mask(plan, c.d_dup, c.d_duppos), []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                st = np.array(st, copy=True); k = min(len(st) // 2, int(round(c.d_dups * len(st))))
                if k > 0:
                    st[len(st) - k:] = st[:k]
            out.append((st, L))
        plan = out
    if c.d_clu > 0 and plan:                         # topic clusters: k-means on hashed token bags; pick rows by cluster from a d_cluk x pool
        r = np.random.default_rng(c.d_seed + 99)
        HB, C = 256, int(c.d_cluC)

        def feat(starts, L):
            X = np.stack([train[s0:s0 + L + 1] for s0 in starts]).astype(np.int64)
            F = np.zeros((len(X), HB))
            np.add.at(F, (np.repeat(np.arange(len(X)), X.shape[1]), ((X * 2654435761) % HB).ravel()), 1.0)
            return F / np.linalg.norm(F, axis=1, keepdims=True)
        F0 = feat(r.integers(0, len(train) - 65, 4096), 64)
        cen = F0[r.choice(len(F0), C, replace=False)]
        for _ in range(10):
            a_ = np.argmax(F0 @ cen.T, 1)
            for j in range(C):
                if (a_ == j).any():
                    v = F0[a_ == j].mean(0); cen[j] = v / np.linalg.norm(v)
        on, out = phase_mask(plan, c.d_clu, c.d_clupos), []
        for t, (st, L) in enumerate(plan):
            if on[t] and len(st) > 1:
                b = len(st)
                cand = np.concatenate([np.asarray(st), r.integers(0, len(train) - L - 1, b * (int(c.d_cluk) - 1))])
                sim = feat(cand, L) @ cen.T
                a_ = np.argmax(sim, 1)
                if c.d_clumode == "mix":             # one row per cluster in turn: the batch spans every cluster
                    ch, used = [], np.zeros(len(cand), bool)
                    for i in range(b):
                        j = i % C
                        idx = np.flatnonzero((a_ == j) & ~used)
                        if len(idx) == 0:
                            idx = np.flatnonzero(~used)
                        ch.append(idx[0]); used[idx[0]] = True
                    st = cand[np.array(ch)]
                else:                                # pure: every row from one target cluster (blk: target held for d_clublk steps)
                    tg = int(r.integers(C)) if c.d_clumode == "pure" else (t // int(c.d_clublk)) % C
                    st = cand[np.argsort(-sim[:, tg], kind="stable")[:b]]
            out.append((st, L))
        plan = out
    if c.d_dd > 0 and plan:                          # dedup: a row sharing more than d_ddt of its d_ddn-grams with earlier rows is redrawn
        r = np.random.default_rng(c.d_seed + 101)
        seen = set()

        def grams(s0, L):
            x = np.asarray(train[s0:s0 + L + 1]).astype(np.int64)
            n = min(int(c.d_ddn), len(x))
            h = np.zeros(len(x) - n + 1, np.int64)
            for j in range(n):
                h = h * 1000003 + x[j:len(x) - n + 1 + j]
            return set(h.tolist())
        on, out = phase_mask(plan, c.d_dd, c.d_ddpos), []
        for t, (st, L) in enumerate(plan):
            st = np.array(st, copy=True)
            for i in range(len(st)):
                g = grams(int(st[i]), L)
                if on[t]:
                    for _ in range(4):
                        if len(g & seen) <= c.d_ddt * len(g):
                            break
                        st[i] = int(r.integers(0, len(train) - L - 1)); g = grams(int(st[i]), L)
                seen |= g
            out.append((st, L))
        plan = out
    return plan


def phase_mask(plan, share, pos):
    """steps whose token-share start lies in the head / tail share, or every step for pos 'all'"""
    tot = sum(len(st) * L for st, L in plan)
    cum = np.cumsum([len(st) * L for st, L in plan]) / tot
    p0 = cum - np.array([len(st) * L for st, L in plan]) / tot
    if pos == "all":
        return np.ones(len(plan), dtype=bool)
    if pos == "tail":
        return p0 >= 1 - share - 1e-12
    return p0 < share - 1e-12


def build_plan(train, cfg, steps):
    for kk, v in DEFAULTS.items():
        if not hasattr(cfg, kk):
            setattr(cfg, kk, v)
    plan = _build_plan(train, cfg, steps)
    if cfg.d_lperm or cfg.d_cont > 0 or (cfg.d_xsrc and cfg.d_xp > 0) or cfg.d_loc > 0 or cfg.d_dsp > 0 \
            or cfg.d_cov > 0 or cfg.d_xs > 0 or cfg.d_ovl > 0 or cfg.d_qr > 0 or cfg.d_rcs > 0 \
            or cfg.d_pool > 0 or cfg.d_reg > 0 or cfg.d_echo > 0 or cfg.d_gz > 0 or cfg.d_bg > 0 or cfg.d_spc > 0 or cfg.d_dup > 0 or cfg.d_clu > 0 or cfg.d_dd > 0:
        plan = post_plan(plan, train, cfg)
    return plan


def _build_plan(train, cfg, steps):
    for kk, v in DEFAULTS.items():
        if not hasattr(cfg, kk):
            setattr(cfg, kk, v)
    c = cfg
    if c.d_seqmode == "li":
        return li_plan(train, c, steps)
    W = base_starts(train, cfg, steps)
    if (c.d_order == "shuffle" and c.d_anneal == 0 and c.d_bsf == 0 and c.d_bsef == 0 and c.d_bs0 == 1
            and not c.d_seq0 and not c.d_seqef and c.d_repf == 0 and c.d_align == 0):
        return [(W[i * c.batch:(i + 1) * c.batch], c.seq_len) for i in range(steps)]
    rng = np.random.default_rng(c.d_seed)
    keys = Keys(train, cfg, W)
    W = align(W, keys, c, rng)
    W = anneal(W, keys, c, train, rng)
    if c.d_anneal > 0 and c.d_order != "shuffle":   # order only the pre-anneal part
        m = int(round(len(W) * c.d_anneal))
        W = np.concatenate([order_windows(W[:len(W) - m], keys, c, rng), W[len(W) - m:]])
    elif c.d_lfrac > 0:                              # order only the last d_lfrac of the run
        m = int(round(len(W) * c.d_lfrac))
        W = np.concatenate([W[:len(W) - m], order_windows(W[len(W) - m:], keys, c, rng)])
    else:
        W = order_windows(W, keys, c, rng)
    n, B, L = len(W), c.batch, c.seq_len
    # batch-size schedule over the token share consumed
    sizes, used = [], 0
    while used < n:
        p = used / n
        b = max(1, int(round(B * bs_mult(c, p))))
        b = min(b, n - used)
        sizes.append(b)
        used += b
    plan, pos = [], 0
    T = len(sizes)
    for s, b in enumerate(sizes):
        st = W[pos:pos + b]
        pos += b
        Ls = L
        if c.d_seqef > 0 and s >= (1 - c.d_seqef) * T:
            Ls = c.d_seqe
            while L % Ls:
                Ls //= 2
        elif c.d_seq0:
            e = c.d_seq1 or L
            p = min(1.0, s / max(1, c.d_seqf * T)) if c.d_seqf > 0 else 1.0
            lg = np.log2(c.d_seq0) + (np.log2(e) - np.log2(c.d_seq0)) * p
            Ls = int(2 ** int(np.floor(lg + 1e-9)))
            Ls = max(1, min(L, Ls))
            while L % Ls:
                Ls //= 2
        if Ls < L:
            st = (st[:, None] + np.arange(0, L, Ls)[None, :]).ravel()
        plan.append((st, Ls))
    if c.d_repf > 0 and c.d_rep > 1:  # re-show: inside the region every batch is shown d_rep times, copies d_repg steps apart
        T = len(plan)
        r0, r1 = (int((1 - c.d_repf) * T), T) if c.d_rep_end else (0, int(c.d_repf * T))
        slots = [None] * (r1 - r0)
        k = 0
        for p in range(len(slots)):
            if slots[p] is not None:
                continue
            q = p
            for _ in range(int(c.d_rep)):
                while q < len(slots) and slots[q] is not None:
                    q += 1
                if q >= len(slots):
                    break
                slots[q] = r0 + k
                q += int(c.d_repg)
            k += 1
        orig = list(plan)
        for i, src in enumerate(slots):
            plan[r0 + i] = orig[src]
    return plan
