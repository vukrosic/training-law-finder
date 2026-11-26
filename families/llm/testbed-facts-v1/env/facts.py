import json, os
"""Synthetic rare-fact world for the facts search (Vuk 2026-11-23: find a formula for how Muon learns rare facts).

N made-up people, each with 5 attributes (birth city, university, major, employer, birth year). Each person gets a
sighting count k from the bins (default 1, 2, 4, ..., 256), equal numbers of people per bin. A bio is the 5
sentences "<First Last> was born in <City>." ... in a random order; every sentence starts with the full name, so the
recall prompt "<First Last> was born in" appears verbatim in training. The k sightings of a person are spread at
uniformly random positions over training (seeded), and bios replace web-text rows in the batch (batch size stays fixed).

Recall test (end of training): for every person and attribute, teacher-forced prompt "<First Last> <relation>" and the
value tokens; a fact counts as recalled when every value token is the argmax. Also reports value loss in bits.
"""
import math
import numpy as np
import torch

SYL_A = ["ka", "vo", "mel", "dra", "tu", "sen", "ri", "lo", "bar", "ne", "fi", "go", "pra", "zu", "hal", "mi",
         "tor", "ve", "qua", "lin", "os", "ru", "den", "pa", "shi", "cor", "ul", "ber", "na", "tis"]
SYL_B = ["n", "r", "s", "k", "l", "th", "m", "x", "d", "v", "", "", ""]
REL = [("city", "was born in", "{v}"), ("univ", "studied at", "{v} University"), ("major", "majored in", "{v} studies"),
       ("company", "worked for", "{v} Corporation"), ("year", "has birth year", "{v}")]
POOL = {"city": 200, "univ": 300, "major": 100, "company": 260, "year": 100}
# ffmt=1 filler: places, not people; other relations, other value words (Vuk 2026-11-24: does filler help across formats)
REL_F = [("country", "is a town in", "{v} Republic"), ("river", "lies on the", "{v} River"), ("mount", "sits below", "Mount {v}"),
         ("dish", "is known for", "{v} soup"), ("year", "was founded in", "{v}")]
POOL_F = {"country": 200, "river": 300, "mount": 100, "dish": 260, "year": 100}
# format ladder (Vuk 2026-11-25: what is a format). ffmt=2: same people style and same value sets, relations reworded;
# ffmt=3: same relations reversed, value first then name (" <City> is the birthplace of <First Last>.")
REL_R = [("city", "is a native of", "{v}"), ("univ", "graduated from", "{v} University"), ("major", "holds a degree in", "{v} studies"),
         ("company", "was employed by", "{v} Corporation"), ("year", "came into the world in", "{v}")]
REL_V = [("city", "{v}", "is the birthplace of"), ("univ", "{v} University", "trained"), ("major", "{v} studies", "were chosen by"),
         ("company", "{v} Corporation", "employed"), ("year", "{v}", "is the birth year of")]


def _words(rng, n, used):
    out = []
    while len(out) < n:
        w = "".join(rng.choice(SYL_A, size=rng.integers(2, 4))) + rng.choice(SYL_B)
        w = w.capitalize()
        if w not in used:
            used.add(w)
            out.append(w)
    return out


def _common(tok):
    """single vocab tokens that are whole lowercase English words with a leading space (' the', ' from', ...)"""
    import re
    return [i for i in range(tok.get_vocab_size()) if re.fullmatch(r" [a-z]{3,}", tok.decode([i]))]


class FactWorld:
    def __init__(self, tok, n_people, bins, steps, seed=1, win=(0.0, 1.0), n_filler=0, filler_k=256, fwin=None, frand=False, frandp=1.0, ffmt=0, ffkind=0, web=None, segs=None, pre=None, fpre=None, resight=None, replay=None, n_ref=0):
        self.V = tok.get_vocab_size()   # 8192 by default; small-vocab data (fast/prep_vocab.py) changes it
        # frand: filler bios get fresh random values at every sighting (same format, nothing learnable)
        # fwin: own window for the filler bios (None = same window as the tested people, the original behaviour)
        # win: bios only appear in steps [win0*T, win1*T); n_filler: extra people seen filler_k times, never tested
        rng = np.random.default_rng(seed)
        used = set()
        firsts, lasts = _words(rng, 700, used), _words(rng, 700, used)
        pools = {a: (list(range(1900, 2000)) if a == "year" else _words(rng, POOL[a], used)) for a in POOL}
        self.bins = bins
        nb = len(bins)
        n_people = (n_people // nb) * nb
        self.n_eval = n_people
        n_total = n_people + n_filler
        names = set()
        while len(names) < n_total:
            names.add(f"{rng.choice(firsts)} {rng.choice(lasts)}")
        names = sorted(names)
        rng.shuffle(names)
        self.names = names
        self.k = np.array([bins[i % nb] for i in range(n_people)] + [filler_k] * n_filler)
        self.vals = {a: rng.integers(0, len(pools[a]), n_total) for a in POOL}
        if ffmt in (5, 6, 7) and n_filler:   # 7: contradict only city and year, the other 3 relations rehearsed   # 2026-11-25 same-cue test: the filler slots are the most-learned tested people
            # again, with their own values (5, rehearse) or other values from the same pools (6, contradict)
            tg = [i for i in range(n_people) if self.k[i] == max(bins)][:n_filler]
            rc = np.random.default_rng(seed + 51)
            for j, t in enumerate(tg):
                names[n_people + j] = names[t]
                for a in POOL:
                    v, n = self.vals[a][t], len(pools[a])
                    keep = ffmt == 5 or (ffmt == 7 and a not in ("city", "year"))
                    self.vals[a][n_people + j] = v if keep else (v + 1 + rc.integers(n - 1)) % n
            self.targets = tg
        self.pools = pools
        self.tok = tok
        self.sep = tok.encode("\n").ids
        # per person: token ids of each sentence (with leading space except at row start, handled by always using " ")
        self.sent = []          # [person][attr] -> ids of " <name> <rel> <value>."
        self.prompt = []        # [person][attr] -> (prompt ids, value ids)
        if ffmt == 1 and n_filler:     # filler places: own single-word names and own value words, disjoint from the people's
            rf = np.random.default_rng(seed + 11)
            pools_f = {a: (list(range(1500, 1600)) if a == "year" else _words(rf, POOL_F[a], used)) for a in POOL_F}
            pnames = _words(rf, n_filler, used)
            vals_f = {a: rf.integers(0, len(pools_f[a]), n_filler) for a in POOL_F}
        if ffmt == 4 and n_filler:   # 2026-11-25 overlap test: same template, names and values made of common web words
            rc = np.random.default_rng(seed + 31)
            cw = _common(tok)
            cname = lambda: " ".join(tok.decode([int(i)]).strip() for i in rc.choice(cw, 2, replace=False))
            cval = lambda: tok.decode([int(rc.choice(cw))]).strip()
            cvals = {p: [cval() for _ in REL] for p in range(n_people, n_total)}
            for p in range(n_people, n_total):
                names[p] = cname()
        for p in range(n_total):
            ss, pr = [], []
            if ffmt == 4 and p >= n_people:
                for (a, rel, fmt), v in zip(REL, cvals[p]):
                    pid, vid = tok.encode(f" {names[p]} {rel}").ids, tok.encode(" " + fmt.format(v=v)).ids
                    ss.append(pid + vid + tok.encode(".").ids)
                    pr.append((pid, vid))
                self.sent.append(ss)
                self.prompt.append(pr)
                continue
            if ffmt == 1 and p >= n_people:
                q = p - n_people
                names[p] = pnames[q]
                for a, rel, fmt in REL_F:
                    pid = tok.encode(f" {pnames[q]} {rel}").ids
                    vid = tok.encode(" " + fmt.format(v=pools_f[a][vals_f[a][q]])).ids
                    ss.append(pid + vid + tok.encode(".").ids)
                    pr.append((pid, vid))
                self.sent.append(ss)
                self.prompt.append(pr)
                continue
            if ffmt in (2, 3) and p >= n_people:
                for (a, rel, fmt), (_, vfmt, vrel) in zip(REL_R, REL_V):
                    v = pools[a][self.vals[a][p]]
                    if ffmt == 2:
                        pid, vid = tok.encode(f" {names[p]} {rel}").ids, tok.encode(" " + fmt.format(v=v)).ids
                    else:
                        pid, vid = tok.encode(" " + vfmt.format(v=v) + f" {vrel}").ids, tok.encode(f" {names[p]}").ids
                    ss.append(pid + vid + tok.encode(".").ids)
                    pr.append((pid, vid))
                self.sent.append(ss)
                self.prompt.append(pr)
                continue
            for a, rel, fmt in REL:
                v = fmt.format(v=pools[a][self.vals[a][p]])
                pid = tok.encode(f" {names[p]} {rel}").ids
                vid = tok.encode(f" {v}").ids
                dot = tok.encode(".").ids
                ss.append(pid + vid + dot)
                pr.append((pid, vid))
            self.sent.append(ss)
            self.prompt.append(pr)
        # occurrence stream: person ids, k copies each, shuffled; spread evenly over training steps
        occ = np.repeat(np.arange(n_total), self.k)
        rng.shuffle(occ)
        self.occ = occ
        self.steps = steps
        self.w0, self.w1 = int(win[0] * steps), max(int(win[0] * steps) + 1, int(win[1] * steps))
        self.streams = [(occ, self.w0, self.w1)]
        self.segs = list(segs) if segs else None
        if segs:   # branch runs (sched wsd + branches): person p gets exactly k*len/T sightings inside each segment,
            # so at a branch end b every person has seen exactly k*b/T bios, as in a standalone run of b steps
            assert win == (0.0, 1.0) and fwin is None, "segs needs the default windows"
            bounds = [0] + list(segs)
            self.streams = []
            for s0, s1 in zip(bounds[:-1], bounds[1:]):
                c = self.k * (s1 - s0)
                assert (c % steps == 0).all(), f"k*{s1 - s0} not divisible by {steps}: scale fkmax/bins"
                o = np.repeat(np.arange(n_total), c // steps)
                rng.shuffle(o)
                self.streams.append((o, s0, s1))
        if fwin is not None and n_filler:
            f0 = int(fwin[0] * steps)
            self.streams = [(occ[occ < n_people], self.w0, self.w1), (occ[occ >= n_people], f0, max(f0 + 1, int(fwin[1] * steps)))]
        if pre is not None:   # savings test (2026-11-25): an extra early dose, each person k more sightings inside `pre`
            po = np.repeat(np.arange(n_people), self.k[:n_people])
            np.random.default_rng(seed + 31).shuffle(po)
            p0 = int(pre[0] * steps)
            self.streams.append((po, p0, max(p0 + 1, int(pre[1] * steps))))
        if fpre is not None and n_filler:   # known-vs-new test (2026-11-25): the filler items also get fk sightings each
            # inside an early window, so by the gap the model already knows them
            fo = np.repeat(np.arange(n_people, n_total), int(fpre[2]))
            np.random.default_rng(seed + 41).shuffle(fo)
            q0 = int(fpre[0] * steps)
            self.streams.append((fo, q0, max(q0 + 1, int(fpre[1] * steps))))
        if resight is not None:   # R7 test (2026-11-25): learned people (k >= max bin / 2) once per step for m steps at each start
            lp_ = np.array([q for q in range(n_people) if self.k[q] >= max(bins) // 2])
            r7 = np.random.default_rng(seed + 51)
            for s0_ in resight[0]:
                o_ = np.concatenate([r7.permutation(lp_) for _ in range(resight[1])])
                self.streams.append((o_, int(s0_), int(s0_) + resight[1]))
            self.resight = [int(x) for x in resight[0]]
        # practical-win test (2026-11-25): replay single fact sentences at given steps; {step: [(person, attr), ...]};
        # emitted after the scheduled bios, consume no rng, so the run is identical to the no-replay run before the first replay
        self.replay = {int(k_): [(int(q_), int(a_)) for q_, a_ in v_] for k_, v_ in replay.items()} if replay else {}
        self.rng = np.random.default_rng(seed + 1)
        self.frand = int(frand)
        self.frandp = float(frandp)   # share of filler sentences that get a random answer (mode 1 and 3)
        # frand 1: random values from the same pools; 2: also a fresh random name each sighting (pure template);
        # 3: random values from a disjoint pool of new words (template + names, but not the answer sets)
        self.firsts, self.lasts = firsts, lasts
        self.name_set = set(names)
        self.rpools = pools
        if self.frand == 3:
            r2 = np.random.default_rng(seed + 7)
            used2 = set(used)
            self.rpools = {a: (list(range(1800, 1900)) if a == "year" else _words(r2, POOL[a], used2)) for a in POOL}
        self.last_step = np.full(n_total, -1)
        self.seen = [[] for _ in range(n_total)]     # steps at which each person's bio was trained on (ftrace)
        self.bio_len = float(np.mean([sum(len(s) for s in ss) for ss in self.sent])) + len(self.sep)
        # ffkind (Vuk 2026-11-24: which kinds of mixed-in text help rare facts): filler slots carry other text, same
        # token count per sighting as that filler bio. 0 facts (default) | 1 fresh random made-up words | 2 fresh random
        # tokens | 3 fixed web snippet per filler item (repeated text) | 4 filler bio tokens shuffled each sighting |
        # 5 fresh random words then the same words again (in-context copy) | 6 fixed random made-up word sequence per item
        self.ffkind = int(ffkind)
        if self.ffkind and n_filler:
            rk = np.random.default_rng(seed + 21)
            self.flen = {p: sum(len(x) for x in self.sent[p]) for p in range(n_people, n_total)}
            self.fwords = [tok.encode(" " + w).ids for w in _words(rk, 2000, used)]
            if self.ffkind == 8:   # 2026-11-25 overlap test: fresh random common English words instead of made-up words
                self.fwords = [[i] for i in _common(tok)]
            if self.ffkind == 9:   # 2026-11-25 conflict test: fresh random sequence of the facts' OWN name and value words
                lp = [q for q in range(n_people) if self.k[q] >= max(bins) // 2]   # the people whose facts are learned
                ws = {w for q in lp for w in self.names[q].split()} | {pools[a][self.vals[a][q]] for q in lp for a in POOL if a != "year"}
                self.fwords = [tok.encode(" " + w).ids for w in sorted(ws)]
            if self.ffkind in (10, 11, 12, 13, 14, 15, 16, 17) or self.ffkind >= 1000:   # 12/13/14: dose test, 25/50/75% of pieces drawn from the fact pieces, rest from the matched unused pieces
                # 2026-11-25 shared-piece test: fresh random single-token strings made of
                # the word pieces in the learned facts' names and values (10), or of id-matched pieces no tested fact uses (11)
                lp = [q for q in range(n_people) if self.k[q] >= max(bins) // 2]
                fp = sorted(t for t in {t for q in lp for (pid, vid) in self.prompt[q] for t in vid} | {t for q in lp for t in tok.encode(" " + self.names[q]).ids} if t >= 300)   # ids < 300: digits, bytes
                self.fp_all = list(fp)   # 2026-11-25 row-freeze test: all learned-fact pieces
                allf = {t for q in range(n_people) for ss in self.sent[q] for t in ss} | {t for q in range(n_people) for t in tok.encode(" " + self.names[q]).ids}
                free, pick = sorted(set(range(300, self.V)) - allf), []
                for t in fp:
                    c = min((x for x in free if x not in pick), key=lambda x: abs(x - t))
                    pick.append(c)
                if self.ffkind in (15, 16, 17):   # 2026-11-25 which-piece test: 25% of pieces from one subset, rest unused pieces
                    # 15: answer (value) pieces of learned people, 16: their name pieces, 17: answer pieces of half the learned people (set A)
                    rare = lambda t: t >= 300
                    vp = lambda qs: {t for q in qs for (pid, vid) in self.prompt[q] for t in vid if rare(t)}
                    npc = {t for q in lp for t in tok.encode(" " + self.names[q]).ids if rare(t)}
                    self.setA = [q for q in lp if (q // 8) % 2 == 0]   # half of each k group
                    S = {15: vp(lp) - npc, 16: npc - vp(lp), 17: vp(self.setA) - npc - vp([q for q in lp if q not in self.setA])}[self.ffkind]
                    self.fmix = (sorted(S), pick, 0.25)
                    self.fwords = [[t] for t in sorted(S)]
                    self.fpieces = sorted(S)
                if self.ffkind >= 1000:   # 2026-11-25 breadth test: ffkind = 1000*percent + n -> fraction percent/100 of gap pieces
                    # drawn from a random n-piece subset of the 82 fact pieces (nested: smaller n is a prefix of one shuffle), rest unused pieces
                    q, m = (self.ffkind // 1000) / 100, self.ffkind % 1000
                    mode, n = m // 100, m % 100   # 2026-11-25 exposure test: mode 1 = the n most-used pieces, mode 2 = pieces used by one fact only (n=0: all)
                    if mode == 0:
                        S = [int(t) for t in np.random.default_rng(seed + 71).permutation(fp)[:n]]
                    else:
                        use = {t: sum(t in (set(vid) | set(tok.encode(" " + self.names[q2]).ids)) for q2 in lp for (pid, vid) in self.prompt[q2]) for t in fp}
                        if mode in (3, 4):   # 2026-11-25 novelty test: 3 = n fact pieces rarest in web text, 4 = n most common
                            ws = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "webshare_v1024.json")))
                            S = sorted(fp, key=lambda t: (ws[t] if mode == 3 else -ws[t], t))
                        else:
                            S = sorted(fp, key=lambda t: (-use[t], t))[:n] if mode == 1 else [t for t in fp if use[t] == 1]
                        S = S[:n] if n else S
                    self.fmix = (sorted(S), pick, q)
                    self.fwords = [[t] for t in sorted(S)]
                    self.fpieces = sorted(S)
                    self.ffkind_n = n
                nf, nu = {10: (1, 0), 11: (0, 1), 12: (1, 3), 13: (1, 1), 14: (3, 1)}.get(self.ffkind, (1, 0))
                if self.ffkind in (10, 11, 12, 13, 14):
                    self.fwords = [[t] for t in fp] * nf + [[t] for t in pick] * nu
                    self.fpieces = sorted({w[0] for w in self.fwords})
            self.fixed = {}
            for p in range(n_people, n_total):
                if self.ffkind == 6:
                    self.fixed[p] = self._randwords(rk, self.flen[p])
                elif self.ffkind == 3:
                    assert web is not None, "ffkind 3 needs web tokens"
                    o = int(rk.integers(0, len(web) - self.flen[p] - 1))
                    self.fixed[p] = [int(t) for t in web[o:o + self.flen[p]]]

        # O76 (2026-11-26, ops): n_ref never-seen reference people (k = 0), same name and value pools, 5 attributes each.
        # Own rng (seed + 104729), built after everything else, so the planted people, values and bio stream are unchanged.
        # Their value bits at a step = the chance level C at that step (works for single-bin runs, no end-of-run leak).
        self.ref_prompt = []
        if n_ref:
            rr = np.random.default_rng(seed + 104729)
            taken, rn = set(self.names), []
            while len(rn) < n_ref:
                nm_ = f"{rr.choice(firsts)} {rr.choice(lasts)}"
                if nm_ not in taken: taken.add(nm_); rn.append(nm_)
            for nm_ in rn:
                pr = []
                for a, rel, fmt in REL:
                    v = fmt.format(v=pools[a][int(rr.integers(0, len(pools[a])))])
                    pr.append((tok.encode(f" {nm_} {rel}").ids, tok.encode(f" {v}").ids))
                self.ref_prompt.append(pr)

    def _randwords(self, rng, L):
        out = []
        while len(out) < L:
            out += self.fwords[rng.integers(len(self.fwords))]
        return out[:L]

    def _alt(self, p):
        L, k, r = self.flen[p], self.ffkind, self.rng
        if k in (3, 6):
            return list(self.fixed[p])
        if k in (15, 16, 17) or k >= 1000:
            S, U, q = self.fmix
            return [int(S[r.integers(len(S))]) if r.random() < q else int(U[r.integers(len(U))]) for _ in range(L)]
        if k in (1, 8, 9, 10, 11, 12, 13, 14):
            return self._randwords(r, L)
        if k == 2:
            return [int(t) for t in r.integers(300, self.V, L)]
        if k == 4:
            t = [x for ss in self.sent[p] for x in ss]
            return [t[i] for i in r.permutation(len(t))]
        if k == 5:
            h = self._randwords(r, L // 2)
            return (h + h + h[:1])[:L]
        if k == 7:   # blank control: one constant token, so the slot only removes web text
            return [300] * L
        raise ValueError(k)

    def step_bios(self, step):
        """token stream of the bios scheduled at this step"""
        out = []
        self.fmask = []   # 1 for each token of a filler item (p >= n_eval), for the filler loss weight ffw (2026-11-24)
        for occ, w0, w1 in self.streams:
            if w0 <= step < w1:
                L, S, j = len(occ), w1 - w0, step - w0
                out += self._bios(occ[(L * j) // S:(L * (j + 1)) // S], step)
        for q_, a_ in self.replay.get(step, ()):
            n0 = len(out); out += self.sent[q_][a_] + self.sep; self.fmask += [0] * (len(out) - n0)
        return out

    def _bios(self, people, step):
        out = []
        for p in people:
            n0 = len(out)
            order = self.rng.permutation(len(REL))
            if self.ffkind and p >= self.n_eval:
                out += self._alt(p) + self.sep
                self.fmask += [1] * (len(out) - n0)
                self.last_step[p] = step
                self.seen[p].append(step)
                continue
            rand = self.frand and p >= self.n_eval
            nm = self.names[p]
            while rand and self.frand == 2 and nm in self.name_set:      # fresh name, never one of the world's people
                nm = f"{self.firsts[self.rng.integers(len(self.firsts))]} {self.lasts[self.rng.integers(len(self.lasts))]}"
            for i in order:
                if rand and (self.frand == 2 or self.frandp >= 1 or self.rng.random() < self.frandp):
                    a, rel, fmt = REL[i]
                    v = fmt.format(v=self.rpools[a][self.rng.integers(len(self.rpools[a]))])
                    out += self.tok.encode(f" {nm} {rel}").ids + self.tok.encode(f" {v}").ids + self.tok.encode(".").ids
                else:
                    out += self.sent[p][i]
            out += self.sep
            self.fmask += [int(p >= self.n_eval)] * (len(out) - n0)
            self.last_step[p] = step
            self.seen[p].append(step)
        return out

    def rows_needed(self, seq_len):
        cov = np.zeros(self.steps + 1)   # bios per step, summed only over streams that overlap (2026-11-25)
        for o, w0, w1 in self.streams:
            cov[w0:w1] += math.ceil(len(o) / (w1 - w0))
        per_step = cov.max() * self.bio_len
        rp = max([sum(len(self.sent[q_][a_]) + len(self.sep) for q_, a_ in v_) for v_ in self.replay.values()] + [0])
        return math.ceil(per_step / (seq_len + 1)) + 1 + math.ceil(rp / (seq_len + 1))

    @torch.no_grad()
    def _score(self, model, device, items, B=128):
        """items: (ids, n_prompt) -> (sum log2-prob of the value tokens, all value tokens argmax)"""
        lpv, hits = np.zeros(len(items)), np.zeros(len(items), dtype=bool)
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            T = max(len(c[0]) for c in chunk)
            x = torch.zeros(len(chunk), T - 1, dtype=torch.long)
            y = torch.full((len(chunk), T - 1), -1, dtype=torch.long)
            for j, (ids, npr) in enumerate(chunk):
                t = torch.tensor(ids)
                x[j, :len(ids) - 1] = t[:-1]
                y[j, npr - 1:len(ids) - 1] = t[npr:]
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            lp = torch.log_softmax(logits.float(), -1)
            m = y >= 0
            yy = y.clamp(min=0)
            tl = lp.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * m
            lpv[i:i + len(chunk)] = (tl.sum(-1) / math.log(2)).cpu().numpy()
            hits[i:i + len(chunk)] = ((lp.argmax(-1) == yy) | ~m).all(-1).cpu().numpy()
        return lpv, hits

    @torch.no_grad()
    def choice_eval(self, model, device, n_alt=9):
        """10-way choice: does the true value get a higher log-prob than 9 other values from the same pool?
        Separates 'association stored' from 'model willing to output a made-up name' (exact-match recall)."""
        model.eval()
        rng = np.random.default_rng(12345)
        ne = self.n_eval
        items, owner = [], []
        for p in range(ne):
            for ai, (a, rel, fmt) in enumerate(REL):
                pid, _ = self.prompt[p][ai]
                true = self.vals[a][p]
                alts = [true] + [v for v in rng.choice(len(self.pools[a]), n_alt + 3, replace=False) if v != true][:n_alt]
                for v in alts:
                    vid = self.tok.encode(" " + fmt.format(v=self.pools[a][v])).ids
                    items.append((self.sep + pid + vid, len(self.sep) + len(pid)))
                owner.append((p, ai))
        lpv, _ = self._score(model, device, items)
        lpv = lpv.reshape(len(owner), n_alt + 1)
        win = lpv[:, 0] > lpv[:, 1:].max(1)
        win = win.reshape(ne, len(REL))
        model.train()
        return {"choice_by_k": {str(k): round(float(win[self.k[:ne] == k].mean()), 4) for k in self.bins},
                "choice_all": round(float(win.mean()), 4), "choice_chance": round(1 / (n_alt + 1), 3)}

    def fact_grad(self, model, device, names, B=128, kmin=0, nmax=0, only=None, sep=False):
        """gradient of the mean value-token loss (nats) of tested facts seen >= kmin times w.r.t. the named params
        (fdrift, 2026-11-25); the loss itself is left in self.last_fact_loss. nmax > 0 (zcheap, 2026-11-25): only nmax
        questions, evenly spaced over the list"""
        items = [(self.prompt[p][a][0] + self.prompt[p][a][1], len(self.prompt[p][a][0])) for p in range(self.n_eval) if self.k[p] >= kmin for a in range(len(REL))]
        if only is not None:   # practical-win screen (2026-11-25): just these (person, attr) facts
            items = [(self.prompt[q_][a_][0] + self.prompt[q_][a_][1], len(self.prompt[q_][a_][0])) for q_, a_ in only]
        if sep:   # fsplit (2026-11-26): same prompts as _score (separator first), so the loss matches the kept bits
            items = [(self.sep + ids, len(self.sep) + npr) for ids, npr in items]
        if nmax and nmax < len(items):
            items = [items[int(i)] for i in np.linspace(0, len(items) - 1, nmax).round()]
        tot = 0.0
        P = dict(model.named_parameters())
        model.zero_grad(set_to_none=True)
        ntok = sum(len(ids) - npr for ids, npr in items)
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            T = max(len(c[0]) for c in chunk)
            x = torch.zeros(len(chunk), T - 1, dtype=torch.long)
            y = torch.full((len(chunk), T - 1), -1, dtype=torch.long)
            for j, (ids, npr) in enumerate(chunk):
                t = torch.tensor(ids)
                x[j, :len(ids) - 1] = t[:-1]
                y[j, npr - 1:len(ids) - 1] = t[npr:]
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-1, reduction="sum") / ntok
            loss.backward()
            tot += loss.item()
        self.last_fact_loss = tot
        g = {n: P[n].grad.detach().float().clone() for n in names}
        model.zero_grad(set_to_none=True)
        return g

    @torch.no_grad()
    def _score(self, model, device, people, prompts=None):
        """exact-match recall and value bits for every (person, attr) of the given people (prompts: other prompt table, O76 ref)"""
        items = []      # (row, attr, ids, n_prompt)
        PT_ = self.prompt if prompts is None else prompts
        for r, p in enumerate(people):
            for ai in range(len(REL)):
                pid, vid = PT_[p][ai]
                items.append((r, ai, self.sep + pid + vid, len(self.sep) + len(pid)))
        ok = np.zeros((len(people), len(REL)), dtype=bool)
        bits = np.zeros((len(people), len(REL)))
        B = 128
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            T = max(len(c[2]) for c in chunk)
            x = torch.zeros(len(chunk), T - 1, dtype=torch.long)
            y = torch.full((len(chunk), T - 1), -1, dtype=torch.long)
            for j, (_, _, ids, npr) in enumerate(chunk):
                t = torch.tensor(ids)
                x[j, :len(ids) - 1] = t[:-1]
                y[j, npr - 1:len(ids) - 1] = t[npr:]
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            lp = torch.log_softmax(logits.float(), -1)
            m = y >= 0
            yy = y.clamp(min=0)
            tl = -lp.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * m
            hit = ((lp.argmax(-1) == yy) | ~m).all(-1)
            for j, (p, ai, _, _) in enumerate(chunk):
                ok[p, ai] = bool(hit[j])
                bits[p, ai] = tl[j].sum().item() / math.log(2)
        return ok, bits

    @torch.no_grad()
    def _filler_tok_bits(self, model, device):
        """mean bits per token on each filler item's own text (fixed kinds: its fixed tokens; facts: its sentences in
        canonical order; fresh kinds: one fresh sample). First token of each item is not scored."""
        tot, n = 0.0, 0
        items = []
        for p in range(self.n_eval, len(self.k)):
            if self.ffkind:
                t = list(self.fixed[p]) if self.ffkind in (3, 6) else self._alt(p)
            else:
                t = [x for ss in self.sent[p] for x in ss]
            t = self.sep + t
            W = getattr(getattr(model, "cfg", None), "seq_len", 64) + 1
            items += [t[j:j + W] for j in range(0, len(t) - 1, W - 1)]   # windows that fit the model context
        for i in range(0, len(items), 64):
            ch = items[i:i + 64]
            T = max(len(c) for c in ch)
            x = torch.zeros(len(ch), T - 1, dtype=torch.long)
            y = torch.full((len(ch), T - 1), -1, dtype=torch.long)
            for j, c in enumerate(ch):
                c = torch.tensor(c)
                x[j, :len(c) - 1] = c[:-1]
                y[j, :len(c) - 1] = c[1:]
            x, y = x.to(device), y.to(device)
            lg = model(x)
            lg = lg[0] if isinstance(lg, tuple) else lg
            lp = torch.log_softmax(lg.float(), -1)
            m = y >= 0
            tot += (-lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * m).sum().item()
            n += int(m.sum())
        return round(tot / max(n, 1) / math.log(2), 4)

    def ref_bits(self, model, device):
        """O76: mean value bits over the never-seen reference people (chance level C now); None without n_ref"""
        if not self.ref_prompt: return None
        _, b_ = self._score(model, device, range(len(self.ref_prompt)), prompts=self.ref_prompt)
        return round(float(b_.mean()), 4)

    @torch.no_grad()
    def evaluate(self, model, device, steps):
        model.eval()
        ne = self.n_eval
        ok, bits = self._score(model, device, range(ne))
        # filler-fact recall (Vuk 2026-11-24, candidate lab self-check): how well the frequent filler facts themselves
        # are learned. Only when the filler slots carry learnable facts (ffkind 0, fixed values).
        fres = None
        if len(self.k) > ne and not self.ffkind and not self.frand:
            fok, fbits = self._score(model, device, range(ne, len(self.k)))
            fres = (round(float(fok.mean()), 4), round(float(fbits.mean()), 3))
        ftb = self._filler_tok_bits(model, device) if len(self.k) > ne else None
        model.train()
        res = {"acc_by_k": {}, "bits_by_k": {}}
        if ftb is not None:
            res["filler_tok_bits"] = ftb   # how well the filler text itself is learned, any ffkind (2026-11-24)
        if fres:
            res["filler_acc"], res["filler_bits"] = fres
        for k in self.bins:
            sel = self.k[:ne] == k
            res["acc_by_k"][str(k)] = round(float(ok[sel].mean()), 4)
            res["bits_by_k"][str(k)] = round(float(bits[sel].mean()), 3)
        # recall by when the person was last seen (quarter of training), pooled over k <= 8
        q = np.clip((self.last_step[:ne] * 4) // max(1, steps), 0, 3)
        low = self.k[:ne] <= 8
        res["acc_lowk_by_lastq"] = [round(float(ok[low & (q == i)].mean()), 4) if (low & (q == i)).any() else None
                                    for i in range(4)]
        res["acc_by_attr"] = {REL[i][0]: round(float(ok[:, i].mean()), 4) for i in range(len(REL))}
        # stored bits: per fact, log2(pool) minus value bits, floored at 0 (pool = chance level)
        cap = 0.0
        for ai, (a, _, _) in enumerate(REL):
            cap += np.clip(math.log2(len(self.pools[a])) - bits[:, ai], 0, None).sum()
        res["stored_bits"] = round(float(cap), 1)
        res["n_people"] = ne
        res["acc_all"] = round(float(ok.mean()), 4)
        res["per_fact_ok"] = "".join("1" if v else "0" for v in ok.reshape(-1))          # person-major, 5 attrs each
        res["per_fact_bits"] = [round(float(v), 2) for v in bits.reshape(-1)]
        if self.ref_prompt:   # O76: never-seen reference people at the final weights (chance level C at the end)
            res["ref_bits"] = self.ref_bits(model, device)
        return res
