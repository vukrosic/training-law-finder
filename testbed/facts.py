"""Planted facts: made-up people with five attributes, mixed into web text.

Each person gets a pseudo-word name and five facts:
    city (200 values), university (300), major (100), company (260), birth year (100).
A biography states all five facts:
    " Zorvek Talimun was born in Brasko. Zorvek Talimun studied at Quelor University. ..."
People are split into sighting bins: bin k holds people whose biography appears
k times during training (k = 1, 2, 4, ..., kmax). Every sighting lands at a
uniformly random training step inside [start, end) of the run, and replaces web
text in that step's batch, so the token budget stays the same.

Recall test (teacher forcing): the prompt " <name> <relation>" is followed by the
value tokens; a fact counts as recalled when every value token is the argmax.
The test also reports the value's bits (-log2 p of its tokens) and the bits the
model stores beyond chance, max(0, log2(pool size) - bits).

Per-person rows also record the step of the person's last sighting.
"""
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

SYL_A = ["ka", "zor", "vel", "mi", "tan", "ru", "bel", "do", "sha", "ni", "gor", "lu",
         "fen", "tra", "qua", "pim", "sor", "xa", "len", "dru", "vo", "ki", "mar", "zu"]
SYL_B = ["vek", "lim", "dor", "sa", "nel", "bri", "tok", "mun", "ras", "ven", "ilo", "gat",
         "pel", "rin", "sko", "mav", "ther", "lo", "dax", "wen", "cor", "fi", "yan", "rek"]
SYL_C = ["bra", "sko", "que", "lor", "tav", "min", "ost", "ral", "pek", "dun", "vas", "ith",
         "gre", "mol", "zan", "hep", "tur", "cav", "lyn", "obe", "nix", "fal", "rup", "sel"]

# (key, phrase before the value, value template, pool size)
RELATIONS = [
    ("city", "was born in", "{v}", 200),
    ("univ", "studied at", "{v} University", 300),
    ("major", "majored in", "{v} studies", 100),
    ("company", "worked for", "{v} Corporation", 260),
    ("year", "has birth year", "{v}", 100),
]
SEP = "\n"


def _word(rng, a, b, n=2):
    s = rng.choice(a) + "".join(rng.choice(b) for _ in range(n - 1))
    return s.capitalize()


def _unique(rng, n, make):
    out, seen = [], set()
    while len(out) < n:
        w = make()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


class FactWorld:
    def __init__(self, tok, n_per_bin=64, kmax=16, start_frac=0.0, end_frac=1.0,
                 steps=1000, seed=0):
        self.tok = tok
        rng = random.Random(seed)
        self.bins = [2 ** i for i in range(int(math.log2(kmax)) + 1)]
        n_people = n_per_bin * len(self.bins)
        names = _unique(rng, n_people, lambda: _word(rng, SYL_A, SYL_B, 2) + " " +
                        _word(rng, SYL_B, SYL_C, 3))
        pools = {}
        for key, _, _, size in RELATIONS:
            if key == "year":
                pools[key] = [str(1900 + i) for i in range(size)]
            else:
                pools[key] = _unique(rng, size, lambda: _word(rng, SYL_C, SYL_A, 2))
        self.pools = pools
        self.people = []
        for i, name in enumerate(names):
            k = self.bins[i // n_per_bin]
            facts = {key: rng.choice(pools[key]) for key, _, _, _ in RELATIONS}
            self.people.append(dict(pid=i, name=name, k=k, facts=facts))
        # schedule sightings at random steps in [s0, s1)
        s0 = int(start_frac * steps)
        s1 = max(s0 + 1, int(end_frac * steps))
        self.by_step = {}
        for p in self.people:
            seen = sorted(rng.randrange(s0, s1) for _ in range(p["k"]))
            p["sightings"] = seen
            p["last_seen"] = seen[-1]
            for s in seen:
                self.by_step.setdefault(s, []).append(p["pid"])
        self.bio_ids = {p["pid"]: tok.encode(self.bio(p)).ids for p in self.people}

    def sentence(self, p, key):
        for k, phrase, tmpl, _ in RELATIONS:
            if k == key:
                return f" {p['name']} {phrase} {tmpl.format(v=p['facts'][key])}."
        raise KeyError(key)

    def bio(self, p):
        return SEP + "".join(self.sentence(p, key) for key, _, _, _ in RELATIONS)

    def step_tokens(self, step):
        """Token ids of every biography scheduled at this step (may be empty)."""
        ids = []
        for pid in self.by_step.get(step, []):
            ids.extend(self.bio_ids[pid])
        return ids

    def recall_items(self):
        """(pid, key, prompt ids, value ids, pool size) for every fact."""
        items = []
        for p in self.people:
            for key, phrase, tmpl, size in RELATIONS:
                prompt = f"{SEP} {p['name']} {phrase}"
                full = prompt + " " + tmpl.format(v=p["facts"][key])
                pi = self.tok.encode(prompt).ids
                fi = self.tok.encode(full).ids
                # the value is whatever the full string adds after the prompt tokens
                if fi[:len(pi)] != pi:
                    pi = fi[:len(pi)]
                items.append((p["pid"], key, pi, fi[len(pi):], size))
        return items

    @torch.no_grad()
    def evaluate(self, model, device, batch=256):
        """Per-person recall. Returns summary dict and per-person rows."""
        model.eval()
        items = self.recall_items()
        res = {}
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            L = max(len(a) + len(b) for _, _, a, b, _ in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for j, (_, _, a, b, _) in enumerate(chunk):
                seq = a + b
                x[j, :len(seq)] = torch.tensor(seq)
            logp = F.log_softmax(model(x.to(device)).float(), -1).cpu()
            for j, (pid, key, a, b, size) in enumerate(chunk):
                pos = torch.arange(len(a) - 1, len(a) + len(b) - 1)
                tgt = torch.tensor(b)
                lp = logp[j, pos]
                ok = bool((lp.argmax(-1) == tgt).all())
                bits = float(-lp.gather(1, tgt[:, None]).sum() / math.log(2))
                res[(pid, key)] = (ok, bits, size)
        model.train()
        rows, by_k = [], {}
        for p in self.people:
            r = [res[(p["pid"], key)] for key, _, _, _ in RELATIONS]
            n_ok = sum(o for o, _, _ in r)
            bits = sum(b for _, b, _ in r)
            stored = sum(max(0.0, math.log2(s) - b) for _, b, s in r)
            rows.append(dict(pid=p["pid"], k=p["k"], last_seen=p["last_seen"],
                             recalled=n_ok, n_facts=len(r), bits=round(bits, 3),
                             stored_bits=round(stored, 3)))
            d = by_k.setdefault(p["k"], [0, 0, 0.0, 0.0])
            d[0] += n_ok
            d[1] += len(r)
            d[2] += bits
            d[3] += stored
        summary = dict(
            acc=round(sum(v[0] for v in by_k.values()) / sum(v[1] for v in by_k.values()), 4),
            stored_bits=round(sum(v[3] for v in by_k.values()), 1),
            acc_by_k={str(k): round(v[0] / v[1], 4) for k, v in sorted(by_k.items())},
            bits_by_k={str(k): round(v[2] / v[1], 3) for k, v in sorted(by_k.items())},
        )
        return summary, rows
