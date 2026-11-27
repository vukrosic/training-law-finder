"""Out-of-sample checks behind CLAIMS.md. Analysis only, no training.

    python3 tools/claims_data.py          # builds work/claims/*.jsonl from the runs table
    python3 tools/claims_fits.py batchseq # one analysis; names below

Every number is a held-out RMSE of kept[-1] (share of facts kept at the end) on runs that
the frozen forgetting problem never saw, unless the analysis says otherwise.
"""
import json
import math
import os
import sys
import zlib

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from problems.forgetting.formulas import end_z2  # noqa: E402

W = os.path.join(ROOT, "work", "claims")


def rows(name="lf.jsonl"):
    return [json.loads(l) for l in open(os.path.join(W, name))]


def clean(R):
    return [r for r in R if r["sched"] in ("cos", "wsd", "const") and r.get("momentum") == 0.95]


def z2(r, S):
    z = end_z2.x(r) @ np.asarray(end_z2.FROZEN[:9])
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(end_z2.FROZEN[9]) * (z + math.log(S)))))


def rmse(e):
    e = np.asarray(e)
    return float(np.sqrt((e ** 2).mean())), float(e.mean())


def half(r):
    return zlib.crc32(r["tag"].encode()) % 2


def report(name, R, f, p):
    e = [f(p, r) - r["kept"] for r in R]
    m, b = rmse(e)
    print(f"  {name:34s} n={len(R):4d} rmse {m:.4f} bias {b:+.4f}")
    return m


def batchseq():
    """Frozen end_z2, with its clock S scaled by batch and sequence length (fixed 10 constants)."""
    R = [r for r in clean(rows()) if not r["in_frozen"]]
    forms = {
        "frozen end_z2 (0 new)": (lambda p, r: z2(r, r["S"]), []),
        "S*(B/64)^-a (1 new)": (lambda p, r: z2(r, r["S"] * (r["batch"] / 64) ** -p[0]), [0.3]),
        "S*(B/64)^-a*(T/64)^c (2 new)": (
            lambda p, r: z2(r, r["S"] * (r["batch"] / 64) ** -p[0] * (r["seq"] / 64) ** p[1]), [0.3, 0.1]),
        "S*noise(B), noise=(1+Bn/B)/(1+Bn/64) (2)": (
            lambda p, r: z2(r, r["S"] * ((1 + math.exp(p[1]) / r["batch"]) / (1 + math.exp(p[1]) / 64)) ** p[0]),
            [1.0, 4.0]),
        "noise(B, Bn grows with width) + seq (4)": (
            lambda p, r: z2(r, r["S"] * ((1 + math.exp(p[1] + p[2] * math.log(r["width"] / 256)) / r["batch"])
                                         / (1 + math.exp(p[1] + p[2] * math.log(r["width"] / 256)) / 64)) ** p[0]
                            * (r["seq"] / 64) ** p[3]), [1.0, 4.0, 0.5, 0.1]),
    }
    splits = {
        "random half A->B": ([r for r in R if half(r) == 0], [r for r in R if half(r) == 1]),
        "batch<=64 -> batch>64": ([r for r in R if r["batch"] <= 64], [r for r in R if r["batch"] > 64]),
        "seq<=256 -> seq>256": ([r for r in R if r["seq"] <= 256], [r for r in R if r["seq"] > 256]),
        "width<=128 -> width>128": ([r for r in R if r["width"] <= 128], [r for r in R if r["width"] > 128]),
    }
    for sname, (tr, te) in splits.items():
        print(f"== fit on {sname}: train {len(tr)}, test {len(te)}")
        for name, (f, p0) in forms.items():
            p = p0
            if p0:
                p = least_squares(lambda q: [f(q, r) - r["kept"] for r in tr], p0).x
            m = report(name, te, f, p)
            if p0:
                print("      params", np.round(p, 3).tolist())


def matched(knob, fixed):
    """Print kept at each value of one knob with every other setting fixed."""
    import collections
    R = clean(rows())
    keys = ["width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched", "peak", "wd",
            "min_lr_frac", "warmup", "decay", "window"]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        g[tuple(r[k] for k in keys if k != knob)][r[knob]].append(r["kept"])
    return g


def shape():
    """Is the error of frozen end_z2 on cosine runs explained by how much of the clock is spent at low lr?"""
    R = [r for r in clean(rows()) if not r["in_frozen"] and r["batch"] == 64 and r["seq"] == 64]
    for s in ("cos", "wsd", "const"):
        Rs = [r for r in R if r["sched"] == s]
        if not Rs:
            continue
        e = np.array([z2(r, r["S"]) - r["kept"] for r in Rs])
        h = np.array([r["share_below_half"] for r in Rs])
        c = np.corrcoef(e, h)[0, 1] if h.std() > 0 else float("nan")
        print(f"  {s:6s} n={len(Rs):4d} rmse {rmse(e)[0]:.4f} bias {e.mean():+.4f} corr(err, share below half) {c:+.2f}")


def recovery():
    """Share of runs whose kept rises after the window (facts come back during the decay)."""
    R = clean(rows())
    for s in ("cos", "wsd", "const"):
        Rs = [r for r in R if r["sched"] == s]
        up = [r["kept_max"] - r["kept"] for r in Rs]
        print(f"  {s:6s} n={len(Rs):4d} mean (max kept - end kept) {np.mean(up):.4f}  share > 0.02: {np.mean(np.array(up) > 0.02):.2f}")



def curves(maxrun=None):
    import collections
    C = collections.defaultdict(list)
    for l in open(os.path.join(W, "curve.jsonl")):
        r = json.loads(l)
        C[r["tag"]].append(r)
    out = {t: sorted(v, key=lambda r: r["i"]) for t, v in C.items() if len(v) >= 8}
    return dict(list(out.items())[:maxrun]) if maxrun else out


def rise():
    """How much kept climbs back from its lowest reading to the end of the run."""
    for s in ("cos", "wsd", "const"):
        u = []
        for t, v in curves().items():
            if v[0]["sched"] != s:
                continue
            k = np.array([r["kept"] for r in v])
            u.append(k[-1] - k.min())
        u = np.array(u)
        print(f"  {s:6s} runs {len(u):4d} mean rise {u.mean():.4f} share > 0.02 {np.mean(u > 0.02):.2f} share > 0.05 {np.mean(u > 0.05):.2f}")


def curveshape():
    """Per-run fit of kept(S_t) inside the run: three 2-constant curves, fit on the first 70% of
    readings, scored on the last 30% (held out in time)."""
    forms = {
        "hill 1/(1+(S/s0)^q)": lambda p, S: 1 / (1 + (S / np.exp(p[0])) ** np.exp(p[1])),
        "stretched exp(-(S/s0)^q)": lambda p, S: np.exp(-(S / np.exp(p[0])) ** np.exp(p[1])),
        "power (1+S/s0)^-q": lambda p, S: (1 + S / np.exp(p[0])) ** -np.exp(p[1]),
        "hill with q=1, 1/(1+S/s0)": lambda p, S: 1 / (1 + S / np.exp(p[0])) + 0 * p[1],
    }
    err = {k: [] for k in forms}
    qs = collections_q = []
    for t, v in curves().items():
        S = np.array([r["S"] for r in v])
        k = np.array([r["kept"] for r in v])
        n = int(0.7 * len(S))
        for name, f in forms.items():
            p = least_squares(lambda p: f(p, S[:n]) - k[:n], [np.log(np.median(S)), 0.0]).x
            err[name].append(np.sqrt(np.mean((f(p, S[n:]) - k[n:]) ** 2)))
            if name.startswith("hill 1/(1+(S"):
                collections_q.append((v[0]["width"], float(np.exp(p[1]))))
    for name, e in err.items():
        print(f"  {name:28s} runs {len(e)} median held-out-in-time rmse {np.median(e):.4f} mean {np.mean(e):.4f}")
    import collections
    by = collections.defaultdict(list)
    for w, q in qs:
        by[w].append(q)
    print("  hill steepness q by width:", {w: round(float(np.median(v)), 2) for w, v in sorted(by.items()) if len(v) >= 5})



def pairs():
    """Matched pairs in the lf table: change one knob, keep every other setting. Prints the mean
    change in final val loss and in kept, and how often the second value has the lower loss.
        python3 tools/claims_fits.py pairs layers wd sched"""
    import collections
    R = [r for r in clean(rows()) if r.get("val_loss")]
    keys = ["width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched", "peak", "wd",
            "min_lr_frac", "warmup", "decay", "window", "momentum"]
    for knob in sys.argv[2:]:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in R:
            g[tuple(r[k] for k in keys if k != knob)][r[knob]].append((r["val_loss"], r["kept"]))
        dl, dk = collections.defaultdict(list), collections.defaultdict(list)
        for v in g.values():
            vals = sorted(v)
            for a, b in zip(vals, vals[1:]):
                dl[(a, b)].append(np.mean([x[0] for x in v[b]]) - np.mean([x[0] for x in v[a]]))
                dk[(a, b)].append(np.mean([x[1] for x in v[b]]) - np.mean([x[1] for x in v[a]]))
        print("==", knob)
        for p in sorted(dl, key=str):
            if len(dl[p]) >= 3:
                print(f"  {p[0]}->{p[1]} pairs {len(dl[p]):3d} loss {np.mean(dl[p]):+.4f} "
                      f"lower in {sum(x < 0 for x in dl[p])}/{len(dl[p])} kept {np.mean(dk[p]):+.4f}")


def other():
    """Matched pairs inside one of the other searches (no fact probe): change one config key,
    keep the rest. python3 tools/claims_fits.py other init norm_mode"""
    import collections
    search, knob = sys.argv[2], sys.argv[3]
    R = [r for r in rows(f"other_{search}.jsonl") if r.get("val_loss") and isinstance(r.get("config"), dict)]
    for r in R:
        c = dict(r["config"])
        c.pop("seed", None)
        c.update(_method=r.get("method"), _steps=r.get("steps"), _data=r.get("data"),
                 _device=r.get("device"), _size=r.get("size"))
        r["c"] = {k: json.dumps(v) for k, v in c.items()}
    keys = sorted(set(k for r in R for k in r["c"]))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        g[tuple(r["c"].get(k) for k in keys if k != knob)][str(r["c"].get(knob))].append(r["val_loss"])
    dl = collections.defaultdict(list)
    for v in g.values():
        for a in v:
            for b in v:
                if a < b:
                    dl[(a, b)].append(np.mean(v[b]) - np.mean(v[a]))
    for p, x in sorted(dl.items(), key=lambda t: -len(t[1])):
        if len(x) >= 4:
            print(f"  {p[0]} -> {p[1]} cells {len(x):3d} loss {np.mean(x):+.4f} second lower in {sum(y < 0 for y in x)}/{len(x)}")



def decay():
    """WSD runs: kept at the start of the final decay and at the end, and over the flat stretch of
    the same length just before it, with the learning-rate sum each stretch spends."""
    L = {r["tag"]: r for r in rows()}
    a = []
    for t, v in curves().items():
        r = L[t]
        if r["sched"] != "wsd":
            continue
        fd = ((1 - r["decay"]) - r["fwhi"]) / (1 - r["fwhi"])
        f = np.array([x["frac"] for x in v])
        k = np.array([x["kept"] for x in v])
        S = np.array([x["S"] for x in v])
        j = int(np.abs(f - fd).argmin())
        j0 = int(np.abs(f - (2 * fd - 1)).argmin())
        a.append((k[j0], k[j], k[-1], S[j] - S[j0], r["S"] - S[j], float(k[-1] > k[j])))
    a = np.array(a)
    print(f"  WSD runs {len(a)}: kept {a[:, 0].mean():.3f} -> {a[:, 1].mean():.3f} over the flat stretch "
          f"(lr sum {a[:, 3].mean():.2f}), -> {a[:, 2].mean():.3f} over the decay (lr sum {a[:, 4].mean():.2f}); "
          f"kept rises during the decay in {a[:, 5].mean():.2f} of runs")


def batchwidth():
    """Kept by batch at each width: cosine, peak 8e-3, wd 0, 3 layers, seq 64, 1024 steps."""
    import collections
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in clean(rows()):
        if (r["sched"], r["peak"], r["wd"], r["layers"], r["seq"], r["steps"], r["window"]) == ("cos", 0.008, 0.0, 3, 64, 1024, "A") \
                and r["min_lr_frac"] == 0.0:
            g[r["width"]][r["batch"]].append(r["kept"])
    for w in sorted(g):
        if len(g[w]) >= 3:
            print(f"  width {w:5d}: " + "  ".join(f"b{b}:{np.mean(v):.3f}" for b, v in sorted(g[w].items())))



def scan():
    """Every config key of every other search: matched pairs that change only that key.
    Prints pairs with 8+ cells where one side wins in 85%+ of cells.
    python3 tools/claims_fits.py scan [search]"""
    import collections
    searches = sys.argv[2:] or ["compute", "data", "init", "learning", "mixing", "muscale", "residual", "target", "vocab"]
    for search in searches:
        R = [r for r in rows(f"other_{search}.jsonl") if r.get("val_loss") and isinstance(r.get("config"), dict)]
        for r in R:
            c = dict(r["config"])
            c.pop("seed", None)
            c.update(_method=r.get("method"), _steps=r.get("steps"), _data=r.get("data"),
                     _device=r.get("device"), _size=r.get("size"))
            r["c"] = {k: json.dumps(v) for k, v in c.items()}
        keys = sorted(set(k for r in R for k in r["c"]))
        for knob in keys:
            if len(set(r["c"].get(knob) for r in R)) < 2:
                continue
            g = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in R:
                g[tuple(r["c"].get(k) for k in keys if k != knob)][str(r["c"].get(knob))].append(r["val_loss"])
            dl = collections.defaultdict(list)
            for v in g.values():
                for a in v:
                    for b in v:
                        if a < b:
                            dl[(a, b)].append(np.mean(v[b]) - np.mean(v[a]))
            for pr, x in sorted(dl.items(), key=lambda t: -len(t[1])):
                w = sum(y < 0 for y in x)
                if len(x) >= 8 and max(w, len(x) - w) >= 0.85 * len(x):
                    print(f"{search:9s} {knob:18s} {pr[0][:14]:>14s} -> {pr[1][:14]:14s} cells {len(x):3d} loss {np.mean(x):+.4f} second lower {w}/{len(x)}")


def detail():
    """One knob pair inside one other search, split by steps and size.
    python3 tools/claims_fits.py detail learning mom 0.85 0.95"""
    import collections
    search, knob, a, b = sys.argv[2:6]
    R = [r for r in rows(f"other_{search}.jsonl") if r.get("val_loss") and isinstance(r.get("config"), dict)]
    for r in R:
        c = dict(r["config"])
        c.pop("seed", None)
        c.update(_method=r.get("method"), _steps=r.get("steps"), _data=r.get("data"),
                 _device=r.get("device"), _size=r.get("size"))
        r["c"] = {k: json.dumps(v) for k, v in c.items()}
    keys = sorted(set(k for r in R for k in r["c"]))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        g[tuple(r["c"].get(k) for k in keys if k != knob)][str(r["c"].get(knob))].append(r)
    by = collections.defaultdict(list)
    for v in g.values():
        if a in v and b in v:
            ra = v[a][0]
            d = np.mean([x["val_loss"] for x in v[b]]) - np.mean([x["val_loss"] for x in v[a]])
            by[(ra.get("steps"), ra.get("size"), ra.get("data"), ra.get("params"))].append(d)
    allx = [x for v in by.values() for x in v]
    print(f"{search} {knob} {a} -> {b}: cells {len(allx)} loss {np.mean(allx):+.4f} second lower {sum(x < 0 for x in allx)}/{len(allx)}")
    for k, x in sorted(by.items(), key=lambda t: str(t[0])):
        print(f"  steps {k[0]} size {k[1]} data {k[2]} params {k[3]}: n {len(x)} {np.mean(x):+.4f}")


RUN_KEYS = ["width", "layers", "heads", "hidden", "n_experts", "batch", "seq", "steps", "method", "sched", "peak",
            "mlr", "wd", "min_lr_frac", "warmup", "decay", "window", "k", "F", "amp", "clip", "momentum", "opt",
            "prec", "device", "eval_seqs"]


def runs_table():
    import gzip
    out = []
    with gzip.open(os.path.join(ROOT, "data/llm/runs.jsonl.gz"), "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("val_loss") and math.isfinite(d["val_loss"]):
                out.append({k: d.get(k) for k in RUN_KEYS + ["val_loss", "rms_w_end", "wnorm", "tag", "protocol", "cfg"]})
    return out


def scanruns():
    """Matched pairs in runs.jsonl.gz (final loss only, no fact probe): change one setting, keep every
    other setting in RUN_KEYS and the device. python3 tools/claims_fits.py scanruns [knob ...]"""
    import collections
    R = runs_table()
    for r in R:
        r["c"] = {k: json.dumps(r.get(k)) for k in RUN_KEYS}
        r["c"]["cfg"] = json.dumps(r.get("cfg"), sort_keys=True) if not isinstance(r.get("cfg"), dict) else json.dumps({k: v for k, v in r["cfg"].items() if k not in RUN_KEYS and k != "seed"}, sort_keys=True)
    knobs = sys.argv[2:] or [k for k in RUN_KEYS if k != "device"]
    for knob in knobs:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in R:
            g[tuple(v for k, v in r["c"].items() if k != knob)][r["c"][knob]].append(r["val_loss"])
        dl = collections.defaultdict(list)
        for v in g.values():
            for a in v:
                for b in v:
                    if a != b:
                        try:
                            ok = float(json.loads(a) if a != "null" else -1) < float(json.loads(b) if b != "null" else -1)
                        except (TypeError, ValueError):
                            ok = a < b
                        if ok:
                            dl[(a, b)].append(np.mean(v[b]) - np.mean(v[a]))
        for pr, x in sorted(dl.items(), key=lambda t: -len(t[1]))[:8]:
            if len(x) >= 8:
                w = sum(y < 0 for y in x)
                print(f"{knob:12s} {pr[0][:16]:>16s} -> {pr[1][:16]:16s} pairs {len(x):4d} loss {np.mean(x):+.4f} second lower {w}/{len(x)}")


FL_KEYS = ["width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched", "wd", "min_lr_frac", "warmup",
           "decay", "momentum", "device"]


def fl_rows():
    import gzip
    with gzip.open(os.path.join(ROOT, "problems/final-loss/data/llm.jsonl.gz"), "rt") as f:
        return [json.loads(x) for x in f]


def steplaw():
    """Final-loss problem runs: chains that differ only in steps (512..4096 or 1000..4000). Ratio of
    successive loss drops per doubling of steps, and the exponent a in L = E + A*T^-a it implies."""
    import collections
    g = collections.defaultdict(dict)
    for r in fl_rows():
        g[tuple(str(r.get(k)) for k in FL_KEYS if k != "steps") + (str(r.get("peak")),)][r["steps"]] = r["val_loss"]
    ratios, n = [], 0
    for v in g.values():
        T = sorted(v)
        dbl = [t for t in T if 2 * t in v and 4 * t in v]
        for t in dbl:
            d1, d2 = v[t] - v[2 * t], v[2 * t] - v[4 * t]
            if d1 > 0:
                ratios.append(d2 / d1)
        n += bool(dbl)
    ratios = np.array(ratios)
    print(f"chains {n}, triples {len(ratios)}: drop ratio per doubling median {np.median(ratios):.3f}"
          f" (IQR {np.percentile(ratios, 25):.3f}-{np.percentile(ratios, 75):.3f}), implied a = {-np.log2(np.median(ratios)):.2f};"
          f" ratio < 1 in {np.mean(ratios < 1):.2f}")


def basin():
    """Final-loss problem runs: learning-rate sweeps (4+ peaks, all else equal). Loss cost of a peak 2x
    above and 2x below the best grid peak."""
    import collections
    g = collections.defaultdict(dict)
    for r in fl_rows():
        g[tuple(str(r.get(k)) for k in FL_KEYS)][r["peak"]] = r["val_loss"]
    hi, lo, curv, W = [], [], [], []
    for k, v in g.items():
        if len(v) < 4:
            continue
        P = sorted(v)
        b = min(P, key=lambda p: v[p])
        if b == P[0] or b == P[-1]:
            continue
        def near(f):
            c = min(P, key=lambda p: abs(math.log(p / (b * f))))
            return v[c] - v[b] if abs(math.log(c / (b * f))) < 0.2 else None
        h, l = near(2), near(0.5)
        if h is not None and l is not None:
            hi.append(h); lo.append(l); W.append(int(k[0]))
        x = np.log([p for p in P]); y = np.array([v[p] for p in P])
        i = P.index(b); s = slice(max(0, i - 2), i + 3)
        if len(x[s]) >= 4:
            c = np.polyfit(x[s] - math.log(b), y[s], 2)[0]
            curv.append((int(k[0]), c))
    hi, lo = np.array(hi), np.array(lo)
    print(f"sweeps with an inside optimum and both 2x neighbours: {len(hi)}; cost 2x high median {np.median(hi):.4f},"
          f" 2x low median {np.median(lo):.4f}; high costs more in {np.mean(hi > lo):.2f}")
    cw = collections.defaultdict(list)
    for w, c in curv:
        cw[w].append(c)
    print("quadratic curvature in ln(peak) by width:", {w: (len(c), round(float(np.median(c)), 4)) for w, c in sorted(cw.items())})
    ws = np.array([w for w, c in curv]); cs = np.array([c for w, c in curv])
    m = cs > 0
    print(f"slope of ln curvature on ln width: {np.polyfit(np.log(ws[m]), np.log(cs[m]), 1)[0]:+.3f} over {m.sum()} sweeps")


def wrms():
    """Weight RMS at the end of training (rms_w_end) against peak lr, steps, width and weight decay,
    least squares on logs. Runs with rms_w_end in runs.jsonl.gz."""
    import gzip
    R = []
    with gzip.open(os.path.join(ROOT, "data/llm/runs.jsonl.gz"), "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("rms_w_end") and d.get("peak") and d.get("opt") in ("muon", None):
                R.append(d)
    def fit(rows, name):
        X = np.array([[1, math.log(r["peak"]), math.log(r["steps"]), math.log(r["width"]), r["wd"] or 0.0,
                       r.get("min_lr_frac") or 0.0] for r in rows])
        y = np.log([r["rms_w_end"] for r in rows])
        h = np.array([zlib.crc32(r["tag"].encode()) % 2 for r in rows])
        c, *_ = np.linalg.lstsq(X[h == 0], y[h == 0], rcond=None)
        e = y[h == 1] - X[h == 1] @ c
        print(f"{name}: n {len(rows)} coef [1, ln peak, ln steps, ln width, wd, min_lr_frac] = {np.round(c, 3)};"
              f" held-out RMSE in ln {np.sqrt(np.mean(e ** 2)):.3f} (sd of ln rms {y[h == 1].std():.3f})")
    fit(R, "all")
    fit([r for r in R if not r["wd"]], "wd 0")


def webcode():
    """Every knob pair of every other search, measured separately on FineWeb text and on Python code
    (pystd). Share of pairs whose mean effect has the same sign on both, and the slope of the code effect
    on the web effect. Pairs need 3+ matched cells on each data set and |effect| > 0.01 on at least one."""
    import collections
    out = []
    for search in ["compute", "data", "init", "learning", "mixing", "muscale", "residual", "target", "vocab"]:
        R = [r for r in rows(f"other_{search}.jsonl") if r.get("val_loss") and isinstance(r.get("config"), dict)]
        for r in R:
            c = dict(r["config"])
            c.pop("seed", None)
            c.update(_method=r.get("method"), _steps=r.get("steps"), _device=r.get("device"), _size=r.get("size"))
            r["c"] = {k: json.dumps(v) for k, v in c.items()}
            r["d"] = "code" if "py" in str(r.get("data")) else "web"
        keys = sorted(set(k for r in R for k in r["c"]))
        for knob in keys:
            if len(set(r["c"].get(knob) for r in R)) < 2:
                continue
            g = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in R:
                g[(r["d"],) + tuple(r["c"].get(k) for k in keys if k != knob)][str(r["c"].get(knob))].append(r["val_loss"])
            dl = collections.defaultdict(lambda: collections.defaultdict(list))
            for gk, v in g.items():
                for a in v:
                    for b in v:
                        if a < b:
                            dl[(a, b)][gk[0]].append(np.mean(v[b]) - np.mean(v[a]))
            for pr, d in dl.items():
                if len(d["web"]) >= 3 and len(d["code"]) >= 3:
                    w, c = float(np.mean(d["web"])), float(np.mean(d["code"]))
                    if max(abs(w), abs(c)) > 0.01:
                        out.append((search, knob, pr, w, c))
    out = [o for o in out if np.isfinite(o[3]) and np.isfinite(o[4])]
    W = np.array([o[3] for o in out]); C = np.array([o[4] for o in out])
    print("data values:", sorted(set(str(r.get("data")) for r in R)))
    big = (np.abs(W) > 0.03) & (np.abs(C) > 0.03)
    print(f"pairs {len(out)}: same sign {np.mean(np.sign(W) == np.sign(C)):.2f}; both |effect|>0.03: {big.sum()} pairs,"
          f" same sign {np.mean(np.sign(W[big]) == np.sign(C[big])):.2f}; slope code on web {np.polyfit(W, C, 1)[0]:.2f} (through 0: {W @ C / (W @ W):.2f}),"
          f" median |code|/|web| {np.median(np.abs(C) / np.maximum(np.abs(W), 1e-4)):.2f}, corr {np.corrcoef(W, C)[0, 1]:.2f}")
    for o in sorted(out, key=lambda o: -abs(o[3] - o[4]))[:8]:
        print(f"  {o[0]:9s} {o[1]:12s} {o[2][0][:10]:>10s}->{o[2][1][:10]:10s} web {o[3]:+.3f} code {o[4]:+.3f}")

if __name__ == "__main__":
    globals()[sys.argv[1]]()
