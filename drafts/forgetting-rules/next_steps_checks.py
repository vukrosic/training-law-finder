"""Checks behind NEXT-STEPS.md. No training: everything is computed from the public table of 375 runs.

    python3 drafts/forgetting-rules/next_steps_checks.py

Needs numpy and scipy. Takes about one minute on a laptop CPU. Prints three parts:
  1. ten constants against nine (no depth term) on seven held-out slices
  2. the formula on the readings inside each run, not only the end point
  3. a schedule picker, tested on groups of runs that differ only in schedule (and peak)
"""
import collections, gzip, json, math, os, warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = os.environ.get("TLF_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rows = [json.loads(l) for l in gzip.open(os.path.join(ROOT, "problems/forgetting/data/llm.jsonl.gz"), "rt")]
FROZEN = [0.6090888304082575, -0.5929143343302276, 0.8482437722622462, 0.5477102470699494, -1.1231337626121998,
          -0.3217546827755414, -0.12444372936842882, 0.17762544026882962, 0.23118595535279032, 0.24836649590480928]
FLOOR = 0.0126  # the noise floor of the problem


def feats(r):  # the nine inputs of end_z2.py; column 6 is the depth term
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return [1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld, math.log(r["layers"] / 8), lp * ld, ld * ld]


X = np.array([feats(r) for r in rows])
lnS = np.array([math.log(sum(r["lr"])) for r in rows])   # S = learning rate summed after the facts
y = np.array([r["kept"][-1] for r in rows])              # share of facts kept at the end
W = np.array([r["width"] for r in rows])
ALL = list(range(9))
NO_DEPTH = [c for c in ALL if c != 6]
rmse = lambda a: float(np.sqrt(np.mean(np.square(a))))


def pred(p, X, lnS, cols=ALL):
    return 1 / (1 + np.exp(np.minimum(50, np.exp(p[-1]) * (X[:, cols] @ p[:-1] + lnS))))


def fit(X, lnS, y, cols=ALL):
    p0 = np.array([FROZEN[c] for c in cols] + [FROZEN[9]])
    return least_squares(lambda p: pred(p, X, lnS, cols) - y, p0).x


def heldout(groups, cols):  # leave one group out, fit on the rest, mean error over the groups
    per = []
    for g in sorted(set(groups), key=str):
        te = np.array([x == g for x in groups])
        per.append(rmse(pred(fit(X[~te], lnS[~te], y[~te], cols), X[te], lnS[te], cols) - y[te]))
    return float(np.mean(per))


print("PART 1. Error on a held-out group, in points: ten constants, and nine (no depth term)")
for name, key in [("width", "width"), ("layers", "layers"), ("steps", "steps"), ("peak", "peak"),
                  ("schedule", "sname"), ("weight decay", "wd"), ("window", "window")]:
    g = [r[key] if (key != "window" or r[key] != "const") else "A" for r in rows]
    a, b = heldout(g, ALL), heldout(g, NO_DEPTH)
    print(f"  hold out {name:13s} ten {100 * a:5.2f}   nine {100 * b:5.2f}   difference {100 * (b - a):+5.2f}")
pairs = collections.Counter((r["width"], r["layers"]) for r in rows)
print("  runs per (width, layers):", dict(sorted(pairs.items())))

print("\nPART 2. Inside the run: one reading every 6th step after the facts stop")
Xi, Si, Yi, Wi = [], [], [], []
for i, r in enumerate(rows):
    cs, kk = np.cumsum(r["lr"]), np.array(r["kept"])
    for j in range(5, len(kk), 6):
        Xi.append(X[i]); Si.append(math.log(cs[j])); Yi.append(kk[j]); Wi.append(r["width"])
Xi, Si, Yi, Wi = np.array(Xi), np.array(Si), np.array(Yi), np.array(Wi)
p_end, p_in = fit(X, lnS, y), fit(Xi, Si, Yi)
print(f"  readings: {len(Yi)}")
print(f"  constants fitted on end points:   error inside the run {100 * rmse(pred(p_end, Xi, Si) - Yi):.2f}, at the end {100 * rmse(pred(p_end, X, lnS) - y):.2f}")
print(f"  constants fitted on all readings: error inside the run {100 * rmse(pred(p_in, Xi, Si) - Yi):.2f}, at the end {100 * rmse(pred(p_in, X, lnS) - y):.2f}")
print(f"  q fitted on end points {math.exp(p_end[-1]):.3f}, on all readings {math.exp(p_in[-1]):.3f}")
ins, end = [], []
for w in sorted(set(W.tolist())):
    p = fit(Xi[Wi != w], Si[Wi != w], Yi[Wi != w])
    ins.append(rmse(pred(p, Xi[Wi == w], Si[Wi == w]) - Yi[Wi == w])); end.append(rmse(pred(p, X[W == w], lnS[W == w]) - y[W == w]))
print(f"  held-out width, constants fitted on all readings: inside the run {100 * np.mean(ins):.2f}, at the end {100 * np.mean(end):.2f}")
e, pr = pred(p_in, Xi, Si) - Yi, pred(p_in, Xi, Si)
for lo, hi in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
    m = (pr >= lo) & (pr < hi)
    print(f"    predicted kept {lo:.1f} to {min(hi, 1):.1f}: {m.sum():6d} readings, error {100 * rmse(e[m]):5.2f}")

print("\nPART 3. Schedule picker: in a group of runs, pick the one with the highest predicted kept")
ph = np.zeros(len(y))  # each run predicted by constants fitted without its width
for w in sorted(set(W.tolist())):
    ph[W == w] = pred(fit(X[W != w], lnS[W != w], y[W != w]), X[W == w], lnS[W == w])
pf = pred(np.array(FROZEN), X, lnS)


def test(keys, score, name, sel=lambda r: True):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if sel(r):
            groups[tuple(r[k] for k in keys)].append(i)
    hit = n = 0; loss, rho = [], []
    for idx in map(np.array, groups.values()):
        if len(idx) < 2 or y[idx].max() - y[idx].min() < FLOOR:
            continue
        n += 1; pick = idx[score[idx].argmax()]
        hit += y[pick] >= y[idx].max() - FLOOR; loss.append(y[idx].max() - y[pick])
        if len(idx) >= 3:
            rho.append(spearmanr(score[idx], y[idx]).correlation)
    print(f"    {name:34s} best run picked in {hit}/{n} groups, mean loss {100 * np.mean(loss):4.1f} points, worst {100 * max(loss):4.1f}, rank correlation {np.nanmean(rho):.2f}")


rnd = np.random.default_rng(0).random(len(y))
SAME = ["width", "layers", "steps", "wd", "window"]
print("  only the schedule differs:")
for s, n in [(ph, "formula, width held out"), (-lnS, "lowest S"), (rnd, "random pick")]:
    test(SAME + ["peak"], s, n)
print("  schedule and peak learning rate differ:")
for s, n in [(ph, "formula, width held out"), (pf, "formula, frozen constants"), (-lnS, "lowest S"), (rnd, "random pick")]:
    test(SAME, s, n)
print("  the same, split by width:")
for sel, label in [(lambda r: r["width"] == 64, "width 64"), (lambda r: r["width"] > 64, "widths 128 to 1,024")]:
    for s, n in [(ph, "formula, width held out"), (pf, "formula, frozen constants")]:
        test(SAME, s, f"{label}, {n[9:]}", sel)


def menu(width, layers, steps, peak, wd, after):
    """Predicted share kept for a few schedules over the `after` steps that follow the facts."""
    t, out = np.arange(after) / max(1, after - 1), []
    for name, pk, lr, mf in [("constant", peak, np.full(after, peak), 1.0),
                             ("constant at half the peak", peak / 2, np.full(after, peak / 2), 1.0),
                             ("linear to 0", peak, peak * (1 - t), 0.0),
                             ("cosine to 0", peak, peak * 0.5 * (1 + np.cos(math.pi * t)), 0.0),
                             ("cosine to 10% of peak", peak, peak * (0.1 + 0.45 * (1 + np.cos(math.pi * t))), 0.1),
                             ("10x drop right after the facts", peak, np.full(after, peak / 10), 0.1)]:
        r = dict(peak=pk, width=width, wd=wd, min_lr_frac=mf, steps=steps, layers=layers)
        S = float(lr.sum())
        out.append((float(pred(np.array(FROZEN), np.array([feats(r)]), np.array([math.log(S)]))[0]), S, name))
    return sorted(out, reverse=True)


print("\n  example menu: width 512, 8 layers, 2,048 steps, peak 8e-3, weight decay 0.1, 1,024 steps after the facts")
for k, S, name in menu(512, 8, 2048, 8e-3, 0.1, 1024):
    print(f"    {name:32s} predicted kept {100 * k:5.1f}%   S {S:.3f}")
print("  A lower S always predicts more kept, so compare schedules that still train the model well.")
