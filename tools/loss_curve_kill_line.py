"""Compute the two kill-line tests of prereg/loss-curve.md on testbed result lines.

    python3 tools/loss_curve_kill_line.py work/loss-curve/small.jsonl [--long work/loss-curve/results_long.jsonl]

Test 1 (collapse): per (width, layers) with 6+ runs, one decreasing curve f(tau) fit by isotonic regression
to all 32 readings of all its runs, tau = sum over steps of (peak * lr_mult)^p, one p shared by all shapes,
searched over 0.5-2. Dead if the median RMSE over shapes is above 5 x floor (stand-in floor 0.0019).

Test 2 (matched clock): per width, WSD vs const at the same peak and step count. For each WSD reading after
80% of its decay steps, take the const reading (same width, peak, steps) with the closest lr area; keep it if
the areas are within 2%. At equal peak, tau is equal exactly when lr area is equal, for every p. Gap = const
loss minus WSD loss. A width counts against the guess if the median gap over its kept readings is above
0.02 nats. Dead if 4 or more of the 6 widths count (8 of 12 in the original grid).
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "problems", "loss-curve"))
from problem import lr_mult  # noqa: E402

FLOOR = 0.0019


def load(path):
    out = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("diverged") or "lc" not in r:
            continue
        c = r["config"]
        T = c["steps"]
        w = round(c["warmup"] * T)
        m = np.array([lr_mult(s, T, w, c["sched"], c["decay"], c["min_lr_frac"]) for s in range(T)])
        out.append({"tag": r["tag"], "width": c["d"], "layers": c["layers"], "steps": T, "sched": c["sched"],
                    "peak": c["lr"], "decay": c["decay"], "mlf": c["min_lr_frac"], "m": m,
                    "rstep": np.array(r["lc"]["step"]), "loss": np.array(r["lc"]["val_loss"])})
    return out


def isotonic_dec(x, y):
    """Least-squares non-increasing fit of y on x. Returns fitted values in the input order."""
    o = np.argsort(x, kind="stable")
    ys = list(-y[o])  # fit non-decreasing on -y
    vals, wts, lens = [], [], []
    for v in ys:
        vals.append(v); wts.append(1.0); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2); wts[-1] += w2; lens[-1] += l2
    fit = np.repeat(vals, lens)
    out = np.empty_like(fit)
    out[o] = -fit
    return out


def test1(runs, ps=np.round(np.arange(0.5, 2.0001, 0.05), 3)):
    shapes = defaultdict(list)
    for r in runs:
        shapes[(r["width"], r["layers"])].append(r)
    shapes = {k: v for k, v in shapes.items() if len(v) >= 6}
    best = None
    for p in ps:
        res = {}
        for k, rs in shapes.items():
            tau, y = [], []
            for r in rs:
                c = np.cumsum((r["peak"] * r["m"]) ** p)
                tau.append(c[r["rstep"] - 1]); y.append(r["loss"])
            tau, y = np.concatenate(tau), np.concatenate(y)
            res[k] = float(np.sqrt(np.mean((isotonic_dec(tau, y) - y) ** 2)))
        med = float(np.median(list(res.values())))
        if best is None or med < best[1]:
            best = (float(p), med, res)
    return best


def test2(runs):
    const = defaultdict(list)
    for r in runs:
        if r["sched"] == "const":
            const[(r["width"], r["layers"], r["peak"], r["steps"])].append(r)
    gaps = defaultdict(list)
    for r in runs:
        if r["sched"] != "wsd" or r["decay"] != 0.2 or r["mlf"] != 0.0:
            continue
        cs = const.get((r["width"], r["layers"], r["peak"], r["steps"]))
        if not cs:
            continue
        T = r["steps"]
        d0 = T - int(round(r["decay"] * T))
        late = d0 + 0.8 * (T - d0)
        area = np.cumsum(r["m"])
        c = cs[0]
        carea = np.cumsum(c["m"])[c["rstep"] - 1]
        for st, y in zip(r["rstep"], r["loss"]):
            if st < late:
                continue
            a = area[st - 1]
            j = int(np.argmin(np.abs(carea - a)))
            if abs(carea[j] - a) / a <= 0.02:
                gaps[(r["width"], r["layers"])].append((float(c["loss"][j] - y), r["tag"]))
    return gaps


def main():
    args = sys.argv[1:]
    long_path = None
    if "--long" in args:
        i = args.index("--long"); long_path = args[i + 1]; del args[i:i + 2]
    runs = load(args[0])
    print(f"{len(runs)} runs")
    p, med, res = test1(runs)
    print(f"Test 1: best p = {p}, median RMSE over shapes = {med:.4f} (kill above {5 * FLOOR:.4f}) -> "
          f"{'DEAD' if med > 5 * FLOOR else 'alive'}")
    for k in sorted(res):
        print(f"  width {k[0]:>3} layers {k[1]}: RMSE {res[k]:.4f}")
    for name, rs in [("small grid", runs)] + ([("long runs", load(long_path))] if long_path else []):
        g = test2(rs)
        main_l = max(set(r["layers"] for r in rs), key=[r["layers"] for r in rs].count)
        widths = sorted(k for k in g if k[1] == main_l)
        if not widths:
            print(f"Test 2 ({name}): no matched pairs"); continue
        bad = 0
        print(f"Test 2 ({name}): const minus WSD at matched lr area, late in decay")
        for k in widths:
            med_g = float(np.median([x for x, _ in g[k]]))
            bad += med_g > 0.02
            print(f"  width {k[0]:>3}: median gap {med_g:+.4f} nats over {len(g[k])} readings, "
                  f"min {min(x for x, _ in g[k]):+.4f}, max {max(x for x, _ in g[k]):+.4f}")
        need = math.ceil(len(widths) * 8 / 12)
        print(f"  {bad} of {len(widths)} widths above 0.02 (dead at {need}+) -> {'DEAD' if bad >= need else 'alive'}")


if __name__ == "__main__":
    main()
