"""Muon batch law with a free minimum loss per (D, w) cell, so only the batch shape is tested:
L(D,w,B) = Lmin[D,w] + p(w,D) * |ln B - ln B*(w,D)|^q  (q free; q=2 is a parabola in log batch)
B* = B0 (w/64)^g_w (D/1M)^g_D,  p = p0 (w/64)^e_w (D/1M)^-e_D.  Leave-one-cell-out checks the shape."""
import json, math, sys, numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
FILES = ["out/cbs.jsonl","out/cbs2.jsonl","out/cbs2_c.jsonl","out/cbs4.jsonl","out/cbs4_c.jsonl",
         "out/cbs5.jsonl","out/cbs5_c.jsonl","out/cbs6.jsonl","out/cbs6_c.jsonl", "out/cbs7_c.jsonl", "out/cbs8.jsonl", "out/cbs8_c.jsonl"]
opt = sys.argv[1] if len(sys.argv) > 1 else "muon"
if opt == "adam": FILES += ["out/cbs3.jsonl", "out/cbs3_c.jsonl"]
best = defaultdict(lambda: 9e9)
for f in FILES:
    try:
        for l in open(f):
            r = json.loads(l)
            if r["opt"] != opt or r["val"] is None: continue
            k = (r["tokens"], r["d"], r["batch"]); best[k] = min(best[k], r["val"])
    except FileNotFoundError: pass
cells = sorted({(k[0], k[1]) for k in best if sum(1 for j in best if j[:2] == k[:2]) >= 4})
K = [k for k in best if k[:2] in cells]
ci = np.array([cells.index(k[:2]) for k in K]); D = np.array([k[0] for k in K]) / 2**20
w = np.array([k[1] for k in K]) / 64; lB = np.log([k[2] for k in K]); y = np.array([best[k] for k in K])
def f(p, ci, D, w, lB, Lm):
    lB0, gw, gD, lp0, ew, eD, q = p[:7]
    s = np.abs(lB - (lB0 + gw*np.log(w) + gD*np.log(D)))
    return Lm[ci] + np.exp(lp0 + ew*np.log(w) - eD*np.log(D)) * s**q
nc = len(cells)
def fit(m, init=None):
    x0 = init if init is not None else np.r_[math.log(16), 0, .2, math.log(.02), .5, .5, 2, [y[ci == i].min() for i in range(nc)]]
    return least_squares(lambda p: f(p, ci[m], D[m], w[m], lB[m], p[7:]) - y[m], x0, max_nfev=50000)
r = fit(np.ones(len(y), bool)); p = r.x
print(f"{opt}: cells={nc} points={len(y)} rms={np.sqrt(np.mean(r.fun**2)):.4f}  (7 shape constants + {nc} cell minima)")
print(f"  B* = {math.exp(p[0]):.1f} * (w/64)^{p[1]:.2f} * (D/1M)^{p[2]:.2f}")
print(f"  penalty = {math.exp(p[3]):.4f} * (w/64)^{p[4]:.2f} * (D/1M)^-{p[5]:.2f} * |ln(B/B*)|^{p[6]:.2f}")
# held-out cell: shape constants fit without the cell; the cell's own minimum is set from its best point only
errs = []
for i in range(nc):
    m = ci != i
    q = fit(m, p).x
    Lm = q[7:].copy(); Lm[i] = 0
    pred0 = f(q, ci[~m], D[~m], w[~m], lB[~m], Lm)
    Lm[i] = np.mean(y[~m] - pred0)          # one free number: the level
    e = f(q, ci[~m], D[~m], w[~m], lB[~m], Lm) - y[~m]
    errs.append(np.sqrt(np.mean(e**2)))
print(f"  held-out-cell shape rms: mean {np.mean(errs):.4f}  max {np.max(errs):.4f}")
