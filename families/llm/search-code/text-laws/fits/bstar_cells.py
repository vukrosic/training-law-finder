"""Per-cell optimal batch: for each (opt, D, w) with >=4 batches, fit L = Lm + p*(ln B - ln B*)^2 to the 4 lowest points
(B* kept inside the tried range), then regress ln B* and ln p on ln w, ln D."""
import json, glob, math, sys, numpy as np
from collections import defaultdict
best = defaultdict(lambda: 9e9)
for f in sorted(glob.glob("out/cbs*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["val"] is None: continue
        k = (r["opt"], r["tokens"], r["d"], r["batch"]); best[k] = min(best[k], r["val"])
cells = defaultdict(list)
for (o, D, d, B), v in best.items(): cells[(o, D, d)].append((B, v))
res = defaultdict(list)
for (o, D, d), pts in sorted(cells.items()):
    if len(pts) < 4: continue
    pts.sort(key=lambda t: t[1]); use = sorted(pts[:4])
    x = np.log([b for b, _ in use]); y = np.array([v for _, v in use])
    c2, c1, c0 = np.polyfit(x, y, 2)
    if c2 <= 0: lbs = x[np.argmin(y)]; c2 = float("nan")
    else: lbs = min(max(-c1 / (2 * c2), math.log(min(b for b, _ in pts))), math.log(max(b for b, _ in pts)))
    res[o].append((D, d, lbs, c2))
    print(f"{o} D={D//1024:5d}k d{d:<4d} B*={math.exp(lbs):6.1f}  curv={c2:.4f}  (B tried {min(b for b,_ in pts)}-{max(b for b,_ in pts)})")
for o, rows in res.items():
    X = np.array([[1, math.log(d / 64), math.log(D / 2**20)] for D, d, _, _ in rows]); yb = np.array([r[2] for r in rows])
    cb = np.linalg.lstsq(X, yb, rcond=None)[0]; eb = np.sqrt(np.mean((X @ cb - yb) ** 2))
    m = np.array([np.isfinite(r[3]) for r in rows]); yc = np.log([r[3] for r in rows if np.isfinite(r[3])])
    cc = np.linalg.lstsq(X[m], yc, rcond=None)[0]
    print(f"{o}: n={len(rows)}  B* = {math.exp(cb[0]):.1f} * (w/64)^{cb[1]:.2f} * (D/1M)^{cb[2]:.2f}  (rms ln {eb:.2f})   curvature = {math.exp(cc[0]):.4f} * (w/64)^{cc[1]:.2f} * (D/1M)^{cc[2]:.2f}")
