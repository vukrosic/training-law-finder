"""Best learning rate per (opt, D, width, depth, batch) from the lr members of the cbs grids: quadratic in ln lr through
the best point and its two neighbours (interior minima only). Then ln lr* = c0 + cB ln B + cw ln w + cD ln D."""
import json, glob, math, sys, numpy as np
from collections import defaultdict
pts = defaultdict(dict)
for f in sorted(glob.glob("out/cbs*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["val"] is None or not math.isfinite(r["val"]): continue
        k = (r["opt"], r["tokens"], r["d"], r["L"], r["batch"]); lr = r["lr"]
        pts[k][lr] = min(pts[k].get(lr, 9e9), r["val"])
for opt in ("muon", "adam"):
    rows = []
    for k, d in pts.items():
        if k[0] != opt or len(d) < 3: continue
        lrs = sorted(d); v = [d[x] for x in lrs]; i = int(np.argmin(v))
        if i == 0 or i == len(lrs) - 1: continue
        x = np.log(lrs[i - 1:i + 2]); y = np.array(v[i - 1:i + 2]); c2, c1, _ = np.polyfit(x, y, 2)
        if c2 <= 0: continue
        rows.append((k, -c1 / (2 * c2), c2))
    X = np.array([[1, math.log(k[4]), math.log(k[2] / 64), math.log(k[1] / 2**20)] for k, _, _ in rows])
    y = np.array([r[1] for r in rows]); c = np.linalg.lstsq(X, y, rcond=None)[0]; e = np.sqrt(np.mean((X @ c - y) ** 2))
    yc = np.log([r[2] for r in rows]); cc = np.linalg.lstsq(X, yc, rcond=None)[0]; ec = np.sqrt(np.mean((X @ cc - yc) ** 2))
    print(f"{opt}: n={len(rows)}  lr* = {math.exp(c[0]):.5f} * B^{c[1]:.2f} * (w/64)^{c[2]:.2f} * (D/1M)^{c[3]:.2f}   rms ln {e:.2f}")
    print(f"      bowl curvature (loss per (ln lr)^2) = {math.exp(cc[0]):.4f} * B^{cc[1]:.2f} * (w/64)^{cc[2]:.2f} * (D/1M)^{cc[3]:.2f}   rms ln {ec:.2f}")
