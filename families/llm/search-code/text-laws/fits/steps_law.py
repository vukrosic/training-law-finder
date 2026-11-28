"""Muon steps law on the big-batch side (B >= best B of the cell), free floor per (D, width, depth) cell:
L = F[cell] + a*(w/64)^e*S^-k,  S = D/(B*seq).  --predict W D: print predicted L(B) - L(B=8) for an unseen width."""
import json, glob, math, sys, numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
best = defaultdict(lambda: 9e9)
for f in sorted(glob.glob("out/cbs*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["val"] is None or r["opt"] != "muon": continue
        k = (r["tokens"], r["d"], r["L"], r["batch"]); best[k] = min(best[k], r["val"])
cells = defaultdict(dict)
for (D, d, L, B), v in best.items(): cells[(D, d, L)][B] = v
P = []
for c, bv in cells.items():
    if len(bv) < 4: continue
    Bb = min(bv, key=bv.get)
    for B, v in bv.items():
        if B >= Bb: P.append((c, B, v))
cl = sorted({p[0] for p in P}); ci = np.array([cl.index(p[0]) for p in P])
D = np.array([p[0][0] for p in P], float); w = np.array([p[0][1] for p in P]) / 64; B = np.array([p[1] for p in P], float)
y = np.array([p[2] for p in P]); S = D / (B * 64)
def res(p): return p[3:][ci] + np.exp(p[0]) * w**p[1] * S**-p[2] - y
r = least_squares(res, np.r_[1, .4, .7, [y[ci == i].min() for i in range(len(cl))]], max_nfev=50000); p = r.x
print(f"cells={len(cl)} points={len(y)} rms={np.sqrt(np.mean(r.fun**2)):.4f}  steps term = {math.exp(p[0]):.3f} * (w/64)^{p[1]:.3f} * S^-{p[2]:.3f}")
if "--predict" in sys.argv:
    W, Dp = float(sys.argv[-2]), float(sys.argv[-1])
    f = lambda Bq: math.exp(p[0]) * (W / 64)**p[1] * (Dp / (Bq * 64))**-p[2]
    print(json.dumps({"width": W, "tokens": Dp, "pred_minus_B16": {B: round(f(B) - f(16), 4) for B in (16, 32, 64, 128)},
                      "law": [round(math.exp(p[0]), 4), round(p[1], 4), round(p[2], 4)]}))
if "--resid" in sys.argv:
    e = res(p)
    for i, c in enumerate(cl):
        m = ci == i; o = np.argsort(B[m])
        print(f"D={c[0]>>10:5d}k d{c[1]}L{c[2]}: " + " ".join(f"B{int(b)}:{v:+.3f}" for b, v in zip(B[m][o], e[m][o])))
