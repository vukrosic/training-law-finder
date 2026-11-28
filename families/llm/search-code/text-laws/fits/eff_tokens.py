"""Big-batch side (B >= best B per cell) as wasted data: L = E[w] + A[w] * De^-beta,  De = D / (1 + Bt/Bn),
Bn = exp(b0) * (w/64)^g * (D/1M)^h  (tokens/step where half of each step's tokens are wasted).  E, A per (width, depth); beta shared.
Prints per-cell residuals at B128 and held-out-width error."""
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
        if B >= Bb and (c[0] > 65536 or "--with64k" in sys.argv): P.append((c, B, v))
ws = sorted({(p[0][1], p[0][2]) for p in P}); wi = np.array([ws.index((p[0][1], p[0][2])) for p in P])
D = np.array([p[0][0] for p in P], float); w = np.array([p[0][1] for p in P]) / 64
Bt = np.array([p[1] for p in P], float) * 64; y = np.array([p[2] for p in P]); nw = len(ws)
def pred(p, m):
    b0, g, h, lbeta = p[:4]; E = p[4:4 + nw]; lA = p[4 + nw:]
    Bn = np.exp(b0) * w[m]**g * (D[m] / 2**20)**h
    De = D[m] / (1 + Bt[m] / Bn)
    return E[wi[m]] + np.exp(lA[wi[m]]) * (De / 2**20)**-np.exp(lbeta)
allm = np.ones(len(y), bool)
x0 = np.r_[math.log(2000), 0, .3, math.log(.3), [3.2] * nw, [math.log(.8)] * nw]
r = least_squares(lambda p: pred(p, allm) - y, x0, max_nfev=100000); p = r.x; e = r.fun
print(f"points={len(y)} widths={nw} rms={np.sqrt(np.mean(e**2)):.4f}  beta={math.exp(p[3]):.3f}")
print(f"  B_noise (tokens/step) = {math.exp(p[0]):.0f} * (w/64)^{p[1]:.2f} * (D/1M)^{p[2]:.2f}   -> in sequences of 64: {math.exp(p[0])/64:.1f}")
for i, c in enumerate(sorted(set(p_[0] for p_ in P))):
    m = np.array([p_[0] == c for p_ in P]); o = np.argsort(Bt[m])
    print(f"   D={c[0]>>10:5d}k d{c[1]}L{c[2]}: " + " ".join(f"B{int(b/64)}:{v:+.3f}" for b, v in zip(Bt[m][o], e[m][o])))
