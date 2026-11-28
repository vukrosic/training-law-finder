"""Whole Muon loss-vs-batch surface: L = E + A*(w/64)^-a + C*(D/1M)^-b + p*(ln B - ln B*)^2
B* = B0*(w/64)^g_w*(D/1M)^g_D,  p = p0*(w/64)^e_w*(D/1M)^-e_D.  Best val over lr per (D, w, B)."""
import json, math, sys, numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
FILES = ["out/cbs.jsonl","out/cbs2.jsonl","out/cbs2_c.jsonl","out/cbs4.jsonl","out/cbs4_c.jsonl",
         "out/cbs5.jsonl","out/cbs5_c.jsonl","out/cbs6.jsonl","out/cbs6_c.jsonl"]
best = defaultdict(lambda: 9e9)
for f in FILES:
    try:
        for l in open(f):
            r = json.loads(l)
            if r["opt"] != "muon" or r["val"] is None: continue
            k = (r["tokens"], r["d"], r["batch"]); best[k] = min(best[k], r["val"])
    except FileNotFoundError: pass
minD = float(sys.argv[1]) if len(sys.argv) > 1 else 0
K = [k for k in best if k[0] >= minD]
D = np.array([k[0] for k in K]) / 2**20; w = np.array([k[1] for k in K]) / 64; B = np.array([k[2] for k in K], float)
y = np.array([best[k] for k in K])
def f(p, D, w, B):
    E, A, a, Cc, b, lB0, gw, gD, lp0, ew, eD = p
    lBs = lB0 + gw*np.log(w) + gD*np.log(D)
    pen = np.exp(lp0 + ew*np.log(w) - eD*np.log(D))
    return E + A*w**-a + Cc*D**-b + pen*(np.log(B) - lBs)**2
p0 = [3, .3, .5, .3, .4, math.log(16), 0, .3, math.log(.03), .3, .3]
res = least_squares(lambda p: f(p, D, w, B) - y, p0, max_nfev=20000)
p = res.x; rms = np.sqrt(np.mean(res.fun**2))
names = "E A a C b lnB0 g_w g_D lnp0 e_w e_D".split()
print(f"n={len(y)} rms={rms:.4f}"); print("  ".join(f"{n}={v:.3f}" for n, v in zip(names, p)))
print(f"B* = {math.exp(p[5]):.1f} * (w/64)^{p[6]:.2f} * (D/1M)^{p[7]:.2f}   curvature p = {math.exp(p[8]):.4f} * (w/64)^{p[9]:.2f} * (D/1M)^-{p[10]:.2f}")
# leave-one-data-size-out
for Dh in sorted(set(D)):
    m = D != Dh
    r2 = least_squares(lambda q: f(q, D[m], w[m], B[m]) - y[m], p, max_nfev=20000)
    e = f(r2.x, D[~m], w[~m], B[~m]) - y[~m]
    print(f"  hold out D={Dh*1024:.0f}k: n={(~m).sum()} rms={np.sqrt(np.mean(e**2)):.4f}  g_w={r2.x[6]:.2f} g_D={r2.x[7]:.2f}")
