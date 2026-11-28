"""Same-batch Muon/Adam token speedup S vs number of steps. Fits S = 1 + c * (steps/1000)^-g, then
checks whether width or tokens explain what is left (log-residual regression)."""
import json, glob, math, collections
import numpy as np
from scipy.optimize import curve_fit
C = {}
for f in glob.glob("out/cbs*.jsonl") + glob.glob("out/spd*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("val") is None: continue
        k = (r["opt"], r["d"], r["L"], r["batch"], r["tokens"]); C[k] = min(C.get(k, 9), r["val"])
A = collections.defaultdict(dict)
for (o, d, L, B, D), v in C.items():
    if o == "adam": A[(d, L, B)][D] = v
def teq(key, v):
    pts = sorted(A[key].items())
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 <= v <= y0: return math.exp(math.log(x0) + (v - y0) * (math.log(x1) - math.log(x0)) / (y1 - y0))
R = []
for (o, d, L, B, D), v in C.items():
    if o != "muon" or len(A.get((d, L, B), {})) < 2: continue
    t = teq((d, L, B), v)
    if t: R.append((d, L, B, D, D // (B * 64), t / D))
R = [r for r in R if r[0] >= 64 or "--all" in __import__("sys").argv]
s = np.array([r[4] for r in R], float); S = np.array([r[5] for r in R])
f = lambda x, c, g: 1 + c * (x / 1000) ** -g
(c, g), _ = curve_fit(f, s, S, p0=[0.5, 0.6])
res = np.log(S) - np.log(f(s, c, g))
print(f"{len(R)} cells (width>=64 unless --all):  S = 1 + {c:.2f} * (steps/1000)^-{g:.2f}   rms(log) {res.std():.3f}")
X = np.c_[np.ones(len(R)), np.log([r[0] / 64 for r in R]), np.log([r[3] / 2**20 for r in R])]
b, *_ = np.linalg.lstsq(X, res, rcond=None)
print(f"residual vs width exponent {b[1]:+.3f}, vs tokens exponent {b[2]:+.3f}  (0 = steps explain it)")
for n in sorted(set(s)):
    m = s == n
    print(f"  steps {int(n):5d}: S {np.round(np.sort(S[m]), 2).tolist()}  fit {f(n, c, g):.2f}")
