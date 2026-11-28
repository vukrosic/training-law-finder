"""Big-batch side: does the excess loss L(B) - Lmin collapse on B/B* (batch rule) or on steps S = D/(B*seq) with a
width-dependent scale (steps rule)? Fit both on Muon points with B > best B, compare rms."""
import json, glob, math, numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
opt = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "muon"
best = defaultdict(lambda: 9e9)
for f in sorted(glob.glob("out/cbs*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["val"] is None or r["opt"] != opt: continue
        k = (r["tokens"], r["d"], r["batch"]); best[k] = min(best[k], r["val"])
cells = defaultdict(dict)
for (D, d, B), v in best.items(): cells[(D, d)][B] = v
P = []
for (D, d), c in cells.items():
    if len(c) < 4: continue
    Bb = min(c, key=c.get); Lm = c[Bb]
    for B, v in c.items():
        if B > Bb: P.append((D, d, B, Bb, v - Lm))
D, w, B, Bb, y = (np.array(z, float) for z in zip(*P)); w = w / 64; S = D / (B * 64)
def batch_rule(p): return np.exp(p[0]) * (np.log(B / Bb)) ** p[1] * w ** p[2] * (D / 2**20) ** p[3]
def steps_rule(p): return np.exp(p[0]) * w ** p[1] * S ** -p[2] - np.exp(p[0]) * w ** p[1] * (D / (Bb * 64)) ** -p[2]
for name, fn, x0 in [("batch rule  a*ln(B/B_best)^k*w^e*D^f", batch_rule, [-2, 1.5, .3, -.3]), ("steps rule  a*w^e*(S^-k - S_best^-k)", steps_rule, [0, .3, .5])]:
    r = least_squares(lambda p: fn(p) - y, x0, max_nfev=20000)
    print(f"{opt} {name}: n={len(y)} rms={np.sqrt(np.mean(r.fun**2)):.4f}  params={np.round(r.x, 3)}")
