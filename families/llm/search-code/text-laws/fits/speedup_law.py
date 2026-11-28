"""Muon-vs-Adam token speedup S(width, tokens) from best cells (min val over batch and lr).
S = Adam tokens needed to reach Muon's loss / Muon tokens (interpolated in log tokens; '*' = extrapolated
from the last two Adam points). Then: fit log S = c + a log(w/64) + b log(D/1M), and compare two joint
loss laws: optimizer shifts only the data constant (shared exponent) vs optimizer has its own exponent."""
import json, glob, math, sys
import numpy as np
from scipy.optimize import least_squares
B = {}
for f in glob.glob("out/cbs*.jsonl") + glob.glob("out/spd*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("val") is None: continue
        k = (r["opt"], r["d"], r["L"], r["tokens"])
        if k not in B or r["val"] < B[k]: B[k] = r["val"]
rows = []
for (o, d, L, D), v in sorted(B.items()):
    if o != "muon": continue
    A = sorted((t, vv) for (oo, dd, LL, t), vv in B.items() if oo == "adam" and dd == d and LL == L)
    if len(A) < 2: continue
    x = [math.log(t) for t, _ in A]; y = [vv for _, vv in A]
    ext = not (min(y) <= v <= max(y))
    if v < min(y): i = len(A) - 2
    elif v > max(y): i = 0
    else: i = max(j for j in range(len(A) - 1) if (y[j] - v) * (y[j + 1] - v) <= 0)
    lt = x[i] + (v - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    S = math.exp(lt) / D
    rows.append((d, L, D, S, ext))
print("width L  tokens    S")
for d, L, D, S, e in rows: print(f"d{d:<4d} L{L} {D:9d} {S:5.2f}{'*' if e else ''}")
R = [r for r in rows if not r[4] or "--ext" in sys.argv]
X = np.array([[1, math.log(d / 64), math.log(D / 2**20)] for d, L, D, S, e in R]); y = np.log([r[3] for r in R])
c, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"\nlog-S fit on {len(R)} cells: S = {math.exp(c[0]):.2f} * (w/64)^{c[1]:.2f} * (D/1M)^{c[2]:.2f}   rms(logS) {np.sqrt(np.mean((X@c-y)**2)):.3f}")
# joint laws over all (opt, d, L, D) cells with L=2 or Adam/Muon d32L1
cells = [(o, d, L, D, v) for (o, d, L, D), v in B.items() if D >= 262_144]
def pred(p, shared):
    E, lA, a, lB, b, sh, db = p
    out = []
    for o, d, L, D, v in cells:
        m = 1.0 if o == "muon" else 0.0
        bb = b if shared else b + m * db
        out.append(E + math.exp(lA) * (d / 64) ** -a + math.exp(lB) * (D / 2**20 * math.exp(m * sh)) ** -bb)
    return np.array(out)
v = np.array([c[4] for c in cells])
for shared in (True, False):
    f = least_squares(lambda p: pred(p, shared) - v, [3, 0, 0.5, 0, 0.3, 0.3, 0.0], max_nfev=20000)
    p = f.x; rms = np.sqrt(np.mean(f.fun ** 2))
    print(f"{'data-constant shift' if shared else 'own exponent      '}: rms {rms:.4f}  E {p[0]:.2f} a {p[2]:.2f} b {p[4]:.2f}"
          f" muon shift x{math.exp(p[5]):.2f}" + ("" if shared else f" muon d_exponent {p[6]:+.3f}"))
