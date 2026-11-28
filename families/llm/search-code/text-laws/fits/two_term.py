"""Whole loss-vs-batch curve as two terms on top of a free per-cell floor F[D,w]:
L = F[D,w] + a*(w/64)^e*S^-k  (too few steps)  +  n*(w/64)^u*(D/1M)^h*(B*seq)^-m  (too noisy steps)
S = D/(B*seq). Held-out: refit without one cell, predict its curve shape (its floor from its points' mean)."""
import json, glob, math, sys, numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
opt = sys.argv[1] if len(sys.argv) > 1 else "muon"
best = defaultdict(lambda: 9e9)
for f in sorted(glob.glob("out/cbs*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["val"] is None or r["opt"] != opt: continue
        if r["L"] == 1 and "--noL1" in sys.argv: continue
        k = (r["tokens"], r["d"], r["L"], r["batch"]); best[k] = min(best[k], r["val"])
cnt = defaultdict(int)
for k in best: cnt[k[:3]] += 1
cells = sorted(c for c in cnt if cnt[c] >= 4)
K = [k for k in best if k[:3] in cells]
ci = np.array([cells.index(k[:3]) for k in K]); D = np.array([k[0] for k in K], float); w = np.array([k[1] for k in K]) / 64
Bt = np.array([k[3] for k in K]) * 64.0; y = np.array([best[k] for k in K]); S = D / Bt; nc = len(cells)
def shape(p, ci, D, w, Bt, S):
    la, e, k, ln, u, h, m = p[:7]
    return np.exp(la) * w**e * S**-k + np.exp(ln) * w**u * (D / 2**20)**h * Bt**-m
def resid(p, m): return p[7:][ci[m]] + shape(p, ci[m], D[m], w[m], Bt[m], S[m]) - y[m]
x0 = np.r_[1, .4, .77, 2, 0, 0, .5, [y[ci == i].min() - .05 for i in range(nc)]]
allm = np.ones(len(y), bool)
r = least_squares(lambda p: resid(p, allm), x0, max_nfev=100000); p = r.x
print(f"{opt}: cells={nc} points={len(y)} rms={np.sqrt(np.mean(r.fun**2)):.4f}")
print(f"  steps term = {math.exp(p[0]):.3f} * (w/64)^{p[1]:.2f} * S^-{p[2]:.2f}")
print(f"  noise term = {math.exp(p[3]):.3f} * (w/64)^{p[4]:.2f} * (D/1M)^{p[5]:.2f} * (tokens/step)^-{p[6]:.2f}")
# implied best tokens/step: d/dB [a w^e (D/Bt)^-k + n w^u D^h Bt^-m] = 0 -> Bt* = (m n w^u D'^h / (k a w^e D^-k))^(1/(k+m))
for d in (32, 128, 512):
    row = []
    for Dk in (65536, 1 << 20, 1 << 22):
        wv = d / 64; a = math.exp(p[0]) * wv**p[1]; n = math.exp(p[3]) * wv**p[4] * (Dk / 2**20)**p[5]
        bt = (p[6] * n / (p[2] * a * Dk**-p[2])) ** (1 / (p[2] + p[6])); row.append(f"D={Dk>>10}k: B*={bt/64:6.1f}")
    print(f"  d{d}: " + "  ".join(row))
errs = []
for i in range(nc):
    m = ci != i; q = least_squares(lambda z: resid(z, m), p, max_nfev=50000).x
    sh = shape(q, ci[~m], D[~m], w[~m], Bt[~m], S[~m]); e = sh + np.mean(y[~m] - sh) - y[~m]; errs.append(np.sqrt(np.mean(e**2)))
print(f"  held-out-cell rms: mean {np.mean(errs):.4f} max {np.max(errs):.4f} (worst {cells[int(np.argmax(errs))]})")

for wd in sorted(set(w)):
    m = w != wd; q = least_squares(lambda z: resid(z, m), p, max_nfev=50000).x
    es = []
    for i in set(ci[~m]):
        mm = (~m) & (ci == i); sh = shape(q, ci[mm], D[mm], w[mm], Bt[mm], S[mm]); es.append(np.sqrt(np.mean((sh + np.mean(y[mm] - sh) - y[mm])**2)))
    print(f"  hold out width {wd*64:.0f}: cells={len(es)} rms={np.mean(es):.4f}  steps term w^{q[1]:.2f} S^-{q[2]:.2f}")
