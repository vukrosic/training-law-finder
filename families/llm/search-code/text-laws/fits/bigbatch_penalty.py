"""Big-batch penalty law: P = L(B128) - min_B L(B), fit P = c * (w/64)^a * (D/1M)^-b per optimizer.
Data: out/cbs*.jsonl via show_cbs best-over-lr table."""
import json, math, numpy as np
from collections import defaultdict
best = defaultdict(lambda: 9e9)
for f in ["out/cbs.jsonl","out/cbs2.jsonl","out/cbs2_c.jsonl","out/cbs3.jsonl","out/cbs3_c.jsonl"]:
    try:
        for l in open(f):
            r = json.loads(l); v = r.get("val", r.get("loss"))
            if v is None or not math.isfinite(v): continue
            k = (r["opt"], r["tokens"], r["d"], r["batch"]); best[k] = min(best[k], v)
    except FileNotFoundError: pass
for opt in ["muon","adam"]:
    X, y, rows = [], [], []
    for (o,D,d,B), v in best.items():
        if o!=opt or B!=128: continue
        m = min(vv for (o2,D2,d2,B2),vv in best.items() if (o2,D2,d2)==(o,D,d))
        P = v - m; X.append([1, math.log(d/64), math.log(D/2**20)]); y.append(math.log(P)); rows.append((D,d,P))
    X, y = np.array(X), np.array(y); c = np.linalg.lstsq(X, y, rcond=None)[0]; pred = X@c
    print(f"{opt}: n={len(y)}  P = {math.exp(c[0]):.3f} * (w/64)^{c[1]:.2f} * (D/1M)^{c[2]:.2f}   rms log err {np.sqrt(np.mean((y-pred)**2)):.2f}")
    for (D,d,P),p in zip(rows,pred): print(f"   D={D//1024}k d{d}: P={P:.3f} fit={math.exp(p):.3f}")
