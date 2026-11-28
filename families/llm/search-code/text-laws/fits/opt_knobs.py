"""Best val per knob (over 3 lrs) vs baseline, and equivalent token speedup from baseline at 0.5/1/2M."""
import json, glob, math, sys
pre = sys.argv[1] if len(sys.argv) > 1 else "opt1"
R = {}
for f in glob.glob(f"out/{pre}*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("val") is None: continue
        k = (r["d"], r["knob"], r["tokens"]); lrm = r["tag"].split("_lr")[-1]
        if k not in R or r["val"] < R[k][0]: R[k] = (r["val"], lrm)
for d in sorted({k[0] for k in R}):
    T = sorted(t for (dd, kn, t) in R if dd == d and kn == "base")
    base = [(math.log(t), R[(d, "base", t)][0]) for t in T]
    def tok_equiv(v):  # tokens baseline needs to reach val v (piecewise linear in log tokens)
        for (x0, y0), (x1, y1) in zip(base, base[1:]):
            if y1 <= v <= y0 or (v > y0 and x0 == base[0][0]) or (v < y1 and x1 == base[-1][0]):
                return math.exp(x0 + (v - y0) * (x1 - x0) / (y1 - y0))
    b1 = R.get((d, "base", 2**20))
    print(f"d{d}  baseline 0.5M/1M/2M:", " ".join(f"{R[(d,'base',t)][0]:.4f}" for t in T))
    for (dd, kn, t), (v, lrm) in sorted(R.items(), key=lambda x: x[1][0]):
        if dd != d or t != 2**20: continue
        sp = tok_equiv(v) / 2**20 if len(base) > 1 else float("nan")
        print(f"  {kn:7s} val {v:.4f}  d={v - b1[0]:+.4f}  best lr x{lrm:4s}  token speedup {sp:.2f}x")
