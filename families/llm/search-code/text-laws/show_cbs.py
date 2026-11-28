"""Table: best val loss (over lr) per optimizer, data size, width, batch."""
import json, glob, collections
edge = {}
tried = collections.defaultdict(set)
best = collections.defaultdict(lambda: (9e9, None))
for f in sorted(glob.glob("out/cbs*.jsonl")):
    try:
        for l in open(f):
            r = json.loads(l)
            k = (r["opt"], r["tokens"], (r["d"], r["L"]), r["batch"])
            tried[k].add(float(r["tag"].split("_lr")[-1]))
            if r["val"] is None: continue
            if r["val"] < best[k][0]: best[k] = (r["val"], r["lr"]); edge[k] = r["tag"].split("_lr")[-1]
    except FileNotFoundError: pass
Bs = [2, 4, 8, 16, 32, 64, 128]
for opt in ["muon", "adam"]:
    for D in sorted({k[1] for k in best if k[0] == opt}):
        print(f"\n{opt} D={D//1024}k   " + "".join(f"B{b:<7d}" for b in Bs) + " best B")
        for d in sorted({k[2] for k in best if k[0] == opt and k[1] == D}):
            v = [best.get((opt, D, d, b), (None,))[0] for b in Bs]
            ok = [(x, b) for x, b in zip(v, Bs) if x]
            print(f"  d{d[0]}L{d[1]:<5d}    " + "".join((f"{x:.3f}" + ("^" if (opt, D, d, b) in edge and float(edge[(opt, D, d, b)]) == max(tried[(opt, D, d, b)]) else "v" if (opt, D, d, b) in edge and float(edge[(opt, D, d, b)]) == min(tried[(opt, D, d, b)]) else " ") + "  ") if x else "  -     " for x, b in zip(v, Bs)) + (f" B{min(ok)[1]}" if ok else ""))
