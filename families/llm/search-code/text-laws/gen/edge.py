"""edge.py OUTNAME OPT SRC...: extend the lr grid of every pack (merged by tokens, width, batch) whose best lr is at the
edge of the multipliers tried so far. Top edge -> next two x2 steps up; bottom -> next two /2 steps down."""
import json, sys, collections
out, opt, srcs = sys.argv[1], sys.argv[2], sys.argv[3:]
DEV = {512: "mps", 256: "mps", 128: "mps", 64: "cpu", 48: "cpu", 32: "cpu"}
best, tried, proto = {}, collections.defaultdict(set), {}
for f in srcs:
    try: lines = open(f).readlines()
    except FileNotFoundError: continue
    for l in lines:
        r = json.loads(l)
        if r["opt"] != opt: continue
        k = (r["tokens"], r["d"], r["batch"], r["L"]); m = float(r["tag"].split("_lr")[-1])
        c = r["lr"] / m; tried[k].add(m); proto.setdefault(k, (r, c))
        if r["val"] is not None and (k not in best or r["val"] < best[k][0]): best[k] = (r["val"], m)
jobs = {"mps": [], "cpu": []}
for k, (v, m) in sorted(best.items()):
    r, c = proto[k]
    if m == max(tried[k]): mult = (round(m * 2, 4), round(m * 4, 4))
    elif m == min(tried[k]): mult = (round(m / 2, 4), round(m / 4, 4))
    else: continue
    name = f"{out}_{opt}_t{k[0]}_d{k[1]}L{k[3]}_B{k[2]}"
    mem = [dict(tag=f"{name}_lr{x}", lr=round(c * x, 7)) for x in mult]
    packs = [mem] if k[1] < 512 else [[x] for x in mem]
    for i, ms in enumerate(packs):
        jobs[DEV[k[1]]].append(dict(pack=f"{name}_{i}", opt=opt, d=k[1], L=r["L"], seq=64, batch=k[2], steps=r["steps"], warm=0.05, members=ms))
for d, v in jobs.items():
    open(f"jobs/{out}_{d}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + ("\n" if v else ""))
print({d: len(v) for d, v in jobs.items()})
