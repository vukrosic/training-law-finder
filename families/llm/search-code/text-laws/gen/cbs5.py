"""cbs5: extend the lr grid of every Muon pack (cbs2, cbs4) whose best lr sat at an edge of x0.7/x1.4/x2.8.
Top edge -> add x5.6, x11.2; bottom edge -> add x0.35, x0.175. Same pack name + "_e" so rows merge by (opt, tokens, d, batch)."""
import json, glob
DEV = {256: "mps", 128: "mps", 64: "cpu", 32: "cpu"}
best = {}
for f in ["out/cbs2.jsonl", "out/cbs2_c.jsonl", "out/cbs4.jsonl", "out/cbs4_c.jsonl"]:
    for l in open(f):
        r = json.loads(l)
        if r["opt"] != "muon" or r["val"] is None: continue
        if r["pack"] not in best or r["val"] < best[r["pack"]][0]:
            best[r["pack"]] = (r["val"], r)
done = set()
for f in ["out/cbs5.jsonl", "out/cbs5_c.jsonl"]:
    try: done |= {json.loads(l)["pack"] for l in open(f)}
    except FileNotFoundError: pass
jobs = {"mps": [], "cpu": []}
for p, (v, r) in sorted(best.items(), key=lambda x: x[1][1]["tokens"]):
    m = r["tag"].split("_lr")[-1]
    if m not in ("2.8", "0.7"): continue
    c = r["lr"] / float(m)
    mult = (5.6, 11.2) if m == "2.8" else (0.35, 0.175)
    name = p + "_e"
    jobs[DEV[r["d"]]].append(dict(pack=name, opt="muon", d=r["d"], L=r["L"], seq=64, batch=r["batch"], steps=r["steps"], warm=0.05,
                                  members=[dict(tag=f"{name}_lr{x}", lr=round(c * x, 6)) for x in mult]))
for k, v in jobs.items():
    open(f"jobs/cbs5_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
