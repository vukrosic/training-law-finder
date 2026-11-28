"""spec3e: for every (d, steps, p) of spec3 whose best hidden lr or embedding lr E is at a grid edge, add a pack one
step beyond it (x2 or /2) on that axis, plus the corner. Appends to jobs/spec3_{cpu,mps}.jsonl (resume skips done packs)."""
import json, glob, collections
R = collections.defaultdict(list); proto = {}
for f in glob.glob("out/spec3*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("val") is None: continue
        t = r["tag"]; m = float(t.split("_lr")[-1]); E = float(t.split("_E")[-1].split("_")[0])
        k = (r["d"], r["steps"], r["pexp"]); R[k].append((r["val"], m, E, r["lr"] / m)); proto[k] = r
jobs = {"cpu": [], "mps": []}
for k, rows in sorted(R.items()):
    v, m, E, c = min(rows); ms = {x[1] for x in rows}; Es = {x[2] for x in rows}
    nm = [m * 2] if m == max(ms) else [m / 2] if m == min(ms) else []
    nE = [E * 2] if E == max(Es) else [E / 2] if E == min(Es) else []
    combos = [(a, E) for a in nm] + [(m, b) for b in nE] + [(a, b) for a in nm for b in nE]
    combos = [(a, b) for a, b in combos if not any(abs(x[1] - a) < 1e-9 and abs(x[2] - b) < 1e-9 for x in rows)]
    if not combos: continue
    d, st, p = k; r = proto[k]
    name = f"spec3_p{p}_t{r['tokens']}_d{d}L2_B16_edge{len(rows)}"
    mem = [dict(tag=f"{name}_E{round(b,5)}_lr{round(a,4)}", lr=round(c * a, 7), emb_mult=round(b / (c * a), 4)) for a, b in combos]
    jobs["cpu" if d == 64 else "mps"].append(dict(pack=name, opt="muon", d=d, L=2, seq=64, batch=16, steps=st, warm=0.05, pexp=p, pnorm="spec", members=mem))
for dev, J in jobs.items():
    open(f"jobs/spec3_{dev}.jsonl", "a").write("".join(json.dumps(j) + "\n" for j in J))
    print(dev, len(J), "edge packs", [j["pack"].split("_t")[0] + f" s{j['steps']}" for j in J])
