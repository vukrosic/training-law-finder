import json,collections,glob,sys
fam=sys.argv[1:] or ["lr","lr2"]
R=[json.loads(l) for f in fam for g in [f"out/{f}.jsonl",f"out/{f}_c.jsonl"] for l in open(g)]
T=collections.defaultdict(dict)
for r in R: T[(r["opt"],r["d"],r["L"])][r["lr"]]=r["val"]
for k in sorted(T):
    best=min((v,l) for l,v in T[k].items() if v)
    print(*k,"best lr",best[1],"val",best[0],"|"," ".join(f"{l}:{T[k][l]:.3f}" for l in sorted(T[k]) if T[k][l]))
