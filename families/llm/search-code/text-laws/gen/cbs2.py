"""cbs2: loss vs batch at fixed tokens, two data sizes (256k and 1M tokens), widths 32-256, Muon then Adam.
3 learning rates per point (x0.7, x1.4, x2.8 of a center growing as sqrt(batch/32)); best is kept.
Rows from cbs (1M tokens, batch 4/16/32, 4 lrs) join the same table."""
import json, math
C = {("muon", 32): 0.014, ("muon", 64): 0.0075, ("muon", 128): 0.0075, ("muon", 256): 0.005,
     ("adam", 32): 0.003, ("adam", 64): 0.002, ("adam", 128): 0.0015, ("adam", 256): 0.001}
SHAPE = {32: 1, 64: 2, 128: 2, 256: 2}
DEV = {256: "mps", 128: "mps", 64: "cpu", 32: "cpu"}
done = {"cbs_muon_d%dL%d_B%d" % (d, SHAPE[d], b) for d in (32, 64, 128, 256) for b in (16, 32)} | {"cbs_muon_d32L1_B4", "cbs_muon_d256L2_B4"}
jobs = {"mps": [], "cpu": []}
plan = []
for opt in ["muon", "adam"]:
    plan += [(opt, 262_144, B) for B in (4, 8, 16, 32, 64, 128)]
    plan += [(opt, 1_048_576, B) for B in (8, 16, 32, 64, 128)]
for opt, D, B in plan:
    for d in (32, 64, 128, 256):
        L = SHAPE[d]
        name = ("cbs" if D == 1_048_576 else "cbsq") + f"_{opt}_d{d}L{L}_B{B}"
        if name in done:
            continue
        c = C[(opt, d)] * math.sqrt(B / 32)
        jobs[DEV[d]].append(dict(pack=name, opt=opt, d=d, L=L, seq=64, batch=B, steps=D // (B * 64), warm=0.05,
                                 members=[dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in (0.7, 1.4, 2.8)]))
for k, v in jobs.items():
    open(f"jobs/cbs2_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
