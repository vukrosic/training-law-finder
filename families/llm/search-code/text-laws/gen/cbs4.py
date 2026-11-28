"""cbs4: Muon loss vs batch at more data sizes, to test the big-batch penalty law
P = L(B128) - min_B L(B) = c * w^a * D^-b (fit on 256k and 1M tokens in fits/bigbatch_penalty.py).
64k tokens for all widths (fast); 2M for d64/d128; 4M for d32. Same lr rule as cbs2."""
import json, math
C = {32: 0.014, 64: 0.0075, 128: 0.0075, 256: 0.005}
SHAPE = {32: 1, 64: 2, 128: 2, 256: 2}
DEV = {256: "mps", 128: "mps", 64: "cpu", 32: "cpu"}
plan = [("cbst", 65_536, d, B) for d in (32, 64, 128, 256) for B in (2, 4, 8, 16, 32, 64, 128)]
plan += [("cbsx", 4_194_304, 32, B) for B in (8, 16, 32, 64, 128)]
plan += [("cbsh", 2_097_152, 64, B) for B in (16, 32, 64, 128)]
plan += [("cbsh", 2_097_152, 128, B) for B in (16, 32, 64, 128)]
jobs = {"mps": [], "cpu": []}
for pre, D, d, B in plan:
    L = SHAPE[d]; name = f"{pre}_muon_d{d}L{L}_B{B}"
    c = C[d] * math.sqrt(B / 32)
    jobs[DEV[d]].append(dict(pack=name, opt="muon", d=d, L=L, seq=64, batch=B, steps=D // (B * 64), warm=0.05,
                             members=[dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in (0.7, 1.4, 2.8)]))
for k, v in jobs.items():
    open(f"jobs/cbs4_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
