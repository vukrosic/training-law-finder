"""spec1: one dial, the spectral power p of the update U S^p V^T (update size fixed; p=0 exact Muon,
p=1 normalized momentum). d64L2, batch 16, steps 64/256/1024/4096 (64k..4M tokens). 5 lrs x0.25..4
around the Muon lr law. Question: does the best p, and the gain of p=0 over p=1, depend on step count?"""
import json
def mlaw(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2**20) ** -0.27
jobs = []
for D in (65_536, 262_144, 1_048_576, 4_194_304):
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        B = 16; c = mlaw(B, 64, D); name = f"spec1_p{p}_t{D}_d64L2_B16"
        mem = [dict(tag=f"{name}_lr{m}", lr=round(c * m, 7)) for m in (0.25, 0.5, 1, 2, 4)]
        groups = [[x] for x in mem] if D >= 4_194_304 else [mem]
        for i, g in enumerate(groups):
            jobs.append(dict(pack=f"{name}_{i}", opt="muon", d=64, L=2, seq=64, batch=B, steps=D // (B * 64),
                             warm=0.05, pexp=p, members=g))
open("jobs/spec1_cpu.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in jobs))
print(len(jobs))
