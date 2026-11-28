"""rep: fixed 2.05M training tokens, unique data U from 2M down to 16k (1..128 epochs).
Each unique chunk starts at a different offset so models never share text."""
import json
BEST = {("muon", 32): 0.014, ("muon", 64): 0.0075, ("muon", 128): 0.0075,
        ("adam", 32): 0.003, ("adam", 64): 0.002, ("adam", 128): 0.0015}
Us = [2_048_000, 1_024_000, 512_000, 256_000, 128_000, 64_000, 32_000, 16_000]
mps, cpu = [], []
for opt in ["muon", "adam"]:
    for d, L, dest in [(64, 2, mps), (128, 2, mps), (32, 1, cpu)]:
        dest.append(dict(pack=f"rep_{opt}_d{d}L{L}", opt=opt, d=d, L=L, seq=64, batch=32, steps=1000,
                         members=[dict(tag=f"rep_{opt}_d{d}L{L}_U{U}", lr=BEST[(opt, d)], U=U,
                                       off=i * 2_100_000) for i, U in enumerate(Us)]))
open("jobs/rep_mps.jsonl", "w").write("\n".join(json.dumps(j) for j in mps) + "\n")
open("jobs/rep_cpu.jsonl", "w").write("\n".join(json.dumps(j) for j in cpu) + "\n")
