"""cbs: critical batch vs width. Fixed 1.0M training tokens (steps = D / (batch*64)),
batch 16..512 at seq 64, widths 32/64/128/256, Muon and Adam, 4 learning rates per point
(x0.35, x0.7, x1.4, x2.8 of a center that grows as sqrt(batch/32)). Best lr per point is
taken, so the batch effect is not a learning-rate miss."""
import json, math
C = {("muon", 32): 0.014, ("muon", 64): 0.0075, ("muon", 128): 0.0075, ("muon", 256): 0.005,
     ("adam", 32): 0.003, ("adam", 64): 0.002, ("adam", 128): 0.0015, ("adam", 256): 0.001}
SHAPE = {32: 1, 64: 2, 128: 2, 256: 2}
D = 1_048_576
mps, cpu = [], []
for opt in ["muon", "adam"]:
    for B in [16, 32, 64, 128, 256, 512]:
        for d, dest in [(256, mps), (128, mps), (64, cpu), (32, cpu)]:
            L, steps = SHAPE[d], D // (B * 64)
            c = C[(opt, d)] * math.sqrt(B / 32)
            name = f"cbs_{opt}_d{d}L{L}_B{B}"
            dest.append(dict(pack=name, opt=opt, d=d, L=L, seq=64, batch=B, steps=steps, warm=0.05,
                             members=[dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in (0.35, 0.7, 1.4, 2.8)]))
open("jobs/cbs_mps.jsonl", "w").write("\n".join(json.dumps(j) for j in mps) + "\n")
open("jobs/cbs_cpu.jsonl", "w").write("\n".join(json.dumps(j) for j in cpu) + "\n")
print(len(mps), len(cpu), "packs")
