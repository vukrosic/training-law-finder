"""cbs11: 8M tokens (S up to 4096 steps) to test B* growth with data and the mined lr law
lr* = 0.00218 * B^0.48 * (w/64)^-0.54 * (D/1M)^-0.27 (fits/lr_star.py, 148 cells). Members x0.6, x1, x1.6 of that prediction.
CPU: d32L2, d48L2. MPS: d128L2 (3 per pack), d256L2 (1 per pack, memory/time)."""
import json, math
D = 8_388_608
pred = lambda B, w: 0.00218 * B**0.48 * (w / 64)**-0.54 * (D / 2**20)**-0.27
jobs = {"mps": [], "cpu": []}
def add(dev, d, B, split):
    name = f"cbs8m_muon_d{d}L2_B{B}"; c = pred(B, d)
    mem = [dict(tag=f"{name}_lr{m}", lr=round(c * m, 7)) for m in (0.6, 1, 1.6)]
    for i, ms in enumerate([[x] for x in mem] if split else [mem]):
        jobs[dev].append(dict(pack=name + (f"_{i}" if split else ""), opt="muon", d=d, L=2, seq=64, batch=B, steps=D // (B * 64), warm=0.05, members=ms))
for B in (256, 128, 64, 32, 16): add("cpu", 32, B, False)
for B in (256, 128, 64, 32): add("cpu", 48, B, False)
for B in (256, 128, 64, 32): add("mps", 128, B, B <= 32)
for B in (256, 128, 64): add("mps", 256, B, True)
for k, v in jobs.items(): open(f"jobs/cbs11_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
