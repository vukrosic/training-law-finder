"""cbs9: test the width factor of the Muon steps term (excess = a*w^0.42*S^-0.74).
CPU: d32 at depth 2 (old d32 runs are depth 1) and d48 depth 2, at 256k / 1M / 4M tokens, B8-128, 4 lrs.
MPS: d512 at 2M (one model per pack, memory), d256 at 4M."""
import json, math
SHAPE = {32: 2, 48: 2, 256: 2, 512: 2}
C = {32: 0.01, 48: 0.009, 256: 0.005, 512: 0.0035}
PRE = {262_144: "cbsq", 1_048_576: "cbs", 2_097_152: "cbsh", 4_194_304: "cbsx"}
jobs = {"mps": [], "cpu": []}
def add(dev, D, d, B, mult, split=False):
    name = f"{PRE[D]}_muon_d{d}L{SHAPE[d]}_B{B}"; c = C[d] * math.sqrt(B / 32)
    mem = [dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in mult]
    for i, ms in enumerate([[x] for x in mem] if split else [mem]):
        jobs[dev].append(dict(pack=name + (f"_{i}" if split else ""), opt="muon", d=d, L=SHAPE[d], seq=64, batch=B,
                              steps=D // (B * 64), warm=0.05, members=ms))
for D in (262_144, 1_048_576, 4_194_304):
    for d in (32, 48):
        for B in (8, 16, 32, 64, 128): add("cpu", D, d, B, (0.5, 1, 2, 4))
for B in (16, 32, 64, 128): add("mps", 4_194_304, 256, B, (0.5, 1, 2))
for B in (16, 32, 64, 128): add("mps", 2_097_152, 512, B, (0.5, 1, 2), split=True)
for k, v in jobs.items():
    open(f"jobs/cbs9_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
