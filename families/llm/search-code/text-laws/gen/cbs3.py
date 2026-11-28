"""cbs3: Adam rerun with higher learning rates (cbs2's Adam grid peaked at its top edge in 20 of 34 packs).
Same packs as cbs2 Adam, members x4 and x8 of the same center (x1.4 too where cbs2 never ran the pack)."""
import json, math
C = {32: 0.003, 64: 0.002, 128: 0.0015, 256: 0.001}
SHAPE = {32: 1, 64: 2, 128: 2, 256: 2}
DEV = {256: "mps", 128: "mps", 64: "cpu", 32: "cpu"}
have = set()
for f in ["out/cbs2.jsonl", "out/cbs2_c.jsonl"]:
    have |= {json.loads(l)["pack"] for l in open(f)}
jobs = {"mps": [], "cpu": []}
for D, pre, Bs in [(262_144, "cbsq", (4, 8, 16, 32, 64, 128)), (1_048_576, "cbs", (8, 16, 32, 64, 128))]:
    for B in Bs:
        for d in (32, 64, 128, 256):
            base = f"{pre}_adam_d{d}L{SHAPE[d]}_B{B}"
            ms = (4, 8) if base in have else (1.4, 4, 8)
            c = C[d] * math.sqrt(B / 32)
            jobs[DEV[d]].append(dict(pack=base + "_hi", opt="adam", d=d, L=SHAPE[d], seq=64, batch=B, steps=D // (B * 64),
                                     warm=0.05, members=[dict(tag=f"{base}_lr{m}", lr=round(c * m, 6)) for m in ms]))
for k, v in jobs.items():
    open(f"jobs/cbs3_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
