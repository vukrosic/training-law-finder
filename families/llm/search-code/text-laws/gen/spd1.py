"""spd1: Adam cells missing for a Muon-vs-Adam speedup law S(width, tokens).
Adam lr centre per width at batch 8 (from cbs best cells), scaled sqrt(B/8), members x0.5, 1, 2.
MPS: d128/d256 at 64k, 2M, 4M; d512 at 256k, 1M, 2M (K=1 packs for d256+ and B>=64).
CPU: d64L2 at 4M; d32L2 and d48L2 at 256k, 1M, 4M (Muon has these, Adam does not)."""
import json, math
C8 = {32: 0.0045, 48: 0.0038, 64: 0.0028, 128: 0.00105, 256: 0.0007, 512: 0.0005}
BS = {65_536: (2, 4, 8), 262_144: (4, 8), 1_048_576: (8, 16), 2_097_152: (8, 16, 32), 4_194_304: (16, 32, 64)}
mps = [(d, D) for d in (128, 256) for D in (65_536, 2_097_152, 4_194_304)] + [(512, D) for D in (262_144, 1_048_576, 2_097_152)]
cpu = [(64, 4_194_304)] + [(d, D) for d in (32, 48) for D in (262_144, 1_048_576, 4_194_304)]
for dev, plan in (("mps", mps), ("cpu", cpu)):
    jobs = []
    for d, D in plan:
        bs = BS[D] if d != 512 else {262_144: (4, 8), 1_048_576: (8, 16), 2_097_152: (16, 32)}[D]
        for B in bs:
            name = f"spd1_adam_t{D}_d{d}L2_B{B}"; c = C8[d] * math.sqrt(B / 8)
            mem = [dict(tag=f"{name}_lr{m}", lr=round(c * m, 7)) for m in (0.5, 1, 2)]
            groups = [[x] for x in mem] if (d >= 256 or B >= 64) and dev == "mps" else [mem]
            for i, g in enumerate(groups):
                jobs.append(dict(pack=f"{name}_{i}", opt="adam", d=d, L=2, seq=64, batch=B,
                                 steps=D // (B * 64), warm=0.05, members=g))
    open(f"jobs/spd1_{dev}.jsonl", "w").write("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(dev, len(jobs))
