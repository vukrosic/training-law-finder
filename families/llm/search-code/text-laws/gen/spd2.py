"""spd2: long runs so Muon's 4M/8M cells can be read against Adam by interpolation, not extrapolation.
MPS: d128, d256 at 8M and 16M; d512 at 4M. CPU: d64 at 8M. Both optimizers.
Adam lr: best 4M lr * sqrt(B/B4) * (D/4M)^-0.25, x0.5,1,2. Muon lr: lr law x0.6,1,1.6."""
import json, math
A4 = {64: (64, 0.00396), 128: (16, 0.001485), 256: (16, 0.00099), 512: (16, 0.0007071 * 0.84)}
def mlaw(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2**20) ** -0.27
M4, M8, M16 = 4 * 2**20, 8 * 2**20, 16 * 2**20
plan = {"mps": [(d, D) for d in (128, 256) for D in (M8, M16)] + [(512, M4)], "cpu": [(64, M8)]}
AB = {M4: (16, 32, 64), M8: (32, 64, 128), M16: (32, 64, 128)}
MB = {M4: (32, 64, 128), M8: (32, 64, 128, 256), M16: (64, 128, 256)}
for dev, cells in plan.items():
    jobs = []
    for d, D in cells:
        for opt in ("adam", "muon"):
            for B in (AB if opt == "adam" else MB)[D]:
                if opt == "adam":
                    b0, l0 = A4[d]; c = l0 * math.sqrt(B / b0) * (D / M4) ** -0.25; ms = (0.5, 1, 2)
                else:
                    c = mlaw(B, d, D); ms = (0.6, 1, 1.6)
                name = f"spd2_{opt}_t{D}_d{d}L2_B{B}"
                mem = [dict(tag=f"{name}_lr{m}", lr=round(c * m, 7)) for m in ms]
                split = dev == "cpu" or d >= 256 or B >= 64
                for i, g in enumerate([[x] for x in mem] if split else [mem]):
                    jobs.append(dict(pack=f"{name}_{i}", opt=opt, d=d, L=2, seq=64, batch=B,
                                     steps=D // (B * 64), warm=0.05, members=g))
    jobs.sort(key=lambda j: j["steps"] * j["d"] ** 2)          # cheap first
    open(f"jobs/spd2_{dev}.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in jobs))
    print(dev, len(jobs))
