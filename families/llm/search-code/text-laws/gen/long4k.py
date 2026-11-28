"""long4k: 4096 steps at batch 16 (4.2M tokens), to see where the best p goes past 1024 steps.
p in {-0.25, 0, 0.25, 0.5}; E in {0.0175, 0.035}; hidden lr = Muon law x {0.5,1,2,4}. p<0 uses Frobenius size (as Muon)."""
import json
def law(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2 ** 20) ** -0.27
J = {"cpu": [], "mps": []}
B, st = 16, 4096
for dev, d in (("cpu", 64), ("mps", 128)):
    D = st * B * 64
    for p in (0.0, 0.25, -0.25, 0.5):
        for E in (0.0175, 0.035):
            c = law(B, d, D); name = f"long4k_p{p}_t{D}_d{d}L2_B{B}_E{E}"
            mem = [dict(tag=f"{name}_lr{x}", lr=round(c * x, 6), emb_mult=round(E / (c * x), 4)) for x in (0.5, 1, 2, 4)]
            J[dev].append(dict(pack=name, opt="muon", d=d, L=2, seq=64, batch=B, steps=st, warm=0.05, pexp=p,
                               pnorm="fro" if p < 0 else "spec", members=mem))
for dev, v in J.items():
    open(f"jobs/long4k_{dev}.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in v)); print(dev, len(v))
