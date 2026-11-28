"""spneg: negative spectral powers (p<0 boosts weak singular directions, like Freon c>1/2) at the long cells.
pnorm 'fro': update rescaled to Frobenius sqrt(min dim), the same size as Muon. Hidden lr = Muon law x {0.5,1,2,4}."""
import json
def law(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2 ** 20) ** -0.27
J = {"cpu": [], "mps": []}
for dev, d in (("cpu", 64), ("mps", 128)):
    for B, st in ((16, 1024), (64, 1024)):
        D = st * B * 64
        for p in (-0.25, -0.5):
            for E in (0.0175, 0.035):
                c = law(B, d, D); name = f"spneg_p{p}_t{D}_d{d}L2_B{B}_E{E}"
                mem = [dict(tag=f"{name}_lr{x}", lr=round(c * x, 6), emb_mult=round(E / (c * x), 4)) for x in (0.5, 1, 2, 4)]
                J[dev].append(dict(pack=name, opt="muon", d=d, L=2, seq=64, batch=B, steps=st, warm=0.05, pexp=p, pnorm="fro", members=mem))
for dev, v in J.items():
    open(f"jobs/spneg_{dev}.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in v)); print(dev, len(v))
