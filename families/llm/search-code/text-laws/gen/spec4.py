"""spec4: spec3 at batch 64 (same steps, 4x tokens) to tell steps from tokens and batch noise. d64 on CPU.
p in {0,.25,.5,.75,1}; steps 64/256/1024 at B16; hidden lr = Muon law x {0.5,1,2,4}; E in {0.0175,0.035,0.07}.
One pack = one (p, steps, E) with 4 hidden lrs. d64 on CPU, d128 on MPS (SVD runs on CPU inside Opt)."""
import json
def law(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2 ** 20) ** -0.27
for dev, d in (("cpu", 64),):
    J = []
    for st in (64, 256, 1024):
        D = st * 64 * 64
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            for E in (0.0175, 0.035, 0.07):
                c = law(64, d, D); name = f"spec4_p{p}_t{D}_d{d}L2_B64_E{E}"
                mem = [dict(tag=f"{name}_lr{x}", lr=round(c * x, 6), emb_mult=round(E / (c * x), 4)) for x in (0.5, 1, 2, 4)]
                J.append(dict(pack=name, opt="muon", d=d, L=2, seq=64, batch=64, steps=st, warm=0.05, pexp=p, pnorm="spec", members=mem))
    open(f"jobs/spec4_{dev}.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in J))
    print(dev, len(J), "packs")
