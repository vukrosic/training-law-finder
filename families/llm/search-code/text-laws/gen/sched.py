"""sched: does sliding p from soft to Muon during training beat every fixed p? Linear p schedule pexp -> pend.
d128 B16 on MPS at 256 and 1024 steps (fixed best: p0.25 4.2910 / 3.7561; Muon 4.3313 / 3.7619)."""
import json
def law(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2 ** 20) ** -0.27
J = []; d, B = 128, 16
for st, Es in ((256, (0.0175, 0.035)), (1024, (0.0175, 0.035))):
    D = st * B * 64
    for p0, p1 in ((0.75, 0.0), (0.5, 0.0), (1.0, 0.0)):
        for E in Es:
            c = law(B, d, D); name = f"sched_p{p0}to{p1}_t{D}_d{d}L2_B{B}_E{E}"
            mem = [dict(tag=f"{name}_lr{x}", lr=round(c * x, 6), emb_mult=round(E / (c * x), 4)) for x in (0.5, 1, 2, 4)]
            J.append(dict(pack=name, opt="muon", d=d, L=2, seq=64, batch=B, steps=st, warm=0.05, pexp=p0, pend=p1, pnorm="spec", members=mem))
open("jobs/sched_mps.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in J)); print(len(J))
