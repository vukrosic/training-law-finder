"""opt1: no-cost Muon knobs vs baseline Muon, at the lr the batch law predicts (x0.6, 1, 1.6).
CPU: d64L2, MPS: d128L2; batch 16, seq 64, 1M tokens. Baseline also at 0.5M and 2M tokens to turn
loss gains into token speedups. Writes jobs/opt1_{cpu,mps}.jsonl."""
import json, os
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def lrlaw(B, w, D): return 0.00218 * B ** 0.48 * (w / 64) ** -0.54 * (D / 2**20) ** -0.27
KN = {"base": {}, "mom90": {"mom": 0.9}, "mom98": {"mom": 0.98}, "nonest": {"nest": False},
      "pe": {"coef": "pe"}, "ns3": {"ns": 3}, "ar": {"mscale": "ar"}, "b2_99": {"b2": 0.99},
      "lin": {"sched": "lin"}, "wsd": {"sched": "wsd"}, "warm0": {"warm": 0.0},
      "emb2": {"_em": 2.0}, "emb05": {"_em": 0.5}}
for dev, d in (("cpu", 64), ("mps", 128)):
    out = []
    runs = [(k, 2**20) for k in KN] + [("base", 2**19), ("base", 2**21)]
    for k, D in runs:
        B = 16; steps = D // (B * 64); kn = dict(KN[k]); em = kn.pop("_em", None)
        lr0 = lrlaw(B, d, D)
        mem = [{"tag": f"opt1_{k}_d{d}_t{D}_lr{m}", "lr": round(lr0 * m, 7), **({"emb_mult": em} if em else {})}
               for m in (0.6, 1, 1.6)]
        out.append({"pack": f"opt1_{k}_d{d}_t{D}", "opt": "muon", "d": d, "L": 2, "seq": 64, "batch": B,
                    "steps": steps, "warm": 0.05, "evals": 8, "knob": k, **kn, "members": mem})
    with open(os.path.join(H, "jobs", f"opt1_{dev}.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in out)
    print(dev, len(out))
