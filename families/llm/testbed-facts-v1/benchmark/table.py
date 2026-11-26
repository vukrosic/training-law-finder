"""Print one table of runs: python3 -I benchmark/table.py experiments/runs/*.jsonl"""
import json, sys

cols = ["method", "tag", "steps", "tokens", "active_params", "val_bpb", "val_loss",
        "load_cv", "maxvio", "drop_at_1.25", "zero_expert", "experts_per_token", "train_seconds", "peak_mb"]
rows = []
for p in sys.argv[1:]:
    for line in open(p):
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda r: (r.get("tag", ""), r.get("val_bpb", 9e9)))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"
    return "" if v is None else str(v)


print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
