"""Predicted vs actual loss for the 6 runs predicted before they ran.

    python3 examples/loss-law/plot_predictions.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    with open(os.path.join(HERE, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


actual = {r["tag"]: r for r in rows("results.jsonl")}
panels = [
    ("predictions_two_power.jsonl", "L = E + A·N^-a + B·D^-b"),
    ("predictions_loglinear.jsonl", "log L = c + α log N + β log D + γ log N·log D"),
]
max_fit_width = max(r["config"]["d"] for t, r in actual.items() if t.startswith("fit_"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
lo, hi = 4.75, 5.45
for ax, (fname, title) in zip(axes, panels):
    preds = rows(fname)
    p = [r["pred"] for r in preds]
    a = [actual[r["tag"]]["val_loss"] for r in preds]
    wide = [actual[r["tag"]]["config"]["d"] > max_fit_width for r in preds]
    mean = sum(a) / len(a)
    r2 = 1 - sum((x - y) ** 2 for x, y in zip(p, a)) / sum((y - mean) ** 2 for y in a)
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", zorder=1)
    for x, y, w in zip(a, p, wide):
        ax.scatter(x, y, s=70, zorder=2, color="#d1495b" if w else "#2e86ab",
                   edgecolor="white", linewidth=1)
    ax.set_title(f"{title}\nR² of predictions = {r2:.3f}", fontsize=10.5)
    ax.set_xlabel("actual loss")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("loss predicted before the run")
handles = [
    plt.Line2D([], [], marker="o", ls="", color="#2e86ab", markersize=8, label=f"width ≤ {max_fit_width} (inside fit range)"),
    plt.Line2D([], [], marker="o", ls="", color="#d1495b", markersize=8, label="width 160 (wider than any fit run)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.suptitle("Both formulas fit the old runs equally well. Predictions made in advance pick one.", fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
out = os.path.join(HERE, "predicted_vs_actual.png")
fig.savefig(out, dpi=150, facecolor="white")
print("wrote", out)
