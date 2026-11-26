"""Rebuild problems/<name>/LEADERBOARD.md from problems/<name>/scores.jsonl.

    python3 core/leaderboard.py forgetting

Keeps the best score per formula sha, sorted by score 1. A row counts as a new best only if it
beats the row above it in constants by 0.005 per extra constant (problems/README.md).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(name):
    d = os.path.join(ROOT, "problems", name)
    best = {}
    for line in open(os.path.join(d, "scores.jsonl")):
        r = json.loads(line)
        if r["sha"] not in best or r["score"] < best[r["sha"]]["score"]:
            best[r["sha"]] = r
    rows = sorted(best.values(), key=lambda r: r["score"])
    out = [f"# Leaderboard: {name}", "",
           "Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py " + name + "`.", "",
           "| # | Formula | Constants | Score 1 | Worst group | In-sample | Author | Date |",
           "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | [{r['name']}](../../{r['formula']}) | {r['constants']} | {r['score']:.4f} | "
                   f"{r['worst']:.4f} ({r['worst_group']}) | {r['in_sample']:.4f} | {r['author']} | {r['date']} |")
    open(os.path.join(d, "LEADERBOARD.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
