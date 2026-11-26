"""Rebuild problems/<name>/LEADERBOARD.md from problems/<name>/scores.jsonl.

    python3 core/leaderboard.py forgetting

Keeps the best score per (model family, formula sha), one table per family, sorted by score 1.
The 0.005-per-extra-constant rule (problems/README.md) is applied by reviewers, not by this script;
the Constants column is there so they can.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(name):
    d = os.path.join(ROOT, "problems", name)
    best = {}
    for line in open(os.path.join(d, "scores.jsonl")):
        r = json.loads(line)
        key = (r.get("family", "all"), r["sha"])
        if key not in best or r["score"] < best[key]["score"]:
            best[key] = r
    out = [f"# Leaderboard: {name}", "",
           "Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py " + name + "`.",
           "\"all\" scores every model family combined; other tables score one family (`--family`)."]
    for fam in sorted({k[0] for k in best}, key=lambda f: (f != "all", f)):
        rows = sorted((r for k, r in best.items() if k[0] == fam), key=lambda r: r["score"])
        out += ["", f"## {fam}", "",
                "| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |",
                "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            out.append(f"| {i} | [{r['name']}](../../{r['formula']}) | {r['constants']} | {r['score']:.4f} | "
                       f"{r['worst']:.4f} ({r['worst_group']}) | {r['in_sample']:.4f} | {r['n_rows']} | {r['author']} | {r['date']} |")
    return "\n".join(out) + "\n"


def main(name):
    text = build(name)
    open(os.path.join(ROOT, "problems", name, "LEADERBOARD.md"), "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1])
