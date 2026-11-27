"""Rebuild problems/<name>/LEADERBOARD.md from problems/<name>/scores.jsonl.

    python3 core/leaderboard.py forgetting

Keeps the best score per (model family, formula sha), one table per family, sorted by score 1.
Formulas scored with --measured (they read measured columns) get their own table per family.
A star marks the formula to beat in each table: the best score among the formulas that pass the
constants rule (rule 2 in problems/README.md) against every formula with fewer constants.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_CONSTANT = 0.005  # rule 2 in problems/README.md


def to_beat(rows):
    """The best-scoring formula whose extra constants each lower score 1 by PER_CONSTANT."""
    ok = [r for r in rows if all(g["score"] - r["score"] >= PER_CONSTANT * (r["constants"] - g["constants"])
                                 for g in rows if g["constants"] < r["constants"])]
    return min(ok, key=lambda r: r["score"])


def build(name):
    d = os.path.join(ROOT, "problems", name)
    best = {}
    for line in open(os.path.join(d, "scores.jsonl")):
        r = json.loads(line)
        key = (r.get("family", "all") + (" (reads measured columns)" if r.get("measured") else ""), r["sha"])
        if key not in best or r["score"] < best[key]["score"]:
            best[key] = r
    out = [f"# Leaderboard: {name}", "",
           "Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py " + name + "`.",
           "\"all\" scores every model family combined; other tables score one family (`--family`).",
           "★ = the formula to beat: the best score among formulas whose constants each earn their place "
           "(rule 2 in [problems/README.md](../README.md#rules))."]
    for fam in sorted({k[0] for k in best}, key=lambda f: (f != "all", f)):
        rows = sorted((r for k, r in best.items() if k[0] == fam), key=lambda r: r["score"])
        out += ["", f"## {fam}", "",
                "| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |",
                "|---|---|---|---|---|---|---|---|---|"]
        star = to_beat(rows)
        for i, r in enumerate(rows, 1):
            out.append(f"| {i} | {'★ ' if r is star else ''}[{r['name']}](../../{r['formula']}) | {r['constants']} | {r['score']:.4f} | "
                       f"{r['worst']:.4f} ({r['worst_group']}) | {r['in_sample']:.4f} | {r['n_rows']} | {r['author']} | {r['date']} |")
    return "\n".join(out) + "\n"


def main(name):
    text = build(name)
    open(os.path.join(ROOT, "problems", name, "LEADERBOARD.md"), "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1])
