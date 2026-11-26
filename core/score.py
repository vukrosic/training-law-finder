"""Score a formula on a problem. One command, same rules for everyone.

    python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py
    python3 core/score.py forgetting my_formula.py --name "my idea" --author yourname --record

A formula file defines three things:

    CONSTANTS = 10                     # how many numbers fit() learns from data
    def fit(rows, y): ... return params  # learn constants from training rows and their targets
    def predict(params, row): ...      # return one float for one row

Score 1 (held-out): the problem splits rows into groups (for example by model width).
Each group is left out in turn, fit() sees the other groups, predict() is scored on the
left-out group. The score is the mean of the per-group RMSE, so a formula has to
extrapolate to settings it never saw. Lower is better. The worst group is printed too.

Rules (problems/README.md): every added constant must lower the score by at least 0.005,
and a score below the problem's FLOOR is measurement noise, not a better formula.
--record appends the result to problems/<name>/scores.jsonl; core/leaderboard.py
turns that file into LEADERBOARD.md.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_problem(name):
    return load_module(os.path.join(ROOT, "problems", name, "problem.py"), f"problem_{name.replace('-', '_')}")


def rmse(pairs):
    return math.sqrt(sum((p - y) ** 2 for p, y in pairs) / len(pairs)) if pairs else float("nan")


def score(problem, formula, rows=None):
    rows = rows if rows is not None else problem.rows()
    groups = {}
    for r in rows:
        groups.setdefault(problem.group(r), []).append(r)
    per = {}
    for g in sorted(groups, key=str):
        train = [r for r in rows if problem.group(r) != g]
        params = formula.fit(train, [problem.target(r) for r in train])
        per[g] = rmse([(float(formula.predict(params, r)), problem.target(r)) for r in groups[g]])
    params = formula.fit(rows, [problem.target(r) for r in rows])
    insample = rmse([(float(formula.predict(params, r)), problem.target(r)) for r in rows])
    return {"score": sum(per.values()) / len(per), "worst": max(per.values()),
            "worst_group": str(max(per, key=per.get)), "per_group": {str(k): round(v, 5) for k, v in per.items()},
            "in_sample": insample, "n_rows": len(rows), "n_groups": len(per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("formula")
    ap.add_argument("--name", default=None)
    ap.add_argument("--author", default="anonymous")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    problem = load_problem(a.problem)
    formula = load_module(a.formula, "formula")
    res = score(problem, formula)
    k = getattr(formula, "CONSTANTS", None)
    print(f"problem   {a.problem}  ({res['n_rows']} rows, {res['n_groups']} groups by {problem.GROUP_BY})")
    print(f"formula   {a.formula}  constants={k}")
    print(f"score 1   {res['score']:.4f}  (mean held-out RMSE per group; floor {problem.FLOOR})")
    print(f"worst     {res['worst']:.4f}  at {problem.GROUP_BY}={res['worst_group']}")
    print(f"in-sample {res['in_sample']:.4f}  (not the score)")
    print("per group " + "  ".join(f"{g}:{v:.4f}" for g, v in res["per_group"].items()))
    if a.record:
        sha = hashlib.sha256(open(a.formula, "rb").read()).hexdigest()[:12]
        rec = {"date": datetime.date.today().isoformat(), "name": a.name or os.path.basename(a.formula),
               "author": a.author, "formula": os.path.relpath(os.path.abspath(a.formula), ROOT),
               "sha": sha, "constants": k, **{x: res[x] for x in ("score", "worst", "worst_group", "in_sample", "n_rows")}}
        with open(os.path.join(ROOT, "problems", a.problem, "scores.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("recorded  problems/%s/scores.jsonl" % a.problem)


if __name__ == "__main__":
    main()
