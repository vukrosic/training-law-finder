"""Write predictions for a prediction round (score 2) before its runs exist.

    python3 core/predict.py forgetting rounds/003-forgetting/settings.jsonl \
        problems/forgetting/formulas/end_z2.py problems/forgetting/formulas/lr_clock.py \
        > rounds/003-forgetting/predictions.jsonl

settings.jsonl has one line per planned run: {"tag": "...", "settings": {"width": 512, ...}}, with
every setting column the formulas read. Each formula is fit on all rows of the problem (or one
--family) exactly as core/score.py fits it, then predicts every planned run. Commit the output
before training the runs; rounds/README.md explains the rest.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.score import load_module, load_problem, visible_columns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("settings")
    ap.add_argument("formulas", nargs="+")
    ap.add_argument("--family", default=None)
    a = ap.parse_args()
    problem = load_problem(a.problem)
    keep = visible_columns(problem)
    rows = problem.rows(a.family)
    train = [{k: v for k, v in r.items() if k in keep} for r in rows]
    y = [problem.target(r) for r in rows]
    fitted = []
    for path in a.formulas:
        f = load_module(path, "formula_" + os.path.basename(path)[:-3])
        fitted.append((os.path.basename(path)[:-3], f, f.fit(train, y)))
    for line in open(a.settings):
        if not line.strip():
            continue
        run = json.loads(line)
        s = {k: v for k, v in run["settings"].items() if k in keep}
        pred = {name: round(float(f.predict(p, s)), 6) for name, f, p in fitted}
        print(json.dumps({"tag": run["tag"], "settings": run["settings"], "pred": pred}))


if __name__ == "__main__":
    main()
