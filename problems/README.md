# Problems

One folder per number a formula predicts (not per model type), with frozen data, a fixed split and one scoring command; one data file per model family:

```text
problems/forgetting/
  README.md          the question, allowed inputs, the published law to beat
  problem.py         rows(family), target(row), group(row), TARGET, GROUP_BY, FLOOR
  data/llm.jsonl.gz  frozen rows per model family (data/cifar.jsonl.gz would hold image models)
  formulas/          mean.py (baseline) and every submitted formula
  scores.jsonl       one line per recorded score
  LEADERBOARD.md     built from scores.jsonl, one table per family plus "all"; ★ marks the formula to beat
```

Open problems and floors: [main README](../README.md#open-problems). Beating ★ is the core of a paper.

## Score it

```bash
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py               # all families
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py --family llm  # one family
```

Start from `core/formula_template.py`.

## Two scores

- **Score 1, held out:** rows are grouped (by model width); each group is left out in turn, the formula is fit on the rest and predicts it. Score 1 = mean RMSE over groups, lower is better.
- **Score 2, predicted before the runs exist:** predictions committed to [rounds/](../rounds/README.md) before the runs are trained; the strongest evidence. A submitted formula does not need it: a maintainer scores it on fresh runs ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Rules

1. **Count constants.** Every number `fit()` learns counts; the leaderboard shows the count.
2. **Each added constant must lower score 1 by at least 0.005.** A 12-constant formula that beats a 10-constant one by 0.004 does not take ★. `core/leaderboard.py` applies this. This is the one place the rule is stated.
3. **Below the floor is noise.** The floor is the error between repeated measurements of the same setting.
4. **Predict from settings only.** `fit()` and `predict()` get only the setting columns of `core/columns.yaml` (width, steps, schedule, ...), not measurements such as final loss or weight norms; `core/score.py` enforces this. A formula that needs a measured quantity is scored with `--measured` and goes in a separate table (`TARGET` stays hidden).
5. **Beat the published law** each problem names (Chinchilla for final loss). Otherwise it is not a finding.

## When a question gets its own folder

Needs all five; otherwise open an issue with the `question` label that collects runs until it qualifies:

1. **A new number** no folder predicts. A new model family for an existing number is a new data file, not a folder.
2. **A floor:** 10+ settings trained twice each (another machine or a restart counts); floor = RMS of (first - second) / sqrt(2) over the pairs, written as `FLOOR` in `problem.py`.
3. **200+ runs** over at least 4 values of the group column, at least 5 runs per group (`MIN_GROUP = 5`; smaller groups are dropped because score 1 weighs groups equally).
4. **A baseline** `formulas/mean.py`, plus the published law to beat if one exists.
5. **One README line on who would use the answer.**

## Add a problem

1. Make the folder as in the tree above; copy `problem.py` from `final-loss/`.
2. Freeze the data: set `SOURCE`, `COLUMNS` and `select()` in `problem.py`, then `python3 tools/build_problem_data.py <name>`. Your own runs go straight into `data/<family>.jsonl.gz` (3 MB at most).
3. Add every new column to `core/columns.yaml` with its kind (setting, measured, target, id).
4. `python3 core/score.py <name> problems/<name>/formulas/mean.py --author <you> --record`, then `python3 core/leaderboard.py <name>`.
5. Add a row to the table in the main README.
6. `python3 tools/check.py`, then open the PR.
