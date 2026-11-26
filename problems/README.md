# Problems

Each folder is one number about training that a formula predicts (for example the share of facts
kept, or the final loss), with frozen data, a fixed split and one scoring command. A folder is named
after the number, not the model type: the same question asked of language models, image models or
world models lives in one folder, with one data file per model family:

```text
problems/forgetting/
  README.md          the question, allowed inputs, the published law to beat
  problem.py         rows(family), target(row), group(row), GROUP_BY, FLOOR
  data/llm.jsonl.gz  frozen rows from language models; data/cifar.jsonl.gz would hold image models
  formulas/          mean.py (baseline) and every submitted formula
  scores.jsonl       one line per recorded score
  LEADERBOARD.md     built from scores.jsonl, one table per model family plus "all"
``` Beat the best formula and you have the core of a paper.

| problem | predict | rows | best score 1 | floor | leaderboard |
|---|---|---|---|---|---|
| [forgetting](forgetting/) | share of facts a model still knows at the end of training | 375 | 0.0441 (end+z2, 10 constants) | 0.0126 | [LEADERBOARD](forgetting/LEADERBOARD.md) |
| [final-loss](final-loss/) | final validation loss from the training settings | 22,231 | 0.1476 (Chinchilla + lr bowl, 8 constants) | 0.0001 | [LEADERBOARD](final-loss/LEADERBOARD.md) |

## Score it

```bash
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py               # all model families combined
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py --family llm  # one family
```

Start a new formula from `core/formula_template.py`.

## Two scores

- **Score 1, held-out.** The rows are split into groups (by model width). Each group is left out in turn, the formula is fit on the rest and predicts the left-out group. Score 1 is the mean RMSE over groups. A formula that only memorizes the table fails here.
- **Score 2, predicted before the runs exist.** You commit the formula's predictions for new settings to `rounds/` before anyone trains them; then the runs are trained and compared. This is the strongest evidence a paper can have. See [rounds/README.md](../rounds/README.md).

## Rules

1. **Count constants.** Every number `fit()` learns counts, and the leaderboard shows the count.
2. **Each added constant must lower score 1 by at least 0.005.** A 12-constant formula that beats a 10-constant one by 0.004 does not take the top spot.
3. **Below the floor is noise.** The floor is the error between repeated measurements of the same setting.
4. **Predict from settings only.** `predict()` may read the run's settings (width, steps, learning rate schedule, ...), not measurements of the run you are predicting (final loss, weight norms). A formula that needs a measured quantity is allowed but listed separately.
5. **Beat the published law.** Each problem names the published formula it must beat (Chinchilla for final loss). A new rule that does not beat it is not a finding.

## When a question gets its own folder

A new folder needs all five. Otherwise open an issue with the `question` label; the issue collects
runs until it qualifies.

1. **A new number.** It predicts something no existing folder predicts. A new model family for an existing number is a new data file in that folder, not a new folder.
2. **Measurable with a floor.** Two runs of the same setting give the floor; without one, measure it first.
3. **At least 200 runs** that measure the number, spread over at least 4 values of the group column.
4. **A baseline formula** in `formulas/mean.py`, plus the published law to beat if one exists.
5. **One line on who would use the answer**, in the README.

## Add a problem

1. Make the folder with the files in the tree above. Copy `problem.py` from `final-loss/`.
2. Freeze the data: list `SOURCE`, `COLUMNS` and `select()` in `problem.py`, then run `python3 tools/build_problem_data.py <name>`. Data from your own runs goes straight into `data/<family>.jsonl.gz` (3 MB at most).
3. Add every new column to `core/columns.yaml` with its kind (setting, measured, target, id).
4. Score the baseline: `python3 core/score.py <name> problems/<name>/formulas/mean.py --author <you> --record`, then `python3 core/leaderboard.py <name>`.
5. Add a row to the table at the top of this file and to the main README.
6. Run `python3 tools/check.py` and open the pull request.
