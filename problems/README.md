# Problems

Each folder is one open question about how neural networks train, with fixed data, a fixed split
and one scoring command. Beat the best formula and you have the core of a paper.

| problem | predict | rows | best score 1 | floor | leaderboard |
|---|---|---|---|---|---|
| [forgetting](forgetting/) | share of facts a model still knows at the end of training | 375 | 0.0441 (end+z2, 10 constants) | 0.0126 | [LEADERBOARD](forgetting/LEADERBOARD.md) |
| [final-loss](final-loss/) | final validation loss from the training settings | 22,068 | 0.1852 (Chinchilla + lr bowl, 8 constants) | 0.0001 | [LEADERBOARD](final-loss/LEADERBOARD.md) |

## Score it

```bash
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py
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

## Add a problem

A problem is a folder with `problem.py` (`rows()`, `target(row)`, `group(row)`, `GROUP_BY`, `FLOOR`),
a `README.md` (the question, which columns are allowed, the published law to beat), and at least
one baseline in `formulas/`. Open questions waiting for a folder are listed in the main README.
