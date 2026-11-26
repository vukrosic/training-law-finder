# Prediction rounds (score 2)

Score 1 can be gamed by trying many formulas on the same table. Score 2 cannot: the formulas
predict runs that do not exist yet, and the git commit time proves the predictions came first.

## How a round works

1. Pick 10-50 new settings, ideally outside the table (wider, longer, a new schedule).
2. Write the settings to `rounds/<NNN>-<problem>/settings.jsonl` (`{"tag": ..., "settings": {...}}` per run),
   then let every competing formula predict each one and commit the result:
   `python3 core/predict.py <problem> rounds/<NNN>-<problem>/settings.jsonl <formula files> > rounds/<NNN>-<problem>/predictions.jsonl`.
   Each line is `{"tag": ..., "settings": {...}, "pred": {"formula_name": value, ...}}`.
3. Only after that commit, train the runs and commit `results.jsonl`: `{"tag": ..., "actual": value}`.
4. Score: `python3 core/round_score.py rounds/<NNN>-<problem>`. It prints each formula's RMSE and bias.

Anyone can propose a round by pull request with predictions only; anyone with compute can train it.
A round's runs then join the problem's data for the next score 1.

## Rounds so far (forgetting)

| round | runs | what | result |
|---|---|---|---|
| internal-1 | 3 | width 512-1024, 3 layers | end+z2 RMSE 0.0258, every run overpredicted (+0.019 to +0.046); schedule-share variant 0.0112 |
| internal-2 | 1 | width 2048, 3 layers (pending) | predictions: end+z2 0.203, end+z2+batch 0.268, schedule-share 0.227, abs-width 0.306 |
| [002](002-forgetting/) | 3 | two runs at width 2048, 12 layers (503M non-embedding params), plus one finished 768-wide run (a known miss) | 503M run measured 0.2416; end+z2 predicted 0.2414 (-0.0002), hA -0.019, a_hA -0.053. With the known miss: RMSE 0.130 / 0.135 / 0.145 |

These two were logged in our own notebook before the runs finished, not in this repo, so they are
history, not proof. Rounds from here on go through git.
