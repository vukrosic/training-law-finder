# Loss curve: predict the whole validation-loss curve from the training settings

**195 real small runs are in `data/llm.jsonl.gz` (widths 16-96, 128-1,024 steps, MacBook CPU). The
single-clock kill line is dead on both tests; see [RESULTS.md](RESULTS.md). The formulas are not yet scored on
the real runs; the scores below are on the made-up fixture and say nothing about real training.**

A language model is trained with a given width, depth, step count, peak learning rate and schedule (cosine,
WSD or constant). **Predict its validation loss at 32 evenly spaced points of training, from the settings
alone.** The [final-loss](../final-loss/) problem asks only for the last point; this one asks for the whole
curve, so a formula must get the shape right: the fast early drop, the slow middle, and the extra drop when
the learning rate decays.

Who would use the answer: anyone who wants to stop a bad run early, or compare schedules, by reading the first
part of a curve and knowing where it should go.

## Status

| part | state |
|---|---|
| logging | `testbed/train.py --curve_points 32` logs the 32 readings, the lr at each, and 4 token-band losses (default off) |
| run grid | [prereg/loss-curve-small.jsonl](../../prereg/loss-curve-small.jsonl): 228 runs of a few seconds each, written before any data, with the guess and kill line in [prereg/loss-curve.md](../../prereg/loss-curve.md). The larger 300-run grid (`loss-curve-grid.jsonl`) is parked |
| real data | `data/llm.jsonl.gz`: 195 of the 228 runs (`build.py` turns testbed result lines into it) |
| kill line | dead on both tests: WSD late in its decay is 0.07-0.09 nats below const at the same lr area, at all 6 widths ([RESULTS.md](RESULTS.md)) |
| fixture | `data/synthetic.jsonl.gz`: 90 made-up runs from an invented law (`make_fixture.py`), only so the scorer and `tools/check.py` run |
| floor | **pending**: the grid has no repeated settings, so `problem.floor()` returns None until two runs share their settings |
| literature | [LITERATURE.md](LITERATURE.md): first pass, gate 0 decides |

## Data

One line per run in `data/<family>.jsonl.gz`. Each run has 32 readings at steps `round(i * steps / 32)`,
i = 1..32; the last reading is the final step. `rows()` in problem.py turns each run into 32 rows, one per
reading.

Per reading a formula may read every run setting plus:

| column | meaning |
|---|---|
| `cv_frac` | position of the reading: i / 32 |
| `cv_step` | steps taken so far |
| `cv_lr` | lr multiplier (share of the peak) at the last step taken |
| `cv_area` | sum of the lr multiplier over the steps taken; `peak * cv_area` = lr area so far |

Target: `cv_loss`, the val loss at that reading. The token-band losses (`lc_band`) are kept in the data file
for later problems but `rows()` does not pass them on, so a formula cannot rebuild the target from them.
Columns: [core/columns.yaml](../../core/columns.yaml).

Tested range (the small grid, once complete): widths 16 to 96, 1, 2 or 4 layers, 128 to 1,024 steps, cosine /
WSD / constant schedules, peak learning rate 0.25x to 4x the best final-loss peak per width, WSD decay share 0.1
to 0.5, cosine to 0 or to 10% of the peak, batch 32, sequence 64, 1,024-token vocabulary, Muon, weight decay 0.
A paper on this problem states this range.

## Scoring

Score 1 = RMSE of `cv_loss` over all 32 readings of a held-out width, averaged over widths
(leave-one-width-out, widths with 5+ runs). So it is the error averaged over the 32 points, on widths the
formula never saw. Lower is better. The floor is computed the final-loss way: pooled spread of the target over
rows with identical visible settings (here: identical settings and the same reading).

## Commands

Problem name for every tool: `loss-curve`.

```bash
python3 core/score.py loss-curve problems/loss-curve/formulas/multi_power.py     # about 35 s on the fixture (M-series laptop, 2026-11-27)
python3 core/score.py loss-curve work/<slug>/formulas/c01.py                     # your candidate
```

When real runs exist:

```bash
python3 problems/loss-curve/build.py results_a.jsonl results_b.jsonl
```

Then delete `data/synthetic.jsonl.gz`, empty `scores.jsonl`, set `FLOOR` in problem.py, score every formula
with `--record`, and run `python3 core/leaderboard.py loss-curve`. `rows()` combines every data file, so the
fixture must go before scoring real runs.

## Starting formulas

| formula | idea | constants |
|---|---|---|
| [mean](formulas/mean.py) | constant | 1 |
| [tissue_anneal](formulas/tissue_anneal.py) | published: lr area plus a decaying-lr "annealing area" term, size form (Tissue et al. 2024) | 7 |
| [multi_power](formulas/multi_power.py) | published: multi-power law (Luo et al. 2025) plus a model-size offset that is ours | 9 |

Fixture scores are in [LEADERBOARD.md](LEADERBOARD.md) under "synthetic". They only show that the code runs.
