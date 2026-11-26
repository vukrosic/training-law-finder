# Example: final loss from width and training steps

A known kind of law, small enough for a laptop CPU, to show the whole loop: random draws, two candidate formulas, held-out scoring, pre-registered predictions.

**Target.** Predict final validation loss of a 3-layer Muon-trained transformer (vocab 4096, sequence 128, batch 16, peak lr 0.02, cosine schedule) from its non-embedding parameter count N and training tokens D.

```bash
sh examples/loss-law/run.sh
```

The result files from our run are committed here, so you can read the numbers without running it.

## Two candidates tie on score 1

20 runs, widths 48-128, 100-700 steps (12-150 s each on 2 CPU threads of a laptop that was busy with other work).

| Candidate | Held out: width | Held out: steps | Score 1 |
|---|---|---|---|
| E + A·N^-a + B·D^-b | 0.964 | 0.970 | **0.964** |
| log L = c + α log N + β log D + γ log N · log D | 0.990 | 0.963 | **0.963** |

The two-power form puts its width exponent on the lower bound (a = 0.05): 48-128 is too narrow a range of widths to pin it down from runs this short.

## Score 2 picks one

Both fits predicted 6 new runs before they ran. 3 of them use width 160, wider than any fit run, and some run up to 1000 steps.

| Candidate | Score 2 (R²) | Mean abs. error | Worst miss |
|---|---|---|---|
| E + A·N^-a + B·D^-b | 0.939 | 0.044 | 0.090 (width 160, 825 steps) |
| log-linear with interaction | **0.985** | **0.019** | 0.059 (width 128, 1000 steps) |

The power-law sum underpredicts how much the wider model gains from longer training. The interaction term log N · log D captures that gain. This is the decision rule the agents follow: when candidates tie on held-out slices, order the runs where their predictions differ, predict them first, and keep the candidate that predicts better.

> [!NOTE]
> The tree "ceiling" from `--ceiling` is low for the step-count slices (0.23) because trees cannot extrapolate beyond the range they were fit on. Read it as a ceiling only for slices inside the training range.

## Files

- `make_jobs.py`: random draws of width and steps
- `jobs_fit.jsonl`, `jobs_predict.jsonl`: the job files
- `results.jsonl`: one row per run
- `fits/*.json`: the fitted candidates
- `predictions_*.jsonl`: predictions, time-stamped before their runs
