# Forgetting curve: how much does a model still know at each step after it saw the facts?

A language model sees 64 facts during one window of training (20%-45% of the steps, or 45%-70%), then keeps
training on plain text and forgets them step by step. **Predict the share of fact knowledge it still has at
any point after the window, from the training settings alone.** The [forgetting](../forgetting/) problem asks
only for the end of training; this one asks for the whole curve, on 3.4 times as many runs.

Who would use the answer: anyone who adds data in phases (code, math, instruction data) and needs to know how
much of an early phase is still there after N more steps at a given learning rate.

## Data

`data/llm.jsonl.gz`: 1,283 real runs (0.08 MB), decoder-only transformers trained with Muon on FineWeb text,
1,024-token vocabulary, from the public run table on Hugging Face. `rows()` in problem.py turns each run into
8 rows, one per reading, so there are 10,264 rows.

| setting | values |
|---|---|
| width | 24 to 2048 (17 widths with 5+ runs; 471 runs at 64, 5 at 2048) |
| layers | 1-12 (668 runs at 3) |
| steps | 512-16,384 (1,053 runs at 1,024) |
| fact window | A = 20%-45% (1,167 runs), B = 45%-70% (116) |
| schedule | cos 981, wsd 188, const 114; min_lr_frac 0-1 |
| weight decay | 0 (1,079 runs), 0.1, 0.3, 1.0 |
| batch | 8-1024 (807 runs at 64) |
| peak learning rate | 0.002-0.032 |

Reading `j` (column `frac`) sits at 1/16, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4 and 1 of the steps after the window.
Per reading a formula may read `t_now` (steps since the window), `S_now` (sum of the learning rate from the
window end to here), `lr_now` (learning rate here), plus every run setting and `S_end` (the learning-rate sum
to the end). Target: `kept_now`, the share of the fact bits learned in the window that are still there, as
the mean of the 5 readings around that step (single readings wobble from one step to the next).
Columns: [core/columns.yaml](../../core/columns.yaml).

Floor 0.0043: pooled spread of `kept_now` over runs with identical settings (384 setting-and-reading groups,
832 rows; most pairs are the same settings on two machines).

Tested range: widths 24 to 2048, 1 to 12 layers, 512 to 16,384 steps, peak learning rate 0.002 to 0.032. A paper on this problem states this range.

To train a new run, use the command in [forgetting/README.md](../forgetting/README.md#train-a-run-from-a-settings-line-score-2);
the runs here use the same protocol.

## Commands

Problem name for every tool: `forgetting-curve`. Formula to beat: `formulas/clock_zb.py` (★). Published baseline: `formulas/chang_power.py`.

```bash
nice -n 10 python3 core/score.py forgetting-curve problems/forgetting-curve/formulas/clock_zb.py     # one score run: about 24 s on a laptop (M-series, 2026-11-27)
nice -n 10 python3 core/score.py forgetting-curve work/<slug>/formulas/c01.py               # your candidate
```

When you add a constant to a copied formula, extend the start vector in its `fit()` too.

Value ranges of the per-reading columns: `S_now` 0.012-21.2 (median 0.94), `t_now` 10-9,011 steps (median 211),
`lr_now` 0-0.019, `S_end` 0.063-21.2 (median 1.95); `fwlo`/`fwhi` are 0.2/0.45 (window A, 1,167 runs) or 0.45/0.7 (window B, 116 runs).

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [clock_zb](formulas/clock_zb.py) | 12 | 0.0612 | learning-rate-sum clock with a settings onset, plus window position and batch |
| [clock_z](formulas/clock_z.py) | 10 | 0.0765 | the forgetting problem's end_z2 read at every point of the curve |
| [lr_clock](formulas/lr_clock.py) | 2 | 0.1297 | the same clock with no settings |
| [chang_power](formulas/chang_power.py) | 2 | 0.1588 | published: power law in steps since the facts (Chang et al. 2024) |
| [mean](formulas/mean.py) | 1 | 0.2077 | constant |

Floor 0.0043. Gap from clock_zb to the floor: 0.057 (14 times the floor). Full table: [LEADERBOARD.md](LEADERBOARD.md).

## Published work to beat

The baseline is Chang et al. (2024), arXiv:2406.11813: knowledge fades
as a power law in steps since it was seen. Here the step count alone scores 0.1588; the learning-rate sum as
the clock scores 0.1297 with the same two constants.
