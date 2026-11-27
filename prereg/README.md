# Pre-registered predictions for grids 1-3 (2026-11-27)

These are predictions for small-scale runs that had not finished when this was written. Every formula and
constant was frozen before any grid result was read. Grid 4 (forgetting over time) has its predictions in
`drafts/forgetting-curve-shape` and is not repeated here.

- `grids-2026-11-27.jsonl`: one prediction per line: `tag`, `claim`, `quantity`, `predicted`, `formula`
  (plus `k` for recall, and `runs_done_before` for a whole sweep).
- `frozen-2026-11-27.json`: every frozen constant.
- `tools/prereg_grids.py`: the script that wrote both files. It reads the run table only to list the grid
  tags without a result and to fit C28 on older runs. It trains nothing.

```
python3 tools/prereg_grids.py --results <run table .jsonl> --grids <tlf-setups grids folder>
```

## What was pending

| grid | runs | done when frozen | predicted runs | prediction lines |
|---|---|---|---|---|
| 1 window-lr (`tlfg1`) | 128 | 0 | 128 | 257 |
| 2 lr-width-depth (`tlfg2`) | 160 | 90 | 70 (and all 20 sweeps) | 517 |
| 3 repeats-width-depth (`tlfg3`) | 54 | 0 | 54 | 525 |

On grid 2, 90 runs had finished, but none of their results were read. The sweep predictions (C26, C11)
cover all 20 sweeps; each sweep line records how many of its 8 runs had finished.

## Frozen formulas

| formula id | claim | what it predicts | where the constants come from |
|---|---|---|---|
| `end_z2` | C06 | kept per run, grids 1 and 2 | frozen 2026-11-26 in `problems/forgetting/formulas/end_z2.py` |
| `end_z2_curve` | C10 | s0 per run and q per grid in kept_t = 1/(1+(S_t/s0)^q) | the same constants: s0 = e^-z, q = 1.284 |
| `end_z2_L0` | C01 | kept per run on grid 2 with no depth term | `end_z2` with the ln(layers/8) term set to 0 |
| `lr_area` | C09 | final val_loss per run, grid 2 | fitted now on all 1,163 final-loss runs |
| `chinchilla_lr` | C09 | the baseline val_loss per run, grid 2 | fitted now on the same runs |
| `c26_line` | C26 | best peak per sweep = e^-1.708 · width^-0.64 | intercept fitted now on 60 cosine sweeps of the final-loss data (free exponent there: -0.67) |
| `lr_area_argmin` | C26 | best peak per sweep = minimum of the frozen `lr_area` | `lr_area` above |
| `lr_area_bowl` | C11 | 2x-high cost, 2x-low cost and curvature per sweep | `lr_area` above |
| `c28_loglin` | C28 | bits per fact at k = 1 to 256 per run, grid 3 | fitted now on 68 older runs of the grid-3 testbed |
| `c01_deeper_lower_loss` | C01 | sign of the val_loss change for one depth step | the claim itself (-1 = deeper is lower) |

Two frozen predictions already disagree with a claim:

- `lr_area` refitted on the 1,163-run data has a width term of -1.24, so its best peak falls as width^-1.13
  on grid 2. It puts the best peak below the grid (under 0.002) at widths 512 and 1024. `c26_line` puts it
  inside the grid at every width (0.0127 at width 64, 0.0021 at width 1024).
- `c28_loglin` fitted on the 68 older runs has a repeats slope of -2.19 bits per doubling and a size slope
  of -0.32. C28 states -0.61 and -1.46. On those runs the interaction term does not lower the held-out
  error (2.87 against 2.85 bits). The log-linear form goes below 0 bits at large k for the largest models;
  the raw value is what is scored.

C03 has no prediction here: batch and sequence length are 64 in every run of grids 1 and 2, so its clock
terms equal 1 and it predicts exactly what `end_z2` predicts.

## How results are measured

- kept: from `facts_eval.gstep.bits` (one list per reading, one reading after every step past the window):
  b = mean of each reading, C = mean of `bits_by_k` at k = 1, 2, 4, kept = 1 - (b[-1] - b[0]) / (C - b[0]).
- q (C10): fit kept_t = 1/(1+(S_t/s0)^q) per run on all readings, S_t = learning rate summed over the
  first t steps after the window.
- best peak (C26): vertex of a parabola in ln(peak) through the grid minimum and up to 2 neighbours on each
  side; a sweep with its minimum at either end of the grid has no vertex.
- 2x costs (C11): loss 2 grid steps above and 2 grid steps below the grid minimum, minus the minimum (the
  peaks are sqrt(2) apart). Curvature: the x² coefficient of the same parabola.

## Kill lines

- **C01** dies if, at any width of grid 2, the mean change in kept for a depth doubling (2→4 or 4→8,
  averaged over the 8 peaks) is larger than 0.01 in size. It also dies if the deeper run has the lower
  val_loss in fewer than 90% of matched pairs: on grid 2 (same width and peak, 2→4, 4→8 and 8→12) or on
  grid 3 (same width and number of facts). The frozen `end_z2` depth term predicts +0.021 kept per doubling
  (median over the pending runs), so `end_z2` and C01 cannot both pass.
- **C06** dies if the frozen `end_z2` has a kept RMSE above 0.05 on the 70 pending runs of grid 2, or on the
  128 runs of grid 1.
- **C09** dies if `lr_area` does not have a lower val_loss RMSE than `chinchilla_lr` on the 70 pending runs
  of grid 2.
- **C10** dies if the median fitted q is outside [0.9, 1.3] over the runs of grid 1, over the runs of
  grid 2, or at any width of grid 1. The frozen q is 1.284, close to the upper end.
- **C11** dies if, at any width of grid 2, the median 2x-high cost and the median 2x-low cost differ by more
  than 0.01 (sweeps with the grid minimum at peak 3 to 6 of 8). It also dies if curvature does not grow
  with width: Spearman correlation of curvature with width at or below 0 over the sweeps that have a
  vertex. `lr_area_bowl` predicts equal costs (0.026 to 0.031) and curvature that grows by only 4% per
  width doubling.
- **C26** dies if the fitted exponent b in ln(best peak) = a + b·ln(width) + c·ln(layers), over the grid 2
  sweeps that have a vertex, is outside [-0.85, -0.45]. It also dies if depth moves the best peak by more
  than 1.5x from 2 to 12 layers (|c| > 0.226), or if the median ratio between the measured best peak and
  `c26_line` is above 1.5 in either direction. `lr_area_argmin` predicts b = -1.13 and c = +0.02.
- **C28** dies if the repeats slope fitted on grid 3 (bits = a + b·log2 k + c·log2 size + d·log2 facts) is
  outside [-0.8, -0.4] bits per doubling, or if adding a log2 k × log2 size term lowers the
  leave-one-width-out RMSE on grid 3. The frozen fit on older runs gives b = -2.19, so C28 as written is
  expected to fail on grid 3.

## Ranges to keep in mind

- `end_z2` was fitted on 3 to 12 layers. Grid 2 also has 2 layers.
- `c28_loglin` was fitted on widths 64 to 512 and 2 to 12 layers. Grid 3 goes to width 1024.
- `lr_area` was fitted on widths 24 to 1024, so grid 2 is inside its width range.
