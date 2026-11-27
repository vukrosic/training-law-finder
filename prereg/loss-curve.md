# Loss-curve grid and kill line (written 2026-11-27, 20:30 SGT, before any run)

No run of this grid exists yet. This file was written before any data, so it cannot be tuned to the results.

- `loss-curve-grid.jsonl`: 300 runs, one line each (`block`, `tag`, `args` for `testbed/train.py`, `machine`, `draw`).
- `tools/loss_curve_grid.py`: the script that wrote it. It trains nothing.
- Problem that uses the runs: [problems/loss-curve](../problems/loss-curve/README.md).

```
python3 tools/loss_curve_grid.py
```

## The grid

Settings follow `problems/final-loss`: vocabulary 1,024, Muon, batch 64, sequence 64, warmup 20 steps,
3 layers, MLP hidden 2 x width, weight decay 0. Each setting appears once. Each run logs val loss and
4 token-frequency band losses at 32 evenly spaced steps (`--curve_points 32`).

| block | what changes | runs | Mac |
|---|---|---|---|
| A | 12 widths (24-1024) x cos / WSD / const x 512, 1,024, 2,048, 4,096 steps, best peak per width | 144 | 54 |
| B | the same 12 widths x 3 schedules at 0.5x and 2x the best peak, 1,024 steps | 72 | 30 |
| C | widths 32, 64, 128, 256 x 3 schedules x 8,192 and 16,384 steps | 24 | 3 |
| D | 12 widths x WSD decay share 0.1 and 0.5 (A uses 0.2), 1,024 steps | 24 | 10 |
| E | 12 widths x cosine down to 0.1 of the peak, 1,024 steps | 12 | 5 |
| F | widths 64, 128, 256, 512 x 1 and 6 layers x 3 schedules, 1,024 steps | 24 | 9 |
| total | | 300 | 111 |

Best peak per width = e^-1.708 x width^-0.64 (`c26_line` in [README.md](README.md)).
`machine = mac` means estimated memory under 2 GB and estimated CPU time under 20 minutes:
111 runs, about 12.8 CPU hours. The other 189 runs need a GPU: about 9.2 A4000 hours. These are rough
estimates from the cost model in the script.

## The guess being tested

One hidden "learning clock" explains learning. The clock is the learning rate summed over the steps so far,
tau(t) = sum over s <= t of lr(s)^p, with one exponent p for all runs (p = 1 is the plain lr area).
If the guess holds, all runs of one model shape (width, layers) fall on one curve L = f(tau), whatever
the schedule, the step count or the peak.

## Kill line

The single-clock guess is dead if either test below fails. The floor is `FLOOR` of problems/loss-curve
once it is measured; until then the stand-in is 0.0019, the final-loss floor.

1. **Collapse test.** For each (width, layers) with 6+ runs, fit one decreasing curve f(tau) (isotonic
   regression in tau) to all 32 points of all its runs, with one p shared by all shapes (searched over 0.5-2).
   The guess is dead if the median RMSE over shapes is above 5 x floor (0.0095 with the stand-in).
2. **Matched-clock test, needs no fit.** At the same width and peak, compare a WSD run late in its decay
   (after 80% of its decay steps) with the const run at the same tau (within 2%). The guess is dead if
   the WSD loss is lower by more than 0.02 nats at 8 or more of the 12 widths.

Across widths, the guess also says the curves of different widths match after one time scale and one loss
offset per width. This is a weaker claim and is checked by the problem's score 1 (held-out widths), not here.

If the guess dies, the next candidate is the clock plus a term for the current learning rate (the shape of
Tissue et al. 2024 and Luo et al. 2025, see `problems/loss-curve/LITERATURE.md`). The token-band rungs then
start from that two-term form.

## What was not decided here

- The floor. The grid has no repeated settings. `floor()` in `problems/loss-curve/problem.py` computes it
  the final-loss way once two runs with the same settings exist (for example the same line run on two machines).
- Which machine runs which line. Each line runs once, on either machine.
