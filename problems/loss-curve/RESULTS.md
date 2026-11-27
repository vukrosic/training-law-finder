# Loss curve: results on the small-run grid

Runs: `prereg/loss-curve-small.jsonl` on a MacBook CPU (2 threads, about 5-12 s per run), 2026-11-27.

## Kill line (prereg/loss-curve.md): the single-clock guess is dead

Computed with `python3 tools/loss_curve_kill_line.py <results>` on the first 97 runs, which hold every
kill-line pair (block S at 1x peak, all widths and step counts) plus 2-layer runs at 0.5x and 2x peak.

| test | result | kill threshold | verdict |
|---|---|---|---|
| 1. collapse onto one f(tau), tau = sum lr^p | median RMSE 0.043 nats over 6 widths (best p = 0.5, the edge of the 0.5-2 search) | 0.0095 | dead |
| 2. WSD vs const at the same lr area, late in decay | const higher by 0.067-0.095 nats (median per width); 6 of 6 widths above 0.02 | 4 of 6 | dead |

Test 2 on the 12 long runs (1,024 steps, widths 24-64, run before the switch to small runs): const higher by
0.091-0.121 nats, 4 of 4 widths above 0.02. Same direction, larger gap.

Plain meaning: two runs that have used the same total amount of learning rate do not reach the same loss.
The run whose learning rate is falling (WSD in its decay) is lower by about 0.07-0.12 nats. So the loss
depends on the current learning rate as well as on the area under it. The prereg's next candidate is the clock
plus a term for the current learning rate, which is the shape of tissue_anneal and multi_power.
