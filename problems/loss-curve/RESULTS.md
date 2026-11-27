# Loss curve: results on the small-run grid

Runs: `prereg/loss-curve-small.jsonl` on a MacBook CPU, 2 threads, 2026-11-27. 195 of 228 runs finished
(median 7.6 s, longest 23.4 s, none diverged) before Vuk stopped the research. All of blocks S, T and L are
done; 33 runs are missing: P 17, F 7, D 6, E 3. The curves are in `data/llm.jsonl.gz` (195 runs: width 16 42,
24 42, 32 39, 48 30, 64 24, 96 18).

## Kill line (prereg/loss-curve.md): the single-clock guess is dead

Computed with `python3 tools/loss_curve_kill_line.py <results>`. First on 97 runs (every kill-line pair:
block S at 1x peak, all widths and step counts, plus some 0.5x and 2x runs), then again on all 195.

| test | result | kill threshold | verdict |
|---|---|---|---|
| 1. collapse onto one f(tau), tau = sum lr^p | median RMSE over 6 widths: 0.043 nats at 97 runs, 0.085 at 195 (best p = 0.5 both times, the edge of the 0.5-2 search) | 0.0095 | dead |
| 2. WSD vs const at the same lr area, late in decay | const higher by 0.067-0.093 nats (median per width, 195 runs); 6 of 6 widths above 0.02 | 4 of 6 | dead |

Test 1 got worse with more runs because the added 0.25x and 4x peak runs spread further from one curve.

Test 2 on the 12 long runs (1,024 steps, widths 24-64, run before the switch to small runs): const higher by
0.091-0.121 nats, 4 of 4 widths above 0.02. Same direction, larger gap.

Plain meaning: two runs that have used the same total amount of learning rate do not reach the same loss.
The run whose learning rate is falling (WSD in its decay) is lower by about 0.07-0.12 nats. So the loss
depends on the current learning rate as well as on the area under it. The prereg's next candidate is the clock
plus a term for the current learning rate, which is the shape of tissue_anneal and multi_power.

## Formula scores on the real curves (partial)

`python3 core/score.py loss-curve <formula> --family llm --record`, 195 runs, 6,240 readings, held-out
widths. Score 1 = RMSE in nats over the 32 readings of a held-out width, averaged over the 6 widths.

| formula | constants | score 1 | worst width | per width 16 / 24 / 32 / 48 / 64 / 96 |
|---|---|---|---|---|
| mean | 1 | 0.536 | 96 (0.598) | 0.472 / 0.489 / 0.517 / 0.559 / 0.578 / 0.598 |
| tissue_anneal | 7 | 0.236 | 96 (0.305) | 0.210 / 0.232 / 0.220 / 0.220 / 0.230 / 0.305 |
| multi_power | 9 | not scored | | |

multi_power was started and stopped before it finished: the Mac was needed for other training, so no
score was recorded. The leaderboard was not rebuilt and `tools/check.py` was not rerun after these two scores.

tissue_anneal misses by 0.24 nats on average, about 3 times the 0.07-0.09 nat WSD gap it is meant to explain.
Its in-sample error is 0.225, so the miss is not from held-out widths; the form does not fit these curves.
Not yet checked why. Two candidates: warmup counted at peak lr (20 of 128 steps in the shortest runs), and
the first readings (loss 5-7 nats at step 4-16 of 128) dominating a least-squares fit on the loss.

## Not done

- Score multi_power with `--family llm --record`, then rebuild the leaderboard (`python3 core/leaderboard.py
  loss-curve`) and rerun `tools/check.py`. Then the per-schedule and per-step breakdown: which readings
  each law misses. FLOOR stays None.
- The 33 missing runs: `python3 work/loss-curve/run_queue.py <seconds>` skips finished tags (the queue script
  is in the git-ignored work folder; it reads the grid file in order, at most 2 runs, nice 10).
- FLOOR stays None: the grid has no repeated settings.
- 2 runs took over 20 s (block L, 1,024 steps at widths 16 and 24, 20.4 and 23.4 s); the rest stayed under 20 s.
