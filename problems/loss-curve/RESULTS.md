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

## Formula scores on the real curves

`sh tools/loss_curve_rescore.sh` (`core/score.py --family llm --record`), 195 runs, 6,240 readings, held-out
widths. Score 1 = RMSE in nats over the 32 readings of a held-out width, averaged over the 6 widths.

| formula | constants | score 1 | in-sample | per width 16 / 24 / 32 / 48 / 64 / 96 |
|---|---|---|---|---|
| [clock_lr](formulas/clock_lr.py): clock + current-lr term (new) | 8 | **0.122** | 0.112 | 0.129 / 0.113 / 0.113 / 0.105 / 0.120 / 0.151 |
| [clock_s2](formulas/clock_s2.py): clock + annealing term, warmup modeled (new) | 9 | 0.150 | 0.133 | 0.161 / 0.142 / 0.133 / 0.131 / 0.141 / 0.191 |
| [multi_power](formulas/multi_power.py) (Luo et al. 2025 + our size offset) | 9 | 0.187 | 0.176 | 0.173 / 0.185 / 0.173 / 0.171 / 0.178 / 0.242 |
| [tissue_anneal](formulas/tissue_anneal.py) (Tissue et al. 2024, size form) | 7 | 0.236 | 0.225 | 0.210 / 0.232 / 0.220 / 0.220 / 0.230 / 0.305 |
| [mean](formulas/mean.py) | 1 | 0.536 | 0.522 | 0.472 / 0.489 / 0.517 / 0.559 / 0.578 / 0.598 |

**Takeaway: the current-lr term is the best of the four laws (0.122, against 0.187 for multi_power), but no
law fits these curves yet: even fitted on every width, the best misses by 0.11 nats, more than the 0.07-0.09
nat gap it is meant to explain.**

Both new candidates use the kill line's clock tau = sum of lr^p (one p from a grid). clock_lr adds
C * N^gamma * (lr in use now); clock_s2 adds tissue_anneal's annealing area with warmup at its real lr and a
fitted decay lambda. Width 96, the widest, is the worst held-out width for every law.

Where the error is (each law fitted on all 195 runs, RMSE in nats):

| formula | first quarter of training | rest | last reading | last reading, mean error (prediction minus loss): const / WSD / cosine |
|---|---|---|---|---|
| clock_lr | 0.185 | 0.073 | 0.096 | -0.009 / **-0.059** / +0.002 |
| multi_power | 0.243 | 0.147 | 0.144 | +0.050 / +0.008 / -0.071 |
| tissue_anneal | 0.321 | 0.182 | 0.192 | +0.044 / +0.089 / +0.088 |

- Most of the error is in the first quarter of training (readings 1-8, loss 5-7 nats), where the loss falls
  fastest. After it, clock_lr misses by 0.073.
- The current-lr term overshoots WSD: at the end of WSD runs clock_lr predicts 0.059 nats too low, while it
  is close for const and cosine. A term proportional to the lr predicts the whole drop at once; the real
  loss may lag behind the lr.
- clock_lr picks p = 0.5, the lowest value in its grid (the kill line's collapse test also picked 0.5). The
  grid should go lower.

## Not done

- The 33 missing runs (training, about 4 minutes with 2 at once), then rebuild the data. The row count
  changes, so every llm score is redone:

  ```bash
  python3 tools/loss_curve_run_queue.py 600
  python3 problems/loss-curve/build.py work/loss-curve/small.jsonl
  sh tools/loss_curve_rescore.sh mean tissue_anneal multi_power clock_lr clock_s2
  ```

- Next formula ideas from the table above: extend clock_lr's p grid below 0.5; let the current-lr term lag
  the lr (an exponential average of the lr instead of the lr itself); fit the first quarter separately or on
  log loss.
- FLOOR stays None: the grid has no repeated settings.
- 2 runs took over 20 s (block L, 1,024 steps at widths 16 and 24, 20.4 and 23.4 s); the rest stayed under 20 s.
