AI notes, not checked by a person.

# Next steps for the forgetting rules

Ideas for anyone who wants to continue this work. They are suggestions: pick one, change it, or bring your
own. Send results as a pull request ([CONTRIBUTING.md](../../CONTRIBUTING.md)) or talk about them on Discord.

The first part lists six checks that are already done, with their results. The second part lists open
directions, grouped by the hardware they need. The third part lists the runs the table is missing.

## Already checked (no training)

Run `python3 drafts/forgetting-rules/next_steps_checks.py` to print the numbers of checks 3 to 6 from the
public table. It takes about one minute on a laptop CPU. Add `--pairs` to also print check 2: it reads the
full run table (17 MB, downloaded once from Hugging Face, no login). "Points" are percentage points of facts kept.

| # | Check | Result |
|---|---|---|
| 1 | Copied rows | 25 of the 375 rows are copies of another run. Without them the held-out error is 4.3 points instead of 4.4. |
| 2 | Noise floor | The floor of 1.3 points comes from one pair of runs with the same settings. The full public run table has 32 such pairs. They differ by 0.8 points. The two pairs at width 256 differ by 2.0 points. |
| 3 | Nine constants against ten | Removing the depth term changes the held-out error by 0.3 points or less on all seven slices. Nine constants are equal or better on four of them. |
| 4 | Inside the run | With constants fitted on end points, the error on the 36,193 readings inside the runs is 8.0 points. Fitted on all readings it is 5.7 points, but the error at the end rises from 3.8 to 5.9. |
| 5 | Schedule picker | When only the schedule differs, the formula picks the best run in 107 of 107 groups. The plain rule "pick the lowest S" does the same. |
| 6 | Published weight decay bound | Bordt et al. (2025) give an upper limit on what weight decay leaves: the product of (1 − learning rate × weight decay) over the steps after the facts. The formula times this limit has no new constant and lowers the error for a new width from 4.4 to 4.1 points. It is [end_z2_wd.py](../../problems/forgetting/formulas/end_z2_wd.py), the formula marked ★ today. |

Checks 3 and 6, held-out error in points:

| Held-out slice | Ten constants | Nine constants | Ten constants times the bound |
|---|---|---|---|
| width | 4.41 | 4.42 | 4.11 |
| layers | 4.31 | 4.37 | 4.17 |
| steps | 5.15 | 5.06 | 5.70 |
| peak learning rate | 4.15 | 4.18 | 3.69 |
| schedule | 2.76 | 2.76 | 2.81 |
| weight decay | 12.12 | 11.97 | 12.90 |
| window position | 8.46 | 8.19 | 8.98 |

The weight decay and window rows are means over these groups:

| Group held out | Runs | Ten constants | Ten constants times the bound |
|---|---|---|---|
| weight decay 0 | 265 | 33.8 | 41.1 |
| weight decay 0.1 | 48 | 2.2 | 2.2 |
| weight decay 0.3 | 38 | 6.1 | 4.4 |
| weight decay 1.0 | 24 | 6.3 | 3.9 |
| early window (20% to 45% of the steps) | 276 | 6.9 | 6.9 |
| late window (45% to 70%) | 99 | 10.0 | 11.1 |

Check 6 on the two runs outside the table ([rounds/002-forgetting](../../rounds/002-forgetting/)); the constants of the bound version were fitted after both runs:

| Run | Measured | Ten constants | Ten constants times the bound |
|---|---|---|---|
| width 2,048, 12 layers, no weight decay | 24.2% | 24.1% | 23.0% |
| width 768, 10 layers, weight decay 0.3, batch 96 | 10.4% | 28.9% | 17.5% |

Check 5, when the schedule and the peak learning rate both differ (34 groups):

| Choice | Best run picked | Mean loss against the best run |
|---|---|---|
| formula, constants fitted without the width | 21 of 34 | 2.9 points |
| formula, the paper's constants | 31 of 34 | 0.4 points |
| lowest S | 30 of 34 | 0.6 points |
| random pick | 9 of 34 | 11.3 points |

### What the checks show

- **Depth cannot be judged from this table.** It has only 10 pairs of width and layers, and the wide models
  are always the deep ones: width 768 only has 10 layers, width 1,024 only has 12.
- **One set of constants does not fit both the end of the run and the curve inside it.** The largest error
  inside the run is where 30% to 70% of the facts are kept: 8.8 points.
- **The picker fails at width 64 when that width is held out.** It finds the best run in 4 of 17 groups,
  because it predicts that a higher peak learning rate keeps more, and the runs show the opposite. For
  widths 128 to 1,024 it finds the best run in 17 of 17 groups.
- **The 12.1 points for weight decay come from one group.** Holding out weight decay 0 removes 265 of the
  375 runs and every width except 64 and 384, so it tests width more than weight decay. For a weight decay
  of 0.1, 0.3 or 1.0 that the fit never saw, the error is 2.2, 6.1 and 6.3 points.
- **The published bound helps where weight decay is on.** It lowers those three errors to 2.2, 4.4 and 3.9
  points and cuts the miss on the width-768 run from 18.4 to 7.1 points. Weight decay still costs more
  than the bound: the weight decay term in the formula falls from 0.85 to 0.41, not to 0.
- **The bound has a cost on two slices.** A run length the fit never saw goes from 5.2 to 5.7 points, and
  the late window from 10.0 to 11.1. The loss in run length is at 4,096 steps (7.5 to 10.4 points), where
  no run has weight decay, so it comes from the other constants moving.
- **Window position is the weakest slice.** Fitted on the early window, the formula is off by 10.0 points
  on the late one. It has no term for when the facts were shown.

## Open directions

### No training (any laptop, minutes)

- A formula that beats the one marked ★ in the [leaderboard](../../problems/forgetting/LEADERBOARD.md) (4.1 points today). Scoring and rules: [problems/forgetting](../../problems/forgetting/).
- A term for learning rate spent late. Adding to z one constant times ln(share of S in the last half of
  the steps after the facts) gives 3.6 points, and seven other ways of writing "late" give 3.6 to 3.7.
  The best is 0.00498 below the ★ formula, just under the 0.005 a new constant must earn. [end_z3.py](../../problems/forgetting/formulas/end_z3.py) has
  this term and one more (12 constants, 3.7 points).
- A formula shape that fits the whole curve inside the run, not only the end point.
- The direction of the peak learning rate at the smallest width.
- A simpler formula. Nine constants already do the job of ten.
- A term for the window position, the slice with the largest error.
- Why weight decay costs more than the published bound allows for.

### Small runs (a laptop CPU, an Apple-silicon laptop or any small GPU; 20 seconds to 10 minutes per run)

- Repeated facts: show the facts again later in the run and see if S still sets what is kept.
- Other optimizers: Adam on every weight, or SGD.
- Settings the formula does not have: batch size, sequence length, warmup. The known miss in the paper used batch 96.
- Weight decay outside the tested values.
- Other kinds of facts, and other places for the facts in the run.
- The cause: what changes in the weights as S grows.

### Larger runs (one 16 GB GPU such as an RTX A4000, or a free Kaggle or Colab T4; 15 minutes to 2 hours per run)

- Wide and shallow models, and narrow and deep ones, so the depth term can be judged.
- Predictions committed before the run for models from 200M to 1B parameters. One exists today, at 503M:
  [rounds/002-forgetting](../../rounds/002-forgetting/).
- Real text and real facts in place of made-up ones.
- Facts added to a model that is already trained (fine-tuning).
- The schedule picker on new runs, with the choice written down before the run.

## Runs the table is missing

Counted from the 375 runs on 2026-11-27. Every run in the table has batch 64 and sequence length 64.

| Setting | What the table has | What is missing |
|---|---|---|
| Weight decay | 0.1 at width 64 (48 runs), 0.3 at widths 64 and 384 (24 and 14), 1.0 at width 64 (24) | any weight decay at widths 128, 256, 512, 768 and 1,024; values below 0.1 |
| Weight decay and run length | 0.1 at 512, 1,024 and 2,048 steps; 0.3 and 1.0 at 1,024 steps only | 0.3 and 1.0 at 512, 2,048 and 4,096 steps |
| Late window | widths 64 to 512 (48, 3, 12, 24 and 12 runs), at 1,024 and 4,096 steps | widths 768 and 1,024; 512 and 2,048 steps; more than 3 runs at width 128 |
| Window positions | two: 20% to 45% and 45% to 70% of the steps | a third position, such as 5% to 30% or 70% to 90% |
| Run length | 512 and 2,048 steps at width 64 only; 4,096 steps at widths 384 and 512 only | 512, 2,048 and 4,096 steps at the other widths |
| Width and layers | 10 pairs; width 768 only with 10 layers, width 1,024 only with 12 | wide and shallow (width 512 or 1,024 with 3 layers); narrow and deep (width 64 or 128 with 12 layers) |
| Peak learning rate | 0.004, 0.008 and 0.016 at every width; 0.032 at width 64 only | 0.032 at the other widths; any peak below 0.004 |

First runs to add, in order. Each set tests one thing the formula marked ★ claims.

1. **Weight decay at new widths (small runs).** Widths 128 and 256, weight decay 0.1, 0.3 and 1.0, peak
   0.004, 0.008 and 0.016, cosine schedule to 0, 1,024 steps, early window: 18 runs. They test the
   published bound where the fit has never seen weight decay.
2. **Small weight decay (small runs).** Width 64, weight decay 0.01 and 0.03, the same three peaks: 6 runs.
   The bound is close to 1 here, so they show how much of the weight decay cost is outside the bound.
3. **A third window (small runs).** Widths 64, 128 and 256, facts at 70% to 90% of the steps, the same
   three peaks: 9 runs. With three positions a window term can be fitted and tested.
4. **Weight decay and run length (small runs).** Width 64, weight decay 0.3, 512 and 2,048 steps, the
   same three peaks: 6 runs.
5. **Weight decay and the late window at large widths (larger runs).** Widths 512, 768 and 1,024 with
   weight decay 0.3, and widths 768 and 1,024 with the late window, peak 0.008: 5 runs.
6. **Depth apart from width (one small set, one larger).** Width 512 with 3 layers and width 128 with 12 layers, the
   same three peaks: 6 runs.

Write the prediction of the ★ formula before each run ([rounds/README.md](../../rounds/README.md)). The
command that trains one run from a settings line is in
[problems/forgetting](../../problems/forgetting/#train-a-run-from-a-settings-line-score-2). New runs go
in as described in [CONTRIBUTING.md](../../CONTRIBUTING.md), section 2.
