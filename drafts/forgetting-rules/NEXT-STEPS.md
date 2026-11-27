AI notes, not checked by a person.

# Next steps for the forgetting rules

Ideas for anyone who wants to continue this work. They are suggestions: pick one, change it, or bring your
own. Send results as a pull request ([CONTRIBUTING.md](../../CONTRIBUTING.md)) or talk about them on Discord.

The first part lists five checks that are already done, with their results. The second part lists open
directions, grouped by the hardware they need.

## Already checked (no training)

Run `python3 drafts/forgetting-rules/next_steps_checks.py` to print the numbers of checks 3 to 5 from the
public table. It takes about one minute on a laptop CPU. "Points" are percentage points of facts kept.

| # | Check | Result |
|---|---|---|
| 1 | Copied rows | 25 of the 375 rows are copies of another run. Without them the held-out error is 4.3 points instead of 4.4. |
| 2 | Noise floor | The floor of 1.3 points comes from one pair of runs with the same settings. The larger internal run log has 43 such pairs. They differ by 0.7 points. The two pairs at width 256 differ by 2.0 points. |
| 3 | Nine constants against ten | Removing the depth term changes the held-out error by 0.3 points or less on all seven slices. Nine constants are equal or better on four of them. |
| 4 | Inside the run | With constants fitted on end points, the error on the 36,193 readings inside the runs is 8.0 points. Fitted on all readings it is 5.7 points, but the error at the end rises from 3.8 to 5.9. |
| 5 | Schedule picker | When only the schedule differs, the formula picks the best run in 107 of 107 groups. The plain rule "pick the lowest S" does the same. |

Check 2 uses runs that are not in the public table, so the script does not print it.

Check 3, held-out error in points:

| Held-out slice | Ten constants | Nine constants |
|---|---|---|
| width | 4.41 | 4.42 |
| layers | 4.31 | 4.37 |
| steps | 5.15 | 5.06 |
| peak learning rate | 4.15 | 4.18 |
| schedule | 2.76 | 2.76 |
| weight decay | 12.12 | 11.97 |
| window position | 8.46 | 8.19 |

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
- **The two weakest slices are weight decay and window position.** A weight decay value or a window
  position the formula never saw costs 12.1 and 8.5 points.

## Open directions

### No training (any laptop, minutes)

- A formula that beats 4.4 points on the public table. Scoring and rules: [problems/forgetting](../../problems/forgetting/).
- A formula shape that fits the whole curve inside the run, not only the end point.
- The direction of the peak learning rate at the smallest width.
- A simpler formula. Nine constants already do the job of ten.
- Weight decay and window position, the two slices with the largest error.

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

New runs go in as described in [CONTRIBUTING.md](../../CONTRIBUTING.md), section 2.
