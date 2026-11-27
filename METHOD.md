# Method

How a few-day project finds one formula that predicts a number about training.

## The two scores

Every report starts with both and names the unit ("score 1 = 0.0441 RMSE", "score 1 = 0.87 R²"). Every order names the score it raises; before the scores existed, sessions reported partial results that never added up.

**Score 1: worst held-out R² on real runs.** For each knob (width, depth, steps, peak learning rate, schedule, model family), leave out one value, or one quantile bin of a knob with many values, refit on the rest and predict the left-out runs. Score 1 is the worst knob's R². `fit/laws.py --holdout` computes it.

**Score 2: R² of pre-registered predictions.** Before a batch exists, the current best fit predicts every run in it and the predictions are saved with a time stamp. `fit/prereg.py` writes and scores them. It cannot be tuned after the fact, so it is the number a reader trusts.

In-sample R² is never a score. On a problem in `problems/`, score 1 is instead the mean held-out RMSE over groups from `core/score.py`, lower is better ([problems/README.md](problems/README.md#two-scores)).

## Phases

**0. Topic and setup (1-2 hours).**
- The manager proposes 8-10 topics ([rule](agents/manager.md#starting-the-project-propose-ideas-then-talk-them-through)), each with a guessed mechanism and fastest test; the human and manager agree on one and keep to it until the scores stop moving. (About 15,000 runs spread over about 2,400 unrelated ideas once beat no baseline.)
- Find the best published law for that number; it is the first candidate to beat or extend.
- Write the target in one sentence in `TARGET.md`, with its range. Example: "final validation loss of Muon-trained dense transformers, widths 64-384, 200-4000 steps".
- Run one job and confirm every input and the target are logged (30 cloud jobs once ran with logging off). Run the largest planned job alone to measure time and memory.

**1. Find which settings matter, then first data (half a day).**
- Quick screens over every candidate setting (learning rate, schedule, warmup, width, depth, batch size, data, optimizer settings, logged quantities such as weight norm or update size). A setting that moves the target less than run-to-run noise is fixed and dropped. The rest are the knobs.
- Cheapest kill test first: one simulator run showed a candidate was proportional to peak learning rate, which removed it as an input and saved 15 GPU runs.
- Random draws over the knobs, not a crossed grid: each run adds a new combination, and held-out slices get many distinct values.
- A learning-rate bracket at a few sizes, so no candidate fits badly tuned runs.
- Fit the published law and a log-linear model with pairwise interactions. Measure the tree ceiling (`--ceiling`).

**2. Candidate loop (1-2 days).**
- One shared table: held-out R² per knob, one row per candidate.
- Keep a new term only if some column rises and none drops by more than 0.01.
- Add terms freely while held-out R² rises; simplify only once the gap to the tree ceiling is under about 0.05.
- When two candidates tie, run where their predictions differ most.
- When a single variable fails, split the target into parts (early and late training, first- and second-order loss change) and fit each.
- Logged quantities enter as extra inputs, not replacements for knobs. As single "clock" replacements, weight size, margin, path length, largest singular value and gradient noise share all failed; as extra inputs some helped.

**3. Pre-registration rounds (throughout, more at the end).**
- Before any larger or longer batch, predict all of it.
- Put some runs outside the fit range (wider, longer, new schedule) to test extrapolation.
- Never rewrite a prediction; a better fit gets a new file for new runs.

**4. Optional simulator.** If real runs take minutes, a 1-10 s stand-in that keeps the effect can screen terms. It keeps its compute only while its findings raise score 1 or 2 on real runs. Check each finding on a few small real runs before trusting the next: one simulator flipped an effect's sign under heavy machine load.

**5. Write** when score 1 is above the agreed bar, score 2 has at least 10 rows close to score 1, and two rounds of new terms changed neither score ([skill](skill/writing-conference-paper/SKILL.md)). The paper: one new formula for a number nobody could predict before, with 3-4 claims (the formula, score 1, score 2 including runs outside the fit range, the range where it holds).

## Example targets

More in [IDEAS.md](IDEAS.md). For fact recall, `testbed/train.py`'s facts option plants made-up biographies and logs per-person recall; `examples/loss-law/` is a two-input final-loss example.
