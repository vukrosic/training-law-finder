# Method

How a few-day project finds one formula that predicts a number about AI training.

## The two scores

Everything the team does is judged by two numbers. Every report starts with them.

**Score 1: worst held-out R² on real runs.** For each knob the formula must handle (width, depth, steps, peak learning rate, schedule, model family), leave out one value of that knob, or one quantile bin of a knob with many values, refit on the rest, and predict the left-out runs. The R² of those out-of-fold predictions is the knob's held-out R². Score 1 is the worst of them. `fit/laws.py --holdout` computes it.

**Score 2: R² of pre-registered predictions.** Before a batch of new runs exists, the current best fit predicts every run in it, and the predictions are saved with a time stamp. After the runs finish, the R² of those predictions is score 2. `fit/prereg.py` writes and scores them. Score 2 cannot be tuned after the fact, so it is the number a reader trusts.

In-sample R² is printed by the tools but is never a score.

## Phases

**0. Pick the topic and set up (1-2 hours).**
- The project starts with one instruction: "Find a formula about training." The manager agent proposes 8-10 topics, at least half of them new (not in the README's table, no published formula), each with a guess at the mechanism and the fastest test. The human talks them through with it until they agree on one: a number the testbed can measure in a short run, with no settled published formula yet.
- Search the literature for existing laws of that number. The formula should beat or extend the best published one; put it in as the first candidate.
- Write the target in one sentence in `TARGET.md`: what number, over what range. Example: "final validation loss of Muon-trained dense transformers, widths 64-384, 200-4000 steps".
- Run one job, open its row, and confirm every input and the target are logged.
- Run the largest planned job alone to measure its time and memory.

**1. Find which settings matter, then first data (half a day).**
- Quick screens over every setting that might matter (learning rate, schedule, warmup, width, depth, batch size, data, optimizer settings, plus logged quantities such as weight norm or update size). A setting that moves the target less than run-to-run noise is fixed and dropped from the search. The ones that remain are the formula's inputs, called knobs below.
- Random draws over the knobs (not a crossed grid): every run adds a new combination, so each knob gets many distinct values for the same compute.
- A learning-rate bracket at a few sizes, so no candidate is fitting badly tuned runs.
- First fit of the published law and of a log-linear model with pairwise interactions. Measure the tree ceiling (`--ceiling`).

**2. Candidate loop (1-2 days).**
- One shared table: held-out R² per knob, one row per candidate.
- A new term is kept only if some column rises and no column drops by more than 0.01.
- Accuracy first: add terms freely while held-out R² rises. Simplify only once the gap to the tree ceiling is under about 0.05.
- When two candidates tie, run where their predictions differ most.
- When a single variable fails, split the target into parts (early and late training, first- and second-order loss change) and fit the parts.
- Logged quantities (weight norm, gradient noise, update size) go into the search as extra inputs, not as replacements for the knobs.

**3. Pre-registration rounds (throughout, more at the end).**
- Before any larger or longer batch, write predictions for all of it.
- Put some runs outside the fit range (a wider model, a longer run, a new schedule). Those test whether the formula extrapolates.
- Never rewrite a prediction. A better fit gets a new prediction file for new runs.

**4. Optional simulator.** If real runs take minutes, a 1-10 s stand-in that keeps the effect can screen candidate terms. It keeps its compute only while its findings raise score 1 or 2 when checked on real runs. Check one finding on a few small real runs before trusting the next ten.

**5. Write.** When score 1 is above the agreed bar, score 2 has at least 10 rows close to score 1, and two rounds of new terms changed neither score, write the paper (`skill/writing-conference-paper/SKILL.md`).

## The human's role

The human starts the project, talks through the agent's ideas and helps pick the topic, checks in every 15 minutes (`agents/README.md`) to push the work where it looks most promising, and works with the writer on the paper. The agents propose the ideas, find which settings matter, and do the literature checks, the runs and the fits.

## The goal of the paper

One new formula that predicts a number about training nobody could predict before, with 3-4 claims: the formula, score 1, score 2 including runs outside the fit range, and the range where it holds.

## Lessons from the project this method came from

- **One topic at a time.** A search spread over about 2,400 unrelated ideas ran about 15,000 runs with no idea beating the baseline. Progress came once the project committed to one topic and every run served the formula for it. So the manager and the human agree on one topic at the start, and the manager keeps the whole team inside it until the scores stop moving.
- **A score keeps the search honest.** Before the two scores existed, sessions reported many partial results that did not add up to a formula. After every order had to name the score it raised, work that could not raise a score stopped being ordered.
- **Check the logging before the batch.** 30 cloud jobs once ran with the logging flag off. One finished row, checked first, would have caught it.
- **Cheapest kill test first.** One simulator run showed that a candidate quantity was proportional to the peak learning rate, which removed it as an independent input and saved 15 GPU runs.
- **A sim can give wrong signs.** One simulator gave the opposite sign for an effect when the machine was heavily loaded and the right sign at low load. Every sim finding needs a real-run check.
- **Random draws beat crossed grids** once the formula has more than two inputs: a grid spends most runs on combinations that teach little, and its held-out slices have few distinct values.
- **Many single "clock" variables failed** (weight size, margin, path length, largest singular value, gradient noise share) as stand-ins for the knobs. As extra inputs to the formula, some of them helped.

## Example targets

- Final loss from width, depth, steps and peak learning rate (the worked example in `examples/loss-law/` is the two-input version).
- The best peak learning rate as a function of width, batch size and run length.
- Loss after a learning-rate schedule change, from the schedule's shape.
- Facts recalled from the number of times each fact was seen, model size and schedule (the facts option in `testbed/train.py` plants made-up biographies and logs per-person recall).
- Loss spike or divergence threshold as a function of learning rate, warmup and width.
