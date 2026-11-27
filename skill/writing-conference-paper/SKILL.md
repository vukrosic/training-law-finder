---
name: writing-conference-paper
description: "Method for a conference paper (ICML/NeurIPS/ICLR style) from topic to finished draft: pick one finding, check the literature, run many fast small experiments with controls, imitate large-scale conditions with small knobs, find general rules or formulas, and write a short paper that teaches. Use when asked to plan, run experiments for, write or revise a machine-learning research paper, its title or abstract."
---

# Writing a conference paper

## 0. Simple research, simply written

Readers want short, simple papers. Simple does not mean shallow: the goal is a real, new finding that is simple to state.

**Research**
- Aim for one general finding a reader can say in one sentence and use. Example: "final loss follows one formula in width, steps and peak learning rate, and predicts runs 2x wider than any fitted run".
- Prefer the simplest mechanism or formula that gets the result.
- Spend experiments on the one finding: its effect, its mechanism, its scaling and its limits. Drop side variants that would become extra claims.
- Search the literature enough to know the finding is new, and no more.

**Writing**
- One main finding, supported by at most 3-4 claims. Anything else goes to the appendix or is dropped.
- Title: the finding in plain words.
- Abstract: at most 150-180 words in short sentences. What the thing is, what it finds (one number at most), why, and the main limit.
- Main text: 3,000-4,500 words and at most 6 figures, one figure per claim, each caption stating the takeaway. The full record goes to an appendix.
- Sentences: short, one idea each, everyday words. Define every term at first use with a small example. No hedging, no restated context, no sentence that only announces what comes next.
- Paragraphs: the claim, the number behind it, and why it holds.
- Self-check before calling a draft ready:
  - Can a smart non-expert say the finding after reading only the abstract?
  - Could any section lose a third of its words without losing a fact? If so, cut it.

**Page layout (HTML or PDF)**
- Under the title: one plain sentence with the main idea and how it works.
- After the abstract: an "In simple words" paragraph, 3-5 sentences, no jargon, one everyday example.
- Next to it: one figure that shows the main idea or mechanism (not a results plot), understandable on its own.
- Paper feel: centered single column (about 720-760 px), serif body font, numbered sections, captions under figures, line height about 1.55, no dashboard styling.

**Audience.** A paper does not have to be useful only to big labs. Small-scale findings for students and one-GPU users count when they hold in at least 2 settings, have a control, and give a rule the reader can apply. Say in one sentence who the paper is for.

**Compare against the current best methods.** The main table includes the strongest published methods for the task, run under the same budget, not only textbook baselines. List the 3-5 strongest current methods and their published numbers in `LITERATURE.md`; run them, or copy published numbers and say which are copied. If a current method beats yours, say so and test whether your idea adds to it.

## 1. Pick one topic

- One mechanism or one question per paper, not a bundle.
- Check the literature first: a `LITERATURE.md` with a table of what is done and what is not. The paper fills a "not done" row.
- Good gaps: a trick many people use that nobody explains; a quantity everybody tunes by hand that a formula could predict.
- Before running a direction, write one line: "Scale guess: helps up to <1-3B / 20B / frontier>, because <reason>." Prefer causes that hold or grow with parameters, tokens, vocabulary and context, and give low priority to effects that only exist in tiny runs.

## 2. Folder layout

```text
<topic>/
  README.md         short: the question, one table, one chart
  LITERATURE.md     sources and the done / not-done table
  <testbed>.py      one script with a flag for every knob, appending JSON rows to results.jsonl
  analysis/*.py     small scripts that load saved runs
  RESULTS.md        rounds of results, each with a table and a one-line takeaway; corrections stated plainly
  PAPER.md, PAPER.html
```

## 3. Experiments

- **Say what will run and why** before a new round.
- **Short runs.** Aim for 5-30 s per small run; a question gets 20-60 runs that each move one condition, not 3 long runs.
- **Modern architecture.** RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head; Muon on hidden matrices, Adam on embeddings, norms and scalars.
- **One quick run first** to check the code and the logging before any sweep.
- **Tune the learning rate of every variant, including the baseline's embedding rate.** An under-tuned baseline can double a measured gain.
- **Controls for every claim:**
  - the same module with frozen random weights (content vs extra weights);
  - a wider plain model with equal weights;
  - the trained model with the module switched off at test time;
  - test-time ablations of the suspected mechanism;
  - the same module placed elsewhere.
- **Trends need several points and a control.** At least 4 points along a knob and a control that removes the obvious other cause. Differences under about 0.01-0.02 loss between single runs are not ordered; drop a claim whose direction flips between neighbouring points.
- **When a result is unclear, add points along the knob or a control** rather than repeating the same run.
- **Write each run command out in full.** In zsh an unquoted `$v` holding several flags is passed as one argument.

### 3a. Search wide with many small runs

- **Design the search before running.** List the scenarios that approximate large models (bigger vocabulary, deeper, more tokens per weight, longer context), the prediction for each, and the result that would break the rule. Then run them back to back.
- **Use reasoning to choose what to run next.** After each block, ask which scenario could break the current rule and run that one. Stop a direction once it stops changing the answer.
- **Build a theory, not a list of results:** general rules plus criteria for when to use the thing, each tied to the runs that test it.

### 3b. Formula papers: sims, scores and speed-ups

Use these when the question is one number to predict from many knobs (see `METHOD.md` in this repo).

- **Two scores.** Score 1 = worst held-out R² on real runs, leaving out each knob's values in turn. Score 2 = R² of predictions written before the runs existed. On an open problem from `problems/` both are the mean held-out RMSE instead, where lower is better. Every report starts with both and names the unit.
- **Measure the ceiling first.** Gradient-boosted trees on all inputs, with the same held-out folds, show roughly the most any formula of those inputs can reach. Score each formula by its gap to that ceiling.
- **Accuracy first, simplify later.** Any number of terms is fine while held-out R² rises. Simplify once the gap to the ceiling is under 0.05.
- **One formula, no trade-offs.** One shared table of held-out R² per slice. Keep a new term only if some slice rises and no slice drops by more than 0.01.
- **Split the effect into parts** when single variables fail (for example a loss change into first-order and second-order parts).
- **Logged quantities are inputs, not stand-in clocks.** Feed weight norms, gradient noise and similar quantities to the formula search as extra inputs.
- **Sim first, real runs decide.** A 1-10 s simulator that keeps the mechanism can screen hundreds of candidates. Check first that it reproduces known effects, and check every sim finding on a few small real runs.
- **Cheapest kill test first, and check the logging before spending compute.**
- **Random draws over the knobs** labs vary, not crossed grids.
- **Speed without shortening.** Never cut the steps after the event you measure. Instead pack small runs on one GPU, branch variants from a shared checkpoint, plant several independent measurement groups in one run, and choose the next runs where the top formulas disagree most. Each speed-up must match a plain run before use.

### 3c. Idea pipeline: when to drop, continue or write

Most ideas die to a tuned baseline. Gate every idea and kill early.

| Gate | Work | Continue only if | Otherwise |
|---|---|---|---|
| 0. Idea card | One sentence; the closest published papers and what differs; the predicted gain; the run that would kill it. Literature search before any run. | No paper already does it, and it changes what the model computes or how it trains, not just a setting. | Kill and log the reason. |
| 1. Screen | Tuned baseline vs the idea at a small default size, equal weights or equal compute, learning rate tuned for both, 2-3 variants. | Gain of 0.02 or more over the tuned baseline, or 0.01 with a clear mechanism, in at least 2 of 3 settings. | Kill. |
| 2. Mechanism | The controls from section 3 and the strongest published competitor at its tuned setting. | The gain is not explained by a tuned rate, extra weights or a published method, and you can say why it works in one sentence. | Kill, or park as an explanation result. |
| 3. Idea page | One page: the idea, one result table, the mechanism, the closest work, the scaling factors to test next. | - | - |
| 4. Study | 60-500 short runs on the scaling grid: every scale factor alone and in pairs, with the tuned baseline and the strongest competitor in every cell, mapping where it wins and where it fails. | The gain survives along at least two factors, with a when-to-use rule. | Write what broke it and stop. |

Rules:
- **Log every idea, killed or kept**, with date, gate reached, gain against the tuned baseline and a one-line reason, so no idea is tested twice.
- **Do not extend a dying idea.** Two variants that miss a gate end it.
- **Start from a measured failure of the baseline** (rare words, recall past some distance, divergence at high learning rate), then invent a mechanism for that failure. Random tweaks to a tuned baseline almost always tie.
- **When the small model cannot show the mechanism, change the data first** (for example synthetic recall tasks for attention ideas).
- **Write the paper only after the study.**

## 4. Imitate large-scale conditions with small knobs

Small runs can stand in for large ones when the knob that matters is moved: vocabulary size, depth, tokens per parameter, context length, data repetition, batch size. For each claim, move each relevant knob over at least 4 points and report whether the effect grows, holds or shrinks. Give a labelled guess for larger scales, never a claim.

## 5. Write the paper

1. Title and one-sentence finding.
2. Abstract (150-180 words) and the "In simple words" paragraph.
3. Main figure of the mechanism.
4. One section per claim: the claim, the figure, the number, the control that rules out the obvious other cause.
5. Limits: every knob that was fixed, the largest scale tested, where the finding fails.
6. Related work: the closest papers and exactly what differs.
7. Appendix: every setting and every run.

Then review it as a strict reviewer would, run the cheapest experiments that fix the weaknesses, and update the draft.
