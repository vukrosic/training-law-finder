# Focused persistence: how we run research here

This is our default strategy since 2026-11-28. One agent stays on one question for days and ends with a blog post. Every run it makes goes to a public dataset.

## The strategy

1. **One question per agent.** Each agent works on one machine, and it keeps the same question for the whole project.
2. **Do not switch.** An early result that looks bad is not a reason to leave. If the idea is wrong, find out why. That reason is the finding.
3. **Many tiny runs.** Two hundred runs of 3 seconds teach more than one run of 30 minutes. An AI's strength is to search, reason and check, many times over.
4. **Start from known work.** A published paper on the question is where you begin, not a reason to quit. Read it, then measure what it left out.
5. **Aim for a formula.** Settings go in and a number comes out, for example the best learning rate from width and steps. A formula can be checked on runs it never saw, so the agent can test its own work without a human.

## Pick the question

- Choose one knob, or one comparison between two methods.
- The number the agent predicts should be one that costs compute to find. Examples: the best learning rate, the decay length, the warmup length, or the gap between two optimizers.
- Write `work/<topic>/TARGET.md` before the first run. It gives the question, the inputs, the output, the tested range, and the closest published formula.

## What counts as a finding

There are three kinds, from strongest to weakest:

1. **A formula.** It predicts held-out runs better than the best published formula. If it is also new and still holds at about 10 times the size, it is a candidate for an ICML paper.
2. **A mechanism.** A control run shows why something happens. For example, turning one part off removes the effect.
3. **A limit.** A published rule breaks in some range, and the runs show where.

All three are worth a blog post. These are not findings:

- a win at one size only;
- a known trick run again with no new rule;
- a sweep with no formula and no reason behind it.

## The loop

Each round:

1. State the current rule in one sentence at the top of the round in `RESULTS.md`.
2. List the runs most likely to break that rule.
3. Run them: 20 to 100 runs per round.
4. Fit, then score the fit on held-out runs (score 1, [METHOD.md](../METHOD.md#the-two-scores)).
5. Write the table and a one-line takeaway in `RESULTS.md`.
6. Pick the next run that could break the rule.

Log every step in `LOG.md`. Update `STATUS.md` every hour with three lines: the current rule, its score, and the next round. Use existing runs (`data/llm/runs.jsonl.gz`) before training new ones.

## The blog post

Every project ends with a blog post, even when it is not paper level. A good post:

- states the finding in its title, in plain words;
- gives the formula or rule a reader can use;
- shows one chart or one table that proves it;
- gives the tested range (widths, steps, batch sizes);
- says how it was checked: the held-out score, against the best published formula;
- names the closest published work and how this differs;
- says what it does not cover;
- tells how to reproduce it: one command and the link to the data.

Write 400 to 1,200 words in short, plain sentences. Put it in `blog/<yyyy-mm-dd>-<slug>/README.md`, with at most 2 images, each under 500 KB.

## Finished means

A project is finished when all of these are true:

1. The blog post is written as described above.
2. Score 1 is in the post. Add score 2 (predicted before the runs existed) when there is time.
3. The rule was tested on at least 2 knobs it was not fitted on, for example wider models and 10 times longer runs, and the post says what happened.
4. All runs, job files, scripts, `LOG.md` and `RESULTS.md` are uploaded to the Hugging Face dataset [vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research) under `studies/<slug>/`.
5. A pull request with the blog post and the small code files is open, and `python3 tools/check.py` passes. No data goes to GitHub.

A negative result is finished by the same rules. "Muon early-training damping does not help, and here is why" is a good post.

## Time box

- Each question gets 2 to 4 days of box time.
- Day 1: closest published work, existing runs, first sweep, first fit.
- Days 2 and 3: break the rule (held-out runs, wider models, longer runs), then explain why it holds or fails.
- Day 4 at the latest: write the blog post and upload the data.
- Stop early only when the rule is settled and the post is done. Do not stop because a result looks bad; write the "no" post instead.

## Supervision

A manager session checks every box about once an hour. It reads `STATUS.md`, the `LOG.md` tail and the run count, then sends one message when it finds one of these:

- **Drift:** the agent started a second question, or is tuning a side knob that does not test the rule. The message names the rule and the next run that tests it.
- **Stall:** no new runs for an hour while the machine is free. The message names the next run.
- **Sweep with no rule:** two rounds passed with no one-sentence rule at the top of `RESULTS.md`. The message asks for the rule before more runs.
- **Done too early:** a post is being written before the held-out score or the 2 untouched knobs. The message names the missing test.
- **Post review:** when a post is ready, the manager checks it against "The blog post" and "Finished means" above. It sends back each missing item by name.

## Data for later AIs

Every run is one JSON line with:

- every setting;
- the loss curve, if it was logged;
- the final loss;
- the time, the hardware and the git hash.

Each study on Hugging Face uses the same layout: `studies/<slug>/runs.jsonl.gz`, `jobs/`, `code/`, `LOG.md`, `RESULTS.md` and `README.md`. A later AI can then load every study at once and look for patterns across them.

## This repo is the home of every rule

The goal is one place that holds every rule of training we find: the formulas, the mechanisms, the limits and the runs behind them.

- Every finished finding gets a row in [CLAIMS.md](../CLAIMS.md): the rule in one sentence, its tested range, its held-out score, a link to the post and a link to the data.
- [blog/README.md](../blog/README.md) lists every post, newest first. A post that lives somewhere else (another GitHub repo, a website) still gets a line there with its link.
- Code stays small here; the runs live on Hugging Face under `studies/<slug>/`. CLAIMS.md and blog/README.md tie the two together.

## Later

Other people will run the same loop on their own machines and add their studies the same way, through a blog-post pull request plus a data upload. Until then, keep each study clean enough for a stranger to rerun.
