# Running the research team

Five AI coding-agent sessions work on one project folder. Each session gets one prompt file from this folder as its first message, plus the target sentence. The human checks in every 15 minutes.

| Session | Prompt | Owns |
|---|---|---|
| manager | [manager.md](manager.md) | `ORDERS.md`, the plan, every decision about compute |
| ops | [ops.md](ops.md) | the queues: what runs where, stale or broken jobs, logs |
| theory | [theory.md](theory.md) | the formula: candidates, fits, score 1, pre-registered predictions |
| sim | [sim.md](sim.md) | an optional cheap simulator, used only while it earns its compute |
| writer | [writer.md](writer.md) | `PAPER.md` and figures, written from finished results only |

Small projects can run with two sessions: manager and theory, where theory also runs the queue. Add ops when there is more than one machine, sim when real runs take minutes, and writer once score 2 has rows.

## Files the team shares

```text
project/
  TARGET.md        one sentence: what number the formula predicts, from which knobs
  ORDERS.md        numbered orders from the manager (template: ORDERS.template.md)
  LOG.md           every session appends "done O12: <numbers>" and short notes
  jobs/*.jsonl     job files (queue/run_jobs.py); withdrawn files move to jobs/withdrawn/
  results.jsonl    one row per finished run; rows are never deleted
  fits/*.json      saved formulas (fit/laws.py --save)
  predictions.jsonl  pre-registered predictions (fit/prereg.py)
  SCORES.md        score 1 and score 2 after each fit, newest first
  PAPER.md         the writer's draft
```

## How to start

1. Write `TARGET.md`. Example: "Predict final validation loss from width, depth, steps and peak learning rate, for Muon-trained dense transformers, at widths not used for fitting."
2. Open one session per role in your coding agent (Claude Code, Codex or similar) in the project folder, and paste the role prompt followed by `TARGET.md`.
3. The manager writes the first orders. Usually: O1 baseline grid and learning-rate check (ops), O2 first candidate list (theory), O3 decide whether a sim is worth building (sim).

## The 15-minute check (human)

Read three things, about 2 minutes:

```bash
tail -20 project/LOG.md
python3 queue/status.py project/jobs/current.jsonl project/results.jsonl
head -5 project/SCORES.md
```

Then answer one question: **did score 1 or score 2 move since the last check?**

- Yes: let it run.
- No for 2-3 checks in a row: tell the manager the direction is stuck and choose the next one together (a new term family, a new knob, a different target).
- A queue is idle, a job keeps failing, or a session is waiting for an answer: tell the manager in one line.

The human picks the direction; the agents do the literature check, the runs and the fitting. A fully autonomous search tends to produce many ideas and few results, because nothing stops it from testing variants that cannot raise the score.
