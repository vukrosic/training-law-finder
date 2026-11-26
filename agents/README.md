# Running the research team

Five AI coding-agent sessions work on one project folder. Each session gets one prompt file from this folder as its first message. The manager picks the topic; the human checks in every 15 minutes.

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
  TARGET.md        written by the manager: what number the formula predicts, and the knobs the screens kept
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

1. Open the manager session in your coding agent (Claude Code, Codex or similar) in the project folder. Paste `manager.md`, then: "Find a formula about training."
2. The manager picks the topic, writes `TARGET.md`, and writes the first orders. Usually: O1 literature check and screens over the settings that might matter (theory, ops), O2 baseline runs and learning-rate check (ops), O3 decide whether a sim is worth building (sim).
3. Open one session per remaining role in the same folder, and paste its role prompt.

## The 15-minute check (human)

Read three things, about 2 minutes:

```bash
tail -20 project/LOG.md
python3 queue/status.py project/jobs/current.jsonl project/results.jsonl
head -5 project/SCORES.md
```

Then answer one question: **did score 1 or score 2 move since the last check?**

- Yes: let it run.
- No for 2-3 checks in a row: tell the manager the direction is stuck. The manager picks the next one (a new term family, a new knob, a different topic).
- A queue is idle, a job keeps failing, or a session is waiting for an answer: tell the manager in one line.

The agents pick the topic and do the literature check, the screens, the runs and the fitting. The one rule that keeps them productive is one topic at a time: a search spread over many unrelated ideas produces many ideas and few results, because nothing stops it from testing variants that cannot raise the score.
