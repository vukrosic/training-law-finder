# Running the research team

One or more AI coding-agent sessions work on one project folder, `work/<topic>/` inside the clone. Each session gets one prompt file from this folder as its first message. The manager proposes ideas and chooses the topic with the human; the human checks in every 15 minutes and helps push the work.

| Session | Prompt | Owns |
|---|---|---|
| manager | [manager.md](manager.md) | `ORDERS.md`, the plan, every decision about compute |
| ops | [ops.md](ops.md) | the queues: what runs where, stale or broken jobs, logs |
| theory | [theory.md](theory.md) | the formula: candidates, fits, score 1, pre-registered predictions |
| sim | [sim.md](sim.md) | an optional cheap simulator, used only while it earns its compute |
| writer | [writer.md](writer.md) | `PAPER.md` and figures, written from finished results only |
| reviewer | [reviewer.md](reviewer.md) | a strict ICML review of the draft, run before any pull request |
| watcher | [watcher.md](watcher.md) | optional: keeps every machine busy on long runs or several machines, restarts crashed runs smaller |

The stages from idea to finished draft, and where the agent stops for the student, are in [WORKFLOW.md](WORKFLOW.md). The research strategy (one question for days, a blog post per study, what finished means) is in [focused-persistence.md](focused-persistence.md). One session can do every role.

Small projects can run with two sessions: manager and theory, where theory also runs the queue. Add ops when there is more than one machine, sim when real runs take minutes, and writer once score 2 has rows.

## Keep your copy of the repo current

The repo changes while you work: fixed bugs, new problem data, new formulas on the leaderboards. Every session keeps its copy current. Put your project in `work/<topic>/` inside the clone: `work/` is ignored by git, so pulling never touches your runs or drafts.

At the start of a session, at every 15-minute check, and before a pull request:

```bash
git fetch origin
git status -sb                      # "behind N" means there are new commits
git pull --rebase --autostash       # bring them in; your own changes stay on top
```

After a pull that changed `core/`, `fit/`, `queue/` or `testbed/`, read `git log --oneline -5` and tell the manager in one line, because a changed script can change results.

If the pull fails with "diverged" or "unrelated histories", the repo history was rewritten (this happened once, on 2026-11-26, to take large data files out of the history). Your `work/` folder is safe either way. Keep your own commits on a branch and move to the new history:

```bash
git branch my-old-main              # keeps your commits
git fetch origin
git reset --hard origin/main        # tracked files now match GitHub; work/ is untouched
git cherry-pick <commit>            # copy back any of your commits from my-old-main
```

Before a pull request, also run `python3 tools/check.py`. If two people added a score to the same problem, git keeps both lines of `scores.jsonl`; rebuild the table with `python3 core/leaderboard.py <problem>`.

## Files the team shares

```text
work/
  STUDENT.md       written by the manager: the student's answers to the start questions
  MACHINE.md       written by the manager: cores, memory, GPUs, and how many runs fit at once
work/<topic>/
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

1. Open the manager session in your coding agent (Claude Code, Codex or similar) in the clone, and have it make `work/<topic>/` once the topic is chosen. Paste `manager.md`, then: "Find a formula about training."
2. The manager proposes 8-10 topics by the rule in [manager.md](manager.md#starting-the-project-propose-ideas-then-talk-them-through). Talk them through with it: ask questions, push back, combine ideas or add your own, until you agree on one.
3. The manager writes `TARGET.md` and the first orders. Usually: O1 literature check and screens over the settings that might matter (theory, ops), O2 baseline runs and learning-rate check (ops), O3 decide whether a sim is worth building (sim).
4. Open one session per remaining role in the same folder, and paste its role prompt.

## The 15-minute check (human)

Read three things, about 2 minutes:

```bash
tail -20 work/<topic>/LOG.md
python3 queue/status.py work/<topic>/jobs/current.jsonl work/<topic>/results.jsonl
head -5 work/<topic>/SCORES.md
git fetch origin && git status -sb   # new commits on GitHub? the manager pulls them
```

Then answer one question: **did score 1 or score 2 move since the last check?** The manager's report also suggests 1-2 places to push next; pick one or suggest your own.

- Yes: let it run.
- No for 2-3 checks in a row: tell the manager the direction is stuck. It proposes 3-5 new directions (a new term family, a new knob, a different topic), and you choose together.
- A queue is idle, a job keeps failing, or a session is waiting for an answer: tell the manager in one line.

The agents propose the ideas and do the literature check, the screens, the runs and the fitting; the human helps choose and push. The one rule that keeps them productive is one topic at a time: a search spread over many unrelated ideas produces many ideas and few results, because nothing stops it from testing variants that cannot raise the score.
