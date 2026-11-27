# Contributing

Every pull request comes with the command that checks it. Before opening one, run `git pull --rebase --autostash` and `python3 tools/check.py` (GitHub also runs it).

## 1. A better formula

1. Copy `core/formula_template.py` to `problems/<problem>/formulas/<your_name>.py`.
2. `python3 core/score.py <problem> problems/<problem>/formulas/<your_name>.py --author <github-name> --record`
3. `python3 core/leaderboard.py <problem>`
4. PR: the formula, its new line in `scores.jsonl`, `LEADERBOARD.md`, and the idea in one sentence.

Imports allowed: math, numpy, scipy, statistics, itertools, functools, collections. No reading the run id or the target (`tools/check.py` enforces both). Scoring rules: [problems/README.md](problems/README.md#rules).

Review: a reviewer reruns step 2 and the score must match to 4 decimals. Then a maintainer trains a few new runs at random settings inside the problem's tested range and scores the formula on them. No need to commit predictions first.

## 2. New runs

1. Train with `families/llm/testbed-facts-v1/` (or your own code, described in the PR).
2. Add the training script's output lines, unchanged, as `data/llm/contrib/<github-name>-<date>.jsonl.gz` (1 MB or less; bigger sets: [DATA.md](DATA.md#how-new-data-flows-in)). A maintainer converts them to [data/llm/SCHEMA.md](data/llm/SCHEMA.md) rows.
3. The PR lists the exact commands and the machine (GPU model or CPU).

A maintainer reruns about 2% of the rows (at least one) on another machine. Reruns agree within 0.0001 loss; a row off by more than 0.01 blocks the PR until explained.

## 3. A new problem

Follow [When a question gets its own folder](problems/README.md#when-a-question-gets-its-own-folder) and [Add a problem](problems/README.md#add-a-problem). A question that fails the five tests goes in an issue with the `question` label.

## Sharing

1. **Where the paper lives:** your own repo (public or private), or `drafts/<slug>/` here ([rules](drafts/README.md)). If it stays in your repo, the PR here adds only the formula, its `scores.jsonl` line, `LEADERBOARD.md` and a link.
2. **Partial work is welcome:** a formula, a set of runs, or a negative result ("adding batch size made score 1 worse by 0.01").
3. **A paper** also passes `python3 tools/check_paper.py <paper.md>` after the self-check in [agents/WORKFLOW.md](agents/WORKFLOW.md) (stage 8).

Add your times to [TIMES.md](TIMES.md) in the same PR.

Each PR gets an AI review, then a short maintainer check. Credit: [CREDIT.md](CREDIT.md). Contributors: [CONTRIBUTORS.md](CONTRIBUTORS.md).
