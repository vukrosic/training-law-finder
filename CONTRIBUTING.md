# Contributing

Three kinds of pull request. Searching is slow and checking is fast, so every contribution comes
with the command that checks it. Before a pull request, update your clone (`git pull --rebase --autostash`). Every pull request must pass `python3 tools/check.py`
(`pip install numpy scipy pyyaml`); GitHub runs it on each PR.

## 1. A better formula

1. Copy `core/formula_template.py` into `problems/<problem>/formulas/<your_name>.py`.
2. Score it: `python3 core/score.py <problem> problems/<problem>/formulas/<your_name>.py --author <github-name> --record`
3. Rebuild the table: `python3 core/leaderboard.py <problem>`
4. Open a PR with the formula file, the new line in `scores.jsonl` and `LEADERBOARD.md`. The PR text says the idea in one sentence.

A formula may import only math, numpy, scipy, statistics, itertools, functools and collections, and may
not read the run's id or the problem's target. `tools/check.py` enforces this.

A reviewer reruns step 2. The score must match to 4 decimals. The rules (constants count, 0.005 per
constant, floor, settings-only inputs) are in [problems/README.md](problems/README.md).

## 2. New runs

1. Train with `families/llm/testbed-facts-v1/` (or your own code, described in the PR).
2. Add the training script's output lines, unchanged, to a new file `data/llm/contrib/<github-name>-<date>.jsonl.gz` (aim for 1 MB or less; bigger sets go to Hugging Face, see [DATA.md](DATA.md)). No conversion is needed: a maintainer turns them into rows of [data/llm/SCHEMA.md](data/llm/SCHEMA.md) and adds them to the Hugging Face tables at the next refresh.
3. The PR lists the exact commands and the machine (GPU model or CPU).

Verification: a maintainer reruns about 2% of the rows, at least one, on another machine. Reruns of the
same config in our table agree within 0.0001 loss, so a row that differs by more than 0.01 blocks the PR
until explained.

## 3. A new problem

Read "When a question gets its own folder" and "Add a problem" in [problems/README.md](problems/README.md).
A question that does not pass the five tests yet goes in an issue with the `question` label.

## Sharing

You choose where your paper lives. Partial work counts: you do not have to finish a whole paper.

1. **Where the paper lives.** Your own repo (public or private), or a pull request into this repo
   (`drafts/<slug>/`, rules in [drafts/README.md](drafts/README.md)). If the paper stays in your own repo,
   a pull request here adds only the formula, its line in `scores.jsonl` (with `LEADERBOARD.md`) and a link
   to the paper.
2. **No forced claims.** Nobody owns an idea. Your AI may open a short issue titled "Working on: <idea>"
   so others can see it. The issue expires after 14 days with no new result; then anyone may take the idea.
3. **Partial work is welcome as a pull request:** a formula, a set of runs, or a negative result (for
   example "adding batch size made score 1 worse by 0.01"). Others may build on it.
4. **Everyone is credited:** as the author of a leaderboard row, or in a credits line of the paper that
   builds on the work.
5. **Before the pull request:** `python3 tools/check.py` for every pull request, and for a paper also
   `python3 tools/check_paper.py <paper.md>` after the self-check in
   [agents/WORKFLOW.md](agents/WORKFLOW.md) (stage 8).

Each pull request gets an AI review first, then a short check by a maintainer.

## Credit

- The leaderboard keeps the author, date and file hash of every score, so who found what first is on record.
- A paper that uses another contributor's formula, runs or problem cites them, with a link to the leaderboard row or PR.
- If a contributor's work is one of a paper's main claims, offer them co-authorship.
- Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).
