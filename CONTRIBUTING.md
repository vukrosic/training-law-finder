# Contributing

Three kinds of pull request. Searching is slow and checking is fast, so every contribution comes
with the command that checks it.

## 1. A better formula

1. Copy `core/formula_template.py` into `problems/<problem>/formulas/<your_name>.py`.
2. Score it: `python3 core/score.py <problem> problems/<problem>/formulas/<your_name>.py --author <github-name> --record`
3. Rebuild the table: `python3 core/leaderboard.py <problem>`
4. Open a PR with the formula file, the new line in `scores.jsonl` and `LEADERBOARD.md`. The PR text says the idea in one sentence.

A reviewer reruns step 2. The score must match to 4 decimals. The rules (constants count, 0.005 per
constant, floor, settings-only inputs) are in [problems/README.md](problems/README.md).

## 2. New runs

1. Train with `families/llm/testbed-facts-v1/` (or your own code, described in the PR).
2. Add rows in the format of [data/llm/SCHEMA.md](data/llm/SCHEMA.md) to a new file `data/llm/contrib/<github-name>-<date>.jsonl.gz`.
3. The PR lists the exact commands and the machine (GPU model or CPU).

Verification: a maintainer reruns about 2% of the rows, at least one, on another machine. Same-seed
reruns in our table agree within 0.0001 loss, so a row that differs by more than 0.01 blocks the PR
until explained.

## 3. A new problem

A folder `problems/<name>/` with `problem.py`, `README.md` (question, allowed inputs, the published
law to beat) and at least one baseline formula. The main README lists problems waiting for a folder.

## Credit

- The leaderboard keeps the author, date and file hash of every score, so who found what first is on record.
- A paper that uses another contributor's formula, runs or problem cites them, with a link to the leaderboard row or PR.
- If a contributor's work is one of a paper's main claims, offer them co-authorship.
- Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).
