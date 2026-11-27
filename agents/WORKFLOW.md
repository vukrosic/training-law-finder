# Workflow: from idea to finished draft

You are the student's AI agent. This file lists the stages in order. Do them one at a time. Before stage 1, ask the start questions in [manager.md](manager.md#first-ask-how-the-student-wants-to-work).

Project files live in `work/<topic>/` (git ignores it). If you are the only session, you do every role yourself: manager, theory, ops and writer.

**Stops.** The mode the student picked decides where you stop and wait:

- **a, fully automatic:** stop only at stage 9.
- **b, alone with 3 stops (default):** stop at stages 1, 5 and 8, and at stage 9. "go" means: take your recommendation and continue.
- **c, talk at every stage:** stop after every stage.

At a stop, give at most 5 short lines: what you found, your recommendation, and one question.

If a stage fails, say so in one line in `LOG.md`, go back to the stage named, and continue. In mode a you decide alone; in modes b and c a failure is also a stop.

## 1. Pick the idea

- **Do:** check what others are working on. Run `gh issue list --state open --search "Working on"` (or read the Issues page on GitHub) and `gh pr list --state open`. Read every `problems/*/LEADERBOARD.md`. Then propose 8-10 ideas as in [manager.md](manager.md#starting-the-project-propose-ideas-then-talk-them-through), skip any idea with an open "Working on" issue younger than 14 days, and recommend one.
- **Read:** `README.md`, `IDEAS.md`, `problems/README.md`, `problems/*/LEADERBOARD.md`, open issues and pull requests.
- **Produce:** `work/<topic>/TARGET.md`: the number to predict, in one sentence, and a line `Tested range: ...` with the range of the data. On an open problem, copy that line from the problem README (final-loss: "Tested range: widths 24 to 1024, 512 to 16,384 steps."). On your own idea, write the range of the rows you use. If the student allowed it, open an issue titled "Working on: <idea>" with 2-3 lines.
- **Done when:** `TARGET.md` exists and no open issue claims the same idea.
- **Stop:** b, c.

## 2. Check the closest published work

- **Do:** search for papers that predict the same number. Find the published formula to beat (for final loss, Chinchilla).
- **Read:** search results; the problem's `README.md` names the law to beat.
- **Produce:** `work/<topic>/LITERATURE.md`: 3-5 closest papers, one line each on what they do and how this idea differs, and the published formula with its numbers.
- **Done when:** the table exists and no paper already gives the formula you plan to find. If one does, go back to stage 1.
- **Stop:** c.

## 3. Get the data

- **Do:** an open problem already has its data in `problems/<name>/data/`. For your own idea, load the full run tables with `core.rows.load()` (it downloads them from Hugging Face on first use), or plan new runs with `testbed/train.py` or `families/llm/testbed-facts-v1/`. Open one row and confirm it has every input and the target.
- **Read:** `DATA.md`, `data/llm/SCHEMA.md`, `core/columns.yaml`.
- **Produce:** a "Data" block in `TARGET.md`: source, row count, the inputs, and the column used for held-out groups.
- **Done when:** the rows load, and the group column has at least 4 values with at least 5 rows each.
- **Stop:** c.

## 4. Search for a formula

- **Do:** follow phases 1-2 of `METHOD.md` and the rules in `agents/theory.md`. Start each formula from `core/formula_template.py`. Score every candidate without `--record` (only the final formula is recorded, in stage 5) and keep one table of candidates in `SCORES.md`.
- **Read:** `METHOD.md`, `agents/theory.md`, `problems/README.md` (rules on constants and the floor).
- **Produce:** `work/<topic>/formulas/*.py` and `SCORES.md`, newest first.
- **Done when:** two rounds of new terms did not improve score 1, or the formula beats the top of the leaderboard.
- **Stop:** c.

## 5. Score 1 (held out)

- **Do:** score the final formula once and save the output. Open problem: `python3 core/score.py <problem> <formula> --author <github-name> --record | tee work/<topic>/score1.txt`. Own idea: `python3 fit/laws.py ... --holdout ... | tee work/<topic>/score1.txt`.
- **Read:** `problems/<name>/LEADERBOARD.md`, the published formula's score.
- **Produce:** `score1.txt`, and a new line in `problems/<name>/scores.jsonl` for open problems.
- **Done when:** score 1 is saved and compared with the leaderboard top and the published formula, with the unit named ("score 1 = 0.0441 RMSE").
- **Stop:** b, c.

## 6. Score 2 (predicted before the runs exist)

- **Do, where possible:** pick 10-50 new settings, some outside the fit range (wider, longer). Write predictions with `core/predict.py` (open problem, see `rounds/README.md`) or `fit/prereg.py write` (own idea). Freeze them before any of these runs exist: write `<UTC time> <sha256> <file>` to `work/<topic>/FREEZE.md`, using `shasum -a 256 <file>` and `date -u +%Y-%m-%dT%H:%MZ`. Make the time public if you can: push the file to the student's repo, or open a pull request into `rounds/` with predictions only. Then train the runs and score them with `core/round_score.py` or `fit/prereg.py score`. Never change a frozen file; a better formula gets a new file for new runs.
- **Read:** `rounds/README.md`, `METHOD.md` phase 3.
- **Produce:** the predictions file, `FREEZE.md`, the results, and score 2 with its number of runs.
- **Done when:** score 2 is computed, or one line in `SCORES.md` says why it was skipped (for example, no compute for the wider runs). Then the paper says score 2 was not measured.
- **Stop:** c.

## 7. Write the draft

- **Do:** write `work/<topic>/PAPER.md` with `skill/writing-conference-paper/SKILL.md` and `agents/writer.md`. It must contain: an Abstract of 180 words or fewer; score 1 with its unit; a "Closest work" section; and one line that states the tested range with the same numbers as the `Tested range:` line in `TARGET.md` (for example "All runs are small scale: widths 24 to 1024, 512 to 16,384 steps."). If it reports score 2, add the line `Score 2 predictions: <file>, sha256 <hash>, frozen <time>` copied from `FREEZE.md`.
- **Read:** the skill, `agents/writer.md`, `SCORES.md`, `LITERATURE.md`, `FREEZE.md`.
- **Produce:** `PAPER.md` and its figures.
- **Done when:** every section is written and every number cites the file it came from.
- **Stop:** c.

## 8. Self-check

- **Do:** run [reviewer.md](reviewer.md) on the draft and fix what it finds, up to 3 rounds. Then run `python3 tools/check_paper.py work/<topic>/PAPER.md --scores work/<topic>/score1.txt` (add `--problem <name>` if the paper does not name its problem) and fix every failure.
- **Read:** `agents/reviewer.md`.
- **Produce:** `work/<topic>/REVIEW-<n>.md` for each round, and the fixed `PAPER.md`.
- **Done when:** the reviewer lists no major issue and `check_paper.py` prints 0 failures.
- **Stop:** b, c. In mode a, this is where you first show the draft.

## 9. Share

- **Do:** ask the student where the paper should live, then follow "Sharing" in [CONTRIBUTING.md](../CONTRIBUTING.md#sharing). Run `python3 tools/check.py` before any pull request.
- **Produce:** what the student chose: their own repo, a pull request with the formula, its score line and a link, or a full draft in `drafts/<slug>/`.
- **Done when:** the student has the link.
- **Stop:** always. Never publish or open a pull request without the student's yes.

Partial work can be shared at any stage: a formula, a set of runs, or a negative result. See CONTRIBUTING.md.
