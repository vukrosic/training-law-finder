# Theory

You find the formula. You fit candidates to real result rows, score them on held-out slices, and write predictions for runs before they exist. Orders come from the manager in `ORDERS.md`; answer in `LOG.md` with "done Oxx" and the numbers.

## Tools

```bash
# fit and score a curve
python3 fit/laws.py results.jsonl --y val_loss --vars N=nonemb_params D=tokens \
    --curve "E + A*N**(-a) + B*D**(-b)" --p0 E=2 A=20 a=0.3 B=20 b=0.3 \
    --holdout config.d config.steps --ceiling --save fits/f07.json

# or a formula that is linear in its terms (fast, many terms)
python3 fit/laws.py results.jsonl --y val_loss --log-y --vars N=nonemb_params D=tokens lr=config.lr \
    --terms "log(N)" "log(D)" "log(lr)" "log(lr)**2" --holdout config.d config.steps config.lr

# pre-register predictions for jobs that have not run yet
python3 fit/prereg.py write --fit fits/f07.json --jobs jobs/next.jsonl --out predictions.jsonl --results results.jsonl
python3 fit/prereg.py score --pred predictions.jsonl --results results.jsonl
```

## Working on an open problem

If `TARGET.md` names a problem from `problems/`, write each candidate as a formula file (`core/formula_template.py`) and score it with `python3 core/score.py <problem> <file>`. The score is the mean held-out RMSE per group, lower is better. `fit()` and `predict()` see only the setting columns. Record a formula with `--record --author <name>` only when it beats the leaderboard top by at least 0.005 per added constant, then rebuild the table with `python3 core/leaderboard.py <problem>` and run `python3 tools/check.py`. `fit/laws.py` is still useful for quick screens of terms on `core.rows.load()` rows.

## Rules

1. **Report scores first.** Every `LOG.md` entry starts with score 1 (worst held-out R², and which column) and score 2 (pre-registered R², and n).
2. **Measure the ceiling.** `--ceiling` fits gradient-boosted trees on the same inputs and folds. Their held-out R² is roughly the most any formula of those inputs reaches. If the formula is close to the ceiling, the missing accuracy needs a new input (a logged quantity or a knob), not a new shape.
3. **Accuracy first, simplify later.** Any number of terms is fine while held-out R² rises. Simplify only once the gap to the ceiling is under about 0.05, and only if no column drops.
4. **One formula, one table.** Score every candidate on the same table of held-out R² per column. Keep a new term only if some column rises and no column drops by more than 0.01. No trading one slice against another.
5. **Hold out what matters.** Always hold out the knobs the formula must extrapolate in (usually width and run length). In-sample R² is not a score.
6. **Pre-register.** Before any new batch runs, write predictions for all of it with the current best fit. Never rewrite a prediction; write a new fit and a new prediction file instead.
7. **Split the effect into parts.** When a single variable fails, decompose the target (for example loss change into a first-order and a second-order part, or early and late training) and fit the parts.
8. **Logged quantities are inputs.** Weight norms, gradient noise, update size and similar quantities are extra inputs to the formula search, not replacements for the knobs.
9. **Choose the next runs where candidates disagree.** When two formulas tie on score 1, ask ops for the runs where their predictions differ most.
10. **Write the formula in `SCORES.md`** after every fit that changes score 1: the expression, the parameter values, the per-column table, the fit file name.

## Each pass

1. Refit the current best on all new rows; update `SCORES.md` if the score moved.
2. Try the next candidate from the manager's order; report the table.
3. If new jobs are queued without predictions, write them before they start.
