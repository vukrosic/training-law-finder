# Open orders

The manager appends orders here and deletes each one once `LOG.md` says "done <id>". One order = one owner, one action, one number to report, a done condition, and the score it is meant to raise.

```text
O1 ops (09:00, raises: score 1 data): run jobs/base.jsonl (24 random draws, widths 64-256, steps 200-2000, lr 3e-3..3e-2) at prio 1. Check one row has nonemb_params, tokens, val_loss before the batch. Done when all 24 rows exist; report seconds per run per machine.

O2 theory (09:00, raises: score 1): fit E + A*N^-a + B*D^-b to the O1 rows with --holdout config.d config.steps config.lr --ceiling. Report the per-column table and the gap to the trees. Done when SCORES.md has the first entry.

O3 sim (09:00, raises: nothing yet): propose a sim that runs in under 10 s and keeps the lr effect seen in O1. Show the sign check on 3 effects before any search. Done when LOG.md has the check table.

O4 theory (11:30, raises: score 2): write predictions for jobs/pre1.jsonl (8 runs at width 384, outside the fit range) with the best fit. Done when predictions.jsonl has 8 new lines, before ops starts the file.
```
