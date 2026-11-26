# Final loss: predict the loss before you train

**Predict a run's final validation loss from its settings:** model size, tokens, learning rate,
schedule, batch.

Why it matters: a formula like this picks the model size and learning rate for a budget without a
sweep. Chinchilla answers it for the best learning rate only; real runs are often not at the best one.

## Data

`data/llm.jsonl.gz`: a frozen copy of every dense Muon run with a 1,024-token vocabulary in the full
run table, with diverged runs removed: 22,231 runs, widths 24-2048, 1-12 layers, 160-4096 steps,
constant / cosine / WSD schedules and many learning rates. Grouped by width. Frozen on 2026-11-26 with
`python3 tools/build_problem_data.py final-loss`; new runs join only when a maintainer rebuilds it
and rescores every formula. Columns: [core/columns.yaml](../../core/columns.yaml).

Allowed inputs: every setting column (see `data/llm/SCHEMA.md`). Not allowed: val_loss, val_bpb,
the fact-probe columns, wnorm.

Caveat: these runs mix facts into the text and the evaluation set size differs between some
families (`eval_seqs`). That is part of the problem; a formula may use those columns.

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [chinchilla_lr](formulas/chinchilla_lr.py) | 8 | 0.1476 | Chinchilla + a bowl in ln(learning rate) + a decay term |
| [chinchilla](formulas/chinchilla.py) | 5 | 0.2552 | L = E + A/N^a + B/D^b (Hoffmann et al. 2022) |
| [mean](formulas/mean.py) | 1 | 0.4398 | constant |

Floor 0.0001 (115 runs repeated on different machines agree within 0.00006). The best formula is
1,800 times the floor away: this problem is wide open. The largest errors are at width 1024 and 768.

## Published work to beat

Chinchilla (Hoffmann et al. 2022) is the baseline. Laws with learning-rate and schedule terms to
try against: "Scaling Law with Learning Rate Annealing" (Tissue et al. 2024), "A Multi-Power Law
for Loss Curve Prediction Across Learning Rate Schedules" (Luo et al. 2025).
