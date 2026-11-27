# Final loss: predict the loss before you train

**Predict a run's final validation loss from its settings:** model size, tokens, learning rate,
schedule, batch.

Why it matters: a formula like this picks the model size and learning rate for a budget without a
sweep. Chinchilla answers it for the best learning rate only; real runs are often not at the best one.

## Data

`data/llm.jsonl.gz`: a frozen copy of the plain dense Muon runs with a 1,024-token vocabulary in the
full run table, with diverged runs removed: 1,163 runs, 16 widths from 24 to 1024 (widths with fewer
than 5 runs left out), 1-12 layers, 512-16,384 steps, constant / cosine / WSD schedules and many
learning rates. Grouped by width. Frozen on 2026-11-27 with
`python3 tools/build_problem_data.py final-loss`; new runs join only when a maintainer rebuilds it
and rescores every formula. Columns: [core/columns.yaml](../../core/columns.yaml).

Which runs count as plain: a run is kept only if nothing changed its training beyond the columns a
formula can see. That means the clean protocol (`protocol` = `lf` in the run table: no branch from a
checkpoint, no replayed facts, no optimizer switch after the fact window), facts shown in window A,
gradient clipping at 1.0, no bf16, and a constant, cosine or WSD schedule (rewarm schedules depend
on settings a formula cannot see). Before 2026-11-27 the problem held 22,221 runs, and runs with the
same visible settings differed by 0.098 in loss, because hidden changes (SGD after the window, other
clipping, replay files) lived only in the run names. The rule is `select()` in `problem.py`.

Allowed inputs: the setting columns in the data (`core/columns.yaml`, kind `setting`): width, layers,
heads, hidden, params, nonemb_params, batch, seq, steps, tokens, sched, peak, mlr, wd, min_lr_frac,
warmup, decay, momentum. Not allowed: val_loss, val_bpb, run names. For `const` runs, min_lr_frac
and decay are stored but do nothing.

Tested range: widths 24 to 1024, 512 to 16,384 steps. A paper on this problem states this range.

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [chinchilla_lr](formulas/chinchilla_lr.py) | 8 | 0.1350 | Chinchilla + a bowl in ln(learning rate) + a decay term |
| [chinchilla](formulas/chinchilla.py) | 5 | 0.1471 | L = E + A/N^a + B/D^b (Hoffmann et al. 2022) |
| [mean](formulas/mean.py) | 1 | 0.2994 | constant |

Floor 0.0019. How it is computed: group the runs that have exactly the same visible settings (36
settings have 2 or more runs, 80 runs in all, mostly the same run on two machines), take each run's
distance from its group mean, and report the pooled standard deviation
sqrt(sum of squared distances / sum of (group size - 1)). The best formula is about 70 times the
floor away: this problem is wide open. The largest errors are at width 1024 and 768.

## Published work to beat

Chinchilla (Hoffmann et al. 2022) is the baseline. Laws with learning-rate and schedule terms to
try against: "Scaling Law with Learning Rate Annealing" (Tissue et al. 2024), "A Multi-Power Law
for Loss Curve Prediction Across Learning Rate Schedules" (Luo et al. 2025).
