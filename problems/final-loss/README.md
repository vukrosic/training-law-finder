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

In this data `peak` = `mlr` in every run, so a formula cannot tell them apart. Every run also saw 64
synthetic biographies in window A (20%-45% of the steps); new runs must include them too (the command
below does).

Tested range: widths 24 to 1024, 512 to 16,384 steps (only 2 runs above 4,096 steps: 8,192 and
16,384, both width 64). A paper on this problem states this range.

## Train a run from a settings line (score 2)

A round (see [rounds/README.md](../../rounds/README.md)) lists settings such as
`{"tag": "r1_w48", "settings": {"width": 48, "layers": 2, ...}}`. This command trains one of them the
same way as the frozen runs. Each `<name>` is that column of the settings line; everything else is
fixed by the frozen data (1,024-token vocabulary, Muon, the facts in window A, clipping at 1.0).

```bash
cd families/llm/testbed-facts-v1
# once: sh data/download.sh && python3 -I tools/prep_vocab.py 1024   (1-6 min, about 1.3 GB memory)
nice -n 15 python3 -I env/train.py --method dense --data fineweb_v1024 --device auto --threads 1 \
  --tag <tag> --steps <steps> --set vocab=1024 opt=muon \
  d_model=<width> n_layers=<layers> n_heads=<heads> dense_hidden=<hidden> batch=<batch> seq_len=<seq> \
  sched=<sched> lr=<peak> mlr=<mlr> mwd=<wd> min_lr_frac=<min_lr_frac> warmup=<warmup> decay=<decay> \
  mom=<momentum> facts=64 fkmax=128 fwlo=0.2 fwhi=0.45 gstep=1 wlog=32 fref=32 clip=1.0 \
  --out experiments/runs/round.jsonl
```

`warmup` is in steps, not a share. If the run stops with "too many bios per step", the setting is too
short or the batch too small to hold the facts; use more steps or a larger batch. Then write the
results file the round needs (`actual` = `val_loss`):

```bash
python3 -c "import json; [print(json.dumps({'tag': r['tag'], 'actual': r['val_loss']})) for r in map(json.loads, open('experiments/runs/round.jsonl')) if 'val_loss' in r]" \
  > ../../../rounds/<NNN>-final-loss/results.jsonl
```

Checked on 2026-11-27 with the frozen run `m2f_w24L1_b64_const` (width 24, 1 layer, batch 64, seq 64,
1,024 steps, `const`, peak 0.008, wd 0, min_lr_frac 1.0, warmup 20, decay 0.2, momentum 0.95, heads 1,
hidden 48): val_loss 4.1042 and val_bpb 2.3032, the same
as the frozen row. It ran on mps in 354 s with two other jobs on the same Mac GPU (the frozen run took 35 s);
peak memory 1.6 GB.

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [lr_area](formulas/lr_area.py) | 12 | 0.0839 | Chinchilla with the schedule's mean learning rate, plus a width bowl. The formula to beat (★ in [LEADERBOARD.md](LEADERBOARD.md)) |
| [width_bowl](formulas/width_bowl.py) | 10 | 0.1041 | Chinchilla + a learning-rate bowl whose center and steepness move with model size |
| [chinchilla_lr](formulas/chinchilla_lr.py) | 8 | 0.1350 | Chinchilla + a bowl in ln(learning rate) + a decay term |
| [chinchilla](formulas/chinchilla.py) | 5 | 0.1471 | L = E + A/N^a + B/D^b (Hoffmann et al. 2022) |
| [mean](formulas/mean.py) | 1 | 0.2994 | constant |

Floor 0.0019. How it is computed: group the runs that have exactly the same visible settings (36
settings have 2 or more runs, 80 runs in all, mostly the same run on two machines), take each run's
distance from its group mean, and report the pooled standard deviation
sqrt(sum of squared distances / sum of (group size - 1)). The best formula is about 44 times the
floor away: this problem is wide open. The largest errors are at width 1024 and 768.

## What we found

[lr_area](formulas/lr_area.py) (12 constants, score 1 0.0839) holds four rules. Each gain below is
how much score 1 gets worse when that rule alone is removed and the rest is refitted (2026-11-27).

- (a) The best learning rate depends on the schedule's average learning rate (peak times the mean
  schedule multiplier), not on the peak. A free exponent on the mean multiplier fits to 1.02.
  Using the peak instead: 0.0938 (gain 0.0099). Novelty: partly known. Luo et al. 2025 and Tissue
  et al. 2024 use the summed learning rate to predict loss, but neither states that the best learning
  rate is set by the average (Haiku paper search, 2026-11-27).
- (b) Tokens count more at a higher average learning rate: without it 0.0899 (gain 0.0060). Known
  (Tissue et al. 2024).
- (c) Model size helps less when there are few tokens: without it 0.0908 (gain 0.0069). Known
  (Kaplan et al. 2020).
- (d) The best learning rate falls as width grows: without it 0.1114 (gain 0.0275). This is close to
  muP-style learning-rate rules. The second half of the rule, a bowl that gets steeper with size,
  does not help: without it 0.0829, slightly better.

Weak spot: the worst group is width 1024 at 0.1536, almost twice the mean.

All runs are FineWeb-Edu with a 1,024-token vocabulary. A second-dataset test (TinyStories, its own
1,024-token tokenizer, same protocol) is running.

## Published work to beat

Chinchilla (Hoffmann et al. 2022) is the baseline. Laws with learning-rate and schedule terms to
try against: "Scaling Law with Learning Rate Annealing" (Tissue et al. 2024), "A Multi-Power Law
for Loss Curve Prediction Across Learning Rate Schedules" (Luo et al. 2025).
