# Forgetting: how much does a model forget?

A language model sees 64 facts during one window of training (20%-45% of the steps, or 45%-70%), then keeps
training on plain text. By the end it has forgotten some of them. **Predict the share it still
knows at the end, from the training settings alone.**

Why it matters: labs add data in phases (code, math, instruction data near the end). A formula for
forgetting tells you when to show data so the model keeps it, without a sweep.

## Data

`data/llm.jsonl.gz`: 375 real runs, decoder-only transformers trained with Muon.

| setting | values |
|---|---|
| width | 64 (227 runs), 128, 256, 384, 512, 768, 1024 |
| layers | 3-12 |
| steps | 512-4096 (295 runs at 1024) |
| fact window | A = 20%-45% (275 runs), B = 45%-70% (99), 1 run with facts shown through all of training (`const`) |
| peak learning rate, weight decay, min_lr_frac, schedule | swept |

Target: `kept[-1]`. Each row also holds the whole curve (`kept` at each reading after the window) and
the learning rate at every step after the window (`lr`; for example 564 values in a 1,024-step run with
window A), so a formula may use the schedule shape. Columns: [core/columns.yaml](../../core/columns.yaml).

Allowed inputs: everything except `kept`, `C`, `b0`, `rms_w` and `wnorm` (those are measured during
the run). Formulas that use `rms_w` are listed separately.

Tested range: widths 64 to 1024, 512 to 4,096 steps. A paper on this problem states this range.

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [end_z2](formulas/end_z2.py) | 10 | 0.0441 | forgetting clock = sum of the learning rate after the window; settings shift when it starts |
| [lr_clock](formulas/lr_clock.py) | 2 | 0.1507 | the same clock with no settings |
| [mean](formulas/mean.py) | 1 | 0.2177 | constant |

Floor 0.0126. Gap from the best formula to the floor: 0.0315. Full table: [LEADERBOARD.md](LEADERBOARD.md).

## Known open parts

- **Wide, shallow models.** end+z2 overpredicts width 1024 with 3 layers by +0.02 to +0.05 on three runs predicted before they ran. Kept at 3 layers stops rising between width 512 and 1024. A width-2048 run is being trained to settle which width term is right.
- **Batch size.** Adding `-0.359 * ln(batch/64)` to the onset halves the error on runs with batch other than 64 (0.0193 vs 0.0362), on few rows.
- **Schedule shape.** Two schedules with the same learning-rate sum forget differently; the share of the schedule spent below half the peak helps on three blind runs.

## Published work to beat

No published formula predicts forgetting of facts shown mid-training from model and schedule
settings. The closest is a power law in the number of steps since the data was seen (for example
"How Do Large Language Models Acquire Factual Knowledge During Pretraining?", 2024). `lr_clock` is
that idea with the learning-rate sum as the clock.
