# Forgetting: how much does a model forget?

A language model sees 64 facts during one window of training (20%-45% of the steps, or 45%-70%), then keeps
training on plain text. By the end it has forgotten some of them. **Predict the share it still
knows at the end, from the training settings alone.**

Why it matters: labs add data in phases (code, math, instruction data near the end). A formula for
forgetting tells you when to show data so the model keeps it, without a sweep.

Draft paper on this problem (AI-written, not yet checked by a person): [Rules for How Language Models Forget Facts During Training](../../drafts/forgetting-rules/), DOI [10.5281/zenodo.24575889](https://doi.org/10.5281/zenodo.24575889).

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

## Train a run from a settings line (score 2)

A round (see [rounds/README.md](../../rounds/README.md)) lists settings such as
`{"tag": "r1_w96", "settings": {"width": 96, "layers": 3, ...}}`. This command trains one of them the
same way as the frozen runs. Each `<name>` is that column of the settings line; everything else is
fixed by the frozen data (1,024-token vocabulary, Muon, 64 facts shown 1 to 128 times, clipping at 1.0).
Use `sched` = `cos`, `wsd` or `const` with `fmarg` 0 and `gsplice` 0: the restart settings of the 26
`crw` runs are not in the data.

```bash
cd families/llm/testbed-facts-v1
# once: sh data/download.sh && python3 -I tools/prep_vocab.py 1024   (1-6 min, about 1.3 GB memory)
nice -n 15 python3 -I env/train.py --method dense --data fineweb_v1024 --device auto --threads 1 \
  --tag <tag> --steps <steps> --set vocab=1024 opt=muon \
  d_model=<width> n_layers=<layers> n_heads=<heads> dense_hidden=<hidden> batch=<batch> seq_len=<seq> \
  sched=<sched> lr=<peak> mlr=<mlr> mwd=<wd> min_lr_frac=<min_lr_frac> warmup=<warmup> decay=0.5 \
  facts=64 fkmax=128 fwlo=<fwlo> fwhi=<fwhi> gstep=1 wlog=32 clip=1.0 \
  --out experiments/runs/round.jsonl
```

`warmup` is in steps, not a share. `decay` (the share of a `wsd` run spent decaying) is not a column
here: all 50 frozen `wsd` runs used 0.5 (`sname` = `wsd50`); `cos` and `const` ignore it. If the run stops with
"too many bios per step", the setting is too short or the batch too small to hold the facts; use
more steps or a larger batch. Then write the results file the round needs. `actual` is the share of
fact bits still there at the end: `1 - (b_end - b0) / (C - b0)`, where b0 and b_end are the fact bits
at the window end and at the last step, and C is the bits of facts seen only 1, 2 or 4 times:

```bash
python3 -c "
import json
for r in map(json.loads, open('experiments/runs/round.jsonl')):
    if 'facts_eval' not in r: continue
    b = [sum(x) / len(x) for x in r['facts_eval']['gstep']['bits']]
    k = r['facts_eval']['bits_by_k']; C = sum(k[i] for i in ('1', '2', '4')) / 3
    print(json.dumps({'tag': r['tag'], 'actual': round(1 - (b[-1] - b[0]) / (C - b[0]), 5)}))
" > ../../../rounds/<NNN>-forgetting/results.jsonl
```

Checked on 2026-11-27 with the frozen run `xg_w64_T512_L3_p1.6e-2_wd01_mf0` (width 64, 3 layers, 1 head,
hidden 128, batch 64, seq 64, 512 steps, `cos`, peak 0.016, wd 0.1, min_lr_frac 0, warmup 20, window A):
actual 0.08508 against 0.08506 in the
frozen row (floor 0.0126), val_loss 3.729 the same. It took 137 s on mps and 171 s on cpu on a busy Mac;
peak memory 1.7 GB on mps, 0.35 GB on cpu.

## Best formulas

| formula | constants | score 1 | idea |
|---|---|---|---|
| [end_z2](formulas/end_z2.py) | 10 | 0.0441 | forgetting clock = sum of the learning rate after the window; settings shift when it starts |
| [end_z2_wd](formulas/end_z2_wd.py) | 10 | 0.0411 | end_z2 times the published weight decay bound (Bordt et al. 2025). No new constant |
| [end_z3](formulas/end_z3.py) | 12 | 0.0366 | end_z2 plus two terms: weight decay costs more at a high peak learning rate, and learning rate spent late forgets less than the same amount spent early |
| [lr_clock](formulas/lr_clock.py) | 2 | 0.1507 | the same clock with no settings |
| [mean](formulas/mean.py) | 1 | 0.2177 | constant |
| [cum_wd](formulas/cum_wd.py) | 0 | 0.8012 | published: cumulative weight decay (Bordt et al. 2025), no fitted constant |

Floor 0.0126. Gap from end_z2_wd to the floor: 0.0285. Full table: [LEADERBOARD.md](LEADERBOARD.md).

end_z2_wd is the formula to beat (★ in the leaderboard). It has the same 10 constants as end_z2, the formula of the paper. end_z3 lowers score 1 by a further 0.0045 with two more constants, less than the constants rule asks ([rule 2](../README.md#rules)).

Two runs outside the table ([rounds/002-forgetting](../../rounds/002-forgetting/)). end_z2 was frozen before the first run; the other two formulas were fitted afterwards.

| run | measured | end_z2 | end_z2_wd | end_z3 |
|---|---|---|---|---|
| width 2048, 12 layers, no weight decay (predicted first) | 0.242 | 0.241 | 0.230 | 0.203 |
| width 768, 10 layers, weight decay 0.3, batch 96 (the known miss) | 0.104 | 0.289 | 0.175 | 0.236 |

## Known open parts

- **Wide, shallow models.** end+z2 overpredicts width 1024 with 3 layers by +0.02 to +0.05 on three runs predicted before they ran. Kept at 3 layers stops rising between width 512 and 1024. A width-2048 run with 12 layers (503M parameters) was predicted before it ran: end+z2 said 0.241 and it measured 0.242 ([rounds/002-forgetting](../../rounds/002-forgetting/)). Width 2048 with 3 layers is not run yet.
- **Batch size.** Adding `-0.359 * ln(batch/64)` to the onset halves the error on runs with batch other than 64 (0.0193 vs 0.0362), on few rows.
- **Schedule shape.** Two schedules with the same learning-rate sum forget differently; the share of the schedule spent below half the peak helps on three blind runs.

## Published work to beat

The closest published work, with what the formulas here add:

- **Power law in steps.** Chang et al. (2024), "How Do Large Language Models Acquire Factual Knowledge
  During Pretraining?" ([arXiv:2406.11813](https://arxiv.org/abs/2406.11813)): facts fade along a power
  law in the number of steps since they were seen. `lr_clock` is that idea with the learning-rate sum as
  the clock.
- **Cumulative weight decay.** Bordt et al. (ICML 2025), "How Much Can We Forget about Data
  Contamination?" ([arXiv:2410.03249](https://arxiv.org/abs/2410.03249)): under Adam with decoupled
  weight decay, the share of an old update left in the weights is the product of
  (1 - learning rate × weight decay) over the later steps, and examples are forgotten at least that
  fast. [`cum_wd`](formulas/cum_wd.py) is that formula. Here its bound holds in all 110 runs with weight
  decay, and as a prediction it scores 0.8012. What ours adds: a number instead of a bound, and a clock
  that needs no weight decay (265 of the 375 runs have none and still keep only 18% on average).
- **Moving average of the learning rate.** Bergsma et al. (ICLR 2026), "Predicting Training
  Re-evaluation Curves Enables Effective Data Curriculums for LLMs"
  ([arXiv:2509.25380](https://arxiv.org/abs/2509.25380)): the final model's loss on each training batch,
  against the time the batch was seen, follows the moving-average weights that the learning rate and
  weight decay give to each step. It predicts the shape of that curve under Adam with weight decay.
  What ours adds: the share of planted facts kept as a number, under Muon, with terms for model size
  and run length, scored on held-out widths.
