# Strategy: which box does what

The manager decides this; Vuk sets the general rules and checks results (2026-11-28). This file holds the goal, the rules for choosing, then the current plan.

## Goal and outputs (2026-11-28 20:05 SGT)

**Science goal.** A set of hyperparameter rules (learning rate, warmup, decay, batch, optimizer), each measured on CPU-size models, written in scale-free terms, and checked against published big-model values. Also a scored list of the CPU methods that recover known big-model laws (the calibration trials).

**Outputs.** Each study ends in its own blog post or paper, chosen when its BLOG.md meets the bar.

**Studies stay independent (Vuk, 2026-11-28 20:20 SGT: "keep all of the CPUs independent").** Each box keeps its own question and its own write-up. They are not merged into one joint paper.

## Rules for choosing a box's question

1. **One question per box for days** ([focused-persistence.md](../agents/focused-persistence.md)). A box switches only when its study ends in BLOG.md, or when two rounds in a row change nothing.
2. **The knob must be one every big run sets**: learning rate, schedule and decay, warmup, batch, optimizer, width and depth ratio, tokens per parameter. No question that only exists at small scale.
3. **Every study passes the small-to-big bar** before its blog post:
   - the rule is written in scale-free terms (ratios such as tokens per parameter, lr over best lr, steps over a time scale), or with a size exponent measured at 4+ sizes;
   - predictions for the largest CPU size are frozen in `prereg/` before those runs, then scored (method M5);
   - the rule is checked against at least one published big-model law from [ANSWER-KEY.md](ANSWER-KEY.md), with agreement or disagreement stated (M1);
   - the cause is named, and tested with a control or in the noise simulator where it applies (M3);
   - the blog post says the big-model prediction in one sentence and where it may break.
4. **A box whose study ends takes the next open item** in this order: an open calibration trial (T2, T3, T4 in [TRIALS.md](TRIALS.md)), then a new question by rule 2.
5. **Each study is also a trial of the methods.** Its result goes in TRIALS.md, so we learn which methods work.

## Current plan (2026-11-28 19:15 SGT)

| Box | Question | Small-to-big check it must pass | Trial |
|---|---|---|---|
| cpu-1 | WSD decay: best decay share | A rule in ratios (decay share vs steps, lr, batch); check vs Hägele et al. (D1: about 20% cooldown matches cosine) and Tissue et al. (D2) | T6 |
| cpu-2 | Warmup length | What sets the ~28-step warmup scale (fixed steps, tokens, lr rise rate or width), at 4+ widths and batches; whether the cost's width exponent 0.77 holds or saturates; check vs Ma and Yarats 2021 (warmup about 2/(1-β2) steps) and big runs (GPT-3 375M tokens, Llama 2 2,000 steps) | T5 |
| cpu-3 | Does the best-lr rule hold on big models? | Explain Step Law (A2) and DeepSeek (A1) with one lr(steps, batch) rule | T1 |
| gpu-3090 | Adam vs Muon gap | The gap vs tokens per parameter at 4+ widths; check vs published Muon claims (about 2x compute efficiency in Moonshot's "Muon is scalable", 2025); Adam lr vs width for T4 | T4, T7 |
| next free box | Calibration: which known big-model laws do CPU runs recover? (T2 Chinchilla, then T3 critical batch) | It is the calibration itself | T2, T3 |

**Why cpu-2 keeps warmup.** It has a fitted formula (score 1: 0.031 vs 0.042 for no effect) and predictions for rounds 4 and 5 frozen before the runs. Switching now would waste them. Warmup is set in every big run, and published rules exist to check against. Its open question (is the 28-step scale a step count or a share of training?) is exactly the question of how warmup scales.

**Update 2026-11-28 20:35 SGT.** The 3090 is shut down; its data is on HF (studies/adam-vs-muon-3090) and its open items (T4, T7) go to the next free CPU box. cpu-3 now has one claim and aims at a paper: with a decay to zero the best peak lr reaches a floor; without decay it falls as 1/sqrt(steps). Its runs go only to tests that could break that claim (decay to 10%/30%, 4+ widths and a 10x model, Adam to 4,096 steps).

**Update 2026-11-28 20:45 SGT (Vuk).** First focus for cpu-3: predict a published big-model paper from CPU runs. Calibrate on Bjorck et al. 2024 (copy its setup at CPU size, get its lr-vs-horizon exponent at 4+ widths), then with the setup unchanged freeze predictions for DeepSeek LLM 2024 and Step Law 2025 and score them. A disagreement is checked for a paper error (small chance) before the setup is changed. The decay-to-zero floor stays the claim this setup is meant to carry to big scale.

**Update 2026-11-28 21:10 SGT.** Two new CPU boxes (2 vCPU ARM each) take the 3090's open items: cpu-4 runs T4 (Adam best lr vs width, work/adam-lr-width), cpu-5 runs T7 (Adam-Muon gap vs tokens per parameter, work/adam-vs-muon-cpu). Both start from the 3090 round 1-3 data.
