# Claims for students

22 claims from the data already in this repo, each sized for one paper. Each claim is stated boldly
enough to fail. If it fails, the backup formula still gives a paper: fit it, score it, report the result. They are ranked by strength of
evidence: matched pairs and held-out runs first, single groups last. All runs are small scale (widths 24-2048,
1-12 layers, up to 16,384 steps, mostly Muon). Say "small scale" in any paper built on them.

Status: **new** (no paper found), **partly known** (the direction is published, our number or setting is not),
**known** (published, or already posted by this repo; holding on our data makes it a confirmation, see
[CREDIT.md](CREDIT.md)). Paper ids point into [literature/papers.jsonl](literature/papers.jsonl).

Count: 3 new, 11 partly known, 8 known.

## How to reproduce

```
python3 tools/claims_data.py                 # builds work/claims/*.jsonl from data/llm/ (gitignored)
python3 tools/claims_fits.py <analysis>      # batchseq, pairs, curveshape, decay, batchwidth, other
python3 core/score.py <problem> <formula>    # score 1 of a listed formula
```

Words used below. **kept**: share of the synthetic facts still recalled at the end of the run (1 = all, 0 =
chance). **S**: sum of the learning rate over every step after the facts stop. **Matched pair**: two runs that
differ in one setting only. **Unseen runs**: the 1,103 clean runs in the runs table that the frozen forgetting
problem (375 runs) never saw.

Test plans use one run per setting. No seeds and no repeated runs.

## Picking a claim

Pick any claim; all are open to everyone. Check open PRs and `LEADERBOARD.md` first. Credit follows
[CREDIT.md](CREDIT.md): whoever pushes to GitHub or posts a preprint first gets it. A result within the noise
floor of a listed formula is a confirmation, credited with your name.

---

## C01. Depth changes loss but not forgetting

**Claim:** Depth does not move forgetting: at any width, a doubling of depth changes kept by less than 0.01, while it lowers loss in over 90% of matched pairs (3→6 layers: loss −0.042 in 134 of 144 pairs, kept +0.005).

**Evidence:** Every depth step lowers loss (1→2: 54/54 pairs, 2→3: 57/58, 6→8: 51/52). Kept changes by
−0.002 to +0.012 at each step. 386 pairs in all. `python3 tools/claims_fits.py pairs layers`

**How to test it:** Width 256 and 1024, cosine, peak 8e-3. Train 1, 2, 4, 8 and 12 layers, one run each. The
claim fails if kept moves by more than 0.01 between neighbouring depths.

**Backup if it fails:** Fit kept = 1/(1+exp(q(z + ln S))) with z linear in ln(layers) and report the depth
slope; the frozen end_z2 has a ln(layers/8) term of −0.12.

**Closest papers:** arXiv:2110.11526 (wider nets forget less; depth not varied), arXiv:2205.10770 (larger
models forget less), arXiv:2605.26097. We add depth at fixed width, with loss and forgetting from the same runs.

**Status:** new. No paper found compares depth with forgetting.

## C02. Batch and sequence length set the forgetting clock

**Claim:** Batch and sequence length act only through the clock: with S·(batch/64)^−0.19·(seq/64)^0.13 and these 2 exponents fixed, the frozen end_z2 formula predicts any batch and sequence length to within 0.05 RMSE (0.049 on 556 unseen runs, down from 0.061).

**Evidence:** Fit the 2 exponents on a random half of the unseen runs, score the other half. Extrapolation:
width>128 0.115→0.084, seq>256 0.108→0.076. The batch term alone gives 0.053; the seq term adds 0.0046, just
under the 0.005 rule on this split. Weak spot: fit on batch ≤64 and tested above 64, the error rises
(0.0527 → 0.0546). `python3 tools/claims_fits.py batchseq`

**How to test it:** 8 runs at width 256, 6 layers, batch 32 and 256 crossed with seq 128 and 512, two peaks.
Write the predictions with `core/predict.py` before training. Fails if the error is not below the frozen
end_z2 error.

**Backup if it fails:** Fit a noise-scale clock, S·((1+Bn/B)/(1+Bn/64)), with Bn the batch where noise stops
mattering (2 constants, held-out 0.050 on the same split).

**Closest papers:** arXiv:2406.11813 (larger batches forget less), arXiv:2405.13698 (optimizer timescale counted
in steps and batch), arXiv:2505.13738. We add explicit exponents on the learning-rate clock and a sequence term.

**Status:** partly known. The batch direction is published; the sequence term is not.

## C03. WSD buys lower loss with fewer kept facts

**Claim:** WSD pays for its lower loss in facts: it keeps fewer early facts than cosine in every matched setting, by about 8.5 points, while its loss is 0.037 lower (107 of 113 pairs).

**Evidence:** Same width, depth, peak, steps and window. `python3 tools/claims_fits.py pairs sched`. Matching on
width, layers, batch, seq, steps, peak, weight decay, final lr and window only, WSD keeps fewer facts in 156 of
156 pairs.

**How to test it:** 6 pairs (cosine vs WSD) at widths 128, 512 and 1024, two peaks. Fails if WSD keeps as many
facts as cosine in 4 or more pairs.

**Backup if it fails:** Predict both schedules with end_z2 from S alone; any gap left is a schedule-shape term to
fit, for example the share of S spent above half the peak.

**Closest papers:** arXiv:2405.18392 and arXiv:2404.06395 (WSD loss), arXiv:2410.05192. None measures facts kept.

**Status:** partly known. The loss side is known; the cost in facts is our measurement.

## C04. One learning-rate clock predicts forgetting

**Claim:** One number predicts forgetting, the sum of the learning rate after the facts (S): kept = 1/(1+exp(q(z + ln S))) with 10 constants scores 0.0441 held-out on 375 runs, against 0.2177 for the mean and 0.8012 for the cumulative weight-decay rule, and stays within 0.05 at width 1024 and above.

**Evidence:** `python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py` (floor 0.0126).
Posted in the draft paper (doi:10.5281/zenodo.24575889).

**How to test it:** Width 1024+ with 3 layers is the weak spot (overpredicted by 0.02-0.05). Predict 2 new runs
there before training them.

**Backup if it fails:** end_z3 (12 constants, 0.0366) or the two-limit ratio of lr sums (6 constants, 0.0526,
forgetting-curve-shape branch).

**Closest papers:** arXiv:2410.03249 (bound from cumulative weight decay), arXiv:2406.11813, arXiv:2605.26097,
arXiv:2207.00099.

**Status:** known (posted by this repo). Further work counts as a confirmation.

## C05. The frozen forgetting formula transfers to unseen runs

**Claim:** Frozen end_z2 needs no refit at batch 64 and seq 64: it scores 0.039 RMSE on 443 unseen runs there, near its in-sample 0.0375, against 0.065 over all 1,103 unseen runs.

**Evidence:** Errors grow with batch away from 64 (batch 16: bias +0.062; batch 1024: −0.087), with seq (seq 2048:
+0.090) and at widths 192-512 (0.10-0.15).
`python3 core/predict.py forgetting work/claims/forget_new/settings.jsonl problems/forgetting/formulas/end_z2.py > work/claims/forget_new/predictions.jsonl`
then `python3 core/round_score.py work/claims/forget_new`.

**How to test it:** Commit predictions for a new round (rounds/README.md) at widths 128-2048, batch 64, seq 64.
Fails if RMSE is above 0.05. Current width groups there: 0.054 at width 128, 0.058-0.070 at 512 and above, so
it may fail.

**Backup if it fails:** Add the C02 clock terms, then a width-by-peak term.

**Closest papers:** arXiv:2406.11813, arXiv:2205.10770, arXiv:2410.17018. No paper reports a frozen forgetting
formula scored on unseen runs.

**Status:** partly known. A confirmation of C04 with a new out-of-sample number.

## C06. The area under the learning-rate curve predicts final loss

**Claim:** Final loss depends on the schedule only through the area under the learning-rate curve: lr_area (12 constants) scores 0.0839 held-out on 1,163 runs, against 0.1471 for the Chinchilla form E + A/N^a + B/D^b, and keeps that lead at width 1024 and above.

**Evidence:** `python3 core/score.py final-loss problems/final-loss/formulas/lr_area.py` (floor 0.0019).

**How to test it:** Predict 6 new runs at width 1024 and 2048 before training. Fails if lr_area is not better than
chinchilla_lr (0.1350) on them.

**Backup if it fails:** width_bowl (10 constants, 0.1041).

**Closest papers:** arXiv:2408.11029 (area-under-lr law), arXiv:2503.12811, arXiv:2203.15556.

**Status:** known (posted problem formula; the lr-area idea is published).

## C07. Facts fade as a power of the learning-rate sum

**Claim:** Forgetting inside a run follows one law at every size: kept = 1/(1+(S_t/s0)^q) with q between 0.9 and 1.3 at any width (1.0-1.2 from width 24 to 2048, 1,452 runs); the power form (1 + S_t/s0)^−q forecasts forward with median error 0.018, against 0.023 for a stretched exponential.

**Evidence:** Fit on the first 70% of readings per run, score on the last 30%. Hill with free q: 0.019; Hill with
q=1: 0.022. `python3 tools/claims_fits.py curveshape`

**How to test it:** 4 runs at width 4096 or 12 layers. Fails if the fitted q leaves 0.9-1.3.

**Backup if it fails:** Let q depend on width, q = q0·(width/256)^c, and fit it across runs.

**Closest papers:** arXiv:2406.11813 (power-law forgetting in steps), arXiv:2401.05605 (shifted power law in
fine-tuning), arXiv:2207.00099.

**Status:** partly known. The power shape is published in steps; the lr-sum variable and the fixed q are ours.

## C08. A final learning rate of 10% trades facts for loss

**Claim:** Any final learning rate above zero costs facts: kept falls steadily as the final lr rises from 0 to the peak (−4.4 points at 10%, −22 points at 100%), while loss is lowest near 10% (−0.010 in 111 of 180 pairs; a constant schedule loses in 79 of 79).

**Evidence:** `python3 tools/claims_fits.py pairs min_lr_frac`

**How to test it:** Final lr at 0, 3%, 10% and 30% of peak at widths 128 and 512. Fails if kept does not fall as
the final lr rises.

**Backup if it fails:** Fit loss and kept as smooth functions of the final lr and report the best final lr for a
given weight on facts.

**Closest papers:** arXiv:2405.18392, arXiv:2403.08763, arXiv:2502.15938. None links the final lr to facts kept.

**Status:** partly known.

## C09. Weight decay 0.1 trades facts for loss

**Claim:** Weight decay always costs facts: kept falls steadily as weight decay rises from 0 (−4.6 points at 0.1), while loss is lowest between 0.1 and 0.3 (0.1 beats 0 in 90 of 128 pairs; 1.0 loses to 0.3 in 24 of 24).

**Evidence:** `python3 tools/claims_fits.py pairs wd`. The draft paper's rule 5 (weight decay speeds
forgetting) covers the kept side.

**How to test it:** Weight decay 0, 0.03, 0.1 and 0.3 at widths 128 and 512. Fails if loss does not dip near
0.1 while kept falls.

**Backup if it fails:** Fit the end_z2 weight-decay term (+0.85 in z) jointly with a loss term in weight decay.

**Closest papers:** arXiv:2410.03249 (weight decay and forgetting of examples), arXiv:2602.11137 (weight decay
helps plasticity), arXiv:2405.13698.

**Status:** partly known.

## C10. Small batch hurts loss and memory together

**Claim:** Below batch 64, each doubling of batch at equal steps lowers loss and raises kept in at least 95% of matched pairs (16→64: loss −0.39, kept +9.3 points, 9 of 9).

**Evidence:** 16→32: loss lower in 20/20, kept +2.3 points; 32→64: 22/23, +2.7 points. Above 64 the loss
result is mixed (64→128 worse in 103/121 pairs, 128→256 better in 103/103), so the claim stops at 64.
`python3 tools/claims_fits.py pairs batch`

**How to test it:** Batch 8, 16, 32, 64 at widths 128 and 512, same steps. Fails if kept does not rise with batch.

**Backup if it fails:** Use tokens seen after the window in place of steps in the clock and refit.

**Closest papers:** arXiv:2406.11813 (larger batch helps retention), arXiv:1812.06162.

**Status:** partly known.

## C11. Longer sequences forget more at the same steps

**Claim:** Longer training sequences forget more at equal steps: kept falls 3-4 times from seq 64 to 1024 (0.14 → 0.04 at batch 64, 0.16 → 0.05 at batch 128, peak 8e-3).

**Evidence:** Mixed: flat at batch 16 and 32, and rising at peak 3.2e-2 and batch 64 (0.09 → 0.16). Pooled
pairs: seq 256→1024 kept −3.5 points (12 pairs). The C02 seq exponent is +0.13. `python3 tools/claims_fits.py pairs seq`

**How to test it:** Seq 64, 256, 1024 at batch 64, two peaks (8e-3 and 3.2e-2), width 256. Fails if kept falls
at one peak only.

**Backup if it fails:** Fit the seq exponent as a function of peak, c = c0 + c1·ln(peak).

**Closest papers:** arXiv:2205.10770, arXiv:2605.26097. No paper found measures fact retention against training
sequence length.

**Status:** new. Evidence is weaker than its rank neighbours.

## C12. The batch past which more batch stops saving facts grows with width

**Claim:** The batch size that saturates fact retention grows with width: kept stops rising after batch 128-256 at width 64-256, but still rises at batch 1024 at width 512 (0.278 at batch 64 → 0.487).

**Evidence:** Cosine, peak 8e-3, 3 layers, seq 64. Width 64: 0.162-0.170 from batch 128 to 1024. Width 1024:
0.262 → 0.440 from batch 64 to 512. `python3 tools/claims_fits.py batchwidth`

**How to test it:** Batch 128, 512, 2048 at widths 256, 1024, 2048. Fails if the knee does not move right with width.

**Backup if it fails:** Fit the noise-scale clock of C02 with Bn = B0·(width/256)^c.

**Closest papers:** arXiv:1812.06162, arXiv:2410.21676 and arXiv:2505.13738 (critical batch set by data size,
not model size). Our direction conflicts with them; their setting varies model size at fixed data, not width.

**Status:** new, and in tension with known critical-batch results.

## C13. A learning-rate cooldown does not bring facts back

**Claim:** A learning-rate cooldown never brings facts back: kept falls during the final decay of WSD runs (0.23 → 0.07 over 188 runs), and no run gains more than 0.03.

**Evidence:** The flat stretch before it drops kept from 0.43 to 0.23 with a similar lr sum (1.13 vs 1.18).
The loss falls in the decay; the facts do not return. `python3 tools/claims_fits.py decay`

**How to test it:** 4 WSD runs with dense fact readings during the decay (every 10 steps), decay 20% and 50%.
Fails if kept rises by more than 0.03 in the decay.

**Backup if it fails:** Fit the rise as a function of how far the lr falls, kept_end − kept_min = a·ln(peak/lr_end).

**Closest papers:** arXiv:2503.02844 (restarts erode knowledge), arXiv:2405.18392 (cooldown), arXiv:2404.06395.

**Status:** partly known.

## C14. Facts shown later are kept more

**Claim:** Window position acts only through S: at equal S, facts shown early and late are kept equally. Without matching S, a later window keeps 35.6% against 10.7% (83 of 83 pairs).

**Evidence:** Draft paper rule 6 (doi:10.5281/zenodo.24575889). S is smaller for the later window, which C04
already uses.

**How to test it:** 3 windows (20%, 50%, 80% of steps) at width 512, with the peak set so S is equal. Fails if
kept differs by more than 0.03.

**Backup if it fails:** Fit kept against S only and check if window position adds anything at fixed S.

**Closest papers:** arXiv:2406.11813, arXiv:2410.03249.

**Status:** known (posted; direction published).

## C15. Flat then linear decay beats cosine

**Claim:** Flat then linear decay over the last 20% beats cosine on final loss at every width (105 of 109 pairs so far); cosine to 10% beats cosine to 0 (91 of 153).

**Evidence:** Posted on the direction-tree branch (TREE.md).

**How to test it:** 4 pairs at width 1024 and 2048. Fails if cosine wins 2 or more.

**Backup if it fails:** Fit loss as a function of decay share and final lr.

**Closest papers:** arXiv:2502.15938 (linear to zero beats cosine), arXiv:2409.04777, arXiv:2405.18392.

**Status:** known.

## C16. Under Muon the best learning rate falls as width^−0.64

**Claim:** Under Muon the best peak learning rate follows width^−0.64 to within a factor of 1.5 out to width 4096, and depth does not move it (−0.64 in sweeps, −0.67 in the fit).

**Evidence:** Posted on the direction-tree branch (TREE.md) and final-loss-findings.

**How to test it:** 5-point lr sweeps at widths 2048 and 4096. Fails if the best lr is off the line by more than
a factor of 1.5.

**Backup if it fails:** Fit ln lr* = a + b·ln(width) + c·ln(width)^2.

**Closest papers:** arXiv:2512.05620, arXiv:2602.20937, arXiv:2601.01306, arXiv:2505.02222.

**Status:** known (posted; width transfer for Muon is published, our exponent is not).

## C17. Fact recall is log-linear in repeats and size

**Claim:** Fact recall is log-linear in repeats and size with no interaction: −0.61 bits per doubling of repeats and −1.46 bits per doubling of model size, held-out 4.84 bits.

**Evidence:** Posted on the direction-tree branch (TREE.md).

**How to test it:** 3 repeat counts at width 2048. Fails if the slope leaves 0.4-0.8 bits or an interaction
term lowers the held-out error.

**Backup if it fails:** Fit bits = a − b·log2(repeats) − c·log2(params) + d·log2(repeats)·log2(params).

**Closest papers:** arXiv:2404.05405 (about 2 bits per parameter), arXiv:2505.24832, arXiv:2202.07646.

**Status:** partly known (posted by this repo).

## C18. The cumulative weight-decay rule does not predict forgetting here

**Claim:** Weight decay is not the clock of forgetting: runs with zero weight decay forget as much as the lr clock predicts, and the Bordt et al. rule scores 0.8012 held-out against 0.0441 for the lr clock.

**Evidence:** `python3 core/score.py forgetting problems/forgetting/formulas/cum_wd.py`

**How to test it:** Weight decay 0 runs with high lr. The rule predicts no forgetting; we measure strong forgetting.
One pair at width 256 decides it.

**Backup if it fails:** Combine both clocks, ln S and ln(cumulative wd), and fit their weights.

**Closest papers:** arXiv:2410.03249.

**Status:** known (a published rule that does not hold on our data).

## C19. The Tissue annealing law scores worse than Chinchilla here

**Claim:** The Tissue et al. annealing law (forward area plus annealing area) scores worse than Chinchilla at every width: 0.1811 against 0.1471 overall.

**Evidence:** Posted on the direction-tree branch (TREE.md).

**How to test it:** Score both on each width group. Fails if it beats Chinchilla at any width.

**Backup if it fails:** Add a width term to its annealing coefficient.

**Closest papers:** arXiv:2408.11029, arXiv:2512.13705.

**Status:** known (a published law that does not hold on our data).

## C20. Peri-LN beats Pre-LN

**Claim:** Peri-LN beats Pre-LN at any training length: normalization before and after each sublayer wins by 0.32 in validation loss in 13 of 13 cells at 150-600 steps, and by more than 0.05 at 2,400 steps.

**Evidence:** Init search, 150-600 steps. The residual search agrees (+0.25, Pre-LN wins 0 of 8).
`python3 tools/claims_fits.py other init norm_mode`

**How to test it:** 2 pairs at 2,400 steps. Fails if the gap falls below 0.05.

**Backup if it fails:** Fit the gap against steps, gap = a·steps^−b.

**Closest papers:** arXiv:2502.02732 (Peri-LN), arXiv:2002.04745.

**Status:** known.

## C21. Gated DeltaNet beats attention at short training

**Claim:** Gated DeltaNet beats softmax attention at small scale for any training length: 0.26 lower validation loss in 21 of 21 cells at 150-600 steps, and at least half that gap at 4x the steps.

**Evidence:** Mixing search, 150-600 steps. `python3 tools/claims_fits.py other mixing mix`

**How to test it:** 2 pairs at 4x the steps. Fails if the gap halves.

**Backup if it fails:** Fit the gap against tokens and find where the curves cross.

**Closest papers:** arXiv:2412.06464 (Gated DeltaNet), arXiv:2607.07953, arXiv:2504.14366.

**Status:** partly known. Small-scale, short-training gaps are not reported.

## C22. Latent attention beats standard attention

**Claim:** Latent attention beats standard attention at small scale, and the gap does not shrink with training: 0.068 lower validation loss in 30 of 34 cells.

**Evidence:** Mixing search. `python3 tools/claims_fits.py other mixing mix`

**How to test it:** 2 pairs at 4x the steps, with and without RoPE on the latent path. Fails if the gap falls
below 0.03.

**Backup if it fails:** Fit the gap against the latent rank.

**Closest papers:** arXiv:2405.04434, arXiv:2412.19437, arXiv:2506.09342 (reports a small loss rise for latent
attention with RoPE, the opposite sign).

**Status:** partly known, with a sign conflict to resolve first.

---

## Credit

C04, C09 (kept side), C14 and the C05 baseline come from the draft paper (doi:10.5281/zenodo.24575889). C15,
C16, C17 and C19 are posted on the direction-tree branch; C06 and the C16 width term on final-loss-findings; the
C04 backup on forgetting-curve-shape. A paper on any of these offers co-authorship to whoever posted it.
