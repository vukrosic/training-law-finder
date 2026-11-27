# Loss-curve laws: literature, first pass (2026-11-27)

First pass found no paper that scores one formula on whole loss curves across both widths and schedules with
held-out widths. Gate 0 decides; this is not a novelty claim.

The first pass was two search agents plus a direct read of the two main papers. "Full text" means the formula
was read in the paper; "abstract" means only the abstract or a search summary was read.

| law | what it predicts | inputs | where it is known to fail | checked |
|---|---|---|---|---|
| Tissue et al. 2024, [arXiv:2408.11029](https://arxiv.org/abs/2408.11029): L = L0 + A·S1^−α − C·S2 | loss at every step | lr per step. S1 = lr area, S2 = annealing area with a fixed decay λ = 0.999. 4 constants; a size form adds 3 (S2 ∝ N^γ is assumed) | warmup not modeled (counted at peak lr); no width sweep; re-warming not tested; fit with Huber loss on log loss | full text |
| Luo et al. 2025, multi-power law, [arXiv:2503.12811](https://arxiv.org/abs/2503.12811) (ICLR 2025): L = L0 + A·(S1+S_W)^−α − LD(t) | loss at every step, across schedules | lr per step, warmup length, peak lr. 7 constants | tested at 25M-400M (1B only in a short setting); width not varied; accuracy drops at higher peak lr; constants depend on batch size; derived for quadratic losses and decaying schedules | full text |
| Qiu et al. 2025, scaling collapse, [arXiv:2507.02119](https://arxiv.org/abs/2507.02119) (ICML 2025) | curve shape after scaling compute and loss to 1 at the end | normalized compute t/t*, fitted L0 | a check, not a full predictor; needs the final loss; breaks without well-tuned hyperparameters across sizes (constant lr without μP) | full text |
| Li et al. 2025, functional scaling laws, [arXiv:2509.19189](https://arxiv.org/abs/2509.19189) (NeurIPS 2025) | whole loss path in kernel regression | intrinsic time = lr area, schedule | theory is kernel regression, not transformers; the LLM section (0.1B-1B) was not checked for held-out tests | full text (theory part) |
| NCPL, [arXiv:2602.10300](https://arxiv.org/abs/2602.10300) (2026) | final and intermediate loss, from a learned model | architecture, tokens, steps, optimizer, schedule as a category, batch | a neural predictor, not a formula; held-out by size above 430M, not by width or new schedule | full text |
| Schaipp et al. 2025, [arXiv:2501.18965](https://arxiv.org/abs/2501.18965) (ICML 2025) | convex-theory bound that tracks WSD and cosine curves | schedule | bound form not confirmed; no warmup | abstract |
| Hägele et al. 2024, [arXiv:2405.18392](https://arxiv.org/abs/2405.18392) (NeurIPS 2024) | WSD with a short cooldown scales like cosine | schedule, length | empirical, no curve formula | abstract |
| Bergsma et al. 2025, [arXiv:2509.25087](https://arxiv.org/abs/2509.25087) | curves of μP model families collapse after normalizing | normalized time and loss | needs optimal hyperparameters; held-out tests not checked | abstract |
| Chang, Tu and Bergen 2024, [arXiv:2308.15419](https://arxiv.org/abs/2308.15419) (TACL) | per-token learning curves: frequent tokens are learned earlier | token frequency | 5 runs at one setup; no formula across settings | abstract |

Related but not curve laws: Bergsma et al. 2025 "Power Lines" (arXiv:2505.13738, weight decay and batch size),
Bergsma et al. 2025 "Straight to Zero" (arXiv:2502.15938, linear decay to zero), Wen et al. 2025 (arXiv:2410.05192,
river-valley picture of WSD), Bjorck et al. 2025 (arXiv:2409.19913, best lr falls with token count).

## What this problem adds, if gate 0 agrees

1. Held-out widths: every published curve law above is fit and tested at fixed sizes or with size as a side
   term. Here a formula must predict curves at widths it never saw.
2. Three schedule shapes on one grid, from 512 to 16,384 steps, at 3 peak learning rates per width.
3. Four token-frequency bands logged at the same 32 points, for a later problem on which tokens drive the
   curve shape.

## Open checks for gate 0

- Full text of Schaipp et al. 2025 and the LLM section of Li et al. 2025: do either score held-out widths?
- Follow-up papers that cite the multi-power law (2025-2026) and test it across model sizes.
- Whether any paper tests a single lr-area clock against WSD and constant runs at matched area (the kill line
  in [prereg/loss-curve.md](../../prereg/loss-curve.md)).
