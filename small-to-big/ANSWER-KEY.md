# Answer key: published big-model numbers

CPU methods are scored against these. "Checked" means the value was confirmed against the paper or a source quoting it on 2026-11-28; "verify" means it comes from training data and must be checked before a trial is scored on it. Most rows use Adam; our runs mostly use Muon, so compare directions and exponents first and absolute values second.

## Learning rate and batch

| ID | Source | Law or number | Range | Optimizer | Status |
|---|---|---|---|---|---|
| A1 | DeepSeek LLM (arXiv:2401.02954) | lr* = 0.3118 · C^-0.125, B* = 0.2920 · C^0.3271 (C in FLOPs, B in tokens) | 1e17-3e20 FLOPs, 8 budgets | Adam | checked (via [survey](https://arxiv.org/pdf/2307.06435)) |
| A2 | Step Law, StepFun (arXiv:2503.04715) | lr* = 1.79 · N^-0.713 · D^0.307, B* = 0.58 · D^0.571 | grid search, dense and MoE | Adam, constant min lr 1e-5, no decay to 0 | checked ([paper](https://arxiv.org/html/2503.04715v4)) |
| A3 | GPT-3 (Brown et al. 2020), Table 2.1 | lr 6e-4 at 125M, 3e-4 at 350M, 2e-4 at 1.3B, 1.2e-4 at 6.7B, 0.6e-4 at 175B; batch 0.5M-3.2M tokens | 125M-175B | Adam | verify |
| A4 | Yang et al. 2022, μP | best Adam lr falls about as 1/width under standard parametrization; constant under μP | widths 128-8192 | Adam | verify |
| A5 | Bjorck et al. 2024 (arXiv:2409.19913) | best lr falls as a power of the token horizon | 50M-7B | Adam | verify |
| A6 | arXiv:2606.29158 | best lr vs steps bends (flattens) at long horizons; no formula given | 22M-707M | Adam | from cpu-3 LITERATURE.md |

**Conflict to resolve (A2 vs cpu-3).** Step Law says the best lr *rises* with data (D^+0.307). cpu-3's rule says it *falls* with steps toward a floor. The two differ in setup: Step Law grows the batch with data (B ∝ D^0.571) and does not decay to 0, while cpu-3 holds the batch fixed and decays to 0. A rule that explains both (lr* as a function of steps and batch together) would be a strong small-to-big result.

## Compute-optimal size and data

| ID | Source | Law or number | Status |
|---|---|---|---|
| B1 | Chinchilla (Hoffmann et al. 2022) | about 20 tokens per parameter at compute-optimal; N* and D* both grow about as C^0.5 | well known |
| B2 | Kaplan et al. 2020 | N* grows as C^0.73; differs from Chinchilla mainly through schedule and embedding-count choices (Porian et al. 2024) | verify |

## Batch and noise

| ID | Source | Law or number | Status |
|---|---|---|---|
| C1 | McCandlish et al. 2018; Kaplan et al. 2020 | the critical batch grows as the loss falls (about B_crit ∝ L^-4.8 in Kaplan) | verify |
| C2 | Zhang et al. 2024 (arXiv:2410.21676) | the critical batch grows with data size, not with model size | verify |

## Schedule

| ID | Source | Law or number | Status |
|---|---|---|---|
| D1 | Hägele et al. 2024 (arXiv:2405.18392) | constant lr plus a short cooldown (about 20%) matches cosine | verify |
| D2 | Tissue et al. 2024 (arXiv:2408.11029) | loss as a function of the lr area and the annealing area | verify |
