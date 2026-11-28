# Trials

One row per trial. Verdict: passed, failed, open. Each trial names the method ([METHODS.md](METHODS.md)) and the answer-key row ([ANSWER-KEY.md](ANSWER-KEY.md)) it is scored on.

| ID | Method | Box | Question | Answer key | Result | Verdict |
|---|---|---|---|---|---|---|
| T0 | M4 | cpu-3 | Does the Bjorck lr-vs-horizon formula predict cpu-3's runs? | A5 | Bjorck error 0.21 vs cpu-3 rule 0.068 in ln lr* (held-out 512-2048 steps) | passed: counterexample at small scale |
| T1 | M1, M3, M5 | cpu-3 | Does lr* = lr_inf + K/T hold on big models, and what does each term scale with? | A1, A2, A5, A6 | started 2026-11-28 18:42 SGT (Adam arm, 10x model, batch floor) | open |
| T2 | M2 | next free box | Do CPU runs recover Chinchilla's ~20 tokens per parameter? Fit the compute-optimal frontier on `data/llm/runs.jsonl.gz` (24,725 runs), no new training | B1 | not started | open |
| T3 | M3 | next free box | Does the noise simulator, and cpu-1's batch runs, show the critical batch growing as the loss falls? | C1, C2 | not started | open |
| T4 | M5 | 3090 data, then cpu-4 (from 2026-11-28 21:10 SGT) | Does Adam's best lr fall about as 1/width? | A4 | First look from 3090 round 1 (3 widths, so below the 4-size bar): at 500 steps best Adam lr 0.0040, 0.0028, 0.0010 at width 64, 128, 256, exponent about -1.0. At 2,000 and 8,000 steps the trend is not monotone (Adam's lr curve has a cliff and 0.02-0.04 wiggles). Needs widths 32-512 at 500 steps on a finer lr grid | open, promising |
| T5 | M1, M2, M5 | cpu-2 | What sets the warmup scale (steps, tokens, lr rise rate, width, optimizer averaging time), and does the cost's width exponent hold? | Ma and Yarats 2021; GPT-3 and Llama 2 warmups | bar sent 2026-11-28 19:15 SGT | open |
| T6 | M1, M2, M5 | cpu-1 | The best WSD decay share in scale-free terms, with the batch term | D1, D2 | bar sent 2026-11-28 19:15 SGT | open |
| T7 | M1, M5 | gpu-3090, then cpu-5 (from 2026-11-28 21:10 SGT) | The Adam-Muon gap vs tokens per parameter at 4+ widths | Moonshot "Muon is scalable" (to add to the answer key) | bar sent 2026-11-28 19:15 SGT | open |

## Notes

- T1 must explain the A2 conflict: Step Law's best lr rises with data, cpu-3's falls with steps. The difference may be that Step Law grows the batch with data and does not decay to 0.
- 3090 round 1 also shows Adam's best lr rising from 2,000 to 8,000 steps at widths 64-256 (0.0028 to 0.0080 at width 64), the same direction as Step Law and the opposite of cpu-3's Muon rule. The cliff makes this uncertain; T1 should check whether it is the optimizer, the schedule or noise.
