# Methods

Each method says how it works, what it can reach, what it cannot, and its calibration test (a known big-model law it must recover from CPU runs before we trust it). Status: untested, testing, passed or failed, with a link to the row in [TRIALS.md](TRIALS.md).

## M1. Big labs' published runs as the test set

**How.** Fit the rule on CPU runs only. Then predict numbers big labs published (best learning rate, batch size, loss) and score the error. The published runs are free data at 100M-100B parameters.

**Reaches.** Any rule whose output a lab reports: learning rate, batch, tokens per parameter, loss vs compute.

**Cannot.** Labs report one tuned setting, not the full curve, and most used Adam, not Muon. So compare trends (exponents, directions, ratios) more than absolute values, and say when the optimizer differs.

**Calibration.** None needed: M1 is the scoring step the other methods use.

Status: testing (cpu-3, trial T1).

## M2. Match big-model ratios, not sizes

**How.** Many rules depend on ratios, not on parameter count: tokens per parameter, batch over the critical batch, decay share of the schedule, learning rate times steps, width over depth. Put CPU models at the ratio big models use (for example 20 tokens per parameter: a 200K model on 4M tokens), and study the rule there.

**Reaches.** Rules that collapse onto one curve when written in ratios. A curve that collapses across 4+ sizes is the argument that it keeps collapsing at bigger sizes.

**Cannot.** Ratios a CPU cannot reach in time, for example 1,000+ tokens per parameter at 1M parameters (1B tokens), or batches of millions of tokens.

**Calibration.** From the 24,725 finished runs in `data/llm/runs.jsonl.gz`, find the compute-optimal tokens per parameter. Chinchilla says about 20. Pass if the CPU estimate lands within 10-40 and has the same direction of change with compute.

Status: untested (trial T2).

## M3. Mechanism plus simulator

**How.** Explain why a rule holds with a cause (for example, the decay at the end of training removes gradient noise), write that cause as a small model in the noise simulator (`families/llm/search-code/text-laws/sim/nqm.py`), and check that the simulator and the real runs agree. The rule then carries to any size where the cause still exists.

**Reaches.** Optimizer, schedule, batch and noise rules, where the cause is curvature and gradient noise, which exist at every size.

**Cannot.** Rules whose cause is the data or the network's features (what the model learns), which the simulator does not have.

**Calibration.** The simulator must reproduce a known big-model law: the critical batch grows as the loss falls (Kaplan 2020; McCandlish 2018). Pass if the simulated critical batch rises as the loss falls, and the CPU runs show the same.

Status: untested (trial T3).

## M4. Break a published rule

**How.** A paper claims a rule holds everywhere. One clean small-scale counterexample, with a control, breaks the claim at every size the paper covers.

**Reaches.** Any rule stated without a size limit.

**Cannot.** Prove a new rule holds at scale. It only removes wrong ones.

**Calibration.** None needed: a counterexample is valid at any size. cpu-3 already did one: its rule beat Bjorck et al.'s formula (error 0.068 vs 0.21 in ln lr*).

Status: passed once (T0).

## M5. Trend across sizes with a frozen extrapolation

**How.** Measure the rule at 4+ sizes (for example width 32, 48, 64, 96, 128, 192). Fit on the smallest sizes only, write predictions for the largest in `prereg/` before running them, then score them. Also check whether the exponent itself drifts across the range: a drifting exponent means the extrapolation to big sizes is unsafe.

**Reaches.** Any rule with a size exponent.

**Cannot.** Rule out a bend beyond the largest CPU size.

**Calibration.** With Adam under standard parametrization, the best learning rate falls about as 1/width (Yang et al., μP, 2022). Pass if CPU runs at 4+ widths give an exponent between -0.7 and -1.3.

Status: untested (trial T4).

## M6. Questions where small is the target

**How.** Pick questions whose users run small models: phone and laptop models, students with one GPU, fast speedruns. The finding is useful as is.

**Reaches.** Small-batch training, short runs, tiny vocabularies, CPU training itself.

**Cannot.** Claim anything about big models. Say so in the post.

**Calibration.** None needed.

Status: in use (the `speedrun/` records).
