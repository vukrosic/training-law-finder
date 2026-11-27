# Baseline: score 19.8 ± 0.9 (val MSE 0.00704)

A small conv U-Net trained with 9.997e13 of the 1e14 FLOPs.

- Model: input the last 4 frames (12 channels); conv 32 channels at 32 x 32, strided convs to 64 channels at
  16 x 16 and 8 x 8, two residual blocks (GroupNorm, SiLU) at 8 x 8, transposed convs back up with skip
  connections. 350,691 parameters.
- Action: FiLM in each residual block (a learned scale and shift per channel), zero at the start.
- Output: last frame plus a learned change; the last conv starts at zero, so the untrained model copies the last frame.
- Training: one step ahead with true frames as input (teacher forcing), 12 examples per rollout, 16 new rollouts
  per step (batch 192), 2293 steps, 36,688 rollouts. Adam (lr 2e-3, betas 0.9, 0.99), constant, then linear to 0
  over the last 40% of steps.
- Evaluation: 12 steps rolled forward on its own predictions, clamped to [0, 1].
- FLOPs per step 4.36e10 (FlopCounterMode).

The one-step training loss ends at 0.00031, but the 12-step rollout error is 0.0070: most of the error comes from
rolling forward on its own predictions, which this baseline never trains for.

Run from `speedrun/tracks/world-model-tiny/`: `python3 records/2026-11-27_baseline/train.py`.
Training took 242 s on a MacBook Pro M4 (mps); evaluation 2 s. Peak memory footprint 1.9 GB.
`per_item.json` holds the score of each of the 512 validation rollouts.
