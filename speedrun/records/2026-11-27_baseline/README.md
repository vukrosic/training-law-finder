# Baseline: 1.932 bpb

A small modern GPT trained with 9.989e13 of the 1e14 FLOPs.

- Model: 4 layers, width 192, 3 heads of 64, SwiGLU hidden 512, 2.16M parameters.
  RMSNorm, RoPE, QK-norm, no biases, untied head, output projections and head start at zero.
- Optimizers: Muon (lr 0.05, momentum 0.95) on the block matrices; Adam (betas 0.8, 0.95) on the
  embedding (lr 0.6), head (lr 0.008), norms (lr 0.04).
- Learning rate constant, then falls linearly to 0 over the last 40% of steps.
- Batch 64 x 256 tokens, 415 steps, 6.8M tokens, default sp1024 tokenizer.
- FLOPs per step 2.407e11 (FlopCounterMode): 6N gives 1.935e11, attention 3.87e10, Muon Newton-Schulz the rest.

Run from `speedrun/`: `python3 records/2026-11-27_baseline/train.py`.
Training took 435 s on a MacBook Pro M4 (mps) with two other jobs using the GPU; evaluation 33 s. Peak memory 1.8 GB.
