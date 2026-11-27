# Baseline: 1.9319 bpb

A small modern GPT trained with 9.986e13 of the 1e14 FLOPs, every op counted (counter all-ops-v1).

- Model: 4 layers, width 192, 3 heads of 64, SwiGLU hidden 512, 2.16M parameters.
  RMSNorm, RoPE, QK-norm, no biases, untied head, output projections and head start at zero.
- Optimizers: Muon (lr 0.05, momentum 0.95) on the block matrices; Adam (betas 0.8, 0.95) on the
  embedding (lr 0.6), head (lr 0.008), norms (lr 0.04).
- Learning rate constant, then falls linearly to 0 over the last 40% of steps.
- Batch 64 x 256 tokens, 412 steps, 6.75M tokens, default sp1024 tokenizer.
- FLOPs per step (fast count, meta device) 2.424e11: matrix multiplies and attention 2.407e11, Muon's
  Newton-Schulz in that; elementwise ops, norms, softmax and the Adam update 1.7e9.
- Full count on mps (every step, `--record`): 1.0005e14, 0.19% above the fast count, from the casts to bf16.
  Under the 1% tolerance, so the fast count is the score.
- Artifact (code + tokenizer + float32 weights, lzma): 7,896,828 bytes of the 16,000,000 allowed.

Run from `speedrun/tracks/lm-tiny/`: `python3 records/2026-11-27_baseline/train.py --record`.
Training took 162 s on a MacBook Pro M4 (mps) with one other job on the machine; evaluation 13 s.

Under the first counter (matrix multiplies only) the same model ran 415 steps and scored 1.932 bpb.
