# Baseline: 9.15 food per game

PPO with a small conv net, trained with 1,998,848 of the 2,000,000 environment steps.

- Net: conv 3 x 3 (4 to 16 channels), conv 3 x 3 (16 to 32), one hidden layer of 128, ReLU; a policy head
  (4 actions, small orthogonal init) and a value head share it. 415,605 parameters.
- PPO: 64 games side by side, 128 steps per update (8192 steps), 244 updates, 4 epochs of 4 minibatches,
  clip 0.2, GAE (gamma 0.99, lambda 0.95), entropy 0.01, value loss 0.5, gradient norm clipped at 0.5.
- Adam (lr 5e-4, eps 1e-5), learning rate falling linearly to 0. No reward shaping.
- Evaluation plays the most likely action.
- FLOPs (reported, not limited): 1.941e11 per update, 4.737e13 in all (FlopCounterMode).

Food per training game (last 200 games) was still rising at the end: 3.3 at 0.8M steps, 5.5 at 1.2M, 8.6 at 2M.
The hand-written greedy policy in `eval.py` eats 17.68, and the board holds about 91.

Run from `speedrun/tracks/rl-tiny/`: `python3 records/2026-11-27_baseline/train.py`.
Training took 379 s on a MacBook Pro M4 (mps); evaluation 3 s. Peak memory footprint 1.7 GB.
