# Track world-model-tiny

Train the best world model you can with 1e14 training FLOPs: from 4 frames and the actions, predict the next 12
frames of a small 2D world. Scored by rollout mean squared error on 512 fixed validation rollouts.

| model | data | metric | constraint | hardware | baseline |
|---|---|---|---|---|---|
| action-conditioned frame predictor | `env.py`: bouncing balls and a square moved by actions, 32 x 32 RGB, generated locally | 12-frame rollout MSE, lower is better | 1e14 training FLOPs | MacBook (M4, mps); the count is the same on any machine | 0.007029 |

The name says only the model and its size (see [speedrun/](../../)); the environment, budget, hardware and metric are the ones above.
The same rules are in `track.json`, which [common/check_submission.py](../../common/check_submission.py) reads. All tracks: [speedrun/](../../).

Closest existing benchmark: Moving MNIST next-frame prediction (Srivastava et al., 2015); this track adds actions,
collisions and a fixed FLOP budget, and it runs on a laptop.

## The world

Each rollout is 16 frames of 3 x 32 x 32 pixels. One to three balls of random size and speed bounce off the
walls, off each other and off an orange 5 x 5 square. 15 actions (stay, up, down, left, right) move the square
1.5 pixels per step; they come from a random policy that changes its action with probability 0.3 per step.
The world is deterministic, so a perfect model scores close to 0. Two things make it hard: the model must
work out each ball's speed from 4 small frames, and its errors grow as it rolls forward on its own predictions.

`env.generate(n, seed)` makes `n` rollouts in plain numpy. The same seed gives the same bytes on every machine
(`env.py` says why). Nothing is downloaded and nothing is stored.

## Leaderboard

| val MSE | FLOPs | date | name | who | what changed |
|---|---|---|---|---|---|
| 0.007029 | 9.997e13 | 2026-11-27 | [baseline](records/2026-11-27_baseline/) | vukrosic | conv U-Net, 351k parameters, one-step teacher forcing, Adam |

Reference scores, no training: an empty frame 0.02487, repeating the last frame 0.03549 (`python3 eval.py`).

## Rules

1. **Budget.** At most 1e14 training FLOPs, about 4 minutes on a MacBook (M4, mps).
   [common/flops.py](../../common/flops.py) counts one full training step (forward, backward and optimizer step)
   with `count_step_inputs`, on the `meta` device, times the number of steps. If your steps differ, count each
   kind. If a step does work the counter cannot see, count it and say how. Generating rollouts and evaluation are
   not part of the budget. FLOPs spent on anything that uses training rollouts (for example a pretrained encoder
   or a separate decoder) count.
2. **Score.** `eval.rollout_mse`: mean over the 512 validation rollouts, the 12 predicted frames, 3 channels and
   all pixels of (prediction - truth)^2, pixels in [0, 1]. Your model is a function
   `predict(context (B, 4, 3, 32, 32), actions (B, 15)) -> (B, 12, 3, 32, 32)`; it never gets a frame it must
   predict. Use `eval.py`, `env.py` and `common/flops.py` as they are. The score is reported to 4 significant digits.
3. **Data.** Train only on rollouts from `env.generate` with seeds other than `eval.VAL_SEED` (20261127). Any
   number of rollouts, any rollout length, any seeds. No network calls during training or evaluation.
4. **Anything else is free:** model size, architecture (conv, transformer, recurrent, diffusion, latent),
   training objective (multi-step, noise, curriculum), optimizer, batch size, precision.
5. **New record.** It must beat the current best by at least 2% (for example 0.006888 against 0.007029).
   A maintainer reruns it before it goes on the board.

## Run

```bash
cd speedrun/tracks/world-model-tiny
pip install torch numpy
python3 test_env.py              # checks the world and the scorer
python3 eval.py                  # reference scores
python3 train.py | tee train.log # cuda, then mps, then cpu
```

The last line of the log is `result {...}` with `val_mse`, `flops`, the device and the wall time.

## Submit

Open a pull request that adds one folder, `records/<date>_<name>/`, with:

1. `README.md`: what you changed and why, in a few lines.
2. `submission.json`: `name`, `github`, `date`, `val_mse`, `flops`, `device`, `train_seconds`, `params`.
3. `train.log`: the log your run printed, trimmed to under 50 KB.
4. `train.py`: the exact script. It must run with `python3 records/<date>_<name>/train.py` from this folder.
   If it needs other packages, add `requirements.txt`.

Add your row to the leaderboard above in the same pull request. `python3 tools/check.py` (from the repository root)
must pass; it runs `speedrun/common/check_submission.py` on every record.
