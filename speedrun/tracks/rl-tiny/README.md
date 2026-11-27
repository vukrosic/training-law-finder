# Track rl-tiny

Train the best Snake agent you can with 2 million environment steps, scored by the mean food eaten on 256 fixed
evaluation games.

| model | environment | metric | constraint | hardware | baseline |
|---|---|---|---|---|---|
| RL agent | `env.py`: Snake on a 10 x 10 grid with random pillars, numpy, runs locally | mean food eaten per game on 256 fixed games, higher is better | 2e6 environment steps | MacBook (M4, mps) or any machine: the step count is the same everywhere | 9.1 ± 0.3 |

The name says only the model and its size (see [speedrun/](../../)); the environment, budget, hardware and metric are the ones above.
The same rules are in `track.json`, which [common/check_submission.py](../../common/check_submission.py) reads. All tracks: [speedrun/](../../).

Closest existing benchmark: MinAtar (Young and Tian, 2019), small 10 x 10 versions of Atari games; this track is
one game with a new layout every game, a fixed step budget and fixed evaluation games.

## The game

The snake starts with length 3 in the middle of a 10 x 10 board, heading up. Actions: up, down, left, right;
turning straight back is ignored. Each food gives reward 1 and makes the snake one cell longer. The game ends when
the head hits the edge, a pillar or its own body, after 100 steps without food, or when the board is full (about
91 food). Each game's seed places 0 to 6 pillars and fixes where every food appears, so a seed and the actions
always give the same game. The agent sees 4 channels of 10 x 10: pillars, food, head, and the body as age / length.

Random play eats 0.1 food per game. A hand-written policy that steps towards the food and avoids walls eats 17.7
(`python3 eval.py`). Eating most of the board needs planning: a long snake traps itself.

## Leaderboard

| food per game | env steps | FLOPs | date | name | who | what changed |
|---|---|---|---|---|---|---|
| 9.1 ± 0.3 | 1,998,848 | 4.737e13 | 2026-11-27 | [baseline](records/2026-11-27_baseline/) | vukrosic | PPO, small conv net, Adam |

## Rules

1. **Budget.** At most 2,000,000 environment steps in training, counted as one per game per `env.step` call
   (64 games stepped once = 64 steps). Every step counts, also steps used only to explore, to collect data for a
   world model or to tune. Evaluation games are not part of the budget. Compute is free; report your training
   FLOPs (`flops` in `submission.json`, counted with [common/flops.py](../../common/flops.py) on the `meta`
   device) so we can study how score grows with compute.
2. **Score.** `eval.score`: the mean food eaten over the 256 evaluation games (seeds 1,000,000,000 to
   1,000,000,255), each played to its end. Your agent is a function `policy(obs (B, 4, 10, 10)) -> actions (B,)`.
   torch is seeded with 0 before evaluation, so a policy that samples is scored the same way each time.
   Use `eval.py` and `env.py` as they are.
3. **Environment.** Train only with `env.Snake` and seeds below `eval.EVAL_SEED`. No changes to the game, no
   reading the game's internal state (food hash, seed) in the policy: the policy gets only the observation.
   Reward shaping inside your training code is allowed; the score is always food eaten.
   No network calls during training or evaluation.
4. **Anything else is free:** algorithm (PPO, DQN, model-based, search at training time), network, number of
   games side by side, optimizer, updates per step.
5. **New record.** Precision, the minimum gain and the track version follow [speedrun/](../../#records); the
   standard error comes from the 256 games, so scores have 1 decimal. A maintainer reruns it before it goes on the board.

## Run

```bash
cd speedrun/tracks/rl-tiny
pip install torch numpy
python3 test_env.py              # checks the game and the scorer
python3 eval.py                  # reference scores: random and hand-written
python3 train.py | tee train.log # cuda, then mps, then cpu
```

The last line of the log is `result {...}` with `mean_return`, `env_steps`, `flops`, the device and the wall time;
the food of each evaluation game goes to `per_item.json` next to `train.py`.

## Submit

Open a pull request that adds one folder, `records/<date>_<name>/`, with:

1. `README.md`: what you changed and why, in a few lines.
2. `submission.json`: `name`, `github`, `date`, `version`, `mean_return`, `mean_return_se`, `env_steps`, `flops`,
   `device`, `train_seconds`, `params`, plus `per_item.json` (under 1 KB).
3. `train.log`: the log your run printed, trimmed to under 50 KB.
4. `train.py`: the exact script. It must run with `python3 records/<date>_<name>/train.py` from this folder.
   If it needs other packages, add `requirements.txt`.

Add your row to the leaderboard above in the same pull request. `python3 tools/check.py` (from the repository root)
must pass; it runs `speedrun/common/check_submission.py` on every record.
