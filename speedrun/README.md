# Speedrun

Competitions with fixed rules. Each one is a track.

A track is one fixed competition: a model type, a dataset, one metric, one constraint and the hardware it is
meant for. Within a track everything else is free, so scores are comparable.

The constraint differs per track. It can be training FLOPs, environment steps, wall time, peak memory, parameter
count or training tokens. Each built track uses a budget that is the same on every machine (FLOPs or environment
steps), so its score does not depend on the machine.

## Tracks

| track | model | data | metric | constraint | hardware | best |
|---|---|---|---|---|---|---|
| [lm-tiny](tracks/lm-tiny/) | language model | FineWeb-Edu | validation bits per byte | 1e14 training FLOPs | MacBook (M4, mps) | 1.932 bpb |
| [world-model-tiny](tracks/world-model-tiny/) | action-conditioned frame predictor | bouncing balls + a square moved by actions, generated locally | 12-frame rollout MSE | 1e14 training FLOPs | MacBook (M4, mps) | 0.007029 |
| [rl-tiny](tracks/rl-tiny/) | RL agent | Snake, 10 x 10 with random pillars, runs locally | mean food per game on 256 fixed games | 2e6 environment steps | MacBook (M4, mps) | 9.15 food |

Planned:

- lm-small: a larger language model track for one GPU. Not built yet.

## Names

A track is named `<model>-<size>`: `lm-tiny` is the tiny language model track. The dataset, budget, hardware
and metric are not in the name; they are in the track's `README.md` and `track.json`. For `lm-tiny` those are
FineWeb-Edu ([vukrosic/training-law-speedrun](https://huggingface.co/datasets/vukrosic/training-law-speedrun)),
1e14 training FLOPs, a MacBook or any laptop, and validation bits per byte.

## Layout

```
speedrun/
  common/                 shared by all tracks
    flops.py              exact FLOP counter (FlopCounterMode on the meta device)
    check_submission.py   checks a record folder against its track.json
  tracks/<track>/
    README.md             rules, data, how to run, leaderboard
    track.json            metric, constraint, budget, hardware, baseline (and extra required fields)
    env.py / prepare_data.py   the environment, or the script that gets the data
    eval.py               the only scorer
    train.py              the baseline
    records/<date>_<name>/
```

## Submit

Pick a track and follow its README. Open a pull request that adds one folder,
`tracks/<track>/records/<date>_<name>/`, with:

1. `submission.json`: your name, GitHub handle, date, device, the track's metric and the track's constraint.
2. `train.py`: the exact script.
3. `train.log`: the log your run printed, under 50 KB.
4. `README.md`: what you changed, in a few lines.

Add your row to the track's leaderboard. From the repository root, `python3 tools/check.py` must pass;
it runs `common/check_submission.py` on every record. A maintainer reruns a new record before it goes on the board.
