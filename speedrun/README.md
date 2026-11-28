# Speedrun

Competitions with fixed rules. Each one is a track.

A track is one fixed competition: a model type, a dataset, one metric, one constraint and the hardware it is
meant for. Within a track everything else is free, so scores are comparable.

The constraint differs per track. It can be training FLOPs, wall time, peak memory, parameter count or
training tokens. The first track uses a FLOP budget that counts every op, so its score does not depend on how
fast the machine is.

## Tracks

| track | model | data | metric | constraint | hardware | best |
|---|---|---|---|---|---|---|
| [lm-tiny](tracks/lm-tiny/) | language model | FineWeb-Edu | validation bits per byte | 1e14 training FLOPs (every op), 16 MB artifact | MacBook (M4, mps) | 1.8466 bpb |

Planned:

- lm-small: a larger language model track for one GPU. Not built yet.
- world-model-tiny: predict the next frames of a video, scored by prediction error. Not built yet.
- rl-tiny: a small reinforcement learning track. Not built yet.

## Names

A track is named `<model>-<size>`: `lm-tiny` is the tiny language model track. The dataset, budget, hardware
and metric are not in the name; they are in the track's `README.md` and `track.json`. For `lm-tiny` those are
FineWeb-Edu ([vukrosic/training-law-speedrun](https://huggingface.co/datasets/vukrosic/training-law-speedrun)),
1e14 training FLOPs, a MacBook or any laptop, and validation bits per byte.

## Layout

```
speedrun/
  common/                 shared by all tracks
    flops.py              FLOP counter: every op of every training step, on the real device
    test_flops.py         checks the counter against hand-worked numbers
    check_submission.py   checks a record folder against its track.json
  tracks/<track>/
    README.md             rules, data, how to run, leaderboard
    track.json            metric, constraint, budget, hardware, baseline
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
