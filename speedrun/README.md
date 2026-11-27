# Speedrun

Competitions with fixed rules. Each one is a track.

A track is one fixed competition: a model type, a dataset, one metric, one constraint and the hardware it is
meant for. Within a track everything else is free, so scores are comparable.

The constraint differs per track. It can be training FLOPs, wall time, peak memory, parameter count or
training tokens. The first track uses a FLOP budget, so its score does not depend on the machine.

## Tracks

| track | model | data | metric | constraint | hardware | best |
|---|---|---|---|---|---|---|
| [lm-fineweb-1e14](tracks/lm-fineweb-1e14/) | language model | FineWeb-Edu | validation bits per byte | 1e14 training FLOPs | MacBook (M4, mps) | 1.932 bpb |

Planned:

- lm-fineweb-1e15: the same language model track with 1e15 FLOPs, for one GPU. Not built yet.
- A world-model track: predict the next frames of a video, scored by prediction error. Not built yet.
- A small RL track. Not built yet.

## Layout

```
speedrun/
  common/                 shared by all tracks
    flops.py              exact FLOP counter (FlopCounterMode on the meta device)
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
