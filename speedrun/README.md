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
| [world-model-tiny](tracks/world-model-tiny/) | action-conditioned frame predictor | bouncing balls + a square moved by actions, generated locally | 12-frame rollout score, 0-100 (100 = repeating the last frame) | 1e14 training FLOPs | MacBook (M4, mps) | 19.8 ± 0.9 |
| [rl-tiny](tracks/rl-tiny/) | RL agent | Snake, 10 x 10 with random pillars, runs locally | mean food per game on 256 fixed games | 2e6 environment steps | MacBook (M4, mps) | 9.1 ± 0.3 food |

Planned:

- lm-small: a larger language model track for one GPU. Not built yet.

## Records

Precision: a score is the mean over a track's fixed evaluation items (rollouts, games), and its standard error comes
from resampling those items 1,000 times with replacement, never from repeated training runs. The score is shown to
the place of the first significant digit of the baseline's standard error (0.9 gives 1 decimal), stored as
`decimals` in `track.json`, so all records of a track use the same precision.

Minimum gain: a new record must beat the current record by more than 2 standard errors of the paired difference,
item by item on the same evaluation set; `common/check_submission.py` reports a smaller gain as
"tie, not a record". Each record stores its per-item scores in `per_item.json` (a few KB); lm-tiny gets this
later, because its per-sequence values need its baseline retrained.

Versions: once a track has outside records, its game, score and budget never change. A real change becomes a new
track with its own leaderboard. An eval bug fix that rescores all records is the only exception, and it is noted
in the track README; each record stores the track `version` it was scored under.

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
    stats.py              bootstrap standard error, decimals, the minimum-gain test
    check_submission.py   checks a record folder against its track.json
  tracks/<track>/
    README.md             rules, data, how to run, leaderboard
    track.json            version, metric, decimals, constraint, budget, hardware, baseline, required fields
    env.py / prepare_data.py   the environment, or the script that gets the data
    eval.py               the only scorer
    train.py              the baseline
    records/<date>_<name>/
```

## Submit

Pick a track and follow its README. Open a pull request that adds one folder,
`tracks/<track>/records/<date>_<name>/`, with:

1. `submission.json`: your name, GitHub handle, date, device, track version, the track's metric and constraint
   (and `per_item.json` where the track asks for it).
2. `train.py`: the exact script.
3. `train.log`: the log your run printed, under 50 KB.
4. `README.md`: what you changed, in a few lines.

Add your row to the track's leaderboard. From the repository root, `python3 tools/check.py` must pass;
it runs `common/check_submission.py` on every record. A maintainer reruns a new record before it goes on the board.
