"""Problem: predict a language model's whole validation-loss curve from its training settings.

Data: data/<family>.jsonl.gz, one line per run with 32 readings of the val loss at evenly spaced steps
(step = round(i * steps / 32), i = 1..32; the last reading is the final step).
Real runs are pending: they come from prereg/loss-curve-grid.jsonl (testbed/train.py --curve_points 32),
and build.py turns their result lines into data/llm.jsonl.gz. Until then the folder holds only
data/synthetic.jsonl.gz, a small made-up fixture (make_fixture.py) that lets the scorer and tools/check.py run.
Its scores say nothing about real training.

rows() turns each run into 32 rows, one per reading. Target: cv_loss, the val loss at that reading.
Score 1 = RMSE over all readings of the held-out widths, averaged over widths: the error averaged over the
32 points, on widths the formula never saw.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import problem_rows  # noqa: E402

GROUP_BY = "width"
TARGET = "cv_loss"   # column target() reads; never shown to a formula
MIN_GROUP = 5        # widths with fewer runs are left out when the data is built
POINTS = 32
FLOOR = None         # pending: floor(rows()) once two runs with the same settings exist (see README)

# Run-level columns kept in the data file. lc_loss (target list) and lc_band / lc_band_share (measured)
# stay in the file for later problems but are not passed on by rows().
COLUMNS = ["tag", "machine", "device", "width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched",
           "peak", "mlr", "wd", "min_lr_frac", "warmup", "decay", "params", "nonemb_params", "tokens",
           "lc_step", "lc_lr", "lc_loss", "lc_band", "lc_band_share"]
RUN_ONLY = ("lc_step", "lc_lr", "lc_loss", "lc_band", "lc_band_share")


def lr_mult(step, steps, warmup, sched, decay, min_lr_frac):
    """testbed/train.py lr_mult with warmup given in steps. step is 0-based."""
    if step < warmup:
        return (step + 1) / warmup
    if sched == "const":
        return 1.0
    if sched == "cos":
        p = (step - warmup) / max(1, steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
    d0 = steps - int(round(decay * steps))
    if step < d0:
        return 1.0
    return 1.0 - (1.0 - min_lr_frac) * (step - d0 + 1) / max(1, steps - d0)


def rows(family=None):
    """One row per (run, reading). Per reading a formula may read:
    cv_frac  reading position i / 32
    cv_step  optimizer steps taken so far
    cv_lr    lr multiplier (share of peak) used at the last step taken
    cv_area  sum of the lr multiplier over the steps taken (peak * cv_area = lr area so far)"""
    out = []
    for r in problem_rows("loss-curve", family):
        base = {k: v for k, v in r.items() if k not in RUN_ONLY}
        area, done = 0.0, 0
        for j, (st, y) in enumerate(zip(r["lc_step"], r["lc_loss"])):
            while done < st:
                area += lr_mult(done, r["steps"], r["warmup"], r["sched"], r["decay"], r["min_lr_frac"])
                done += 1
            d = dict(base)
            d.update(cv_frac=(j + 1) / len(r["lc_step"]), cv_step=st, cv_lr=r["lc_lr"][j], cv_area=round(area, 6),
                     cv_loss=y)
            out.append(d)
    return out


def target(r):
    return r["cv_loss"]


def group(r):
    return r["width"]


def floor(rs, skip=("tag", "machine", "device", "model_family", TARGET)):
    """Noise floor the final-loss way: pooled SD of the target over rows with identical visible settings
    (here: identical settings and the same reading). Returns (floor, settings groups, rows), or None
    when no two rows share their settings."""
    groups = {}
    for r in rs:
        key = tuple(sorted((k, str(v)) for k, v in r.items() if k not in skip))
        groups.setdefault(key, []).append(target(r))
    ss, dof, n_rows, n_groups = 0.0, 0, 0, 0
    for ys in groups.values():
        if len(ys) > 1:
            m = sum(ys) / len(ys)
            ss += sum((y - m) ** 2 for y in ys)
            dof += len(ys) - 1
            n_rows += len(ys)
            n_groups += 1
    return (math.sqrt(ss / dof), n_groups, n_rows) if dof else None


if __name__ == "__main__":
    rs = rows()
    print(f"{len(rs)} rows, floor {floor(rs)}")
