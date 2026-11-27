"""Score a world model on 512 fixed validation rollouts: the 12-frame rollout error, on a 0-100 scale.

    mse   = mean over rollouts, the 12 predicted frames, 3 channels and 32 x 32 pixels of (prediction - truth)^2
    score = 100 * mse / mse of repeating the last seen frame, on the same rollouts

with pixels in [0, 1]. Lower is better: repeating the last frame scores 100.0, a perfect model 0.0. The model gets the first 4 frames of each rollout and all 15 actions, and returns the
next 12 frames: predict(context, actions) -> (B, 12, 3, 32, 32). It never sees a frame it has to predict.
The validation rollouts come from env.generate(512, VAL_SEED); val_set() checks their sha256, so every machine
scores on the same bytes. The error is summed in float64 on the cpu. The score is the mean of the per-rollout scores
100 * mse_i / copy_last_mse; its standard error is bootstrapped over the 512 rollouts (common/stats.py).
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

import env

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import stats  # noqa: E402

TRACK = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "track.json")))

CONTEXT, HORIZON = 4, 12  # frames the model sees, frames it predicts
VAL_SEED, VAL_N = 20261127, 512  # never train on VAL_SEED
VAL_SHA256 = "c1b5ff8c7e31112d4f24f204fbad72b47f92ae0ca7fb3077df5cc8c1770c77d3"


def val_set():
    frames, actions = env.generate(VAL_N, VAL_SEED, CONTEXT + HORIZON)
    h = hashlib.sha256(frames.tobytes() + actions.tobytes()).hexdigest()
    assert h == VAL_SHA256, f"validation rollouts differ from the reference (sha256 {h})"
    return frames, actions


@torch.no_grad()
def rollout_errors(predict, device, batch=64, data=None):
    """Per-rollout mse, float64 (512,). predict(context float (B, 4, 3, 32, 32) in [0, 1], actions long (B, 15))
    -> (B, 12, 3, 32, 32)."""
    frames, actions = data if data is not None else val_set()
    out = []
    for i in range(0, len(frames), batch):
        f = torch.from_numpy(frames[i:i + batch]).double() / 255
        ctx = f[:, :CONTEXT].float().to(device)
        act = torch.from_numpy(actions[i:i + batch]).to(device)
        pred = predict(ctx, act)
        assert pred.shape == f[:, CONTEXT:].shape, f"predict returned {tuple(pred.shape)}"
        out.append((pred.detach().cpu().double() - f[:, CONTEXT:]).pow(2).flatten(1).mean(1).numpy())
    return np.concatenate(out)


def rollout_mse(predict, device, batch=64, data=None):
    return float(rollout_errors(predict, device, batch, data).mean())


def copy_last_mse(data=None):
    """MSE of repeating the last seen frame: the 100 of the score. 0.035486 on the validation rollouts."""
    frames, _ = data if data is not None else val_set()
    f = frames.astype(np.float64) / 255
    return float(((f[:, CONTEXT:] - f[:, CONTEXT - 1:CONTEXT]) ** 2).mean())


def score(predict, device, batch=64, data=None):
    """The track's score: val_score (mean per-rollout score), its bootstrap SE, the text "19.83 ± 0.04" at the
    track's decimals, val_mse, and per_item (each rollout's score, 4 significant digits)."""
    data = data if data is not None else val_set()
    err = rollout_errors(predict, device, batch, data)
    item = 100 * err / copy_last_mse(data)
    se = stats.se_of_mean(item)
    d = TRACK.get("decimals", stats.decimals(se))
    return dict(val_score=round(float(item.mean()), d), val_score_se=round(se, d), val_score_text=stats.fmt(item.mean(), se, d),
                val_mse=float(f"{err.mean():.4g}"), per_item=[float(f"{x:.4g}") for x in item])


def copy_last(ctx, actions):
    """Reference: repeat the last frame the model saw."""
    return ctx[:, -1:].expand(-1, HORIZON, -1, -1, -1)


def black(ctx, actions):
    """Reference: predict an empty frame."""
    return torch.zeros(ctx.shape[0], HORIZON, *ctx.shape[2:], device=ctx.device)


if __name__ == "__main__":
    data = val_set()
    for name, fn in (("copy last frame", copy_last), ("black frame", black)):
        r = score(fn, "cpu", data=data)
        print(f"{name}: score {r['val_score_text']}, mse {r['val_mse']}")
