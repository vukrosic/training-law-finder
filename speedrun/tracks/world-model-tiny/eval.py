"""Score a world model: mean squared error of a 12-frame rollout on 512 fixed validation rollouts.

    score = mean over rollouts, the 12 predicted frames, 3 channels and 32 x 32 pixels of (prediction - truth)^2

with pixels in [0, 1]. The model gets the first 4 frames of each rollout and all 15 actions, and returns the
next 12 frames: predict(context, actions) -> (B, 12, 3, 32, 32). It never sees a frame it has to predict.
The validation rollouts come from env.generate(512, VAL_SEED); val_set() checks their sha256, so every machine
scores on the same bytes. The error is summed in float64 on the cpu.
"""
import hashlib

import torch

import env

CONTEXT, HORIZON = 4, 12  # frames the model sees, frames it predicts
VAL_SEED, VAL_N = 20261127, 512  # never train on VAL_SEED
VAL_SHA256 = "c1b5ff8c7e31112d4f24f204fbad72b47f92ae0ca7fb3077df5cc8c1770c77d3"


def val_set():
    frames, actions = env.generate(VAL_N, VAL_SEED, CONTEXT + HORIZON)
    h = hashlib.sha256(frames.tobytes() + actions.tobytes()).hexdigest()
    assert h == VAL_SHA256, f"validation rollouts differ from the reference (sha256 {h})"
    return frames, actions


@torch.no_grad()
def rollout_mse(predict, device, batch=64, data=None):
    """predict(context float (B, 4, 3, 32, 32) in [0, 1], actions long (B, 15)) -> (B, 12, 3, 32, 32)."""
    frames, actions = data if data is not None else val_set()
    total = 0.0
    for i in range(0, len(frames), batch):
        f = torch.from_numpy(frames[i:i + batch]).double() / 255
        ctx = f[:, :CONTEXT].float().to(device)
        act = torch.from_numpy(actions[i:i + batch]).to(device)
        pred = predict(ctx, act)
        assert pred.shape == f[:, CONTEXT:].shape, f"predict returned {tuple(pred.shape)}"
        total += (pred.detach().cpu().double() - f[:, CONTEXT:]).pow(2).sum().item()
    return total / (frames[:, CONTEXT:].size)


def copy_last(ctx, actions):
    """Reference: repeat the last frame the model saw."""
    return ctx[:, -1:].expand(-1, HORIZON, -1, -1, -1)


def black(ctx, actions):
    """Reference: predict an empty frame."""
    return torch.zeros(ctx.shape[0], HORIZON, *ctx.shape[2:], device=ctx.device)


if __name__ == "__main__":
    for name, fn in (("copy last frame", copy_last), ("black frame", black)):
        print(f"{name}: {rollout_mse(fn, 'cpu'):.6f}")
