"""Checks the environment and the scorer.  python3 test_env.py

1. A seed gives the same bytes every time, and the validation rollouts match the sha256 in eval.py.
2. Balls stay inside the frame; the square moves by the action and stops at the walls.
3. The scorer gives the numbers we can work out by hand: 0 for the true frames, the mean of the squared pixels for
   an empty frame, and it never shows the model a frame it has to predict.
"""
import numpy as np
import torch

import env
import eval as ev


def test_same_bytes():
    f1, a1 = env.generate(8, 123)
    f2, a2 = env.generate(8, 123)
    assert (f1 == f2).all() and (a1 == a2).all()
    f3, _ = env.generate(8, 124)
    assert not (f1 == f3).all()
    ev.val_set()  # asserts the sha256


def test_physics():
    rng = np.random.default_rng(0)
    agent, pos, vel, r, active = env._start(rng, 256)
    for _ in range(200):
        env._step(agent, pos, vel, r, active, rng.integers(0, env.N_ACTIONS, 256))
        assert (pos >= r[..., None] - 1e-9).all() and (pos <= env.SIZE - r[..., None] + 1e-9).all()
    a = np.array([[10.0, 10.0]])
    env._step(a, pos[:1].copy(), vel[:1].copy(), r[:1], active[:1] & False, np.array([4]))
    assert np.allclose(a, [[10 + env.AGENT_SPEED, 10]])  # action 4 moves right
    a = np.array([[env.SIZE - env.AGENT_HALF, 10.0]])
    env._step(a, pos[:1].copy(), vel[:1].copy(), r[:1], active[:1] & False, np.array([4]))
    assert np.allclose(a, [[env.SIZE - env.AGENT_HALF, 10]])  # stops at the wall


def test_scorer():
    frames, actions = env.generate(16, 7, ev.CONTEXT + ev.HORIZON)
    data = (frames, actions)
    truth = torch.from_numpy(frames).float() / 255
    seen = []

    def perfect(ctx, act):
        seen.append(ctx.shape[1])
        i = len(seen) - 1
        assert torch.equal(ctx, truth[i * 8:(i + 1) * 8, :ev.CONTEXT])
        return truth[i * 8:(i + 1) * 8, ev.CONTEXT:]

    assert ev.rollout_mse(perfect, "cpu", batch=8, data=data) < 1e-14  # float32 rounding only
    assert seen == [ev.CONTEXT, ev.CONTEXT]
    x = frames[:, ev.CONTEXT:].astype(np.float64) / 255
    assert abs(ev.rollout_mse(ev.black, "cpu", data=data) - (x ** 2).mean()) < 1e-12


if __name__ == "__main__":
    test_same_bytes()
    test_physics()
    test_scorer()
    print("world-model-tiny tests pass")
