"""Score an agent: mean food eaten per game over 256 fixed evaluation games of Snake.

    score = mean over the 256 games (seeds EVAL_SEED, EVAL_SEED + 1, ...) of the food eaten before the game ends

The agent is policy(obs float (B, 4, 10, 10)) -> actions long (B,). Every game runs to its end (env.py says when a
game ends, so it always does). torch is seeded with 0 first, so a policy that samples its actions also gives the
same score again on the same machine. Evaluation steps are not part of the environment-step budget.
"""
import numpy as np
import torch

from env import MOVES, N_ACTIONS, SIZE, Snake

EVAL_SEED, EVAL_N = 10**9, 256  # training seeds must be below EVAL_SEED


@torch.no_grad()
def mean_return(policy, device="cpu", n=EVAL_N):
    torch.manual_seed(0)
    env = Snake(n)
    obs = env.reset(EVAL_SEED + np.arange(n))
    while not env.done.all():
        a = policy(torch.from_numpy(obs).to(device))
        obs, _, _ = env.step(np.asarray(a.cpu(), dtype=np.int64))
    return float(env.eaten.mean())


def random_policy(obs):
    """Reference: a uniformly random action."""
    return torch.randint(0, N_ACTIONS, (obs.shape[0],))


def greedy(obs):
    """Reference written by hand: step towards the food, never into a pillar, the body or the edge if it can."""
    o = obs.cpu().numpy()
    out = np.zeros(len(o), dtype=np.int64)
    for i, x in enumerate(o):
        hr, hc = np.argwhere(x[2] > 0)[0] if x[2].any() else (0, 0)
        fr, fc = np.argwhere(x[1] > 0)[0] if x[1].any() else (hr, hc)
        tail = x[3] == x[3][x[3] > 0].min() if (x[3] > 0).any() else np.zeros_like(x[3], dtype=bool)
        best, best_d = 0, None
        for a, (dr, dc) in enumerate(MOVES):
            r, c = hr + dr, hc + dc
            blocked = not (0 <= r < SIZE and 0 <= c < SIZE) or x[0, r, c] > 0 or (x[3, r, c] > 0 and not tail[r, c])
            d = abs(r - fr) + abs(c - fc) + (1000 if blocked else 0)
            if best_d is None or d < best_d:
                best, best_d = a, d
        out[i] = best
    return torch.from_numpy(out)


if __name__ == "__main__":
    for name, fn in (("random", random_policy), ("greedy towards food", greedy)):
        print(f"{name}: {mean_return(fn):.2f}")
