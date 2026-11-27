"""The world-model-tiny environment: balls bouncing in a box, plus one square the actions move.

    frames, actions = generate(n, seed)       # frames uint8 (n, T, 3, 32, 32), actions int64 (n, T - 1)

action[:, k] moves the square between frame k and frame k + 1. Each rollout has 1 to 3 balls with a random
size, speed and direction. Balls bounce off the walls, off each other (elastic, equal mass) and off the square.
The square moves 1.5 pixels per step in the action's direction and stops at the walls. The actions come from a
random policy that keeps its action and changes it with probability 0.3 at each step.

Everything is plain float64 arithmetic (+, *, /, sqrt, clip, compare), which IEEE 754 makes exact and the same on
every machine, so a seed gives the same bytes everywhere; eval.py checks that with a sha256.
Generating data is not part of the FLOP budget. Nothing is downloaded and nothing is stored.
"""
import numpy as np

SIZE = 32  # frames are SIZE x SIZE pixels, 3 colour channels
T = 16  # frames per rollout: 4 the model sees, 12 it predicts (eval.py)
N_ACTIONS = 5  # 0 stay, 1 up, 2 down, 3 left, 4 right
MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.float64)  # (dx, dy) per action
AGENT_HALF, AGENT_SPEED = 2.5, 1.5  # the square is 5 x 5 pixels
MAX_BALLS = 3
RADIUS = (2.0, 3.5)
SPEED = (0.7, 1.8)
P_CHANGE = 0.3  # chance per step that the random policy picks a new action
COLORS = np.array([[1.0, 0.3, 0.1],  # square: orange
                   [0.1, 0.9, 0.3],  # ball 1: green
                   [0.2, 0.5, 1.0],  # ball 2: blue
                   [0.9, 0.9, 0.2]])  # ball 3: yellow


def _start(rng, n):
    agent = rng.uniform(AGENT_HALF, SIZE - AGENT_HALF, (n, 2))
    n_balls = rng.integers(1, MAX_BALLS + 1, n)
    active = np.arange(MAX_BALLS)[None] < n_balls[:, None]  # (n, 3)
    r = rng.uniform(*RADIUS, (n, MAX_BALLS))
    pos = rng.uniform(0, 1, (n, MAX_BALLS, 2)) * (SIZE - 2 * r[..., None]) + r[..., None]
    angle = rng.uniform(0, 2 * np.pi, (n, MAX_BALLS))
    speed = rng.uniform(*SPEED, (n, MAX_BALLS))
    vel = np.stack([np.cos(angle), np.sin(angle)], -1) * speed[..., None]
    return agent, pos, vel, r, active


def _actions(rng, n, t):
    a = np.empty((n, t - 1), dtype=np.int64)
    a[:, 0] = rng.integers(0, N_ACTIONS, n)
    change, new = rng.uniform(size=(n, t - 1)) < P_CHANGE, rng.integers(0, N_ACTIONS, (n, t - 1))
    for k in range(1, t - 1):
        a[:, k] = np.where(change[:, k], new[:, k], a[:, k - 1])
    return a


def _walls(pos, vel, r):
    for d in range(2):
        lo, hi = r, SIZE - r
        x, v = pos[..., d], vel[..., d]
        below, above = x < lo, x > hi
        x[:] = np.where(below, 2 * lo - x, np.where(above, 2 * hi - x, x))
        v[:] = np.where(below, np.abs(v), np.where(above, -np.abs(v), v))
        x[:] = np.clip(x, lo, hi)  # a very fast ball could still be outside after one reflection


def _ball_pairs(pos, vel, r, active):
    for i in range(MAX_BALLS):
        for j in range(i + 1, MAX_BALLS):
            d = pos[:, j] - pos[:, i]
            dist2 = (d * d).sum(-1)
            hit = active[:, i] & active[:, j] & (dist2 < (r[:, i] + r[:, j]) ** 2) & (dist2 > 0)
            dist = np.sqrt(np.where(hit, dist2, 1.0))
            nrm = d / dist[:, None]
            approach = ((vel[:, i] - vel[:, j]) * nrm).sum(-1)
            dv = np.where(hit & (approach > 0), approach, 0.0)[:, None] * nrm
            vel[:, i] -= dv
            vel[:, j] += dv
            push = np.where(hit, (r[:, i] + r[:, j] - dist) / 2, 0.0)[:, None] * nrm
            pos[:, i] -= push
            pos[:, j] += push


def _ball_agent(agent, pos, vel, r):
    h = AGENT_HALF
    for i in range(MAX_BALLS):
        p, v = pos[:, i], vel[:, i]
        closest = np.clip(p, agent - h, agent + h)
        d = p - closest
        dist2 = (d * d).sum(-1)
        inside = dist2 == 0  # centre inside the square: push out along the axis it is furthest along
        off = p - agent
        axis_x = np.abs(off[:, 0]) >= np.abs(off[:, 1])
        sx = np.where(off[:, 0] >= 0, 1.0, -1.0)
        sy = np.where(off[:, 1] >= 0, 1.0, -1.0)
        n_in = np.stack([np.where(axis_x, sx, 0.0), np.where(axis_x, 0.0, sy)], -1)
        dist = np.sqrt(np.where(inside, 1.0, dist2))
        nrm = np.where(inside[:, None], n_in, d / dist[:, None])
        hit = inside | (dist2 < r[:, i] ** 2)
        edge = np.where(inside[:, None], agent + n_in * h, closest)
        p[:] = np.where(hit[:, None], edge + nrm * r[:, i, None], p)
        vn = (v * nrm).sum(-1)
        v[:] = np.where((hit & (vn < 0))[:, None], v - 2 * vn[:, None] * nrm, v)


def _step(agent, pos, vel, r, active, a):
    agent += MOVES[a] * AGENT_SPEED
    np.clip(agent, AGENT_HALF, SIZE - AGENT_HALF, out=agent)
    pos += vel
    _walls(pos, vel, r)
    _ball_pairs(pos, vel, r, active)
    _ball_agent(agent, pos, vel, r)
    _walls(pos, vel, r)


def _render(agent, pos, r, active):
    c = np.arange(SIZE, dtype=np.float64) + 0.5  # pixel centres
    h = AGENT_HALF
    ax = np.clip(h + 0.5 - np.abs(c[None] - agent[:, :1]), 0, 1)  # (n, W): coverage of each pixel column
    ay = np.clip(h + 0.5 - np.abs(c[None] - agent[:, 1:]), 0, 1)  # (n, H)
    cover = [ay[:, :, None] * ax[:, None, :]]
    for i in range(MAX_BALLS):
        dx = c[None, None, :] - pos[:, i, 0, None, None]
        dy = c[None, :, None] - pos[:, i, 1, None, None]
        cov = np.clip(r[:, i, None, None] + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1)
        cover.append(np.where(active[:, i, None, None], cov, 0.0))
    img = np.zeros((len(agent), 3, SIZE, SIZE))
    for k, cov in enumerate(cover):
        img = np.maximum(img, COLORS[k][None, :, None, None] * cov[:, None])
    return np.round(img * 255).astype(np.uint8)


def generate(n, seed, t=T):
    """n rollouts of t frames from one seed: frames uint8 (n, t, 3, SIZE, SIZE), actions int64 (n, t - 1)."""
    rng = np.random.default_rng(seed)
    agent, pos, vel, r, active = _start(rng, n)
    actions = _actions(rng, n, t)
    _ball_agent(agent, pos, vel, r)  # no ball starts inside the square
    _walls(pos, vel, r)
    frames = np.empty((n, t, 3, SIZE, SIZE), dtype=np.uint8)
    frames[:, 0] = _render(agent, pos, r, active)
    for k in range(t - 1):
        _step(agent, pos, vel, r, active, actions[:, k])
        frames[:, k + 1] = _render(agent, pos, r, active)
    return frames, actions
