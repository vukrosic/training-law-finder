"""Checks the environment and the scorer.  python3 test_env.py

1. A seed and a list of actions give the same game every time.
2. Pillars never cut the board into parts, and never sit on the snake's start.
3. Eating makes the snake longer and gives reward 1; hitting the edge or the body ends the game; turning straight
   back is ignored; 100 steps without food end the game.
4. The scorer gives the same number twice for a policy that samples, and a hand-written policy beats random.
"""
import numpy as np
import torch

import env
import eval as ev


def play(seed, actions):
    g = env.Snake(1)
    g.reset([seed])
    out = []
    for a in actions:
        o, r, d = g.step([a])
        out.append((o.copy(), r[0], d[0]))
    return out


def test_same_game():
    acts = np.random.default_rng(1).integers(0, 4, 50)
    a, b = play(42, acts), play(42, acts)
    assert all((x[0] == y[0]).all() and x[1:] == y[1:] for x, y in zip(a, b))


def test_board_connected():
    g = env.Snake(2000)
    g.reset(np.arange(2000))
    counts = g.wall.sum((1, 2))
    assert counts.min() == 0 and counts.max() == env.MAX_PILLARS
    for w in g.wall[:300]:
        free = {(r, c) for r in range(env.SIZE) for c in range(env.SIZE) if not w[r, c]}
        stack, seen = [next(iter(free))], set()
        while stack:
            r, c = stack.pop()
            if (r, c) in seen or (r, c) not in free:
                continue
            seen.add((r, c))
            stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        assert seen == free
        assert not any(w[r, c] for r, c in env.START)


def test_rules():
    g = env.Snake(1)
    g.reset([3])
    g.food[0] = (3, 4)  # put food right above the head
    _, r, d = g.step([0])
    assert r[0] == 1 and not d[0] and g.length[0] == 4 and g.eaten[0] == 1
    _, r, d = g.step([1])  # straight back: ignored, keeps going up
    assert tuple(g.head[0]) == (2, 4) and not d[0]
    g.reset([3])
    for _ in range(4):
        _, r, d = g.step([0])
    assert tuple(g.head[0]) == (0, 4) and not d[0]
    _, r, d = g.step([0])  # into the edge
    assert d[0] and r[0] == 0
    g.reset([3])
    g.food[0] = (0, 0)
    g.wall[0] = False
    acts = [3, 1, 2, 0] * 30  # a small square loop, never reaching the food
    for k, a in enumerate(acts):
        _, r, d = g.step([a])
        if d[0]:
            break
    assert k + 1 == env.HUNGER and g.eaten[0] == 0


def test_scorer():
    s1 = ev.mean_return(ev.random_policy, n=64)
    s2 = ev.mean_return(ev.random_policy, n=64)
    assert s1 == s2
    assert ev.mean_return(ev.greedy, n=64) > 5 * max(s1, 0.1)


if __name__ == "__main__":
    test_same_game()
    test_board_connected()
    test_rules()
    test_scorer()
    print("rl-tiny tests pass")
