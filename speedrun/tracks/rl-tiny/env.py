"""The rl-tiny environment: Snake on a 10 x 10 grid with random pillars, many games at once in numpy.

    env = Snake(n)                       # n games side by side
    obs = env.reset(seeds)               # one seed per game: it fixes the pillars and every food position
    obs, reward, done = env.step(actions)
    obs = env.reset_where(done, seeds)   # start new games where done is True

Rules. The snake starts with length 3 at the centre, heading up. Actions: 0 up, 1 down, 2 left, 3 right; turning
straight back is ignored (the snake keeps going). Eating food gives reward 1 and makes the snake one cell longer.
The game ends when the head hits the edge, a pillar or the body, after 100 steps without food, or when the board
is full. Each seed places 0 to 6 pillars on cells whose row and column are both odd (1, 3, 5, 7), which never
cuts the board into parts. Food appears on an empty cell chosen by a hash of the seed and the food's number, so
a seed and a list of actions always give the same game, on every machine.

Observation: float32 (n, 4, 10, 10): pillars, food, head, body. The body channel holds age / length: 1 at the
head, falling towards the tail, 0 elsewhere, so the observation shows which way the snake moves.
"""
import numpy as np

SIZE = 10
N_ACTIONS = 4  # 0 up, 1 down, 2 left, 3 right
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # (row, column) step per action
OPPOSITE = np.array([1, 0, 3, 2])
HUNGER = 100  # steps without food before the game ends
MAX_PILLARS = 6
PILLAR_CELLS = np.array([(r, c) for r in (1, 3, 5, 7) for c in (1, 3, 5, 7)])
START = [(4, 4), (5, 4), (6, 4)]  # head first; column 4 is even, so no pillar is there


def mix(x):
    """splitmix64: a fixed hash of uint64 numbers, the same everywhere."""
    x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Snake:
    def __init__(self, n):
        self.n = n
        self.wall = np.zeros((n, SIZE, SIZE), dtype=bool)
        self.age = np.zeros((n, SIZE, SIZE), dtype=np.int32)  # steps until each body cell is free; 0 = empty
        self.food = np.zeros((n, 2), dtype=np.int64)
        self.head = np.zeros((n, 2), dtype=np.int64)
        self.dir = np.zeros(n, dtype=np.int64)
        self.length = np.zeros(n, dtype=np.int64)
        self.hunger = np.zeros(n, dtype=np.int64)
        self.eaten = np.zeros(n, dtype=np.int64)  # food eaten this game = the game's return
        self.seed = np.zeros(n, dtype=np.uint64)
        self.done = np.ones(n, dtype=bool)
        self.rows = np.arange(n)

    def reset(self, seeds):
        return self.reset_where(np.ones(self.n, dtype=bool), seeds)

    def reset_where(self, mask, seeds):
        """Start a new game in every slot where mask is True; seeds has one seed per True slot."""
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return self.obs()
        s = np.asarray(seeds, dtype=np.uint64).reshape(-1)
        assert len(s) == len(idx), "one seed per game being reset"
        self.seed[idx] = s
        count = (mix(s * np.uint64(1000003)) % np.uint64(MAX_PILLARS + 1)).astype(np.int64)
        keys = mix(s[:, None] * np.uint64(1000003) + np.arange(1, 17, dtype=np.uint64)[None])
        rank = np.argsort(np.argsort(keys, axis=1, kind="stable"), axis=1, kind="stable")
        self.wall[idx] = False
        for k, (r, c) in enumerate(PILLAR_CELLS):
            self.wall[idx, r, c] = rank[:, k] < count
        self.age[idx] = 0
        for k, (r, c) in enumerate(START):
            self.age[idx, r, c] = len(START) - k
        self.head[idx] = START[0]
        self.dir[idx] = 0
        self.length[idx] = len(START)
        self.hunger[idx] = 0
        self.eaten[idx] = 0
        self.done[idx] = False
        self._place_food(idx)
        return self.obs()

    def _place_food(self, idx):
        """Put food on the k-th empty cell, k = hash(seed, food number) mod number of empty cells."""
        if len(idx) == 0:
            return np.zeros(0, dtype=bool)
        empty = (~self.wall[idx] & (self.age[idx] == 0)).reshape(len(idx), -1)
        n_empty = empty.sum(1)
        full = n_empty == 0
        h = mix(self.seed[idx] * np.uint64(7919) + self.eaten[idx].astype(np.uint64) + np.uint64(1 << 40))
        k = (h % np.maximum(n_empty, 1).astype(np.uint64)).astype(np.int64)
        cell = np.argmax(np.cumsum(empty, 1) > k[:, None], 1)
        self.food[idx] = np.stack([cell // SIZE, cell % SIZE], 1)
        return full

    def obs(self):
        o = np.zeros((self.n, 4, SIZE, SIZE), dtype=np.float32)
        o[:, 0] = self.wall
        o[self.rows, 1, self.food[:, 0], self.food[:, 1]] = 1
        o[self.rows, 2, self.head[:, 0], self.head[:, 1]] = 1
        o[:, 3] = self.age / self.length[:, None, None]
        o[self.done] = 0
        return o

    def step(self, actions):
        """Move every running game one step. Games already done stay done and get reward 0."""
        a = np.asarray(actions, dtype=np.int64)
        live = ~self.done
        a = np.where(a == OPPOSITE[self.dir], self.dir, a)
        new = self.head + MOVES[a]
        out = ((new < 0) | (new >= SIZE)).any(1)
        nr, nc = np.clip(new[:, 0], 0, SIZE - 1), np.clip(new[:, 1], 0, SIZE - 1)
        eat = live & ~out & (nr == self.food[:, 0]) & (nc == self.food[:, 1])
        shrink = live & ~eat
        self.age[shrink] = np.maximum(self.age[shrink] - 1, 0)  # the tail moves on, unless the snake eats
        dead = live & (out | self.wall[self.rows, nr, nc] | (self.age[self.rows, nr, nc] > 0))
        move = live & ~dead
        self.length += eat
        self.age[self.rows[move], nr[move], nc[move]] = self.length[move]
        self.head[move] = np.stack([nr, nc], 1)[move]
        self.dir[move] = a[move]
        self.hunger = np.where(eat, 0, self.hunger + 1)
        self.eaten += eat
        full = np.zeros(self.n, dtype=bool)
        full[np.flatnonzero(eat)] = self._place_food(np.flatnonzero(eat))
        self.done |= dead | (live & (self.hunger >= HUNGER)) | full
        return self.obs(), eat.astype(np.float32), self.done.copy()
