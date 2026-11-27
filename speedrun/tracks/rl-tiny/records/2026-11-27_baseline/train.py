"""Speedrun baseline: train a Snake agent with PPO until the environment-step budget is used, then report the mean
food eaten on the 256 fixed evaluation games.

    python3 train.py                    # from speedrun/tracks/rl-tiny/
    python3 train.py --budget 2e5       # shorter run, not a record

Agent: a small conv net (two 3 x 3 convs, one hidden layer) with a policy head and a value head. PPO with clipping,
GAE, Adam, learning rate falling linearly to 0. 64 games run side by side; each game that ends starts again
with the next training seed (0, 1, 2, ..., always below eval.EVAL_SEED). The evaluation plays the most likely
action. Learning runs on the GPU (cuda or mps); on the cpu the conv backward pass is about 20x slower.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRACK = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(TRACK, "eval.py")):  # a copy in records/<run>/ finds the track folder
    TRACK = os.path.dirname(TRACK)
sys.path[:0] = [TRACK, os.path.join(os.path.dirname(os.path.dirname(TRACK)), "common")]  # env, eval, flops
import env  # noqa: E402
import eval as ev  # noqa: E402
import flops  # noqa: E402

BUDGET = 2e6  # the speedrun budget, in environment steps (README.md)
N_ENVS, N_STEPS = 64, 128  # 8192 steps per update
EPOCHS, MINIBATCHES = 4, 4
LR, GAMMA, LAMBDA, CLIP, ENT, VF, MAX_GRAD = 5e-4, 0.99, 0.95, 0.2, 0.01, 0.5, 0.5
MEM_GB = 1.2  # cap on GPU memory for this process


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * env.SIZE * env.SIZE, 128)
        self.pi = nn.Linear(128, env.N_ACTIONS)
        self.v = nn.Linear(128, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs):
        h = F.relu(self.c2(F.relu(self.c1(obs))))
        h = F.relu(self.fc(h.flatten(1)))
        return self.pi(h), self.v(h).squeeze(-1)


def build():
    model = Agent()
    return model, [torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)]


def update_step(model, opts, obs, act, old_logp, adv, ret):
    logits, v = model(obs)
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(1, act[:, None]).squeeze(1)
    ratio = (logp - old_logp).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
    entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
    loss = pg + VF * F.mse_loss(v, ret) - ENT * entropy
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
    for o in opts:
        o.step()
    model.zero_grad(set_to_none=True)
    return loss.detach()


def make_inputs():
    n = N_ENVS * N_STEPS // MINIBATCHES
    return (torch.zeros(n, 4, env.SIZE, env.SIZE), torch.zeros(n, dtype=torch.long), torch.zeros(n),
            torch.zeros(n), torch.zeros(n))


def act_flops():
    with torch.device("meta"):
        model, _ = build()
        obs = torch.zeros(N_ENVS, 4, env.SIZE, env.SIZE)
    return flops.count(lambda: model(obs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(min(1.0, MEM_GB * 1e9 / torch.mps.recommended_max_memory()))
    torch.manual_seed(0)
    model, opts = build()  # built on cpu so the starting weights are the same on every machine
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_iters = int(args.budget // (N_ENVS * N_STEPS))
    steps = n_iters * N_ENVS * N_STEPS
    upd, actf = flops.count_step_inputs(build, update_step, make_inputs), act_flops()
    iter_flops = EPOCHS * MINIBATCHES * upd + (N_STEPS + 1) * actf
    print(f"device {device}, params {n_params:,}, budget {args.budget:.3g} env steps, updates {n_iters}, steps {steps:,}")
    print(f"FLOPs per update {iter_flops:.4g} (FlopCounterMode: acting {actf:.4g} per batch of {N_ENVS} games, "
          f"learning {upd:.4g} per minibatch)")

    games = env.Snake(N_ENVS)
    next_seed = N_ENVS
    obs = torch.from_numpy(games.reset(np.arange(N_ENVS))).to(device)
    finished = []
    t0 = time.time()
    for it in range(n_iters):
        for g in opts[0].param_groups:
            g["lr"] = LR * (1 - it / n_iters)
        b_obs = torch.zeros(N_STEPS, N_ENVS, 4, env.SIZE, env.SIZE, device=device)
        b_act = torch.zeros(N_STEPS, N_ENVS, dtype=torch.long, device=device)
        b_logp, b_val, b_rew, b_done = (torch.zeros(N_STEPS, N_ENVS, device=device) for _ in range(4))
        for t in range(N_STEPS):
            with torch.no_grad():
                logits, v = model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, a, dist.log_prob(a), v
            o, r, d = games.step(a.cpu().numpy())
            finished += games.eaten[d].tolist()
            b_rew[t], b_done[t] = torch.from_numpy(r).to(device), torch.from_numpy(d.astype(np.float32)).to(device)
            n_new = int(d.sum())
            assert next_seed + n_new <= ev.EVAL_SEED
            o = games.reset_where(d, np.arange(next_seed, next_seed + n_new))
            next_seed += n_new
            obs = torch.from_numpy(o).to(device)
        with torch.no_grad():
            _, last_v = model(obs)
        adv = torch.zeros(N_STEPS, N_ENVS, device=device)
        gae = torch.zeros(N_ENVS, device=device)
        for t in reversed(range(N_STEPS)):
            nv = last_v if t == N_STEPS - 1 else b_val[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_rew[t] + GAMMA * nv * nonterm - b_val[t]
            gae = delta + GAMMA * LAMBDA * nonterm * gae
            adv[t] = gae
        ret = adv + b_val
        n = N_ENVS * N_STEPS
        f_obs, f_act, f_logp, f_adv, f_ret = (x.reshape(n, *x.shape[2:]) for x in (b_obs, b_act, b_logp, adv, ret))
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=device)
            for mb in perm.chunk(MINIBATCHES):
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                update_step(model, opts, f_obs[mb], f_act[mb], f_logp[mb], a_mb, f_ret[mb])
        if it % 25 == 0 or it == n_iters - 1:
            recent = finished[-200:]
            print(f"update {it}/{n_iters} steps {(it + 1) * n:,} games {len(finished)} "
                  f"food per game (last 200) {np.mean(recent) if recent else 0:.2f} time {time.time() - t0:.0f}s",
                  flush=True)
    if device == "mps":
        torch.mps.synchronize()
    train_time = time.time() - t0

    model.eval()
    t1 = time.time()
    score = ev.score(lambda o: model(o)[0].argmax(-1), device)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_item.json"), "w") as f:
        json.dump(score.pop("per_item"), f, separators=(",", ":"))
    result = dict(**score, version=ev.TRACK["version"], env_steps=steps, budget=args.budget,
                  flops=float(f"{n_iters * iter_flops:.6g}"), updates=n_iters, train_games=len(finished),
                  params=n_params, device=device, train_seconds=round(train_time, 1),
                  eval_seconds=round(time.time() - t1, 1))
    print("result " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
