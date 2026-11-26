"""Batched branch trainer (2026-11-25, facts-scale-theory, Paper CEO order): N small dense models of one shape in ONE process,
stacked weights (N, ...) and batched matmuls, all branching from one saved phase-1 checkpoint (train.py gckpt). Each model
gets its own replay schedule and its own phase-2 learning-rate schedule.

    python3 -I env/batched.py --jobs sweeps/X.jsonl --out results_mac.jsonl [--device cuda] [--chunk 64] [--threads 4]

--jobs: the usual job file, lines {"tag", "args"} as for run_jobs.py / train.py --sweep. Jobs are grouped by trunk (every
config field except the branch fields below); each group runs as one batch of up to --chunk models. Tags already in --out
are skipped. Rows have the train.py row format (facts_eval with per_fact_bits, acc_by_k, gstep bits, val_loss, ...) plus
"batched": {n, chunk_seconds, group}.

Branch fields that may differ inside a group:
  rpfile                 replay schedule {"steps": {step: [[person, attr], ...]}} (facts.py replay, no rng used)
  lr, mlr, sched, min_lr_frac, warmup, decay   phase-2 lr = train.py lr_at(step) with these values (steps >= w1 only)
  mwd, gaplr             Muon weight decay, gap lr multiplier (as train.py)
  gstep                  1 = learned-fact bits after every phase-2 step (batched scoring)
  sm2file                online SM-2 review rule, as train.py _sm2 (needs gstep=1; added 2026-11-25, replay-ceiling)
  job field "lrmul"      optional list, one multiplier per phase-2 step (len T - w1), replacing lr_at(step) / lr; any shape
                         (re-warmup, WSD, constant, steps) without a new trunk
Phase 1 (steps 0..w1-1) is whatever the checkpoint holds: if gckpt does not exist yet, the group's first job runs once
through train.main (which writes the checkpoint and that job's row), and the rest branch from it.

Supported: dense SwiGLU models (router dense), opt muon or base, no MoE / row rules / splice / probes. wlog is ignored.
Matches single train.py runs that branch from the same gckpt (check: batched_check.py)."""
import argparse, copy, json, math, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "benchmark"))
import train as T   # noqa: E402
from model import GPT   # noqa: E402
from opt import ns5, moon_scale, mats_params   # noqa: E402
import facts as factsmod   # noqa: E402
from score import evaluate, token_bytes   # noqa: E402

BRANCH = ("rpfile", "lr", "mlr", "sched", "min_lr_frac", "warmup", "decay", "rwp", "rww", "mwd", "gaplr", "gstep", "wlog", "wsig", "sm2file")
VK = ("ffkind", "ffmt", "gaplr", "fmarg", "gsplice", "gfreeze", "gckpt", "fdrift", "flog", "gstep", "gmom", "wlog", "wsig", "zcheap",
      "zfdil", "zrare", "rsp", "rsn", "rsm", "rsx", "rpfile", "rpadd", "pfscreen", "sm2file")   # train.py trunk-key exclusions (same list)
OFF = ("gsplice", "fmarg", "fdrift", "flog", "gfreeze", "gmom", "zcheap", "zfree", "zrare", "pfscreen", "rsp", "fevery", "ftrace",
       "fchoice", "seen_c", "skip_until", "skip_from", "type_balance", "amp", "ce_chunk")


def parse(args):
    """train.py argument parsing and config build, without training"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max_seconds", type=float, default=0); ap.add_argument("--eval_seqs", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default="fineweb"); ap.add_argument("--split", default="val")
    ap.add_argument("--size", default="S"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args(args)
    c = dict(T.BASE, **T.PRESETS[a.method]); sz = dict(T.SIZES[a.size]); wm = sz.pop("width_mult", 1.0); c.update(sz)
    for k in ("dense_hidden", "expert_hidden", "shared_hidden"):
        c[k] = int(round(c[k] * wm))
    for kv in a.set:
        k, v = kv.split("=", 1); assert k in c, f"unknown field {k}"; c[k] = T.parse_value(v)
    return a, c


def trunk_key(a, c):
    return json.dumps(dict({k: v for k, v in c.items() if k not in BRANCH + VK}, _steps=a.steps, _data=a.data, _split=a.split,
                           _size=a.size, _method=a.method, _eval=a.eval_seqs), sort_keys=True)


def lr_fn(c, steps):
    """train.py lr_at for config c (peak c['lr'])"""
    def f(s):
        if s < c["warmup"]:
            return c["lr"] * (s + 1) / c["warmup"]
        if c["sched"] == "wsd":
            d0 = steps - int(round(c["decay"] * steps))
            return c["lr"] if s < d0 else c["lr"] * (1 - (1 - c["min_lr_frac"]) * (s - d0 + 1) / (steps - d0))
        if c["sched"] == "const":
            return c["lr"]
        if c["sched"] == "cosrw":
            P = int(round(c["rwp"] * steps)); mn = c["lr"] * c["min_lr_frac"]
            if s < P:
                p = (s - c["warmup"]) / max(1, P - c["warmup"])
                return mn + (c["lr"] - mn) * 0.5 * (1 + math.cos(math.pi * p))
            if s < P + c["rww"]:
                return mn + (c["lr"] - mn) * (s - P + 1) / c["rww"]
            p = (s - P - c["rww"]) / max(1, steps - P - c["rww"])
            return mn + (c["lr"] - mn) * 0.5 * (1 + math.cos(math.pi * p))
        assert c["sched"] == "cos", c["sched"]
        p = (s - c["warmup"]) / max(1, steps - c["warmup"])
        return c["lr"] * (c["min_lr_frac"] + (1 - c["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))
    return f


# ------------------------------------------------------------------ batched dense GPT (same math as model.py)
def rms(x, w, eps=1e-6):   # x (N, ..., D), w (N, D)
    sh = (w.shape[0],) + (1,) * (x.dim() - 2) + (w.shape[1],)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.view(sh)


def lin(x, W):   # x (N, B, T, in), W (N, out, in) -> (N, B, T, out); nn.Linear without bias
    N, B, Tn, I = x.shape
    return torch.bmm(x.reshape(N, B * Tn, I), W.transpose(1, 2)).view(N, B, Tn, W.shape[1])


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


def forward(P, cfg, idx, cos, sin):
    """P: name -> (N, ...) stacked params; idx (N, B, T) -> hidden after the final norm (N, B, T, D)"""
    N, B, Tn = idx.shape
    D, H = cfg.d_model, cfg.n_heads; dh = D // H
    x = torch.gather(P["emb.weight"], 1, idx.reshape(N, -1, 1).expand(-1, -1, D)).view(N, B, Tn, D)
    c, s = cos[:Tn], sin[:Tn]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        h = rms(x, P[p + "n1.w"])
        q, k, v = lin(h, P[p + "attn.qkv.weight"]).view(N * B, Tn, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k = rope(q, c, s), rope(k, c, s)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + lin(y.transpose(1, 2).reshape(N, B, Tn, D), P[p + "attn.o.weight"])
        h = rms(x, P[p + "n2.w"])
        x = x + lin(F.silu(lin(h, P[p + "ffn.w1.weight"])) * lin(h, P[p + "ffn.w3.weight"]), P[p + "ffn.w2.weight"])
    return rms(x, P["norm.w"])


def token_loss(P, cfg, idx, tgt, cos, sin):
    """per-token CE (N, B, T) in nats, fp32 logits as model.py"""
    h = forward(P, cfg, idx, cos, sin)
    N, B, Tn, D = h.shape
    lg = torch.bmm(h.reshape(N, B * Tn, D), P["head.weight"].transpose(1, 2)).float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(N, B, Tn)


@torch.no_grad()
def item_bits(P, cfg, x, y, cos, sin):
    """value bits per item for every model: x, y (I, L) with y = -1 outside the value -> (N, I)"""
    N = P["emb.weight"].shape[0]
    out = []
    for i0 in range(0, x.shape[0], 128):
        xx = x[i0:i0 + 128].unsqueeze(0).expand(N, -1, -1).contiguous(); yy = y[i0:i0 + 128]
        h = forward(P, cfg, xx, cos, sin)
        lp = torch.log_softmax(torch.bmm(h.reshape(N, -1, h.shape[-1]), P["head.weight"].transpose(1, 2)).float(), -1)
        lp = lp.view(N, xx.shape[1], xx.shape[2], -1)
        m = (yy >= 0).unsqueeze(0); g = lp.gather(-1, yy.clamp(min=0).unsqueeze(0).expand(N, -1, -1).unsqueeze(-1)).squeeze(-1)
        out.append((-(g * m).sum(-1) / math.log(2)))
    return torch.cat(out, 1)


# ------------------------------------------------------------------ one group
def run_group(jobs, out, dev, threads):
    a0, c0 = parse(jobs[0]["args"])
    if c0["attn"] != "softmax" or c0["router"] != "dense":   # 2026-11-26 spacing-learn r5: forward() here is dense softmax only;
        for j in jobs:                                         # other architectures run one model at a time through train.py
            T.main([*j["args"], "--tag", j["tag"], "--threads", str(threads), "--out", out, "--device", str(dev)])
        return
    assert c0["router"] == "dense" and c0["erule"] == "none" and c0["opt"] in ("muon", "base") and c0["facts"] and c0["gckpt"], \
        "batched.py: dense facts runs with opt muon/base and a gckpt"
    for k in OFF:
        assert not c0.get(k), f"batched.py does not support {k}"
    assert c0["ffw"] == 1.0 and c0["branches"] in ("", None, 0) and c0["expert_lr"] == 1.0
    ck = c0["gckpt"]
    if not os.path.exists(ck):   # phase 1 once, through train.py: writes the checkpoint and the first job's row
        print(f"[batched] no {ck}: training the trunk with {jobs[0]['tag']}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(ck)), exist_ok=True)
        T.main([*jobs[0]["args"], "--tag", jobs[0]["tag"], "--threads", str(threads), "--out", out, "--device", str(dev)])
        jobs = jobs[1:]
        if not jobs:
            return
    sv = torch.load(ck, weights_only=False)
    cfg = types.SimpleNamespace(**c0); cfg.l1_coeff_state = [cfg.l1_init]
    torch.manual_seed(cfg.seed)
    tmpl = GPT(cfg).to(dev); tmpl.load_state_dict(sv["model"])
    tk_ = dict({k: v for k, v in c0.items() if k not in VK}, _steps=a0.steps, _data=a0.data, _split=a0.split, _dev=str(dev))
    diff = {k for k in set(tk_) | set(sv["tkey"]) if tk_.get(k) != sv["tkey"].get(k) and k not in BRANCH and k != "_dev"
            and not (k not in sv["tkey"] and tk_.get(k) == T.BASE.get(k))}   # fields added to train.py after the checkpoint, at default
    assert not diff, f"gckpt trunk differs from this group in {sorted(diff)}"
    N = len(jobs); steps = a0.steps
    cs = [parse(j["args"])[1] for j in jobs]

    DATA = T.DATASETS[a0.data]
    TOK0 = os.path.join(HERE, "..", "data", "tokenizer", "bpe8192.json")
    TOKF = os.path.join(DATA, "tokenizer.json") if os.path.exists(os.path.join(DATA, "tokenizer.json")) else TOK0
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOKF)
    bins = [0] if getattr(cfg, 'fzero', 0) else [cfg.fkmin * 2 ** i for i in range(int(math.log2(cfg.fkmax // cfg.fkmin)) + 1)]
    fw = factsmod.FactWorld(tk, cfg.facts, bins, steps, seed=cfg.fseed, win=(cfg.fwlo, cfg.fwhi), n_filler=cfg.ffill,
                            filler_k=cfg.ffillk, fwin=None if cfg.ffwlo < 0 else (cfg.ffwlo, cfg.ffwhi), frand=int(cfg.ffrand),
                            frandp=float(cfg.ffrandp), ffmt=int(cfg.ffmt), ffkind=int(cfg.ffkind),
                            web=train if int(cfg.ffkind) == 3 else None,
                            pre=None if cfg.fprelo < 0 else (cfg.fprelo, cfg.fprehi),
                            fpre=None if cfg.ffprelo < 0 else (cfg.ffprelo, cfg.ffprehi, int(cfg.ffprek)))
    fw.rng.bit_generator.state = sv["frng"]; fw.last_step[:] = sv["last"]; fw.seen = sv["seen"]
    rng = np.random.default_rng(cfg.seed); rng.bit_generator.state = sv["rng"]
    w1 = fw.w1
    replay = []
    for c in cs:
        rp = json.load(open(c["rpfile"]))["steps"] if c["rpfile"] else {}
        replay.append({int(k): [(int(q), int(t)) for q, t in v] for k, v in rp.items()})
        assert not replay[-1] or min(replay[-1]) >= w1, "replay before the branch point: not in the checkpoint"
    W_ = cfg.seq_len + 1
    for r in replay:   # train.py assert: bios + replay rows fit the batch
        fw.replay = r; assert fw.rows_needed(cfg.seq_len) < cfg.batch, "too many bios per step"
    fw.replay = {}
    RA = 0   # 2026-11-26 r8 control arm b (rpadd=1): replay rows added on top of the cfg.batch web rows; RA = most rows any model needs
    if c0.get("rpadd"):
        assert all(c.get("rpadd") for c in cs), "rpadd: every model of a group or none"
        RA = max([-(-sum(len(fw.sent[q][t]) + len(fw.sep) for q, t in v) // W_) for r in replay for v in r.values()] + [1])
    lrs = []
    for j, c in zip(jobs, cs):
        f = lr_fn(c, steps)
        if j.get("lrmul") is not None:
            m = j["lrmul"]; assert len(m) == steps - w1, f"lrmul needs {steps - w1} values"
            lrs.append([c["lr"] * float(m[s - w1]) for s in range(w1, steps)])
        else:
            lrs.append([f(s) for s in range(w1, steps)])
    LR = torch.tensor(lrs, dtype=torch.float32, device=dev)   # (N, steps - w1) Adam-group lr before gaplr
    PEAK = torch.tensor([c["lr"] for c in cs], device=dev); MLR = torch.tensor([c["mlr"] for c in cs], device=dev)
    MWD = torch.tensor([float(c["mwd"]) for c in cs], device=dev); GLR = torch.tensor([float(c["gaplr"]) for c in cs], device=dev)

    # stacked params and optimizer state
    names = [n for n, _ in tmpl.named_parameters()]
    P = {n: w.detach().unsqueeze(0).repeat((N,) + (1,) * w.dim()).contiguous().requires_grad_(True) for n, w in tmpl.named_parameters()}
    mnames = [n for n, _ in mats_params(tmpl)] if cfg.opt == "muon" else []
    anames = [n for n in names if n not in mnames]
    ost = sv["opt"]["state"]; assert len(sv["opt"]["param_groups"]) == 1
    order = sv["opt"]["param_groups"][0]["params"]
    AS = {}
    for idx_, n in zip(order, anames):
        s_ = ost[idx_]
        AS[n] = dict(m=s_["exp_avg"].to(dev).unsqueeze(0).repeat((N,) + (1,) * s_["exp_avg"].dim()).contiguous(),
                     v=s_["exp_avg_sq"].to(dev).unsqueeze(0).repeat((N,) + (1,) * s_["exp_avg_sq"].dim()).contiguous())
    astep = float(ost[order[0]]["step"])
    MS = {}
    if mnames:
        mst = sv["mopt"]["state"]; morder = sv["mopt"]["param_groups"][0]["params"]
        for idx_, n in zip(morder, mnames):
            MS[n] = mst[idx_]["m"].to(dev).unsqueeze(0).repeat((N,) + (1,) * mst[idx_]["m"].dim()).contiguous()
    b1, b2, eps = 0.9, 0.95, 1e-8
    cos, sin = tmpl.cos, tmpl.sin

    # gstep: learned-fact items (people with k >= max bin / 2, person-major, 5 relations), as train.py _gbits
    gsp = [q for q in range(fw.n_eval) if fw.k[q] >= max(fw.bins) // 2]
    items = [(fw.sep + fw.prompt[p][ai][0] + fw.prompt[p][ai][1], len(fw.sep) + len(fw.prompt[p][ai][0])) for p in gsp for ai in range(5)]
    L_ = max(len(i[0]) for i in items)
    gx = torch.zeros(len(items), L_ - 1, dtype=torch.long); gy = torch.full((len(items), L_ - 1), -1, dtype=torch.long)
    for j, (ids, npr) in enumerate(items):
        t = torch.tensor(ids); gx[j, :len(ids) - 1] = t[:-1]; gy[j, npr - 1:len(ids) - 1] = t[npr:]
    gx, gy = gx.to(dev), gy.to(dev)
    want_g = [bool(c["gstep"]) for c in cs]
    S2s = [None] * N   # 2026-11-25 replay-ceiling: online SM-2 per model, same rule and order as train.py _sm2
    for i, c in enumerate(cs):
        if c.get("sm2file"):
            assert want_g[i], "sm2file needs gstep=1"
            j_ = json.load(open(c["sm2file"])); fl = [tuple(f) for f in j_["facts"]]
            S2s[i] = dict(f=fl, u=float(j_["unit"]), cap=int(j_["budget"]), used=0, n={f: 0 for f in fl}, I={f: 0.0 for f in fl},
                          ef={f: 2.5 for f in fl}, due={f: w1 + int(round(float(j_["first"][k]) * float(j_["unit"]))) for k, f in enumerate(fl)},
                          C=j_["chance"], log=[], maxper=int(j_.get("maxper", 120)))
    gbits = [[] for _ in range(N)]
    if any(want_g):
        b0 = item_bits(P, cfg, gx, gy, cos, sin).cpu().numpy()
        for i in range(N):
            gbits[i].append([round(float(v), 2) for v in b0[i]])

    bio_tok = np.full(N, sv["bio"], dtype=np.int64)
    t0 = time.time(); last_loss = None
    PROF = int(os.environ.get("BATCHED_PROF_STEPS", "0"))   # timing only: stop after this many phase-2 steps, no rows
    tg = 0.0
    for step in range(w1, steps):
        if PROF and step - w1 == PROF:
            print(f"[batched] PROF N={N} {PROF} steps {time.time() - t0:.2f}s (gstep scoring {tg:.2f}s)", flush=True); return
        starts = rng.integers(0, len(train) - cfg.seq_len - 1, cfg.batch)
        base = np.stack([train[s:s + W_] for s in starts]).astype(np.int64)
        bt0 = fw.step_bios(step)
        if RA:
            assert not bt0, "rpadd: bios after the branch point"
            st2 = np.random.default_rng([cfg.seed, step]).integers(0, len(train) - W_, RA)
            exb = np.stack([train[s:s + W_] for s in st2]).astype(np.int64)
            MK = np.ones((N, RA + cfg.batch, W_ - 1), dtype=np.float32); MK[:, :RA] = 0
        X = np.empty((N, RA + base.shape[0], W_), dtype=np.int64)
        for i in range(N):
            bt = bt0 + [t for q, t_ in replay[i].get(step, ()) for t in fw.sent[q][t_] + fw.sep]
            xy = base.copy()
            if RA:
                ex = exb.reshape(-1).copy()
                if bt:
                    ex[:len(bt)] = bt; MK[i, :RA] = np.arange(RA * W_).reshape(RA, W_)[:, 1:] < len(bt)
                X[i] = np.concatenate([ex.reshape(RA, W_), xy]); bio_tok[i] += len(bt); continue
            if bt:
                r = -(-len(bt) // W_); buf = xy[:r].reshape(-1).copy(); buf[:len(bt)] = bt; xy[:r] = buf.reshape(r, W_)
            X[i] = xy; bio_tok[i] += len(bt)
        X = torch.from_numpy(X).to(dev)
        lr = LR[:, step - w1]
        gp = cfg.ffwlo > 0 and step >= cfg.ffwlo * steps
        glr = GLR if gp else torch.ones_like(GLR)
        alr = lr * glr                                   # Adam-group lr (mult 1)
        mlr = lr / PEAK * MLR * glr                      # Muon lr
        if RA:   # per-token weight 1/(batch*seq_len), as when fact rows replace web rows
            loss = (token_loss(P, cfg, X[:, :, :-1], X[:, :, 1:], cos, sin) * torch.from_numpy(MK).to(dev)).sum((1, 2)) / (cfg.batch * cfg.seq_len)
        else:
            loss = token_loss(P, cfg, X[:, :, :-1], X[:, :, 1:], cos, sin).mean((1, 2))   # (N,)
        loss.sum().backward()
        last_loss = loss.detach()
        with torch.no_grad():
            gn = torch.zeros(N, device=dev)
            for n in names:
                gn += P[n].grad.float().pow(2).reshape(N, -1).sum(1)
            coef = (cfg.clip / (gn.sqrt() + 1e-6)).clamp(max=1.0)
            astep += 1; bc1 = 1 - b1 ** astep; bc2 = 1 - b2 ** astep
            for n in anames:
                w, g = P[n], P[n].grad * coef.view((N,) + (1,) * (P[n].dim() - 1))
                st = AS[n]; sh = (N,) + (1,) * (w.dim() - 1)
                st["m"].lerp_(g, 1 - b1); st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (st["v"].sqrt() / math.sqrt(bc2)).add_(eps)
                w.sub_((alr / bc1).view(sh) * st["m"] / denom)
            for n in mnames:
                w, g = P[n], (P[n].grad * coef.view(N, 1, 1)).float()
                if n.endswith("attn.qkv.weight"):
                    G = g.view(N, 3, g.shape[1] // 3, g.shape[2]); Wm = w.view(N, 3, w.shape[1] // 3, w.shape[2])
                else:
                    G = g.unsqueeze(1); Wm = w.unsqueeze(1)
                m = MS[n]; m.lerp_(G, 1 - cfg.mom); M = G.lerp(m, cfg.mom)
                U = mlr.view(N, 1, 1, 1) * moon_scale(G) * ns5(M)
                Wm.mul_((1 - mlr * MWD).view(N, 1, 1, 1))
                Wm.sub_(U.to(Wm.dtype))
            for n in names:
                P[n].grad = None
        if any(want_g):
            tq = time.time()
            bb = item_bits(P, cfg, gx, gy, cos, sin).cpu().numpy()
            for i in range(N):
                if want_g[i]:
                    gbits[i].append([round(float(v), 2) for v in bb[i]])
            tg += time.time() - tq
        pi_ = {q: r for r, q in enumerate(gsp)}
        for i in range(N):
            S2 = S2s[i]
            if S2 is None: continue
            b0_a = np.array(gbits[i][0]).reshape(-1, 5); b = np.array(gbits[i][-1]).reshape(-1, 5)
            for f in [f for f in S2["f"] if S2["due"][f] <= step + 1][:S2["maxper"]]:
                if S2["used"] >= S2["cap"]: break
                r_, a_ = pi_[f[0]], f[1]; b0_ = b0_a[r_, a_]; C_ = S2["C"][a_]
                kp = 1 - (b[r_, a_] - b0_) / max(1e-6, C_ - b0_)
                q = 5 if kp >= 0.9 else 4 if kp >= 0.75 else 3 if kp >= 0.5 else 2 if kp >= 0.3 else 1
                if q < 3: S2["n"][f], S2["I"][f] = 0, 1.0
                else:
                    S2["n"][f] += 1
                    S2["I"][f] = 1.0 if S2["n"][f] == 1 else 6.0 if S2["n"][f] == 2 else S2["I"][f] * S2["ef"][f]
                    S2["ef"][f] = max(1.3, S2["ef"][f] + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
                S2["due"][f] = step + 1 + max(1, int(round(S2["I"][f] * S2["u"])))
                replay[i].setdefault(step + 1, []).append((int(f[0]), int(f[1]))); S2["used"] += 1; S2["log"].append([step + 1, f[0], f[1], q])
        if (step - w1) % 50 == 0:
            print(f"[batched] step {step} loss mean {loss.mean().item():.4f} min {loss.min().item():.4f} max {loss.max().item():.4f} "
                  f"{time.time() - t0:.1f}s", flush=True)
    secs = time.time() - t0

    # rows: weights of model i into the template, then train.py's end-of-run evaluation
    val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)
    if a0.split == "val":
        val = val[:T.FINEWEB_TEST_SEQ * cfg.seq_len + 1] if a0.data.startswith("fineweb") else val
    elif a0.data.startswith("fineweb"):
        val = val[T.FINEWEB_TEST_SEQ * cfg.seq_len:]
    tb = token_bytes(TOKF) if TOKF != TOK0 else token_bytes()
    total = sum(p.numel() for p in tmpl.parameters())
    for i, (j, c) in enumerate(zip(jobs, cs)):
        with torch.no_grad():
            for n, w in tmpl.named_parameters():
                w.copy_(P[n][i])
        ci = types.SimpleNamespace(**dict(c0, **{k: c[k] for k in BRANCH}))
        res = evaluate(tmpl, val, tb, ci, a0.eval_seqs, device=dev)
        if cfg.freq_eval:
            fr = np.bincount(train[:4_000_000], minlength=cfg.vocab)
            rk = np.empty(cfg.vocab, dtype=np.int64); rk[np.argsort(-fr)] = np.arange(cfg.vocab)
            bk = torch.from_numpy(np.digitize(rk, [256, 1024])).to(dev)
            n_ = min(a0.eval_seqs, (len(val) - 1) // cfg.seq_len)
            xy = torch.from_numpy(val[:n_ * cfg.seq_len + 1].astype(np.int64))
            x, y = xy[:-1].view(n_, cfg.seq_len).to(dev), xy[1:].view(n_, cfg.seq_len).to(dev)
            tmpl.eval()
            with torch.no_grad():
                l = torch.cat([tmpl(x[k:k + 32], y[k:k + 32])[1].reshape(-1, cfg.seq_len) for k in range(0, n_, 32)])
            tmpl.train(); fe = {}
            for nm, ids in (("in", x), ("tgt", y)):
                b = bk[ids]
                for jj, lab in enumerate("FMR"):
                    mm = b == jj; fe[f"{nm}{lab}"] = round(l[mm].mean().item(), 4); fe[f"{nm}{lab}_frac"] = round(mm.float().mean().item(), 3)
            res["freq_eval"] = fe
        res["facts_eval"] = fw.evaluate(tmpl, dev, steps)
        res["facts_eval"]["curve"] = []
        hid = [w for n_, w in tmpl.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_]
        res["facts_eval"]["wnorm"] = round(float(sum(w.float().pow(2).mean().item() for w in hid) / max(1, len(hid))) ** 0.5, 5)
        if want_g[i]:
            res["facts_eval"]["gstep"] = {"people": gsp, "bits": gbits[i]}
        if S2s[i] is not None: res["facts_eval"]["sm2"] = {"used": S2s[i]["used"], "log": S2s[i]["log"]}
        res["facts_eval"]["bio_frac"] = round(int(bio_tok[i]) / (steps * cfg.batch * (cfg.seq_len + 1)), 4)
        res["nonemb_params"] = sum(p.numel() for n_, p in tmpl.named_parameters() if p.dim() >= 2 and "emb" not in n_ and "head" not in n_)
        row = dict(method=a0.method, tag=j["tag"], branch_of=None, data=a0.data, split=a0.split, size=a0.size, steps=steps,
                   train_seconds=round(secs / N, 2), threads=threads, device=str(dev), tokens=steps * cfg.batch * cfg.seq_len,
                   params=total, active_params=total, **res, config=c,
                   last_train=dict(step=steps - 1, loss=round(float(last_loss[i]), 4)),
                   batched=dict(n=N, chunk_seconds=round(secs, 1), from_step=w1, gckpt=ck, lrmul=j.get("lrmul") is not None))
        print(json.dumps({k: v for k, v in row.items() if k not in ("config", "facts_eval")}), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64, help="models per batch")
    ap.add_argument("--only", default=""); ap.add_argument("--tag", default="")
    s, extra = ap.parse_known_args(argv)
    if extra: print(f"[batched] ignoring {extra}", flush=True)
    torch.set_num_threads(s.threads)
    jf = s.jobs
    if not os.path.isabs(jf):
        for b in (os.getcwd(), os.path.join(HERE, "..", ".."), HERE):
            if os.path.exists(os.path.join(b, jf)):
                jf = os.path.join(b, jf); break
    done = set()
    if os.path.exists(s.out):
        for l in open(s.out):
            try: done.add(json.loads(l)["tag"])
            except Exception: pass
    jobs = [json.loads(l) for l in open(jf) if l.strip()]
    jobs = [j for j in jobs if j["tag"] not in done and (not s.only or any(x in j["tag"] for x in s.only.split(",")))]
    groups = {}
    for j in jobs:
        a, c = parse(j["args"]); groups.setdefault(trunk_key(a, c), []).append(j)
    dev = torch.device(s.device); t_all = time.time(); n_ok = 0
    for g in groups.values():
        for i in range(0, len(g), s.chunk):
            ch = g[i:i + s.chunk]
            try:
                run_group(ch, s.out, dev, s.threads); n_ok += len(ch)
            except Exception as e:
                print(f"BATCHED FAIL {[j['tag'] for j in ch]}: {e!r}", flush=True)
                raise
    print(f"BATCHED done {n_ok}/{len(jobs)} in {time.time() - t_all:.1f}s", flush=True)
    if s.tag:
        with open(s.out, "a") as f:
            f.write(json.dumps(dict(tag=s.tag, sweep_file=s.jobs, sweep_ok=n_ok, sweep_jobs=len(jobs), sweep_seconds=round(time.time() - t_all, 1))) + "\n")


if __name__ == "__main__":
    main()
