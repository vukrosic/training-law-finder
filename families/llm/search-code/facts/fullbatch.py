"""Full-run batched trainer (ops 2026-11-26, manager speed task): R dense XS facts runs in ONE process, from step 0 to the
end, each with its own peak / schedule. Same structure as sim.py's R runs per process: weights stacked (R, ...), one
bmm per layer for all R models. train.py is not changed; batched.py (branches from a gckpt) is not changed.

What may differ inside one group (everything else must be equal, else the runs go to different groups):
  lr, mlr, sched, min_lr_frac, warmup, decay, rwp, rww, rwhold, rwcos, mwd, gaplr.
Because the data, bios and fact world depend only on the shared fields, every model sees exactly the batches its own
train.py run would see; only the step sizes differ.

Supported: dense softmax attention, opt muon or base, facts runs, gstep 1/2 (bits curve after the window), wlog (no wsig).
Not supported (those jobs run one at a time through train.py instead): gsplice, fmarg, fevery, rpfile, sm2file, gckpt,
replay / resight / gap-freeze tests, gbs, glrx, gnoclip, momentum swaps, fsplit, amp, ce_chunk, other models.

usage: python3 -I env/fullbatch.py --jobs X.jsonl --out F [--device cuda] [--chunk 16] [--threads 1] [--prec fp32|tf32|bf16]
       --prec (O72, 2026-11-26): fp32 = exact train.py math (default); tf32 = TF32 matmuls; bf16 = bf16 autocast for the training
       forward/backward plus TF32 for the remaining fp32 matmuls. Evaluation stays fp32. Rows get "prec".
       rows are train.py rows plus "batched": {"n", "full": true, "loss_curve": per-step train loss}.
"""
import argparse, json, math, os, resource, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as T   # noqa: E402
from model import GPT   # noqa: E402
from opt import ns5, moon_scale, mats_params   # noqa: E402
import facts as factsmod   # noqa: E402
from score import evaluate, token_bytes   # noqa: E402
import batched as B   # noqa: E402   forward / token_loss / item_bits / parse (same math as model.py)

PER_MODEL = ("lr", "mlr", "sched", "min_lr_frac", "warmup", "decay", "rwp", "rww", "rwhold", "rwcos", "mwd", "gaplr")
PREC = "fp32"   # set by --prec in main()
IGNORE = ("wlog",)   # handled per model, does not change training
MUST_DEFAULT = ("gsplice", "fmarg", "fevery", "ftrace", "fchoice", "rpfile", "rpadd", "sm2file", "gckpt", "gfreeze", "fdrift", "flog",
                "pfscreen", "rsp", "zcheap", "zfree", "zrare", "gmom", "gmb", "gb1", "gbs", "gsgd", "gnoclip", "glrx", "fsplit",
                "wsig", "amp", "ce_chunk", "branches", "seen_c", "skip_until", "skip_from", "type_balance", "prof", "tf32")


def group_key(a, c):
    return json.dumps(dict({k: v for k, v in c.items() if k not in PER_MODEL + IGNORE}, _steps=a.steps, _data=a.data,
                           _split=a.split, _size=a.size, _method=a.method, _eval=a.eval_seqs), sort_keys=True, default=str)


def supported(c):
    if c["attn"] != "softmax" or c["router"] != "dense" or c["erule"] != "none" or c["opt"] not in ("muon", "base"):
        return False
    if not c["facts"] or c["ffw"] != 1.0 or c["expert_lr"] != 1.0 or c["ht"] != "none" or c["emb_init"]:
        return False
    return all(c.get(k) == T.BASE.get(k) for k in MUST_DEFAULT)


def lr_fn(c, steps):
    """train.py lr_at, every sched it knows (copy; train.py keeps it inside main)"""
    def f(s):
        if s < c["warmup"]:
            return c["lr"] * (s + 1) / c["warmup"]
        if c["sched"] == "wsd":
            d0 = steps - int(round(c["decay"] * steps))
            return c["lr"] if s < d0 else c["lr"] * (1 - (1 - c["min_lr_frac"]) * (s - d0 + 1) / (steps - d0))
        if c["sched"] == "const":
            return c["lr"]
        if c["sched"] == "crw":
            P = int(round(c["rwp"] * steps)); mn = c["lr"] * c["min_lr_frac"]
            if s < P:
                if not c["rwcos"]: return c["lr"]
                if c["rwcos"] == 2: return mn
                p = (s - c["warmup"]) / max(1, P - c["warmup"]); return mn + (c["lr"] - mn) * 0.5 * (1 + math.cos(math.pi * p))
            if s < P + c["rww"]:
                return mn if c["rwhold"] else mn + (c["lr"] - mn) * (s - P + 1) / c["rww"]
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


def run_group(jobs, out, dev, threads):
    a0, c0 = B.parse(jobs[0]["args"])
    cs = [B.parse(j["args"])[1] for j in jobs]
    N, steps = len(jobs), a0.steps
    cfg = types.SimpleNamespace(**c0); cfg.l1_coeff_state = [cfg.l1_init]
    torch.manual_seed(cfg.seed)                      # same init as train.py: every model of the group starts from these weights
    tmpl = GPT(cfg).to(dev)
    DATA = T.DATASETS[a0.data]
    TOK0 = os.path.join(HERE, "..", "data", "tokenizer", "bpe8192.json")
    TOKF = os.path.join(DATA, "tokenizer.json") if os.path.exists(os.path.join(DATA, "tokenizer.json")) else TOK0
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(cfg.seed)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOKF)
    bins = [0] if cfg.fzero else [cfg.fkmin * 2 ** i for i in range(int(math.log2(cfg.fkmax // cfg.fkmin)) + 1)]
    fw = factsmod.FactWorld(tk, cfg.facts, bins, steps, seed=cfg.fseed, win=(cfg.fwlo, cfg.fwhi), n_filler=cfg.ffill,
                            filler_k=cfg.ffillk, fwin=None if cfg.ffwlo < 0 else (cfg.ffwlo, cfg.ffwhi), frand=int(cfg.ffrand),
                            frandp=float(cfg.ffrandp), ffmt=int(cfg.ffmt), ffkind=int(cfg.ffkind),
                            web=train if int(cfg.ffkind) == 3 else None,
                            pre=None if cfg.fprelo < 0 else (cfg.fprelo, cfg.fprehi),
                            fpre=None if cfg.ffprelo < 0 else (cfg.ffprelo, cfg.ffprehi, int(cfg.ffprek)), n_ref=int(cfg.fref))
    fws = [fw]
    if cfg.fgroups:   # O82 (2026-11-26): fgroups as train.py, G worlds (seed fseed+j), windows of length fwhi-fwlo spaced in [fglo, fghi]
        L_ = cfg.fwhi - cfg.fwlo
        _all = [factsmod.FactWorld(tk, cfg.facts, bins, steps, seed=cfg.fseed + j, win=(float(s_), float(s_) + L_), n_ref=int(cfg.fref))
                for j, s_ in enumerate(np.linspace(cfg.fglo, cfg.fghi - L_, int(cfg.fgroups)))]
        fws = [_all[int(cfg.fgonly)]] if cfg.fgonly >= 0 else _all
        fw = fws[0]
    assert sum(w_.rows_needed(cfg.seq_len) for w_ in fws) < cfg.batch, "too many bios per step"
    w1 = fw.w1
    GI = []   # fgroups: per group (start item, people, n learned items, k==max mask, n ref items); one item tensor for all groups
    if cfg.fgroups:
        git = []
        for j_, w_ in enumerate(fws):
            pp_ = [q for q in range(w_.n_eval) if w_.k[q] >= max(w_.bins) // 2]
            it_ = [(w_.sep + w_.prompt[p][ai][0] + w_.prompt[p][ai][1], len(w_.sep) + len(w_.prompt[p][ai][0])) for p in pp_ for ai in range(5)]
            rf_ = [(w_.sep + pr[0] + pr[1], len(w_.sep) + len(pr[0])) for rp in w_.ref_prompt for pr in rp]
            GI.append((len(git), pp_, len(it_), np.array([w_.k[q] == max(w_.bins) for q in pp_]), len(rf_))); git += it_ + rf_
        LG = max(len(i[0]) for i in git)
        ggx = torch.zeros(len(git), LG - 1, dtype=torch.long); ggy = torch.full((len(git), LG - 1), -1, dtype=torch.long)
        for j, (ids, npr) in enumerate(git):
            t = torch.tensor(ids); ggx[j, :len(ids) - 1] = t[:-1]; ggy[j, npr - 1:len(ids) - 1] = t[npr:]
        ggx, ggy = ggx.to(dev), ggy.to(dev)
    GT = [[None] * len(fws) for _ in range(N)]

    def grp_read(step, pre=False):   # fgroups: one reading per group whose window has ended (as train.py _gtr; no rng, no grads);
        due = [(step == w_.w1) if pre else (step >= w_.w1) for w_ in fws]   # pre: at a group's w1 before that step's update
        if not any(due): return
        with torch.no_grad():
            gb = B.item_bits(P, cfg, ggx, ggy, cos, sin).cpu().numpy()
        for j_, w_ in enumerate(fws):
            if not due[j_]: continue
            s0, pp_, nl_, km_, nr_ = GI[j_]
            for i in range(N):
                if GT[i][j_] is None:
                    GT[i][j_] = {"group": int(cfg.fgonly) if cfg.fgonly >= 0 else j_, "w0": w_.w0, "w1": w_.w1, "people": pp_, "b": [], "bmax": []}
                r_ = GT[i][j_]; b_ = gb[i, s0:s0 + nl_].reshape(-1, 5)
                if nr_: r_.setdefault("ref", []).append(round(float(gb[i, s0 + nl_:s0 + nl_ + nr_].mean()), 4))
                r_["b"].append(round(float(b_.mean()), 4)); r_["bmax"].append(round(float(b_[km_].mean()), 4))
    print(f"[fullbatch] {N} models, {steps} steps, window end {w1}, tags {jobs[0]['tag']} .. {jobs[-1]['tag']}", flush=True)

    fs = [lr_fn(c, steps) for c in cs]
    LR = torch.tensor([[f(s) for s in range(steps)] for f in fs], dtype=torch.float32, device=dev)   # (N, steps)
    PEAK = torch.tensor([c["lr"] for c in cs], device=dev); MLR = torch.tensor([c["mlr"] for c in cs], device=dev)
    MWD = torch.tensor([float(c["mwd"]) for c in cs], device=dev); GLR = torch.tensor([float(c["gaplr"]) for c in cs], device=dev)

    names = [n for n, _ in tmpl.named_parameters()]
    P = {n: w.detach().unsqueeze(0).repeat((N,) + (1,) * w.dim()).contiguous().requires_grad_(True) for n, w in tmpl.named_parameters()}
    mnames = [n for n, _ in mats_params(tmpl)] if cfg.opt == "muon" else []
    anames = [n for n in names if n not in mnames]
    AS = {n: dict(m=torch.zeros_like(P[n]), v=torch.zeros_like(P[n])) for n in anames}
    MS = {}
    for n in mnames:
        MS[n] = torch.zeros((N, 3, P[n].shape[1] // 3, P[n].shape[2]) if n.endswith("attn.qkv.weight") else (N, 1) + tuple(P[n].shape[1:]),
                            device=dev, dtype=torch.float32)
    hidn = [n for n, w in tmpl.named_parameters() if w.ndim == 2 and "emb" not in n and "head" not in n]
    b1, b2, eps = 0.9, 0.95, 1e-8
    cos, sin = tmpl.cos, tmpl.sin
    W_ = cfg.seq_len + 1

    gsp = []
    if cfg.gstep:
        gsp = [q for q in range(fw.n_eval) if fw.k[q] >= max(fw.bins) // 2]
        if cfg.gstep == 2:
            gsp += [q for q in range(fw.n_eval) if fw.k[q] <= 4]
        items = [(fw.sep + fw.prompt[p][ai][0] + fw.prompt[p][ai][1], len(fw.sep) + len(fw.prompt[p][ai][0])) for p in gsp for ai in range(5)]
        NL_ = len(items)   # O76: never-seen reference items ride in the same forward, split off after it (ref_bits per reading)
        items += [(fw.sep + pr[0] + pr[1], len(fw.sep) + len(pr[0])) for rp in fw.ref_prompt for pr in rp]
        L_ = max(len(i[0]) for i in items)
        gx = torch.zeros(len(items), L_ - 1, dtype=torch.long); gy = torch.full((len(items), L_ - 1), -1, dtype=torch.long)
        for j, (ids, npr) in enumerate(items):
            t = torch.tensor(ids); gx[j, :len(ids) - 1] = t[:-1]; gy[j, npr - 1:len(ids) - 1] = t[npr:]
        gx, gy = gx.to(dev), gy.to(dev)
    gbits = [[] for _ in range(N)]
    grefs = [[] for _ in range(N)]   # O76 ref_bits per reading

    WL = int(cfg.wlog); wl = [{"step": [], "rms": [], "drel": [], "dw1": [], "cos_prev": []} for _ in range(N)]
    wl_prev = wl_w1 = None

    @torch.no_grad()
    def hid_stats(ref):   # per model: sum sq, count, and (if ref) sum sq diff, ref sum sq, dot
        sq = sum(P[n].detach().float().pow(2).flatten(1).sum(1) for n in hidn); cnt = sum(P[n][0].numel() for n in hidn)
        if ref is None:
            return sq, cnt, None, None, None
        d2 = sum((P[n].detach().float() - ref[n]).pow(2).flatten(1).sum(1) for n in hidn)
        pq = sum(ref[n].pow(2).flatten(1).sum(1) for n in hidn)
        dot = sum((P[n].detach().float() * ref[n]).flatten(1).sum(1) for n in hidn)
        return sq, cnt, d2, pq, dot

    def snap_hid():
        return {n: P[n].detach().float().clone() for n in hidn}

    def do_wlog(n_):
        nonlocal wl_prev
        sq, cnt, d2, pq, dot = hid_stats(wl_prev)
        _, _, e2, q1, _ = hid_stats(wl_w1) if wl_w1 is not None else (None,) * 5
        for i in range(N):
            r = wl[i]
            if wl_prev is not None:
                r["drel"].append(round((float(d2[i]) / max(float(pq[i]), 1e-30)) ** 0.5, 6))
                r["cos_prev"].append(round(float(dot[i]) / max(1e-30, (float(sq[i]) * float(pq[i])) ** 0.5), 7))
            else:
                r["drel"].append(None); r["cos_prev"].append(None)
            r["dw1"].append(round((float(e2[i]) / max(float(q1[i]), 1e-30)) ** 0.5, 6) if wl_w1 is not None else None)
            r["step"].append(n_); r["rms"].append(round((float(sq[i]) / max(1, cnt)) ** 0.5, 6))
        wl_prev = snap_hid()

    bio_tok = 0; loss_curve = [[] for _ in range(N)]
    t0 = time.time()
    for step in range(steps):
        if cfg.fgroups: grp_read(step, pre=True)
        if gsp and step == w1:
            b0 = B.item_bits(P, cfg, gx, gy, cos, sin).cpu().numpy()
            for i in range(N):
                gbits[i].append([round(float(v), 2) for v in b0[i][:NL_]])
                if NL_ < b0.shape[1]: grefs[i].append(round(float(b0[i][NL_:].mean()), 4))
        starts = rng.integers(0, len(train) - cfg.seq_len - 1, cfg.batch)
        xy = np.stack([train[s:s + W_] for s in starts]).astype(np.int64)
        bt = fw.step_bios(step) if len(fws) == 1 else [t_ for w_ in fws for t_ in w_.step_bios(step)]
        if bt:
            r = -(-len(bt) // W_); buf = xy[:r].reshape(-1).copy(); buf[:len(bt)] = bt; xy[:r] = buf.reshape(r, W_)
            bio_tok += len(bt)
        X = torch.from_numpy(xy).to(dev).unsqueeze(0).expand(N, -1, -1)
        gp = cfg.ffwlo > 0 and step >= cfg.ffwlo * steps
        glr = GLR if gp else torch.ones_like(GLR)
        lr = LR[:, step]
        alr = lr * glr
        mlr = lr / PEAK * MLR * glr
        with (torch.autocast("cuda", dtype=torch.bfloat16) if PREC == "bf16" and dev.type == "cuda" else torch.autocast("cpu", enabled=False)):
            loss = B.token_loss(P, cfg, X[:, :, :-1].contiguous(), X[:, :, 1:].contiguous(), cos, sin).float().mean((1, 2))   # (N,)
        loss.sum().backward()
        with torch.no_grad():
            gn = torch.zeros(N, device=dev)
            for n in names:
                gn += P[n].grad.float().pow(2).reshape(N, -1).sum(1)
            coef = (cfg.clip / (gn.sqrt() + 1e-6)).clamp(max=1.0)
            bc1 = 1 - b1 ** (step + 1); bc2 = 1 - b2 ** (step + 1)
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
        lc = loss.detach().cpu().numpy()
        for i in range(N):
            loss_curve[i].append(round(float(lc[i]), 5))
        if gsp and step >= w1:
            bb = B.item_bits(P, cfg, gx, gy, cos, sin).cpu().numpy()
            for i in range(N):
                gbits[i].append([round(float(v), 2) for v in bb[i][:NL_]])
                if NL_ < bb.shape[1]: grefs[i].append(round(float(bb[i][NL_:].mean()), 4))
        if cfg.fgroups: grp_read(step)
        if WL and step + 1 == w1 and wl_w1 is None:
            wl_w1 = snap_hid()
        if WL and ((step + 1) % WL == 0 or step + 1 == w1):
            do_wlog(step + 1)
        if step % 100 == 0 or step == steps - 1:
            print(f"[fullbatch] step {step} loss mean {lc.mean():.4f} min {lc.min():.4f} max {lc.max():.4f} {time.time() - t0:.1f}s", flush=True)
    secs = time.time() - t0

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
        ci = types.SimpleNamespace(**c); ci.l1_coeff_state = [ci.l1_init]
        res = evaluate(tmpl, val, tb, ci, a0.eval_seqs, device=dev)
        if cfg.freq_eval:
            fr = np.bincount(train[:4_000_000], minlength=cfg.vocab)
            rk = np.empty(cfg.vocab, dtype=np.int64); rk[np.argsort(-fr)] = np.arange(cfg.vocab)
            bk = torch.from_numpy(np.digitize(rk, [256, 1024])).to(dev)
            n_ = min(a0.eval_seqs, (len(val) - 1) // cfg.seq_len)
            xyv = torch.from_numpy(val[:n_ * cfg.seq_len + 1].astype(np.int64))
            x, y = xyv[:-1].view(n_, cfg.seq_len).to(dev), xyv[1:].view(n_, cfg.seq_len).to(dev)
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
        if gsp:
            res["facts_eval"]["gstep"] = {"people": gsp, "bits": gbits[i]}
            if grefs[i]: res["facts_eval"]["gstep"]["ref"] = grefs[i]   # O76
        if cfg.fgroups:   # per group C = mean bits of its people seen <= 4 times at the end (as train.py)
            tmpl.eval()
            for j_, w_ in enumerate(fws):
                if GT[i][j_] is None: continue
                _, b_ = w_._score(tmpl, dev, [q for q in range(w_.n_eval) if w_.k[q] <= 4]); GT[i][j_]["C"] = round(float(b_.mean()), 4)
            tmpl.train()
            res["facts_eval"]["groups"] = [g_ for g_ in GT[i] if g_ is not None]
        res["facts_eval"]["bio_frac"] = round(bio_tok / (steps * cfg.batch * (cfg.seq_len + 1)), 4)
        hw = [w for n_, w in tmpl.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_]
        res["facts_eval"]["wnorm"] = round(float(sum(w.float().pow(2).mean().item() for w in hw) / max(1, len(hw))) ** 0.5, 5)
        if WL:
            res["facts_eval"]["wtrace"] = wl[i]
        res["nonemb_params"] = sum(p.numel() for n_, p in tmpl.named_parameters() if p.dim() >= 2 and "emb" not in n_ and "head" not in n_)
        row = dict(method=a0.method, tag=j["tag"], branch_of=None, data=a0.data, split=a0.split, size=a0.size, steps=steps,
                   train_seconds=round(secs / N, 2), threads=threads, device=str(dev), tokens=steps * cfg.batch * cfg.seq_len, passes=round(steps * cfg.batch * cfg.seq_len / len(train), 3), prec=PREC,
                   params=total, active_params=total,
                   peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1),
                   gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
                   **res, prof=None, qk_clips=0, smax_peak=0.0, config=c, last_train=dict(step=steps - 1, loss=round(loss_curve[i][-1], 4)),
                   batched=dict(n=N, full=True, group_seconds=round(secs, 1), loss_curve=loss_curve[i]))
        print(json.dumps({k: v for k, v in row.items() if k not in ("config", "facts_eval", "batched")}), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16, help="models per process batch")
    ap.add_argument("--only", default="")
    ap.add_argument("--prec", default="fp32", choices=("fp32", "tf32", "bf16"))
    s, extra = ap.parse_known_args(argv)
    global PREC
    PREC = s.prec
    if PREC in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if extra: print(f"[fullbatch] ignoring {extra}", flush=True)
    torch.set_num_threads(s.threads)
    done = set()
    if os.path.exists(s.out):
        for l in open(s.out):
            try: done.add(json.loads(l)["tag"])
            except Exception: pass
    jobs = [json.loads(l) for l in open(s.jobs) if l.strip()]
    jobs = [j for j in jobs if j["tag"] not in done and (not s.only or any(x in j["tag"] for x in s.only.split(",")))]
    groups, single = {}, []
    for j in jobs:
        a, c = B.parse(j["args"])
        (groups.setdefault(group_key(a, c), []) if supported(c) else single).append(j)
    dev = torch.device(s.device); t_all = time.time()
    for g in groups.values():
        for i in range(0, len(g), s.chunk):
            run_group(g[i:i + s.chunk], s.out, dev, s.threads)
    for j in single:   # unsupported options: one train.py run each, unchanged
        print(f"[fullbatch] {j['tag']}: unsupported options, running train.py alone", flush=True)
        T.main([*j["args"], "--tag", j["tag"], "--threads", str(s.threads), "--out", s.out, "--device", s.device])
    print(f"FULLBATCH done {len(jobs)} jobs ({sum(len(g) for g in groups.values())} batched, {len(single)} alone) in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
