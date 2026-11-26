"""Train one model from scratch on FineWeb-Edu tokens and score it on the MOE-PRETRAIN benchmark.

    python3 -I env/train.py --method topk_aux --steps 60 --out experiments/runs/mine.jsonl

Methods (laptop presets, matched active parameters per token; see BASELINES.md):
    dense, topk_aux, global_aux, dsv3, expert_choice, remoe
Any preset field can be overridden with --set key=value (e.g. --set aux_coeff=0.001).
Writes one JSON line with the config, val loss, bits per byte and load-balance metrics.
"""
import argparse, json, math, os, resource, sys, time, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "benchmark"))
from model import GPT, MoE              # noqa: E402
from score import evaluate, token_bytes  # noqa: E402

DATASETS = {   # name -> folder with train.bin / val.bin (/ test.bin)
    "fineweb": os.path.join(HERE, "..", "data", "fineweb_edu"),
    "pystd": os.path.join(HERE, "..", "..", "data", "pystd"),
    "fineweb_full": os.path.join(HERE, "..", "..", "data", "fineweb_full"),   # GPU runs, prepare_fineweb_full.py   # video folder, prepare_pystd.py
}
FINEWEB_TEST_SEQ = 1024   # FineWeb-Edu has no test.bin: test = val.bin sequences from 1024 on
SIZES = {   # widths scale by 1.25 for M; every preset keeps active FFN width = dense_hidden
    # berserker size: one run of ~100 steps in 5-10 s on a laptop CPU
    "XS": dict(d_model=64, n_layers=3, n_heads=2, seq_len=64, width_mult=0.25),
    "S": dict(),
    "M": dict(d_model=160, n_layers=6, n_heads=5, width_mult=1.25),
    # GPU sizes for SCALE-PLAN.md (head dim 32 / 64, 16 experts top-3 kept)
    "L": dict(d_model=256, n_layers=8, n_heads=8, width_mult=2.0),
    "XL": dict(d_model=384, n_layers=12, n_heads=6, width_mult=3.0),
    "XXL": dict(d_model=512, n_layers=16, n_heads=8, width_mult=4.0),
}

BASE = dict(vocab=8192, d_model=128, n_layers=4, n_heads=4, seq_len=256, batch=16,
            lr=3e-3, warmup=20, min_lr_frac=0.1, clip=1.0, seed=0,
            router="dense", dense_hidden=512,
            n_experts=8, expert_hidden=256, topk=2, shared_hidden=0,
            pre_softmax=False, capacity_factor=0.0, aux_coeff=0.0, z_coeff=0.0, lbl_ranks=8,
            seq_aux_coeff=0.0, bias_rate=0.0, route_scale=None, n_groups=0, group_topk=0,
            ec_capacity=2.0, l1_init=1e-8, l1_mult=1.2, dm_beta=0.9, ckpt_ffn=False,
            route_input="x", route_norm=False, freeze_gate=False,
            std_route=False, std_rate=0.05, type_balance=False,
            score_fn="sigmoid", balance="auxfree", hash_layers=0, n_shared=1,
            hash_k=0, id_bias=False, id_bias_layers=0, id_bias_init=0.0, hash_sets=0,
            hash_pos=0, hash_mode="uni", hash_gate="learned", inv="plain",
            hyp_rank=1, hyp_where="w1", hyp_gen="lin", hyp_base="dense", hyp_width=0, hyp_off=False, hyp_squash="none", hyp_gain=False, hyp_prev=False, hyp_pinit=0.02,
            bg_win=1, bg_route="x", bg_mode="cat", bg_emb="none", hg_size=8, hk="uuu", he="none", ht="none", ht_n=4000000, emb_init=False, ht_src="file", ht_g=16, ht_top=1024, rare_mode="none", freq_eval=True, skip_sel="rare", rare_k=0, seen_c=0, skip_until=0, skip_from=0, expert_lr=1.0, emb_var="base", emb_scale=1.0, svd_alpha=0.1, svd_mode="logp", svd_rank=0, svd_side="both", mix="attn", mxo="")

PRESETS = {
    # dense SwiGLU, hidden 512 = the active FFN width of every MoE preset
    "dense": dict(router="dense"),
    # Switch / GShard load-balancing loss (coeff 0.01, Switch), top-2 softmax (Mixtral / Megatron
    # default: softmax over the selected logits), router z-loss 0.001 (ST-MoE), dropless;
    # loss per micro-batch of 2 sequences (8 virtual data-parallel ranks)
    "topk_aux": dict(router="topk_aux", aux_coeff=0.01, z_coeff=0.001, lbl_ranks=8),
    # Qiu et al. 2025: the same loss with expert counts over the global batch (16 sequences)
    "global_aux": dict(router="global_aux", aux_coeff=0.01, z_coeff=0.001),
    # DeepSeek-V3: 16 fine-grained routed experts (hidden 128) top-3 + 1 shared (hidden 128),
    # sigmoid, bias update rate 0.001, sequence-wise loss 0.0001, route scale 2.5
    "dsv3": dict(router="dsv3", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128,
                 bias_rate=1e-3, seq_aux_coeff=1e-4, route_scale=2.5),
    # 2026 routers, same layout as dsv3 (16 x 128 routed top-3, shared width 128, route scale 2.5)
    # Kimi K3 (Jul 2026, tech report sec 2.3.3): sigmoid, Quantile Balancing, 2 shared experts
    # (2 x 64: same active width as dsv3), no balance loss
    "k3qb": dict(router="gen", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128,
                 n_shared=2, score_fn="sigmoid", balance="quantile", route_scale=2.5),
    # DeepSeek-V4 (arXiv 2606.19348): Sqrt(Softplus), aux-free bias 0.001 + sequence loss 1e-4,
    # 1 shared expert, hash routing by token id in the first MoE layer (paper: first 3 of 61)
    # invention search (IDEAS.md): FFN from inventions.py named by inv=; same expert budget as dsv4
    "inv": dict(router="inv", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128),
    "dsv4": dict(router="gen", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128,
                 score_fn="sqrtsoftplus", balance="auxfree", bias_rate=1e-3, seq_aux_coeff=1e-4,
                 route_scale=2.5, hash_layers=1),
    # DeepSeek-V4.1-Flash (Sep 2026): as dsv4 without hash routing; V4.1 states no score
    # function, so the V4 Sqrt(Softplus) is used
    "dsv41": dict(router="gen", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128,
                  score_fn="sqrtsoftplus", balance="auxfree", bias_rate=1e-3, seq_aux_coeff=1e-4,
                  route_scale=2.5),
    # Expert Choice: capacity factor 2 (each token gets 2 experts on average)
    "expert_choice": dict(router="expert_choice", ec_capacity=2.0),
    # ReMoE: ReLU router, target sparsity 1 - 2/8, L1 coefficient init 1e-8, multiplier 1.2
    "remoe": dict(router="remoe", l1_init=1e-8, l1_mult=1.2),
    # SHARED-ROUTE (ours): dsv3, but the router reads the shared expert's hidden activations
    "shared_route": dict(router="dsv3", n_experts=16, expert_hidden=128, topk=3, shared_hidden=128,
                         bias_rate=1e-3, seq_aux_coeff=1e-4, route_scale=2.5, route_input="shared_h"),
    # Default MoE (Panda et al. 2025): global_aux router + EMA default vectors (beta 0.9) for the
    # unselected experts; softmax over all experts, top-k weights not renormalized (official code)
    "default_moe": dict(router="default_moe", aux_coeff=0.01, z_coeff=0.001, pre_softmax=True,
                        dm_beta=0.9),
}


def parse_value(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max_seconds", type=float, default=0, help="stop training early (0 = off)")
    ap.add_argument("--eval_seqs", type=int, default=64, help="validation sequences of 256 tokens")
    ap.add_argument("--set", nargs="*", default=[], help="override preset fields: key=value")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default="fineweb", choices=list(DATASETS))
    ap.add_argument("--split", default="val", choices=["val", "test"], help="test only for final numbers")
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--device", default="cpu", help="cpu or cuda; the only change for a GPU run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    c = dict(BASE, **PRESETS[a.method])
    sz = dict(SIZES[a.size])
    wm = sz.pop("width_mult", 1.0)
    c.update(sz)
    for k in ("dense_hidden", "expert_hidden", "shared_hidden"):
        c[k] = int(round(c[k] * wm))
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in c, f"unknown field {k}"
        c[k] = parse_value(v)
    cfg = types.SimpleNamespace(**c)
    cfg.l1_coeff_state = [cfg.l1_init]
    torch.manual_seed(cfg.seed)
    dev = torch.device(a.device)
    model = GPT(cfg).to(dev)
    total, active = model.count_params()
    print(f"{a.method}: {total:,} params, {active:,} active per token", flush=True)

    DATA = DATASETS[a.data]
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    if cfg.type_balance:
        freq = np.bincount(train, minlength=cfg.vocab).astype(np.float64)
        cfg.tok_weight = torch.from_numpy(1.0 / np.maximum(freq, 1.0)).float().to(dev)
    if cfg.router == "inv":
        freq = np.bincount(train, minlength=cfg.vocab).astype(np.float64)
        order = np.argsort(-freq)
        top = np.zeros(cfg.vocab, dtype=bool)
        top[order[:np.searchsorted(np.cumsum(freq[order]), freq.sum() / 2) + 1]] = True
        model.ctx["freq_rank_top"] = torch.from_numpy(top).to(dev)          # ids covering half of tokens
    if cfg.ht != "none" or cfg.emb_init:
        if True:
            import inventions
            src = train
            if cfg.ht_src == "batches":                  # exactly the tokens this run trains on
                r2 = np.random.default_rng(cfg.seed)
                st = [r2.integers(0, len(train) - cfg.seq_len - 1, cfg.batch) for _ in range(a.steps)]
                src = np.concatenate([np.concatenate([train[s:s + cfg.seq_len + 1], [0]]) for ss in st for s in ss])
                cfg.ht_n = len(src)
            if cfg.ht in ("none", "rand_emb"):
                cfg.ht = "rand_emb"
            inventions.build_tables(src, cfg, model)
    rng = np.random.default_rng(cfg.seed)
    if cfg.expert_lr != 1.0:   # routed-expert weights (MoE w1/w2/w3) get lr x expert_lr
        ex = [p for n, p in model.named_parameters() if n.split(".")[-1] in ("w1", "w2", "w3") and p.dim() == 3]
        exid = {id(p) for p in ex}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in exid], mult=1.0),
                  dict(params=ex, mult=cfg.expert_lr)]
        print(f"expert lr x{cfg.expert_lr} on {len(ex)} tensors", flush=True)
    else:
        groups = [dict(params=list(model.parameters()), mult=1.0)]
    opt = torch.optim.Adam(groups, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, a.steps - cfg.warmup)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    moes = model.moe_layers()
    if cfg.seen_c:   # SEEN-COUNT GATE: an id uses the routed experts once it has appeared seen_c times in training
        seen = torch.zeros(cfg.vocab, device=dev)
        cfg.rare_mode = cfg.rare_mode if cfg.rare_mode != "none" else "skip"
        for m in moes:
            m.rare_mask = torch.ones(cfg.vocab, dtype=torch.bool, device=dev)
    t0, step, hist = time.time(), 0, []
    model.train()
    for step in range(a.steps):
        starts = rng.integers(0, len(train) - cfg.seq_len - 1, cfg.batch)
        xy = torch.from_numpy(np.stack([train[s:s + cfg.seq_len + 1] for s in starts]).astype(np.int64)).to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["mult"]
        _, loss = model(xy[:, :-1], xy[:, 1:])
        lm = loss.mean()
        aux, parts = model.aux_loss()
        (lm + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for m in moes:
            m.update_expert_bias()
        if cfg.skip_until and step + 1 == cfg.skip_until:
            cfg.rare_mode = "off"                        # rare ids use the routed experts from here on
        if cfg.skip_from and step == 0:
            cfg._rm, cfg.rare_mode = cfg.rare_mode, "off"
        if cfg.skip_from and step + 1 == cfg.skip_from:
            cfg.rare_mode = cfg._rm
        if cfg.seen_c:
            seen += torch.bincount(xy[:, :-1].reshape(-1), minlength=cfg.vocab).float()
            for m in moes:
                m.rare_mask = seen < cfg.seen_c
        for m in model.modules():
            if hasattr(m, "step_update"):
                m.step_update()
        if cfg.router == "remoe":
            # ReMoE fork training.py: zeroth-order update of the L1 coefficient
            sp = sum(m.last_sparsity for m in moes) / len(moes)
            target = 1 - cfg.topk / cfg.n_experts
            cfg.l1_coeff_state[0] *= cfg.l1_mult if sp < target else 1 / cfg.l1_mult
            parts["sparsity"] = sp
            parts["l1_coeff"] = cfg.l1_coeff_state[0]
        hist.append(dict(step=step, loss=round(lm.item(), 4), **{k: round(v, 6) for k, v in parts.items()}))
        if step % 10 == 0 or step == a.steps - 1:
            print(f"step {step} loss {lm.item():.4f} {parts} {time.time() - t0:.1f}s", flush=True)
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            break
    train_s = time.time() - t0

    if a.split == "val":
        val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)
        if a.data.startswith("fineweb"):
            val = val[:FINEWEB_TEST_SEQ * cfg.seq_len + 1]
    elif a.data.startswith("fineweb"):
        val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)[FINEWEB_TEST_SEQ * cfg.seq_len:]
    else:
        val = np.fromfile(os.path.join(DATA, "test.bin"), dtype=np.uint16)
    res = evaluate(model, val, token_bytes(), cfg, a.eval_seqs, device=dev)
    if cfg.freq_eval:   # analysis only: mean loss (nats) by frequency rank of the input id and the target id
        fr = np.bincount(train[:4_000_000], minlength=cfg.vocab)
        rk = np.empty(cfg.vocab, dtype=np.int64); rk[np.argsort(-fr)] = np.arange(cfg.vocab)
        bk = torch.from_numpy(np.digitize(rk, [256, 1024])).to(dev)            # 0: top 256, 1: 256-1023, 2: rest
        n = min(a.eval_seqs, (len(val) - 1) // cfg.seq_len)
        xy = torch.from_numpy(val[:n * cfg.seq_len + 1].astype(np.int64))
        x, y = xy[:-1].view(n, cfg.seq_len).to(dev), xy[1:].view(n, cfg.seq_len).to(dev)
        model.eval()
        with torch.no_grad():
            l = torch.cat([model(x[i:i + 32], y[i:i + 32])[1].reshape(-1, cfg.seq_len) for i in range(0, n, 32)])
        fe = {}
        for nm, ids in (("in", x), ("tgt", y)):
            b = bk[ids]
            for j, lab in enumerate("FMR"):
                m = b == j
                fe[f"{nm}{lab}"] = round(l[m].mean().item(), 4)
                fe[f"{nm}{lab}_frac"] = round(m.float().mean().item(), 3)
        res["freq_eval"] = fe
        # analysis only: loss by position, by "target already seen in context" (rep) and by
        # "same (input, target) pair already seen earlier in context" (ind, an induction/copy case)
        L = cfg.seq_len
        lo = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev), 0)
        st = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev), -1)
        rep = torch.zeros_like(x, dtype=torch.bool); ind = torch.zeros_like(x, dtype=torch.bool)
        for i in range(0, n, 16):
            xb, yb = x[i:i + 16], y[i:i + 16]
            rep[i:i + 16] = ((yb[:, :, None] == xb[:, None, :]) & lo).any(-1)
            ind[i:i + 16] = ((xb[:, :, None] == xb[:, None, :]) & (yb[:, :, None] == yb[:, None, :]) & st).any(-1)
        pe = {}
        pos = torch.arange(L, device=dev).expand_as(x)
        for lab, a0, a1 in (("p0", 0, 16), ("p1", 16, 64), ("p2", 64, 256), ("p3", 256, 1 << 30)):
            m = (pos >= a0) & (pos < a1)
            if m.any():
                pe[lab] = round(l[m].mean().item(), 4); pe[lab + "_frac"] = round(m.float().mean().item(), 3)
        for lab, m in (("rep1", rep), ("rep0", ~rep), ("ind1", ind), ("ind0", ~ind)):
            pe[lab] = round(l[m].mean().item(), 4); pe[lab + "_frac"] = round(m.float().mean().item(), 3)
        res["pos_eval"] = pe
        # analysis only (mixing 2026-11-23): which exact repeats (ind) are still missed. For each ind target, jp = most
        # recent earlier position with the same (input, target) pair. d = t - jp; m = how many tokens before t also match
        # before jp (0, 1-3, 4+); f = target frequency bucket; last = whether the most recent earlier copy of the input
        # id was followed by the same target (1) or by another token (0, a copy conflict). Fractions are of all tokens.
        ar = torch.arange(L, device=dev)
        me = {}
        acc = {}
        for i in range(0, n, 16):
            xb, yb, lb = x[i:i + 16], y[i:i + 16], l[i:i + 16]
            ex = (xb[:, :, None] == xb[:, None, :]) & st
            ep = ex & (yb[:, :, None] == yb[:, None, :])
            jl = torch.where(ex, ar[None, None, :], -1).amax(-1)              # last earlier copy of the input id
            jp = torch.where(ep, ar[None, None, :], -1).amax(-1)              # last earlier copy of the pair
            hp = jp >= 0
            lastok = (jl >= 0) & (yb.gather(1, jl.clamp(min=0)) == yb)
            d = (ar[None, :] - jp).float()
            mlen = torch.zeros_like(jp)
            alive = hp.clone()
            for kk in range(1, 9):
                a_ = ar[None, :] - kk; b_ = jp - kk
                ok = alive & (a_ >= 0) & (b_ >= 0)
                same = xb.gather(1, a_.clamp(min=0).expand_as(jp)) == xb.gather(1, b_.clamp(min=0))
                alive = ok & same
                mlen = mlen + alive.long()
            groups = {}
            for lab, a0, a1 in (("d0", 0, 16), ("d1", 16, 64), ("d2", 64, 256), ("d3", 256, 1 << 30)):
                groups[lab] = hp & (d >= a0) & (d < a1)
            groups["m0"] = hp & (mlen == 0); groups["m1"] = hp & (mlen >= 1) & (mlen <= 3); groups["m4"] = hp & (mlen >= 4)
            fb = bk[yb]
            for j, lab in enumerate("FMR"):
                groups["f" + lab] = hp & (fb == j)
            groups["last1"] = hp & lastok; groups["last0"] = hp & ~lastok
            groups["nrep_conf"] = ~hp & (jl >= 0)                               # input seen before, never with this target
            for lab, m in groups.items():
                a0 = acc.setdefault(lab, [0.0, 0])
                a0[0] += lb[m].sum().item(); a0[1] += int(m.sum().item())
        tot = n * L
        for lab, (sm, cn) in acc.items():
            me[lab] = round(sm / max(cn, 1), 4); me[lab + "_frac"] = round(cn / tot, 4)
        res["miss_eval"] = me
    row = dict(method=a.method, tag=a.tag, data=a.data, split=a.split, size=a.size, steps=step + 1, train_seconds=round(train_s, 1), threads=a.threads, device=getattr(a, "device", "cpu"),
               tokens=(step + 1) * cfg.batch * cfg.seq_len, params=total, active_params=active,
               peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1),
               gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
               **res, config={k: v for k, v in c.items()}, last_train=hist[-1])
    print(json.dumps({k: v for k, v in row.items() if k != "config"}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
