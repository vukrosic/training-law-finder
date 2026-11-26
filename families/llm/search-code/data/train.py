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
import schedule                          # noqa: E402  search 7: data order + batch schedule

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
            bg_win=1, bg_route="x", bg_mode="cat", bg_emb="none", hg_size=8, hk="uuu", he="none", ht="none", ht_n=4000000, emb_init=False, ht_src="file", ht_g=16, ht_top=1024, rare_mode="none", freq_eval=True, skip_sel="rare", rare_k=0, seen_c=0, skip_until=0, skip_from=0, expert_lr=1.0, emb_var="base", emb_scale=1.0, svd_alpha=0.1, svd_mode="logp", svd_rank=0, svd_side="both", ca=0, cb=0.0, cc="none", pe="none", pe_scale=1.0, pe_gap=2, ex="none", exa=0, exb=0.0, **schedule.DEFAULTS)

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


def pe_init(src, cfg, model):
    """init the previous-token embedding: rows from the skip-bigram log P(x_{t+1} | x_{t-1}) (pe_gap=2)"""
    V, d, gp = cfg.vocab, cfg.d_model, cfg.pe_gap
    sc = 0.02 * cfg.pe_scale
    if cfg.pe == "rand":
        E = torch.randn(V, d, generator=torch.Generator().manual_seed(21))
    else:
        t = np.asarray(src).astype(np.int64)
        Mf = np.zeros(V * V, np.float32)
        np.add.at(Mf, t[:-gp] * V + t[gp:], 1.0)
        M = torch.from_numpy(Mf).view(V, V)
        rs = M.sum(1, keepdim=True)
        M.add_(cfg.svd_alpha).log_().sub_(torch.log(rs + cfg.svd_alpha * V))
        M -= M.mean(1, keepdim=True)
        if cfg.pe == "skip":                      # least squares onto the head: E Wh^T ~ L
            Wh = model.head.weight.detach().float().cpu()
            E = M @ Wh @ torch.linalg.inv(Wh.t() @ Wh)
        else:                                     # own rank-d factor, not aligned to the head
            U, S_, _ = torch.svd_lowrank(M, q=d, niter=3)
            E = U * S_.sqrt()
        del M
    with torch.no_grad():
        model.emb_prev.weight.copy_((E / E.std() * sc).to(model.emb_prev.weight.device))


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
    xtra = None
    if getattr(cfg, "d_xsrc", ""):   # idea path: extra token source (synthetic or other domain), see schedule.extra_tokens
        xtra = schedule.extra_tokens(cfg, train, DATASETS, a.data)
        cfg.d_xoff, cfg.d_xlen = len(train) + 1, len(xtra)
    if cfg.type_balance:
        freq = np.bincount(train, minlength=cfg.vocab).astype(np.float64)
        cfg.tok_weight = torch.from_numpy(1.0 / np.maximum(freq, 1.0)).float().to(dev)
    if cfg.router == "inv":
        freq = np.bincount(train, minlength=cfg.vocab).astype(np.float64)
        order = np.argsort(-freq)
        top = np.zeros(cfg.vocab, dtype=bool)
        top[order[:np.searchsorted(np.cumsum(freq[order]), freq.sum() / 2) + 1]] = True
        model.ctx["freq_rank_top"] = torch.from_numpy(top).to(dev)          # ids covering half of tokens
        rank = np.empty(cfg.vocab, dtype=np.int64); rank[order] = np.arange(cfg.vocab)
        model.ctx["freq_rank"] = torch.from_numpy(rank).to(dev)            # 0 = most frequent id
    plan = schedule.build_plan(train, cfg, a.steps)          # every arm: steps * batch * seq_len target tokens
    cum = np.cumsum([0] + [len(st) * L for st, L in plan])
    n_tok = int(cum[-1])
    assert n_tok == a.steps * cfg.batch * cfg.seq_len, (n_tok, a.steps * cfg.batch * cfg.seq_len)
    if cfg.ht != "none" or cfg.emb_init:
        if True:
            import inventions
            src = train
            if cfg.ht_src == "batches":                  # exactly the tokens this run trains on
                iv = sorted((int(s), int(L)) for ss, L in plan for s in ss)   # chunks of one window are merged
                segs, cs, ce = [], None, None                                    # back into the window (no extra separators)
                for s, L in iv:
                    if cs is not None and L < cfg.seq_len and s == ce - 1 and ce - cs < cfg.seq_len + 1:
                        ce = s + L + 1
                        continue
                    if cs is not None:
                        segs.append((cs, ce))
                    cs, ce = s, s + L + 1
                segs.append((cs, ce))
                src = np.concatenate([np.concatenate([train[a:b], [0]]) for a, b in segs])
                cfg.ht_n = len(src)
            if cfg.ht in ("none", "rand_emb"):
                cfg.ht = "rand_emb"
            inventions.build_tables(src, cfg, model)
            if getattr(cfg, "pe", "none") in ("skip", "skipsvd", "rand"):
                pe_init(src, cfg, model)
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

    def lr_at(s):   # schedule over the token share (same as the bed when every step has batch * seq_len tokens)
        u = cum[s] / (cfg.batch * cfg.seq_len)          # tokens before this step, in bed steps
        if cfg.d_wsd > 0:   # baseline only (MiniCPM WSD): warmup, constant, linear decay to 0 over the last d_wsd share
            if u < cfg.warmup:
                return cfg.lr * (u + 1) / cfg.warmup
            d0 = a.steps * (1 - cfg.d_wsd)
            return cfg.lr if u < d0 else cfg.lr * max(0.0, 1 - (u - d0) / (a.steps - d0))
        if u < cfg.warmup:
            return cfg.lr * (u + 1) / cfg.warmup
        p = (u - cfg.warmup) / max(1, a.steps - cfg.warmup)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    moes = model.moe_layers()
    if cfg.seen_c:   # SEEN-COUNT GATE: an id uses the routed experts once it has appeared seen_c times in training
        seen = torch.zeros(cfg.vocab, device=dev)
        cfg.rare_mode = cfg.rare_mode if cfg.rare_mode != "none" else "skip"
        for m in moes:
            m.rare_mask = torch.ones(cfg.vocab, dtype=torch.bool, device=dev)
    t0, step, hist = time.time(), 0, []
    model.train()
    if cfg.d_online:   # online pure reorder: each step picks its batch from the remaining windows by current loss
        assert all(L == cfg.seq_len and len(st) == cfg.batch for st, L in plan), "online order needs the plain plan"
        pool = list(np.concatenate([st for st, _ in plan]))
        orng = np.random.default_rng(cfg.d_seed)
        n_on = int(round(len(plan) * cfg.d_onf))
    if cfg.d_gns:   # adaptive batch: windows per step = d_gns x simple gradient noise scale (McCandlish 2018), tokens fixed
        assert all(L == cfg.seq_len for st, L in plan), "adaptive batch needs full-length windows"
        gpool = np.concatenate([st for st, _ in plan])
        gstate = dict(pos=0, b=cfg.batch, ns=0.0, ng=0.0, hist=[])
        cum = [0]

        def gen_plan():
            while gstate["pos"] < len(gpool):
                b = min(gstate["b"], len(gpool) - gstate["pos"])
                st = gpool[gstate["pos"]:gstate["pos"] + b]
                gstate["pos"] += b
                cum.append(cum[-1] + b * cfg.seq_len)
                yield st, cfg.seq_len
        plan_iter = gen_plan()
    else:
        plan_iter = plan
    arng = np.random.default_rng(cfg.d_seed + 71)
    if cfg.d_on2 == "adl":   # idea path: loss-driven length warmup; length +8 once the train-loss EMA has not improved for d_onp steps
        tot_, Lf = a.steps * cfg.batch * cfg.seq_len, cfg.seq_len
        astate = dict(L=max(8, int(cfg.d_seq0)), best=1e9, wait=0, loss=None)
        cum = [0]

        def gen_adl():
            used = 0
            while used < tot_:
                if astate["loss"] is not None and astate["L"] < Lf:
                    if astate["loss"] < astate["best"] - 1e-3:
                        astate["best"], astate["wait"] = astate["loss"], 0
                    else:
                        astate["wait"] += 1
                    if astate["wait"] >= cfg.d_onp:
                        astate["L"], astate["best"], astate["wait"] = min(Lf, astate["L"] + int(cfg.d_onstep)), 1e9, 0
                Ls = Lf if used >= cfg.d_seqf * tot_ else astate["L"]
                b = min(cfg.batch, (tot_ - used) // Ls)
                if b == 0:
                    Ls, b = tot_ - used, 1
                st = arng.integers(0, len(train) - Ls - 1, b)
                used += b * Ls
                cum.append(used)
                yield st, Ls
        plan_iter = gen_adl()
    fbuf = []
    shadow = None
    shadow2 = None
    if cfg.d_on2 in ("dis", "prog", "lagl", "pacc", "prel", "ptok", "rband", "sharp", "fut", "flip"):   # idea path: shadow copy of the model (EMA for dis, lagged snapshot for prog)
        import copy
        shadow = copy.deepcopy(model).eval()
        for q in shadow.parameters():
            q.requires_grad_(False)
        if cfg.d_on2 in ("pacc", "fut"):   # second snapshot: older copy (pacc) or extrapolated copy 2*cur - lag (fut)
            shadow2 = copy.deepcopy(shadow).eval()
    rb_ema = np.zeros(int(cfg.d_rbR))
    rare_rank = [None]; uni_lp = [None]
    ov_m = [None]

    step_now = [0]

    def on2_phase(step):
        p0 = cum[step] / n_tok
        if cfg.d_on2pos == "all":
            return True
        if cfg.d_on2pos == "tail":
            return p0 >= 1 - cfg.d_on2f - 1e-12
        return p0 < cfg.d_on2f - 1e-12

    def on2_select(starts, L):   # idea path: pick the step's rows from a d_on2k x pool by a model-based score
        b = len(starts)
        kc = int(cfg.d_on2k) if not cfg.d_on2k1 else max(2, int(round(cfg.d_on2k + (cfg.d_on2k1 - cfg.d_on2k) * cum[step_now[0]] / n_tok)))
        if cfg.d_on2 == "fgt":   # forgetting replay: d_onr share of rows = buffered windows whose loss rose most since trained
            k = max(1, int(round(cfg.d_onr * b)))
            pool = [e for e in fbuf if e[1] == L]
            if len(pool) < k:
                return starts
            ci = arng.choice(len(pool), min(len(pool), k * int(cfg.d_on2k)), replace=False)
            cs = np.array([pool[i][0] for i in ci], dtype=np.int64)
            old = np.array([pool[i][2] for i in ci])
            cx = torch.from_numpy(np.stack([train[s:s + L + 1] for s in cs]).astype(np.int64)).to(dev)
            with torch.no_grad():
                model.eval()
                cur = model(cx[:, :-1], cx[:, 1:])[1].float().mean(1).cpu().numpy()
                model.train()
            take = cs[np.argsort(-(cur - old) * cfg.d_on2dir, kind="stable")[:k]]
            starts = np.array(starts, copy=True)
            starts[:k] = take
            return starts
        if cfg.d_on2 == "rband":   # region bandit: extra pool rows come from corpus regions weighted by their progress EMA
            hi = len(train) - L - 1; R = len(rb_ema)
            w = np.exp((rb_ema - rb_ema.max() if cfg.d_on2dir > 0 else rb_ema.min() - rb_ema) / cfg.d_rbtau)
            rg = arng.choice(R, b * (kc - 1), p=w / w.sum())
            ex = (rg * (hi // R) + arng.integers(0, hi // R, len(rg))).astype(np.int64)
            cand = np.concatenate([np.asarray(starts), ex])
        else:
            cand = np.concatenate([np.asarray(starts), arng.integers(0, len(train) - L - 1, b * (kc - 1))])
        cx = torch.from_numpy(np.stack([train[s:s + L + 1] for s in cand]).astype(np.int64)).to(dev)
        with torch.no_grad():
            model.eval()
            if cfg.d_on2 == "rdiv":   # embedding diversity: farthest-point picks on the mean input embedding (dir -1: nearest)
                f = torch.nn.functional.normalize(model.emb(cx[:, :-1]).float().mean(1), dim=1).cpu().numpy()
                first = int(arng.integers(len(cand)))
                d = 1 - f @ f[first]
                if cfg.d_on2dir < 0:
                    ch = list(np.argsort(d, kind="stable")[:b])
                else:
                    ch, d = [first], d.copy()
                    d[first] = -np.inf
                    while len(ch) < b:
                        j = int(np.argmax(d))
                        ch.append(j)
                        d = np.minimum(d, 1 - f @ f[j])
                        d[ch] = -np.inf
                sel = np.array(ch)
            elif cfg.d_on2 in ("pent", "gnorm", "dis", "prog", "lstr", "spl", "ebal", "ctx", "loss", "lagl", "marg", "acc", "ece", "lmax", "rare", "pacc", "prel", "ptok", "rband", "lnorm", "sharp", "fut", "flip", "cwrong", "alt", "onov", "ediv", "ind", "robust", "pshuf", "depth", "hrank", "uworse"):
                logits, l = model(cx[:, :-1], cx[:, 1:])
                logits, l = logits.float(), l.float()
                if cfg.d_on2 == "pent":    # mean predictive entropy of the model on the window
                    lp = torch.log_softmax(logits, -1)
                    sc = (-(lp.exp() * lp).sum(-1)).mean(1).cpu().numpy() * cfg.d_on2dir
                    sel = np.argsort(-sc, kind="stable")[:b]
                elif cfg.d_on2 == "gnorm":  # output-gradient norm per token: |softmax - onehot|
                    pr = torch.softmax(logits, -1)
                    py = pr.gather(-1, cx[:, 1:, None]).squeeze(-1)
                    sc = ((pr * pr).sum(-1) - 2 * py + 1).clamp_min(0).sqrt().mean(1).cpu().numpy() * cfg.d_on2dir
                    sel = np.argsort(-sc, kind="stable")[:b]
                elif cfg.d_on2 in ("sharp", "fut", "flip"):
                    sl, sll = shadow(cx[:, :-1], cx[:, 1:])
                    if cfg.d_on2 == "sharp":   # loss under a weight-noised copy minus current loss (dir -1: flattest rows)
                        sc = (sll.float().mean(1) - l.mean(1)).cpu().numpy()
                    elif cfg.d_on2 == "flip":  # share of tokens whose top-1 prediction changed since the lagged snapshot
                        sc = (sl.argmax(-1) != logits.argmax(-1)).float().mean(1).cpu().numpy()
                    else:                      # future progress: current loss minus loss of the extrapolated copy 2*cur - lag
                        s2 = shadow2(cx[:, :-1], cx[:, 1:])[1].float()
                        sc = (l.mean(1) - s2.mean(1)).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "alt":   # alternating steps: odd steps take the hardest rows, even steps the easiest
                    sc = l.mean(1).cpu().numpy()
                    sel = np.argsort(-sc * (1 if step_now[0] % 2 else -1), kind="stable")[:b]
                elif cfg.d_on2 == "onov":  # output novelty: KL(row mean prediction || running mean prediction of trained rows)
                    rp = torch.softmax(logits, -1).mean(1)
                    if ov_m[0] is None:
                        ov_m[0] = rp.mean(0)
                    sc = (rp * (torch.log(rp + 1e-9) - torch.log(ov_m[0] + 1e-9))).sum(1).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                    ov_m[0] = 0.9 * ov_m[0] + 0.1 * rp[torch.from_numpy(np.asarray(sel)).to(rp.device)].mean(0)
                elif cfg.d_on2 == "ediv":  # expert diversity: mean entropy of the row's expert-use counts over MoE layers
                    T_ = cx.shape[1] - 1
                    ent = 0
                    for mm in moes:
                        c_ = mm.last_map.reshape(T_, len(cand), -1).float().sum(0)
                        p_ = c_ / c_.sum(1, keepdim=True).clamp_min(1)
                        ent = ent + (-(p_ * torch.log(p_ + 1e-9)).sum(1))
                    sc = ent.cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 in ("marg", "acc", "ece", "cwrong"):
                    pr = torch.softmax(logits, -1)
                    tp = pr.topk(2, -1)
                    hit = (tp.indices[..., 0] == cx[:, 1:]).float()
                    if cfg.d_on2 == "marg":   # top-1 minus top-2 probability (dir -1: least decided rows)
                        sc = (tp.values[..., 0] - tp.values[..., 1]).mean(1)
                    elif cfg.d_on2 == "acc":  # next-token accuracy (dir -1: least accurate rows)
                        sc = hit.mean(1)
                    elif cfg.d_on2 == "cwrong":  # share of confidently wrong tokens (p_max > 0.5 and wrong)
                        sc = ((tp.values[..., 0] > 0.5) & (hit == 0)).float().mean(1)
                    else:                     # calibration gap |p_max - correct| (dir +1: worst calibrated rows)
                        sc = (tp.values[..., 0] - hit).abs().mean(1)
                    sel = np.argsort(-sc.cpu().numpy() * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 in ("ind", "robust", "pshuf", "depth", "hrank", "uworse"):
                    X_, Y_ = cx[:, :-1], cx[:, 1:]
                    if cfg.d_on2 == "ind":     # copy failure: loss on targets already seen in the row minus loss on first-seen targets
                        T_ = X_.shape[1]
                        seen = ((Y_[:, :, None] == X_[:, None, :]) & torch.ones(T_, T_, dtype=torch.bool, device=X_.device).tril()[None]).any(-1).float()
                        lr_ = (l * seen).sum(1) / seen.sum(1).clamp_min(1)
                        ln_ = (l * (1 - seen)).sum(1) / (1 - seen).sum(1).clamp_min(1)
                        sc = torch.where(seen.sum(1) > 0, lr_ - ln_, torch.zeros_like(lr_)).cpu().numpy()
                    elif cfg.d_on2 in ("robust", "pshuf"):
                        g_ = torch.Generator(device="cpu").manual_seed(int(step_now[0]) + 17)
                        Xn = X_.clone().cpu()
                        if cfg.d_on2 == "robust":   # loss with 15% of input tokens replaced by random ids, minus clean loss
                            m_ = torch.rand(Xn.shape, generator=g_) < 0.15
                            Xn[m_] = torch.randint(0, cfg.vocab, (int(m_.sum()),), generator=g_)
                        else:                       # loss with 25% of adjacent input pairs swapped, minus clean loss
                            T_ = Xn.shape[1]
                            for i_ in range(0, T_ - 1, 2):
                                m_ = torch.rand(Xn.shape[0], generator=g_) < 0.5
                                a_ = Xn[m_, i_].clone(); Xn[m_, i_] = Xn[m_, i_ + 1]; Xn[m_, i_ + 1] = a_
                        ln2 = model(Xn.to(dev), Y_)[1].float()
                        sc = (ln2.mean(1) - l.mean(1)).cpu().numpy()
                    elif cfg.d_on2 in ("depth", "hrank"):
                        hk_ = {}
                        bi_ = len(model.blocks) // 2 - 1 if cfg.d_on2 == "depth" else len(model.blocks) - 1
                        hd_ = model.blocks[bi_].register_forward_hook(lambda m, i, o: hk_.__setitem__("h", o))
                        model(X_, Y_)
                        hd_.remove()
                        h_ = hk_["h"].float()
                        if cfg.d_on2 == "depth":    # loss read from the middle layer (logit lens) minus final loss
                            lm_ = torch.nn.functional.cross_entropy(model.head(model.norm(h_)).float().transpose(1, 2), Y_, reduction="none")
                            sc = (lm_.mean(1) - l.mean(1)).cpu().numpy()
                        else:                       # effective rank (singular-value entropy) of the row's last hidden states
                            sv_ = torch.linalg.svdvals(h_ - h_.mean(1, keepdim=True))
                            q_ = sv_ / sv_.sum(1, keepdim=True).clamp_min(1e-9)
                            sc = (-(q_ * torch.log(q_ + 1e-9)).sum(1)).cpu().numpy()
                    else:                           # share of targets where the model loss exceeds the unigram surprisal
                        if uni_lp[0] is None:
                            cnt_ = np.bincount(np.asarray(train[::16]).astype(np.int64), minlength=cfg.vocab).astype(np.float64) + 0.5
                            uni_lp[0] = torch.from_numpy(-np.log(cnt_ / cnt_.sum())).float().to(dev)
                        sc = (l > uni_lp[0][Y_]).float().mean(1).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "lnorm":  # mean L2 norm of the output logits (dir +1: most confident-scale rows)
                    sel = np.argsort(-logits.norm(dim=-1).mean(1).cpu().numpy() * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "lmax":   # peak token loss in the row (dir -1: rows without a surprise spike)
                    sel = np.argsort(-l.max(1).values.cpu().numpy() * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "rare":   # loss on targets outside the d_rareR most frequent tokens (dir +1: hardest rare tokens)
                    if rare_rank[0] is None:
                        cnt_ = np.bincount(np.asarray(train[::16]).astype(np.int64), minlength=65536)
                        rk = np.empty(len(cnt_), np.int64); rk[np.argsort(-cnt_, kind="stable")] = np.arange(len(cnt_))
                        rare_rank[0] = torch.from_numpy(rk).to(dev)
                    m_ = (rare_rank[0][cx[:, 1:]] >= int(cfg.d_rareR)).float()
                    sc = ((l * m_).sum(1) / m_.sum(1).clamp_min(1)).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 in ("pacc", "prel", "ptok", "rband"):
                    sll = shadow(cx[:, :-1], cx[:, 1:])[1].float()
                    if cfg.d_on2 == "pacc":   # progress change: (lag - cur) - (older - lag) (dir -1: slowing rows)
                        s2 = shadow2(cx[:, :-1], cx[:, 1:])[1].float()
                        sc = ((sll.mean(1) - l.mean(1)) - (s2.mean(1) - sll.mean(1))).cpu().numpy()
                    elif cfg.d_on2 == "prel":  # relative progress (lag - cur) / lag
                        sc = ((sll.mean(1) - l.mean(1)) / sll.mean(1).clamp_min(1e-6)).cpu().numpy()
                    elif cfg.d_on2 == "ptok":  # share of tokens whose loss fell since the snapshot
                        sc = (sll > l).float().mean(1).cpu().numpy()
                    else:
                        sc = (sll.mean(1) - l.mean(1)).cpu().numpy()
                        hi_ = len(train) - L - 1; R = len(rb_ema)
                        rr = np.minimum(cand // (hi_ // R), R - 1)
                        for q_ in np.unique(rr):
                            rb_ema[q_] = 0.8 * rb_ema[q_] + 0.2 * float(sc[rr == q_].mean())
                    if cfg.d_on2 == "rband" and cfg.d_rbsel == "rand":
                        sel = arng.choice(len(cand), b, replace=False)
                    else:
                        sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "loss":   # control: current mean loss (dir -1: easiest rows)
                    sc = l.mean(1).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 == "lagl":   # control: loss of the lagged copy alone (dir -1: easiest for the old model)
                    sl, sll = shadow(cx[:, :-1], cx[:, 1:])
                    sc = sll.float().mean(1).cpu().numpy()
                    sel = np.argsort(-sc * cfg.d_on2dir, kind="stable")[:b]
                elif cfg.d_on2 in ("dis", "prog"):
                    sl, sll = shadow(cx[:, :-1], cx[:, 1:])
                    if cfg.d_on2 == "dis":  # KL(current || EMA model) per token
                        lp, lq = torch.log_softmax(logits, -1), torch.log_softmax(sl.float(), -1)
                        sc = (lp.exp() * (lp - lq)).sum(-1).mean(1).cpu().numpy()
                    else:                   # learning progress: lagged-model loss minus current loss
                        sc = (sll.float().mean(1) - l.mean(1)).cpu().numpy()
                    key = -sc * cfg.d_on2dir
                    if cfg.d_ptrim > 0:      # drop the top d_ptrim share of rows by current loss before picking
                        m_ = l.mean(1).cpu().numpy()
                        key = np.where(m_ > np.quantile(m_, 1 - cfg.d_ptrim), np.inf, key)
                    if cfg.d_on2tau > 0:     # softmax sampling instead of top-b
                        w_ = np.exp(-(key - key.min()) / cfg.d_on2tau); w_[~np.isfinite(key)] = 0
                        sel = arng.choice(len(cand), b, replace=False, p=w_ / w_.sum())
                    else:
                        sel = np.argsort(key, kind="stable")[:b]
                elif cfg.d_on2 == "lstr":   # loss-stratified: one row per loss quantile bin (dir -1: one contiguous loss band)
                    o = np.argsort(l.mean(1).cpu().numpy(), kind="stable")
                    kk = len(o)
                    if cfg.d_on2dir < 0:
                        s0 = int(arng.integers(0, kk - b + 1))
                        sel = o[s0:s0 + b]
                    else:
                        edges = np.linspace(0, kk, b + 1).astype(int)
                        sel = np.array([o[int(arng.integers(edges[i], max(edges[i] + 1, edges[i + 1])))] for i in range(b)])
                elif cfg.d_on2 == "spl":    # self-paced: random rows under a loss quantile that rises from d_splq0 to 1 over the phase
                    m = l.mean(1).cpu().numpy()
                    q = cfg.d_splq0 + (1 - cfg.d_splq0) * min(1.0, cum[step_now[0]] / max(1, cfg.d_on2f * n_tok))
                    ok = np.flatnonzero(m <= np.quantile(m, q))
                    sel = ok[arng.permutation(len(ok))[:b]] if len(ok) >= b else np.argsort(m, kind="stable")[:b]
                elif cfg.d_on2 == "ebal":   # expert balance: greedy rows that keep the summed expert load flat (dir -1: most uneven)
                    T_ = cx.shape[1] - 1
                    cnt = torch.cat([mm.last_map.reshape(T_, len(cand), -1).float().sum(0) for mm in moes], 1).cpu().numpy()
                    load, ch, left = np.zeros(cnt.shape[1]), [], list(range(len(cand)))
                    for _ in range(b):
                        nl = load[None, :] + cnt[left]
                        v = nl.var(1) * cfg.d_on2dir
                        j = left.pop(int(np.argmin(v)))
                        ch.append(j); load = load + cnt[j]
                    sel = np.array(ch)
                else:                       # ctx: context gain = loss of the second half alone minus its loss inside the window
                    h = (cx.shape[1] - 1) // 2
                    l2 = model(cx[:, h:-1], cx[:, h + 1:])[1].float()
                    sc = (l2.mean(1) - l[:, h:].mean(1)).cpu().numpy() * cfg.d_on2dir
                    sel = np.argsort(-sc, kind="stable")[:b]
            else:
                l = model(cx[:, :-1], cx[:, 1:])[1].float()
                if cfg.d_on2 == "lvar":   # per-token loss spread inside the window (dir -1: most uniform)
                    sc = l.std(1).cpu().numpy() * cfg.d_on2dir
                else:                      # med: windows whose mean loss is closest to the pool median
                    m = l.mean(1).cpu().numpy()
                    sc = -np.abs(m - np.median(m))
                sel = np.argsort(-sc, kind="stable")[:b]
            model.train()
        return cand[sel]

    def gsq():
        return sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)

    for step, (starts, L) in enumerate(plan_iter):
        if cfg.d_online:
            if step < n_on:
                k = min(len(pool), cfg.batch * int(cfg.d_onk))
                ci = orng.choice(len(pool), k, replace=False)
                cand = np.array([pool[i] for i in ci])
                cx = torch.from_numpy(np.stack([train[s:s + L + 1] for s in cand]).astype(np.int64)).to(dev)
                with torch.no_grad():
                    model.eval()
                    cl = model(cx[:, :-1], cx[:, 1:])[1].reshape(k, -1).mean(1).cpu().numpy()
                    model.train()
                take = np.argsort(-cl * cfg.d_online, kind="stable")[:cfg.batch]   # d_online>0: hardest first
            else:
                ci = orng.choice(len(pool), min(len(pool), cfg.batch), replace=False)
                take = np.arange(len(ci))
                cand = np.array([pool[i] for i in ci])
            starts = cand[take]
            for i in sorted(ci[take].tolist(), reverse=True):
                pool.pop(i)
        step_now[0] = step
        if cfg.d_on2 in ("rdiv", "lvar", "med", "fgt", "pent", "gnorm", "dis", "prog", "lstr", "spl", "ebal", "ctx", "loss", "lagl", "marg", "acc", "ece", "lmax", "rare", "pacc", "prel", "ptok", "rband", "lnorm", "sharp", "fut", "flip", "cwrong", "alt", "onov", "ediv", "ind", "robust", "pshuf", "depth", "hrank", "uworse") \
                and len(starts) > 1 and on2_phase(step):
            starts = on2_select(starts, L)
        if xtra is not None:   # idea path: starts >= d_xoff read the extra source
            xo = cfg.d_xoff
            xy = torch.from_numpy(np.stack([train[s:s + L + 1] if s < xo else xtra[s - xo:s - xo + L + 1] for s in starts]).astype(np.int64)).to(dev)
        else:
            xy = torch.from_numpy(np.stack([train[s:s + L + 1] for s in starts]).astype(np.int64)).to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["mult"]
        if cfg.d_gns and len(xy) >= 2:   # two half-batch backwards: |g_half|^2 and |g_full|^2 give the noise scale
            h = len(xy) // 2
            parts_l = []
            for j, (lo, hi) in enumerate(((0, h), (h, len(xy)))):
                _, loss = model(xy[lo:hi, :-1], xy[lo:hi, 1:])
                lmj = loss.mean()
                aux, parts = model.aux_loss()
                (((hi - lo) / len(xy)) * (lmj + aux)).backward()
                parts_l.append(lmj.detach() * (hi - lo) / len(xy))
                if j == 0:
                    n_small = gsq() * (len(xy) / h) ** 2
            n_big = gsq()
            lm = sum(parts_l)
            bs_s, bs_b = h * L, len(xy) * L                      # tokens in the small and the big batch
            g2 = (bs_b * n_big - bs_s * n_small) / (bs_b - bs_s)
            tr = (n_small - n_big) / (1 / bs_s - 1 / bs_b)
            e = cfg.d_gnsema
            gstate["ng"] = e * gstate["ng"] + (1 - e) * g2
            gstate["ns"] = e * gstate["ns"] + (1 - e) * tr
            ns_, ng_ = gstate["ns"], gstate["ng"]
            bn = ns_ / ng_ if (ns_ > 0 and ng_ > 0) else (1e9 if ns_ > 0 else -1.0)   # simple noise scale in tokens; -1: no estimate
            if step + 1 >= cfg.d_gns0 and bn >= 0:                                      # no estimate: keep the last batch
                gstate["b"] = int(np.clip(round(cfg.d_gns * bn / cfg.seq_len), cfg.d_gnsmin, cfg.d_gnsmax))
            gstate["hist"].append(round(min(bn, 1e5), 1))
        else:
            _, loss = model(xy[:, :-1], xy[:, 1:])
            lm = loss.mean()
            aux, parts = model.aux_loss()
            (lm + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for m in moes:
            m.update_expert_bias()
        if shadow is not None:
            with torch.no_grad():
                if cfg.d_on2 == "dis":
                    for qs, qm in zip(shadow.parameters(), model.parameters()):
                        qs.lerp_(qm, 1 - cfg.d_emad)
                    for bs_, bm_ in zip(shadow.buffers(), model.buffers()):
                        bs_.copy_(bm_)
                elif cfg.d_on2 == "prog" and cfg.d_lagema > 0:   # EMA snapshot instead of a lagged copy
                    for qs, qm in zip(shadow.parameters(), model.parameters()):
                        qs.lerp_(qm, 1 - cfg.d_lagema)
                    for bs_, bm_ in zip(shadow.buffers(), model.buffers()):
                        bs_.copy_(bm_)
                elif (step + 1) % int(cfg.d_lagg) == 0:
                    if cfg.d_on2 == "fut":
                        for q2, qm, qs in zip(shadow2.parameters(), model.parameters(), shadow.parameters()):
                            q2.copy_(2 * qm - qs)
                        for b2, bm_ in zip(shadow2.buffers(), model.buffers()):
                            b2.copy_(bm_)
                    elif shadow2 is not None:
                        shadow2.load_state_dict(shadow.state_dict())
                    shadow.load_state_dict(model.state_dict())
                    if cfg.d_on2 == "sharp":
                        for qs in shadow.parameters():
                            qs.add_(torch.randn_like(qs) * qs.abs() * cfg.d_shs)
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
        if cfg.d_on2 == "adl":
            astate["loss"] = lm.item() if astate["loss"] is None else 0.8 * astate["loss"] + 0.2 * lm.item()
        if cfg.d_on2 == "fgt" and not cfg.d_gns:
            fbuf.extend(zip(np.asarray(starts).tolist(), [L] * len(starts), loss.detach().float().mean(1).cpu().numpy().tolist()))
            del fbuf[:-int(cfg.d_fbuf)]
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
    row = dict(method=a.method, tag=a.tag, data=a.data, split=a.split, size=a.size, steps=step + 1, train_seconds=round(train_s, 1), threads=a.threads, device=getattr(a, "device", "cpu"),
               tokens=int(cum[step + 1]), bed_steps=a.steps, params=total, active_params=active,
               peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1),
               gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
               **res, config={k: v for k, v in c.items()}, last_train=hist[-1],
               gns_hist=(gstate["hist"][::5] if cfg.d_gns else None))
    print(json.dumps({k: v for k, v in row.items() if k != "config"}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
