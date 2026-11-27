"""Train one model from scratch on FineWeb-Edu tokens and score it on the MOE-PRETRAIN benchmark.

    python3 -I env/train.py --method topk_aux --steps 60 --out experiments/runs/mine.jsonl

Methods (laptop presets, matched active parameters per token; see BASELINES.md):
    dense, topk_aux, global_aux, dsv3, expert_choice, remoe
Any preset field can be overridden with --set key=value (e.g. --set aux_coeff=0.001).
Writes one JSON line with the config, val loss, bits per byte and load-balance metrics.
"""
import argparse, contextlib, copy, json, math, os, resource, sys, time, types
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
for _v in (512, 1024, 2048, 4096, 8192):   # vocab-size test (2026-11-24): fast/prep_vocab.py; folder holds its own tokenizer.json, pass --set vocab=V
    DATASETS[f"fineweb_v{_v}"] = os.path.join(HERE, "..", "data", f"fineweb_v{_v}")
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
            bg_win=1, bg_route="x", bg_mode="cat", bg_emb="none", hg_size=8, hk="uuu", he="none", ht="none", ht_n=4000000, emb_init=False, ht_src="file", ht_g=16, ht_top=1024, rare_mode="none", freq_eval=True, skip_sel="rare", rare_k=0, seen_c=0, skip_until=0, skip_from=0, expert_lr=1.0, emb_var="base", emb_scale=1.0, svd_alpha=0.1, svd_mode="logp", svd_rank=0, svd_side="both",
            erule="none", ea=0.5, eb=0.05, eeps=1e-8, opt="base", mlr=3e-3, mom=0.95, b2=0.95, tau=100.0, soap_f=10, mwd=0.0,
            k1=0.0, k2=0.0, k3=0.0, k4=0.0, ks="",
            facts=0, fkmax=256, fseed=1, fwlo=0.0, fwhi=1.0, ffill=0, ffillk=256, ffwlo=-1.0, ffwhi=1.0, ffrand=0, ffrandp=1.0, ffmt=0, ffkind=0, ffw=1.0, flog=0, fprelo=-1.0, fprehi=-1.0, ffprelo=-1.0, ffprehi=-1.0, ffprek=64, fdrift=0, fevery=0, fchoice=0, ftrace=0, fkmin=1, fzero=0, gfreeze=0, gsplice=0, gaplr=1.0, fmarg=0, gckpt='', gstep=0, wlog=0, wsig=0, fsplit=0, fgroups=0, fglo=0.1, fghi=0.6, fgonly=-1, fref=0, gmom=0, gmb=0.0, gb1=0.0, gbs=0, gsgd=0.0, gsgdm=0.9, gnoclip=0, glrx=0.0, zcheap=0, zfree=0, zfdil=0, zrare=0, rsp=0, rsn=0, rsm=4, rsx=1.0, rpfile='', rpadd=0, pfscreen='', rwp=0.45, rww=20, rwhold=0, rwcos=0, sm2file='',
            # 2026-11-24 speedups: sched wsd = warmup, flat lr, linear decay over the last `decay` share of the run;
            # branches "2000,4000" = also finish a decayed copy at those step counts (one row each, fact sightings
            # exact per branch); amp 1 = bf16 autocast on cuda; ce_chunk N = exact loss in chunks of N tokens
            attn="softmax",   # attn=linear: causal linear attention, model.LinearAttention (2026-11-25 replay-ceiling search)
            sched="cos", decay=0.2, branches="", amp=0, ce_chunk=0, prof=0, tf32=0)

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


_CK = {}   # sweep mode (2026-11-25 speed): gap-start checkpoints loaded once per process, deep-copied per variant


def main(argv=None):
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
    ap.add_argument("--device", default="cpu", help="cpu, cuda, mps, or auto (cuda, then mps, then cpu)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
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
    if a.device == "auto":   # NVIDIA GPU, then Mac GPU, then CPU; the row records the device used
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"device {a.device}", flush=True)
    dev = torch.device(a.device)
    model = GPT(cfg).to(dev)
    total, active = model.count_params()
    print(f"{a.method}: {total:,} params, {active:,} active per token", flush=True)

    DATA = DATASETS[a.data]
    TOK0 = os.path.join(HERE, "..", "data", "tokenizer", "bpe8192.json")
    TOKF = os.path.join(DATA, "tokenizer.json") if os.path.exists(os.path.join(DATA, "tokenizer.json")) else TOK0
    if TOKF != TOK0:
        import json as _j; _V = len(_j.load(open(TOKF))["model"]["vocab"])
        assert cfg.vocab == _V, f"--data {a.data} needs --set vocab={_V}"
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
    mopt = None
    if cfg.opt != "base":   # learning search: matrix weights get the rule cfg.opt (opt.py)
        import opt as optmod
        named = optmod.mats_params(model)
        mid = {id(p) for _, p in named}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in mid]
        cfg.T = a.steps
        mopt = optmod.MatOpt(named, cfg, model)
        print(f"matrix rule {cfg.opt} on {len(named)} tensors, mlr {cfg.mlr}", flush=True)
    eopt = None
    if cfg.erule != "none":   # learning search round 10: emb and head rows get the rare-row rule cfg.erule (embopt.py)
        import embopt
        eid = {id(model.emb.weight), id(model.head.weight)}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in eid]
        eopt = embopt.EmbOpt(model.emb.weight, model.head.weight, cfg)
        print(f"row rule {cfg.erule} ea {cfg.ea} eb {cfg.eb} on emb and head", flush=True)
    opt = torch.optim.Adam(groups, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(s, T=None):
        T = T or a.steps
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        if cfg.sched == "wsd":
            d0 = T - int(round(cfg.decay * T))
            if s < d0:
                return cfg.lr
            return cfg.lr * (1 - (1 - cfg.min_lr_frac) * (s - d0 + 1) / (T - d0))
        if cfg.sched == "const":   # 2026-11-25 schedule search: flat lr after warmup
            return cfg.lr
        if cfg.sched == "crw":   # 2026-11-26 lr-clock tests (facts-mac-theory): before P = rwp*T flat peak (rwcos=1: cosine peak -> min);
            P = int(round(cfg.rwp * T)); mn = cfg.lr * cfg.min_lr_frac   # at P drop to min, then over rww steps ramp to peak (rwhold=1: stay at min,
            if s < P:                                                     # then jump to peak); after that flat peak. rwhold=1 + rww >= T = step drop
                if not cfg.rwcos: return cfg.lr
                if cfg.rwcos == 2: return mn   # 2026-11-26 09:55: rwcos=2 = flat at min before P (facts learned at a low flat lr)
                p = (s - cfg.warmup) / max(1, P - cfg.warmup); return mn + (cfg.lr - mn) * 0.5 * (1 + math.cos(math.pi * p))
            if s < P + cfg.rww:
                return mn if cfg.rwhold else mn + (cfg.lr - mn) * (s - P + 1) / cfg.rww
            return cfg.lr
        if cfg.sched == "cosrw":   # 2026-11-25 schedule search: cosine to min over [0, P), P = rwp*T; then re-warm
            P = int(round(cfg.rwp * T))   # linearly from min to peak over rww steps and cosine to min again at T
            mn = cfg.lr * cfg.min_lr_frac
            if s < P:
                p = (s - cfg.warmup) / max(1, P - cfg.warmup)
                return mn + (cfg.lr - mn) * 0.5 * (1 + math.cos(math.pi * p))
            if s < P + cfg.rww:
                return mn + (cfg.lr - mn) * (s - P + 1) / cfg.rww
            p = (s - P - cfg.rww) / max(1, T - P - cfg.rww)
            return mn + (cfg.lr - mn) * 0.5 * (1 + math.cos(math.pi * p))
        assert cfg.sched == "cos", cfg.sched
        p = (s - cfg.warmup) / max(1, T - cfg.warmup)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    moes = model.moe_layers()
    seen, bio_tok = None, 0
    if cfg.seen_c:   # SEEN-COUNT GATE: an id uses the routed experts once it has appeared seen_c times in training
        seen = torch.zeros(cfg.vocab, device=dev)
        cfg.rare_mode = cfg.rare_mode if cfg.rare_mode != "none" else "skip"
        for m in moes:
            m.rare_mask = torch.ones(cfg.vocab, dtype=torch.bool, device=dev)
    brs = sorted(int(b) for b in str(cfg.branches).split(",") if str(b).strip()) if cfg.branches not in ("", None, 0) else []
    if brs:
        assert cfg.sched == "wsd" and all(cfg.warmup < b - int(round(cfg.decay * b)) and b < a.steps for b in brs), brs
        assert cfg.router == "dense" and cfg.erule == "none" and not (cfg.seen_c or cfg.skip_until or cfg.skip_from or cfg.fevery), \
            "branches support dense runs without row rules, gates or curves"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (cfg.amp and dev.type == "cuda") else contextlib.nullcontext
    if cfg.tf32 and dev.type == "cuda":   # opt-in (2026-11-24): fp32 matmuls on Ampere tensor cores, 10-bit mantissa inputs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    t0, step, hist = time.time(), 0, []
    model.train()
    fw = None
    fws = []
    _PN = [n_ for n_, _ in model.named_parameters()]   # fsplit groups: E embeddings, H head, h hidden matrices (Muon), o the rest (norm gains)
    _GP = {n_: "E" if "emb" in n_ else "H" if "head" in n_ else "h" if w.ndim == 2 else "o" for n_, w in model.named_parameters()}
    if cfg.facts:   # facts search (Vuk 2026-11-23): synthetic bios with sighting bins replace web rows
        import facts as factsmod
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(TOKF)
        bins = [0] if cfg.fzero else [cfg.fkmin * 2 ** i for i in range(int(math.log2(cfg.fkmax // cfg.fkmin)) + 1)]   # fzero (spacing-learn 2026-11-25): no bio sightings, every exposure comes from rpfile
        rs_starts = []
        if cfg.rsp and cfg.rsn:   # 2026-11-25 R7 replay-spacing test: after the window, re-show the learned facts rsn times
            # (rsm consecutive steps, one bio per person per step). rsp 1: equal S = sum lr/peak gaps; 2: equal step gaps.
            T_, w1_ = a.steps, max(int(cfg.fwlo * a.steps) + 1, int(cfg.fwhi * a.steps)); n_, m_ = int(cfg.rsn), int(cfg.rsm)
            if int(cfg.rsp) == 2:
                gs_ = (T_ - w1_ - n_ * m_) / (n_ + 1)
                rs_starts = [int(round(w1_ + (j + 1) * gs_ + j * m_)) for j in range(n_)]
            else:
                rel_ = [lr_at(t) / cfg.lr for t in range(T_)]
                x_ = float(cfg.rsx) if int(cfg.rsp) == 3 else 1.0   # rsp 3: expanding S gaps, gap_j = gap_0 * rsx^j (R7b)
                def walk(g):
                    t, st = w1_, []
                    for j_ in range(n_):
                        acc = 0.0
                        while t < T_ and acc + rel_[t] <= g * x_ ** j_: acc += rel_[t]; t += 1
                        st.append(t); t += m_
                    return st, sum(rel_[t:T_]) if t < T_ else 0.0
                lo, hi = 0.0, sum(rel_[w1_:])
                for _ in range(60):
                    g = (lo + hi) / 2; st, rest = walk(g)
                    lo, hi = (g, hi) if rest > g * x_ ** n_ else (lo, g)
                rs_starts = walk((lo + hi) / 2)[0]
            print(f"resight starts {rs_starts} (rsp {cfg.rsp}, {m_} steps each)", flush=True)
        fw = factsmod.FactWorld(tk, cfg.facts, bins, a.steps, seed=cfg.fseed, win=(cfg.fwlo, cfg.fwhi),
                                n_filler=cfg.ffill, filler_k=cfg.ffillk,
                                fwin=None if cfg.ffwlo < 0 else (cfg.ffwlo, cfg.ffwhi), frand=int(cfg.ffrand), frandp=float(cfg.ffrandp), ffmt=int(cfg.ffmt),
                                ffkind=int(cfg.ffkind), web=train if int(cfg.ffkind) == 3 else None,
                                segs=(brs + [a.steps]) if brs else None, pre=None if cfg.fprelo < 0 else (cfg.fprelo, cfg.fprehi),
                                fpre=None if cfg.ffprelo < 0 else (cfg.ffprelo, cfg.ffprehi, int(cfg.ffprek)),
                                resight=(rs_starts, int(cfg.rsm)) if rs_starts else None,
                                replay=json.load(open(cfg.rpfile))["steps"] if cfg.rpfile else None, n_ref=int(cfg.fref))   # fref: O76 never-seen reference people
        fws = [fw]   # worlds whose bios are planted; fw stays the primary world for every older measure
        if cfg.fgroups:   # fgroups=G (2026-11-26, facts-paper-writer, O54): G independent 64-fact groups, own people and values
            # (seed fseed+j), each planted in its own window of length fwhi-fwlo, starts evenly spaced so all windows lie in
            # [fglo, fghi]; fgonly=j plants only group j (same people, same window) for the alone-vs-together check
            assert not (brs or cfg.ffill or cfg.fprelo >= 0 or cfg.rsp or cfg.rpfile or cfg.flog or cfg.ffw != 1.0 or cfg.rpadd
                        or cfg.gckpt or cfg.sm2file or cfg.gsplice or cfg.fdrift or cfg.pfscreen), "fgroups: plain fact runs only"
            L_ = cfg.fwhi - cfg.fwlo
            _all = [factsmod.FactWorld(tk, cfg.facts, bins, a.steps, seed=cfg.fseed + j, win=(float(s_), float(s_) + L_), n_ref=int(cfg.fref))
                    for j, s_ in enumerate(np.linspace(cfg.fglo, cfg.fghi - L_, int(cfg.fgroups)))]
            fws = [_all[int(cfg.fgonly)]] if cfg.fgonly >= 0 else _all
            fw = fws[0]
        fcurve, ftr_steps, ftr_bits = [], [], []
        flog_rows = []
        need = sum(w_.rows_needed(cfg.seq_len) for w_ in fws)   # fgroups: upper bound (windows may overlap)
        print(f"facts: {len(fw.names)} people, bins {bins}, {len(fw.occ)} bios, ~{need} of {cfg.batch} rows per step", flush=True)
        assert need < max(cfg.batch, cfg.gbs), "too many bios per step"   # 2026-11-26: phase-2 gbs rows allow more replay rows
        bio_tok = 0
    PT = {}   # prof=1: seconds per phase (cuda-synced), for the speed audit

    def tick(name, t):
        if cfg.prof:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            n = time.perf_counter()
            PT[name] = PT.get(name, 0.0) + n - t
            return n
        return t

    def one_step(step, T_end, trunk=True):
        nonlocal bio_tok, seen
        tp = tick("other", time.perf_counter()) if cfg.prof else 0.0
        lw = None
        fm = None   # flog (2026-11-24, theory of forgetting): per-step loss on filler and fact tokens
        nb = cfg.gbs if (cfg.gbs and fw is not None and step >= fw.w1) else cfg.batch   # 2026-11-26 phase-2 batch rows (count vs share test)
        starts = rng.integers(0, len(train) - cfg.seq_len - 1, nb)
        xy = np.stack([train[s:s + cfg.seq_len + 1] for s in starts]).astype(np.int64)
        if fw is not None:
            bt = fw.step_bios(step) if len(fws) == 1 else [t_ for w_ in fws for t_ in w_.step_bios(step)]
            if bt and cfg.rpadd:   # 2026-11-26 r8 control arm b: fact rows ADDED on top of the nb web rows (no web text displaced);
                W = cfg.seq_len + 1   # extra rows' web tail is masked; loss weight per token = 1/(nb*seq_len) exactly as when they replace web rows
                assert cfg.ffw == 1.0 and not cfg.flog, "rpadd: no ffw/flog"
                r = -(-len(bt) // W)
                st2 = np.random.default_rng([cfg.seed, step]).integers(0, len(train) - W, r)
                ex = np.stack([train[s:s + W] for s in st2]).astype(np.int64).reshape(-1)
                ex[:len(bt)] = bt; ex = ex.reshape(r, W)
                mk = np.concatenate([(np.arange(r * W).reshape(r, W)[:, 1:] < len(bt)).astype(np.float32), np.ones((xy.shape[0], W - 1), np.float32)])
                xy = np.concatenate([ex, xy])
                lw = torch.from_numpy(mk * (xy.shape[0] / nb)).to(dev)
                bio_tok += len(bt); bt = []
            if bt:
                W = cfg.seq_len + 1
                r = -(-len(bt) // W)
                buf = xy[:r].reshape(-1).copy()        # tail of the last bio row stays web text
                buf[:len(bt)] = bt
                xy[:r] = buf.reshape(r, W)
                if cfg.ffw != 1.0:   # filler loss weight (2026-11-24): 0 = filler tokens give no learning signal
                    wb = np.ones(xy.size, dtype=np.float32); wb[:len(bt)] = np.where(np.array(fw.fmask) > 0, cfg.ffw, 1.0)
                    lw = torch.from_numpy(wb.reshape(xy.shape)[:, 1:]).to(dev)
                bio_tok += len(bt)
                if cfg.flog:
                    mb = np.zeros(xy.size, dtype=np.float32); mb[:len(bt)] = np.array(fw.fmask, dtype=np.float32) + 2.0 * (1 - np.array(fw.fmask, dtype=np.float32))
                    fm = mb.reshape(xy.shape)[:, 1:]
            if trunk and cfg.fevery and (step + 1) % max(1, a.steps // cfg.fevery) == 0 and step + 1 < a.steps:
                e = fw.evaluate(model, dev, a.steps)
                if cfg.ftrace:     # per-fact value bits; weights here have had `step` updates
                    ftr_steps.append(step); ftr_bits.append(e["per_fact_bits"])
                fcurve.append({"step": step + 1, "acc_all": e["acc_all"], "acc_by_k": e["acc_by_k"], "per_fact_ok": e["per_fact_ok"]})
                if cfg.fchoice:
                    fcurve[-1].update(fw.choice_eval(model, dev))
        gp = cfg.ffwlo > 0 and step >= cfg.ffwlo * a.steps   # 2026-11-25 learning-vs-forgetting test: inside the gap
        if cfg.gsplice and gp and fw is not None and bt:   # keep the gap batches to score them before / after the gap
            mb2 = np.zeros(xy.size, dtype=np.int8); mb2[:len(bt)] = 1
            _dr.setdefault("gb", []).append((xy.copy(), mb2.reshape(xy.shape)[:, 1:].copy()))
        if cfg.zfree and fw is not None and trunk and fw.w1 - 8 <= step < fw.w1:   # 2026-11-25 R6 fact-free screen: keep the last 8 training batches
            rr = np.zeros(xy.shape[0], dtype=np.int8); rr[:(-(-len(bt) // (cfg.seq_len + 1)) if bt else 0)] = 1   # rows holding bio text (diagnostic only)
            _dr.setdefault("rb", []).append((xy.copy(), rr))
        xy = torch.from_numpy(xy).to(dev)
        tp = tick("data", tp)
        glr = (cfg.gaplr if gp else 1.0) * (cfg.glrx if (cfg.glrx and fw is not None and step >= fw.w1) else 1.0)   # glrx (2026-11-26): phase-2 lr multiplier, batch-size lr scaling test
        gap = cfg.gfreeze and step >= cfg.ffwlo * a.steps   # 2026-11-25 mechanism test: 1 = freeze embeddings/head/norms (Adam group) in the gap, 2 = freeze hidden matrices (Muon)
        for g in opt.param_groups:
            g["lr"] = 0.0 if gap and cfg.gfreeze == 1 else lr_at(step, T_end) * g["mult"] * glr
        spl = None
        if cfg.fsplit and fw is not None and trunk:   # fsplit=1 (2026-11-26, facts-paper-writer, 13:00 order): after a world's window,
            # gradient of the learned facts' mean value-token loss (nats) at the weights before this step's update; dot with the
            # actual update below gives the first-order part of the loss change; no rng, no effect on training
            spl = []
            for j_, w_ in enumerate(fws):
                if step >= w_.w1:
                    pp_ = [q for q in range(w_.n_eval) if w_.k[q] >= max(w_.bins) // 2]
                    g_ = w_.fact_grad(model, dev, _PN, only=[(q, ai) for q in pp_ for ai in range(5)], sep=True)
                    spl.append((j_, g_, w_.last_fact_loss))
            spl = spl or None
        with amp():
            _, loss = model(xy[:, :-1], xy[:, 1:])
        lm = (loss * lw).mean() if lw is not None else loss.mean()
        if fm is not None:   # filler tokens marked 1, fact tokens 2, web 0; sums in nats
            with torch.no_grad():
                L_ = loss.detach().float().cpu().numpy()
                flog_rows.append([step, float(L_[fm == 1].sum()), int((fm == 1).sum()), float(L_[fm == 2].sum()), int((fm == 2).sum()), float(L_[fm == 0].mean()), lr_at(step, T_end)])
        aux, parts = model.aux_loss()
        tp = tick("fwd", tp)
        (lm + aux).backward()
        tp = tick("bwd", tp)
        pb_ = {n_: w.detach().float().clone() for n_, w in model.named_parameters()} if spl else None
        sg = _dr.get("sgd")   # 2026-11-26 phase-2 plain SGD with momentum: no clipping, no update normalisation
        if sg is None and not (cfg.gnoclip and fw is not None and step >= fw.w1):   # gnoclip: no clipping in phase 2
            _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            if os.environ.get("GNLOG") and fw is not None and step >= fw.w1: print(f"GNLOG {step - fw.w1} {float(_gn):.3f}", flush=True)   # 2026-11-26 pre-clip grad norm per phase-2 step (diagnostic)
        if gap and cfg.gfreeze in (3, 4):   # 3 = keep the fact pieces' emb/head rows fixed in the gap, 4 = keep every other row fixed
            rid = torch.tensor(fw.fp_all, device=dev)
            if cfg.gfreeze == 4:
                m_ = torch.ones(model.emb.weight.shape[0], dtype=torch.bool, device=dev); m_[rid] = False; rid = m_.nonzero().squeeze(1)
            keep = [(w, w.data[rid].clone()) for w in (model.emb.weight, model.head.weight)]
        if gap and cfg.gfreeze == 5:   # 5 = keep every Adam-group tensor except emb/head fixed (norm gains, biases)
            eh = {id(model.emb.weight), id(model.head.weight)}
            keep5 = [(p, p.data.clone()) for g in opt.param_groups for p in g["params"] if id(p) not in eh]
        if sg is not None:
            for g_ in sg.param_groups: g_["lr"] = lr_at(step, T_end) / cfg.lr * cfg.gsgd
            sg.step(); sg.zero_grad(set_to_none=True)
        else:
            opt.step()
        if gap and cfg.gfreeze in (3, 4):
            for w, v in keep:
                w.data[rid] = v
        if gap and cfg.gfreeze == 5:
            for p, v in keep5:
                p.data.copy_(v)
        tp = tick("clip_adam", tp)
        if eopt is not None:
            eopt.lr = lr_at(step, T_end)
            eopt.set_batch(xy)
            eopt.step()
            eopt.zero_grad()
        if mopt is not None and sg is None:
            for g in mopt.param_groups:
                g["lr"] = 0.0 if gap and cfg.gfreeze == 2 else lr_at(step, T_end) / cfg.lr * g["base"] * glr
            mopt.step()
            tp = tick("muon", tp)
        if spl:   # u = weights before - weights after (every optimizer, clipping and freeze included), as p.sub_(u) in formula/sim
            P_ = dict(model.named_parameters()); un2 = unh2 = 0.0; U_ = {}
            for n_, w in P_.items():
                U_[n_] = pb_[n_] - w.detach().float(); q_ = float(U_[n_].pow(2).sum()); un2 += q_
                if _GP[n_] == "h": unh2 += q_
            FS = _dr.setdefault("fs", {})
            for j_, g_, ce_ in spl:
                d_ = {"h": 0.0, "E": 0.0, "H": 0.0, "o": 0.0}; g2 = gh2 = 0.0
                for n_ in P_:
                    d_[_GP[n_]] += float((g_[n_] * U_[n_]).sum()); q_ = float(g_[n_].pow(2).sum()); g2 += q_
                    if _GP[n_] == "h": gh2 += q_
                r_ = FS.setdefault(j_, {"t0": step, "dot": [], "doth": [], "dotH": [], "dotE": [], "doto": [], "un": [], "unh": [], "g": [], "gh": [], "ce": [], "lr": []})
                for k_, v_ in (("dot", sum(d_.values())), ("doth", d_["h"]), ("dotH", d_["H"]), ("dotE", d_["E"]), ("doto", d_["o"]),
                               ("un", un2 ** 0.5), ("unh", unh2 ** 0.5), ("g", g2 ** 0.5), ("gh", gh2 ** 0.5), ("ce", ce_), ("lr", lr_at(step, T_end))):
                    r_[k_].append(float(f"{v_:.6g}"))
            del pb_, U_, spl
        opt.zero_grad(set_to_none=True)
        if mopt is not None:
            mopt.zero_grad(set_to_none=True)
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
        tp = tick("other", tp)
        hist.append(dict(step=step, loss=round(lm.item(), 4), **{k: round(v, 6) for k, v in parts.items()}))
        if step % 10 == 0 or step == a.steps - 1:
            print(f"step {step} loss {lm.item():.4f} {parts} {time.time() - t0:.1f}s", flush=True)
        return bool(a.max_seconds and time.time() - t0 > a.max_seconds)

    _dr = {}   # fdrift (2026-11-25): how far the hidden weights move after the fact window ends (theory test)

    def _gaploss():   # mean loss (nats) on the gap batches: [gap-text tokens, web tokens]
        model.eval(); tot = [0.0, 0, 0.0, 0]
        with torch.no_grad():
            for xb, mk in _dr["gb"]:
                for i0 in range(0, xb.shape[0], 16):
                    x_ = torch.from_numpy(xb[i0:i0 + 16]).to(dev)
                    lg = model(x_[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                    l_ = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), x_[:, 1:].reshape(-1), reduction="none").cpu().numpy()
                    m_ = mk[i0:i0 + 16].reshape(-1)
                    tot[0] += float(l_[m_ == 1].sum()); tot[1] += int((m_ == 1).sum()); tot[2] += float(l_[m_ == 0].sum()); tot[3] += int((m_ == 0).sum())
        model.train()
        return [round(tot[0] / max(1, tot[1]), 5), round(tot[2] / max(1, tot[3]), 5), tot[1]]

    def _margin_probe(endw, hid):
        P = dict(model.named_parameters()); sp = _dr["sp"]
        lp = [q for q in range(fw.n_eval) if fw.k[q] >= max(fw.bins) // 2]
        items = [(fw.sep + fw.prompt[q][a_][0] + fw.prompt[q][a_][1], len(fw.sep) + len(fw.prompt[q][a_][0]), q * 5 + a_) for q in lp for a_ in range(5)]
        T_ = max(len(c[0]) for c in items)
        x_ = torch.zeros(len(items), T_ - 1, dtype=torch.long); y_ = torch.full((len(items), T_ - 1), -1, dtype=torch.long)
        for j_, (ids, npr, _) in enumerate(items):
            t_ = torch.tensor(ids); x_[j_, :len(ids) - 1] = t_[:-1]; y_[j_, npr - 1:len(ids) - 1] = t_[npr:]
        x_, y_ = x_.to(dev), y_.to(dev); msk = y_ >= 0
        fid = torch.tensor([c[2] for c in items], device=dev).unsqueeze(1).expand_as(y_)[msk]

        def setw(base, d=None, e=0.0):
            with torch.no_grad():
                for n_, w in P.items():
                    w.copy_(base[n_] + e * d[n_] if d is not None and n_ in d else base[n_])

        def margins():
            model.eval()
            with torch.no_grad():
                lg = model(x_); lg = (lg[0] if isinstance(lg, tuple) else lg).float()
                zy = lg.gather(-1, y_.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                lo = lg.scatter(-1, y_.clamp(min=0).unsqueeze(-1), float("-inf")).logsumexp(-1)
            model.train()
            return (zy - lo)[msk]

        def ddir(d, e=0.01):   # directional derivative of every margin along d at gap-start weights
            setw(sp, d, e); a1 = margins(); setw(sp, d, -e); a2 = margins()
            return (a1 - a2) / (2 * e)

        nrm = lambda d: sum(float(v.float().pow(2).sum()) for v in d.values()) ** 0.5
        dW = {n_: endw[n_] - sp[n_] for n_ in P}
        dH = {n_: v for n_, v in dW.items() if id(P[n_]) in hid}; dA = {n_: v for n_, v in dW.items() if id(P[n_]) not in hid}
        out = {"fid": fid.tolist(), "dWn": [round(nrm(dW), 5), round(nrm(dH), 5), round(nrm(dA), 5)]}
        setw(sp); out["m_pre"] = margins()
        for t_ in (0.25, 0.5, 0.75):
            setw(sp, dW, t_); out[f"m_t{t_}"] = margins()
        setw(endw); out["m_end"] = margins()
        out["lin_all"], out["lin_hid"], out["lin_adam"] = ddir(dW), ddir(dH), ddir(dA)
        # cheap directions from one backward at gap-start weights: G = gradient of the gap-text-only loss on the gap batches
        setw(sp); model.zero_grad(set_to_none=True); ntk = sum(int(mk.sum()) for _, mk in _dr.get("gb", []))
        for xb, mk in _dr.get("gb", []):
            for i0 in range(0, xb.shape[0], 16):
                xx = torch.from_numpy(xb[i0:i0 + 16]).to(dev); mm = torch.from_numpy(mk[i0:i0 + 16]).to(dev).reshape(-1).float()
                if mm.sum() == 0: continue
                lg = model(xx[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                l_ = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="none")
                ((l_ * mm).sum() / max(1, ntk)).backward()
        G = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}
        model.zero_grad(set_to_none=True)
        gq = torch.Generator(device="cpu").manual_seed(5)

        def msign(M):
            if M.ndim != 2 or M.abs().sum() == 0: return torch.sign(M)
            U, _, V = torch.linalg.svd(M.float(), full_matrices=False); return (U @ V).to(M.dtype)
        unit = lambda d: (lambda n0: {k_: v / max(n0, 1e-12) for k_, v in d.items()})(nrm(d))
        dirs = {"g_raw": G,
                "g_ms": {n_: (msign(v) if id(P[n_]) in hid else torch.sign(v)) for n_, v in G.items()},
                "rnd": {n_: (msign(torch.randn(v.shape, generator=gq).to(v.device)) if id(P[n_]) in hid else torch.sign(torch.randn(v.shape, generator=gq)).to(v.device)) for n_, v in G.items()},
                "dW": dW}
        for nm_, d in dirs.items():
            u = unit(d); out["u_" + nm_] = ddir(u, 0.01 * max(1.0, nrm(dW)))
            uh = unit({k_: v for k_, v in d.items() if id(P[k_]) in hid}); out["uh_" + nm_] = ddir(uh, 0.01 * max(1.0, nrm(dH)))
            ua = unit({k_: v for k_, v in d.items() if id(P[k_]) not in hid}); out["ua_" + nm_] = ddir(ua, 0.01 * max(1.0, nrm(dA)))
        out["Gn"] = [round(nrm(G), 6), round(nrm({k_: v for k_, v in G.items() if id(P[k_]) in hid}), 6)]
        # 2026-11-25 (Paper CEO relay) cosine of the gap-text gradient G with the learned facts' own loss gradient and with the
        # gradient of their summed answer margins, at gap-start weights; [all params, hidden, Adam group]
        setw(sp)
        gF = fw.fact_grad(model, dev, list(P.keys()), kmin=max(fw.bins) // 2); out["fact_loss_pre"] = round(fw.last_fact_loss, 5)
        model.zero_grad(set_to_none=True); model.eval()
        lg = model(x_); lg = (lg[0] if isinstance(lg, tuple) else lg).float()
        zy = lg.gather(-1, y_.clamp(min=0).unsqueeze(-1)).squeeze(-1); lo = lg.scatter(-1, y_.clamp(min=0).unsqueeze(-1), float("-inf")).logsumexp(-1)
        ((zy - lo)[msk].sum() / max(1, int(msk.sum()))).backward(); model.train()
        gM = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}
        model.zero_grad(set_to_none=True)
        def cosg(a, b, sel):
            ks = [k_ for k_ in P if sel(k_)]
            na = sum(float(a[k_].pow(2).sum()) for k_ in ks) ** 0.5; nb = sum(float(b[k_].pow(2).sum()) for k_ in ks) ** 0.5
            return round(sum(float((a[k_] * b[k_]).sum()) for k_ in ks) / max(1e-20, na * nb), 6), round(nb, 6)
        for nm_, gg in (("cosF", gF), ("cosM", gM)):
            out[nm_] = [cosg(G, gg, lambda k_: True)[0], cosg(G, gg, lambda k_: id(P[k_]) in hid)[0], cosg(G, gg, lambda k_: id(P[k_]) not in hid)[0]]
            out[nm_ + "_dW"] = [cosg(dW, gg, lambda k_: True)[0], cosg(dW, gg, lambda k_: id(P[k_]) in hid)[0], cosg(dW, gg, lambda k_: id(P[k_]) not in hid)[0]]
            out[nm_ + "_n"] = cosg(G, gg, lambda k_: True)[1]
        out["cos_G_dW"] = round(sum(float((G[k_] * dW[k_]).sum()) for k_ in P) / max(1e-12, nrm(G) * nrm(dW)), 5)
        if cfg.zcheap:   # 2026-11-25 cheapest screen (facts-scale-theory): cosF from fewer gap sequences, fewer fact questions, per tensor
            setw(sp); zc = {"t": {k_: [round(float((G[k_] * gF[k_]).sum()), 9), round(float(G[k_].pow(2).sum()), 9), round(float(gF[k_].pow(2).sum()), 9)] for k_ in P}}
            for nseq in (8, 32, 128):
                model.zero_grad(set_to_none=True); used = 0
                for xb, mk in _dr.get("gb", []):
                    for i0 in range(0, xb.shape[0], 8):
                        if used >= nseq: break
                        xx = torch.from_numpy(xb[i0:i0 + 8]).to(dev); mm = torch.from_numpy(mk[i0:i0 + 8]).to(dev).reshape(-1).float()
                        if mm.sum() == 0: continue
                        lg = model(xx[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                        l_ = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="none")
                        (l_ * mm).sum().backward(); used += xx.shape[0]
                Gs = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}
                zc[f"G{nseq}"] = [cosg(Gs, gF, lambda k_: True)[0], used]
            model.zero_grad(set_to_none=True)
            for nq in (5, 10, 20, 40):
                gq_ = fw.fact_grad(model, dev, list(P.keys()), kmin=max(fw.bins) // 2, nmax=nq)
                zc[f"F{nq}"] = cosg(G, gq_, lambda k_: True)[0]
            model.zero_grad(set_to_none=True); out["zc"] = zc
        if cfg.zfree:   # 2026-11-25 R6 (facts-scale-theory): reference gradients that need no fact list, at gap-start weights
            setw(sp); rb = _dr.get("rb") or []
            assert len(rb) == 8, "zfree needs the 8 stored pre-gap batches (new gckpt)"
            def lgrad(rows):   # summed token loss gradient over rows (n, T+1)
                model.zero_grad(set_to_none=True)
                for i0 in range(0, rows.shape[0], 16):
                    xx = torch.from_numpy(rows[i0:i0 + 16]).to(dev)
                    lg = model(xx[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                    torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="sum").backward()
                g = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}
                model.zero_grad(set_to_none=True); return g
            allr = np.concatenate([x for x, _ in rb]); isb = np.concatenate([m for _, m in rb]).astype(bool)
            model.eval(); rl = []
            with torch.no_grad():
                for i0 in range(0, allr.shape[0], 32):
                    xx = torch.from_numpy(allr[i0:i0 + 32]).to(dev); lg = model(xx[:, :-1]); lg = (lg[0] if isinstance(lg, tuple) else lg).float()
                    rl.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="none").reshape(xx.shape[0], -1).mean(1).cpu())
            model.train(); rl = torch.cat(rl).numpy(); od = np.argsort(rl); web = np.where(~isb)[0]; odw = web[np.argsort(rl[web])]
            nl = rb[-1][0].shape[0]; last = np.arange(allr.shape[0] - nl, allr.shape[0])
            refs = {"last": last, "last_web": last[~isb[last]], "last8": np.arange(allr.shape[0]), "low32": od[:32], "low8": od[:8], "low32_web": odw[:32]}
            if cfg.zfdil:   # dilution control: all web rows of the 8 batches + only 8 bio rows (one per batch), ~2.5% bio, as in a real batch
                bi = np.where(isb)[0]; pick = np.array([bi[(bi >= j * nl) & (bi < (j + 1) * nl)][0] for j in range(8) if ((bi >= j * nl) & (bi < (j + 1) * nl)).any()])
                pool = np.concatenate([web, pick]); odp = pool[np.argsort(rl[pool])]
                refs.update(dil_all=pool, dil_low8=odp[:8], dil_low32=odp[:32])
            model.zero_grad(set_to_none=True); used = 0
            for xb, mk in _dr.get("gb", []):
                for i0 in range(0, xb.shape[0], 8):
                    if used >= 8: break
                    xx = torch.from_numpy(xb[i0:i0 + 8]).to(dev); mm = torch.from_numpy(mk[i0:i0 + 8]).to(dev).reshape(-1).float()
                    if mm.sum() == 0: continue
                    lg = model(xx[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                    (torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="none") * mm).sum().backward(); used += xx.shape[0]
            G8 = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}; model.zero_grad(set_to_none=True)
            w2 = [k_ for k_ in P if k_.endswith("blocks.0.ffn.w2.weight") or k_.endswith("blocks.0.ffn.w2")]
            zf = {"rowloss": [round(float(rl.mean()), 4), round(float(rl[isb].mean()) if isb.any() else -1, 4), round(float(rl[~isb].mean()), 4)], "w2": w2}
            for nm_, ix in refs.items():
                gR = lgrad(allr[ix])
                zf[nm_] = [cosg(G, gR, lambda k_: True)[0], cosg(G, gR, lambda k_: k_ in w2)[0], cosg(G8, gR, lambda k_: True)[0],
                           cosg(G8, gR, lambda k_: k_ in w2)[0], round(float(isb[ix].mean()), 3), len(ix)]
            if cfg.zrare:   # 2026-11-25 (facts-mac-theory, Pythia Test C arm 1): random earlier data weighted by token rarity, no fact list.
                # unigram counts over the pool itself; weight of each target token = -log p ("log") or 1 only for the rarest 5% of positions ("top")
                def lgradw(rows, wt):
                    model.zero_grad(set_to_none=True)
                    for i0 in range(0, rows.shape[0], 16):
                        xx = torch.from_numpy(rows[i0:i0 + 16]).to(dev); ww = torch.from_numpy(wt[i0:i0 + 16]).to(dev).reshape(-1).float()
                        lg = model(xx[:, :-1]); lg = lg[0] if isinstance(lg, tuple) else lg
                        (torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xx[:, 1:].reshape(-1), reduction="none") * ww).sum().backward()
                    g = {n_: (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w)) for n_, w in P.items()}
                    model.zero_grad(set_to_none=True); return g
                pools = {"web": web}
                if cfg.zfdil: pools["dil"] = pool
                for pn, ix in pools.items():
                    rows = allr[ix]; tg = rows[:, 1:]; cnt = np.bincount(tg.reshape(-1), minlength=int(tg.max()) + 1).astype(np.float64)
                    lp = -np.log((cnt + 1) / (cnt.sum() + len(cnt))); c_at = cnt[tg]
                    for wn, wt in (("log", lp[tg]), ("top", (c_at <= np.quantile(c_at, 0.05)).astype(np.float64))):
                        gR = lgradw(rows, wt)
                        zf[f"rare_{pn}_{wn}"] = [cosg(G, gR, lambda k_: True)[0], cosg(G, gR, lambda k_: k_ in w2)[0], cosg(G8, gR, lambda k_: True)[0],
                                                 cosg(G8, gR, lambda k_: k_ in w2)[0], round(float((wt * isb[ix][:, None]).sum() / max(wt.sum(), 1e-12)), 3), len(ix)]
            zf["cosF_G8"] = cosg(G8, gF, lambda k_: True)[0]
            out["zf"] = zf; model.zero_grad(set_to_none=True)
        setw(endw)
        return {k_: ([round(float(v), 4) for v in vv] if torch.is_tensor(vv) else vv) for k_, vv in out.items()}

    def _gbits():   # value bits of the learned facts only (no rng, no grads); person-major, 5 attrs each
        model.eval(); _, b_ = fw._score(model, dev, _dr["gsp"])
        if fw.ref_prompt: _dr.setdefault("gr", []).append(fw.ref_bits(model, dev))   # O76: chance level C at this same step
        model.train()
        return [round(float(v), 2) for v in b_.reshape(-1)]

    def _sm2(step):   # facts due at step+1 are graded from their bits now (kept vs chance), reviewed at step+1, rescheduled
        S2 = _dr.get("sm2")   # by SM-2: I1 = 1 unit, I2 = 6 units, then I *= EF; q < 3 resets to I1; budget cap on sentences
        if S2 is None:
            j = json.load(open(cfg.sm2file)); fl = [tuple(f) for f in j["facts"]]; u = float(j["unit"])
            S2 = _dr["sm2"] = dict(f=fl, u=u, cap=int(j["budget"]), used=0, ph={f: float(j["first"][i]) for i, f in enumerate(fl)},
                                   n={f: 0 for f in fl}, I={f: 0.0 for f in fl}, ef={f: 2.5 for f in fl}, due={},
                                   C=j["chance"], b0=None, log=[], maxper=int(j.get("maxper", 120)))
            for f in fl: S2["due"][f] = fw.w1 + int(round(S2["ph"][f] * u))
            S2["b0"] = np.array(_dr["gs"][0]).reshape(-1, 5)
        b = np.array(_dr["gs"][-1]).reshape(-1, 5); pi = {q: r for r, q in enumerate(_dr["gsp"])}
        nxt = [f for f in S2["f"] if S2["due"][f] <= step + 1][:S2["maxper"]]
        for f in nxt:
            if S2["used"] >= S2["cap"]: break
            r_, a_ = pi[f[0]], f[1]; b0_ = S2["b0"][r_, a_]; C_ = S2["C"][a_]
            kp = 1 - (b[r_, a_] - b0_) / max(1e-6, C_ - b0_)
            q = 5 if kp >= 0.9 else 4 if kp >= 0.75 else 3 if kp >= 0.5 else 2 if kp >= 0.3 else 1
            if q < 3: S2["n"][f], S2["I"][f] = 0, 1.0
            else:
                S2["n"][f] += 1
                S2["I"][f] = 1.0 if S2["n"][f] == 1 else 6.0 if S2["n"][f] == 2 else S2["I"][f] * S2["ef"][f]
                S2["ef"][f] = max(1.3, S2["ef"][f] + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            S2["due"][f] = step + 1 + max(1, int(round(S2["I"][f] * S2["u"])))
            fw.replay.setdefault(step + 1, []).append(f); S2["used"] += 1; S2["log"].append([step + 1, f[0], f[1], q])

    def _gtr(j_):   # fgroups: append one bits reading for group j_ (no rng, no grads)
        r_, w_ = _dr["gt"][j_], fws[j_]
        model.eval(); _, b_ = w_._score(model, dev, r_["people"])
        if w_.ref_prompt: r_.setdefault("ref", []).append(w_.ref_bits(model, dev))   # O76: this group's own never-seen C at this step
        model.train()
        km = np.array([w_.k[q] == max(w_.bins) for q in r_["people"]])
        r_["b"].append(round(float(b_.mean()), 4)); r_["bmax"].append(round(float(b_[km].mean()), 4))

    def _hid():
        return {n_: w.detach().float().clone() for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_}

    def _wlog(n):   # wlog=N (2026-11-25, facts-scale-theory): every N steps, hidden-matrix rms, relative move since the last log
        now = _hid()   # and since the fact-window end (fw.w1); read-only, no rng, rows otherwise unchanged
        sq = sum(v.pow(2).sum().item() for v in now.values()); cnt = sum(v.numel() for v in now.values())
        wl = _dr.setdefault("wl", {"step": [], "rms": [], "drel": [], "dw1": [], "cos_prev": []})
        pv = _dr.get("wl_prev")
        if pv is not None:
            d2 = sum((now[k] - pv[k]).pow(2).sum().item() for k in now); pq = sum(v.pow(2).sum().item() for v in pv.values())
            dot = sum((now[k] * pv[k]).sum().item() for k in now)
            wl["drel"].append(round((d2 / max(pq, 1e-30)) ** 0.5, 6)); wl["cos_prev"].append(round(dot / max(1e-30, (sq * pq) ** 0.5), 7))
        else:
            wl["drel"].append(None); wl["cos_prev"].append(None)
        w1 = _dr.get("wl_w1")
        if w1 is not None:
            d2 = sum((now[k] - w1[k]).pow(2).sum().item() for k in now); q1 = sum(v.pow(2).sum().item() for v in w1.values())
            wl["dw1"].append(round((d2 / max(q1, 1e-30)) ** 0.5, 6))
        else:
            wl["dw1"].append(None)
        wl["step"].append(n); wl["rms"].append(round((sq / max(1, cnt)) ** 0.5, 6))
        if cfg.wsig:   # wsig=1 (2026-11-26, facts-paper-writer): per hidden matrix, sigma_max by 5 warm-started power-iteration steps
            # and rms; own generator on the model device, so the training rng and the rows are unchanged
            nit = 5   # first tick is a cold start: 5 steps read only 91-94 % of sigma_max on 256-1024 matrices, so give it 30
            if "wsv" not in _dr:
                nit = 30
                g_ = torch.Generator(device=next(iter(now.values())).device); g_.manual_seed(0)
                _dr["wsv"] = {k: torch.randn(v.shape[1], generator=g_, device=v.device) for k, v in now.items()}
                wl["sig_names"] = list(now.keys()); wl["sig"] = []; wl["mrms"] = []
            sg, mr = [], []
            for k, v in now.items():
                x = _dr["wsv"][k]
                for _ in range(nit):
                    u = v @ x; u = u / u.norm().clamp_min(1e-30); x = v.t() @ u; x = x / x.norm().clamp_min(1e-30)
                _dr["wsv"][k] = x
                sg.append(round((v @ x).norm().item(), 6)); mr.append(round(v.pow(2).mean().item() ** 0.5, 7))
            wl["sig"].append(sg); wl["mrms"].append(mr)
        _dr["wl_prev"] = now

    def finish(n_done, tag):
        train_s = time.time() - t0

        if a.split == "val":
            val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)
            if a.data.startswith("fineweb"):
                val = val[:FINEWEB_TEST_SEQ * cfg.seq_len + 1]
        elif a.data.startswith("fineweb"):
            val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)[FINEWEB_TEST_SEQ * cfg.seq_len:]
        else:
            val = np.fromfile(os.path.join(DATA, "test.bin"), dtype=np.uint16)
        res = evaluate(model, val, token_bytes(TOKF) if TOKF != TOK0 else token_bytes(), cfg, a.eval_seqs, device=dev)
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
        if fw is not None:
            res["facts_eval"] = fw.evaluate(model, dev, n_done)
            res["facts_eval"]["curve"] = fcurve
            if "sp" in _dr:   # splice: end weights for one group, gap-start weights for the other (no retraining)
                hid = {id(p) for _, p in optmod.mats_params(model)} if mopt is not None else {id(w) for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_}
                endw = {n_: w.detach().clone() for n_, w in model.named_parameters()}
                spl = {}
                dn = [0.0, 0.0, 0.0, 0.0]   # 2026-11-25 learning-rate-free test: squared weight change in the gap [hidden, adam group] and squared gap-start norms
                for n_, w in model.named_parameters():
                    d2 = float((endw[n_] - _dr["sp"][n_]).float().pow(2).sum()); w2 = float(_dr["sp"][n_].float().pow(2).sum())
                    i_ = 0 if id(w) in hid else 1
                    dn[i_] += d2; dn[2 + i_] += w2
                spl["dw2"] = [round(x, 6) for x in dn]
                for nm_, back in (("pre", lambda w: True), ("hid_only", lambda w: id(w) not in hid), ("adam_only", lambda w: id(w) in hid))[:1 if cfg.gsplice == 2 else 3]:
                    with torch.no_grad():
                        for n_, w in model.named_parameters():
                            w.copy_(_dr["sp"][n_] if back(w) else endw[n_])
                    spl[nm_] = fw.evaluate(model, dev, n_done)["per_fact_bits"]
                    if nm_ == "pre" and _dr.get("gb") and cfg.gsplice != 2:
                        spl["gaploss_pre"] = _gaploss()
                with torch.no_grad():
                    for n_, w in model.named_parameters():
                        w.copy_(endw[n_])
                if _dr.get("gb") and cfg.gsplice != 2:
                    spl["gaploss_end"] = _gaploss()
                res["facts_eval"]["splice"] = spl
                if cfg.fmarg:   # 2026-11-25 margin theory test: per value token of learned facts, log-odds margin m = z_y - lse_{j!=y} z
                    # (CE = softplus(-m)) at gap start / end / along the straight line, first-order change along the gap move
                    # (central differences, split by group), and directional derivatives along cheap pre-gap directions.
                    res["facts_eval"]["marg"] = _margin_probe(endw, hid)
            if _dr.get("w0"):
                w0, now = _dr["w0"], _hid()
                n0 = sum(v.pow(2).sum().item() for v in w0.values()) ** 0.5
                disp = sum((now[k] - w0[k]).pow(2).sum().item() for k in w0) ** 0.5
                res["facts_eval"]["drift"] = round(disp / n0, 6)            # straight-line move, relative to weight size
                res["facts_eval"]["drift_path"] = round(_dr["path"] / n0, 6)  # sum of per-step moves, relative
                res["facts_eval"]["drift_steps"] = _dr["n"]
                g = _dr["g"]
                gn = sum(v.pow(2).sum().item() for v in g.values()) ** 0.5
                dot = sum((g[k] * (now[k] - w0[k])).sum().item() for k in w0)
                res["facts_eval"]["drift_dloss"] = round(dot, 6)                 # first-order rise in fact loss (nats/token)
                res["facts_eval"]["drift_cos"] = round(dot / max(1e-12, gn * disp), 6)   # alignment of the move with the fact gradient
                grp = {}
                for k in w0:   # which layer types move: qkv, attn out, mlp in (w1/w3), mlp out (w2)
                    typ = "qkv" if "qkv" in k else "w2" if ".w2" in k else "w13" if (".w1" in k or ".w3" in k) else "attn_out" if ("proj" in k or "out" in k or ".o" in k) else "other"
                    a_ = grp.setdefault(typ, [0.0, 0.0, 0.0])
                    a_[0] += (now[k] - w0[k]).pow(2).sum().item(); a_[1] += w0[k].pow(2).sum().item(); a_[2] += (g[k] * (now[k] - w0[k])).sum().item()
                res["facts_eval"]["drift_by_type"] = {kk: [round((v[0] / v[1]) ** 0.5, 5), round(v[2], 6)] for kk, v in grp.items()}
                if "gL" in _dr:   # same alignment for the learned facts only, plus their actual loss before and after the gap
                    gL = _dr["gL"]; gLn = sum(v.pow(2).sum().item() for v in gL.values()) ** 0.5
                    dL = sum((gL[k] * (now[k] - w0[k])).sum().item() for k in w0)
                    fw.fact_grad(model, dev, list(w0.keys()), kmin=max(fw.bins) // 2)
                    res["facts_eval"]["drift_dloss_L"] = round(dL, 6)
                    res["facts_eval"]["drift_cos_L"] = round(dL / max(1e-12, gLn * disp), 6)
                    res["facts_eval"]["loss_L"] = [round(_dr["L0"], 5), round(fw.last_fact_loss, 5)]   # learned-fact loss at gap start, at end
                    grpL = {}
                    for k in w0:
                        typ = "qkv" if "qkv" in k else "w2" if ".w2" in k else "w13" if (".w1" in k or ".w3" in k) else "attn_out" if ("proj" in k or "out" in k or ".o" in k) else "other"
                        grpL[typ] = grpL.get(typ, 0.0) + (gL[k] * (now[k] - w0[k])).sum().item()
                    res["facts_eval"]["drift_dloss_L_by_type"] = {kk: round(v, 6) for kk, v in grpL.items()}
            res["facts_eval"]["wnorm"] = round(float(sum(w.float().pow(2).mean().item() for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_) / max(1, sum(1 for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and "head" not in n_))) ** 0.5, 5)   # mean-square of hidden matrices, rms (2026-11-25)
            if "wl" in _dr:
                res["facts_eval"]["wtrace"] = _dr["wl"]
            if cfg.flog:
                res["facts_eval"]["flog"] = flog_rows
            if "gt" in _dr:   # fgroups: per group, C = mean bits of its people seen <= 4 times at the end (chance level, as lfa.py)
                for j_, r_ in _dr["gt"].items():
                    w_ = fws[j_]; model.eval(); _, b_ = w_._score(model, dev, [q for q in range(w_.n_eval) if w_.k[q] <= 4]); model.train()
                    r_["C"] = round(float(b_.mean()), 4)
                res["facts_eval"]["groups"] = [_dr["gt"][j_] for j_ in sorted(_dr["gt"])]
            if "fs" in _dr:   # fsplit: per world, one entry per step from its window end (t0); ce has one more entry (final weights)
                for j_, r_ in _dr["fs"].items():
                    w_ = fws[j_]; pp_ = [q for q in range(w_.n_eval) if w_.k[q] >= max(w_.bins) // 2]
                    w_.fact_grad(model, dev, _PN[:1], only=[(q, ai) for q in pp_ for ai in range(5)], sep=True)
                    r_["ce"].append(float(f"{w_.last_fact_loss:.6g}")); r_["group"] = int(cfg.fgonly) if cfg.fgonly >= 0 else j_
                model.train(); res["facts_eval"]["fsplit"] = [_dr["fs"][j_] for j_ in sorted(_dr["fs"])]
            if "gs" in _dr:   # gstep=1: bits[0] at gap start, bits[n] after n gap steps; people = eval ids (k >= max bin / 2)
                res["facts_eval"]["gstep"] = {"people": _dr["gsp"], "bits": _dr["gs"]}
                if "gr" in _dr: res["facts_eval"]["gstep"]["ref"] = _dr["gr"]   # O76: ref_bits per reading, same length as bits
                if "sm2" in _dr: res["facts_eval"]["sm2"] = {"used": _dr["sm2"]["used"], "log": _dr["sm2"]["log"]}   # 2026-11-25 schedule search
            if os.environ.get("FPROBE"):   # gather vs read-out probe (fast/probe.py, 2026-11-24); opt-in
                sys.path.insert(0, os.path.join(HERE, "..", "..", "fast")); import probe as _probe
                res["facts_eval"]["probe"] = _probe.run(model, fw, dev)
            if cfg.ftrace and a.out:   # trace file: per-fact bits at every checkpoint + each person's sighting steps
                ftr_steps.append(n_done); ftr_bits.append(res["facts_eval"]["per_fact_bits"])
                ne = fw.n_eval
                sl = [fw.seen[p] for p in range(ne)]
                tp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "traces", (tag or "run") + ".npz")
                os.makedirs(os.path.dirname(tp), exist_ok=True)
                np.savez_compressed(tp, steps=np.array(ftr_steps), bits=np.array(ftr_bits, dtype=np.float32).reshape(len(ftr_steps), ne, 5),
                                    k=fw.k[:ne], seen=np.concatenate([np.array(x, dtype=np.int32) for x in sl]),
                                    seen_off=np.cumsum([0] + [len(x) for x in sl]),
                                    pool=np.array([len(fw.pools[r[0]]) for r in factsmod.REL]))
                res["facts_eval"]["trace_file"] = tp
            if cfg.fchoice:
                res["facts_eval"].update(fw.choice_eval(model, dev))
            if n_done != a.steps:   # branch row: relabel bins by the sightings actually seen by step n_done (k*n_done/T)
                fe = res["facts_eval"]
                for key in ("acc_by_k", "bits_by_k", "choice_by_k"):
                    if key in fe:
                        fe[key] = {str(int(k) * n_done // a.steps): v for k, v in fe[key].items()}
                fe["k_scale"] = round(n_done / a.steps, 6)
                fe["ffillk_eff"] = cfg.ffillk * n_done // a.steps
            res["facts_eval"]["bio_frac"] = round(bio_tok / (n_done * cfg.batch * (cfg.seq_len + 1)), 4)
            res["nonemb_params"] = sum(p.numel() for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n and "head" not in n)
        row = dict(method=a.method, tag=tag, branch_of=(a.tag if tag != a.tag else None), data=a.data, split=a.split, size=a.size, steps=n_done, train_seconds=round(train_s, 1), threads=a.threads, device=getattr(a, "device", "cpu"),
                   tokens=n_done * cfg.batch * cfg.seq_len, params=total, active_params=active,
                   peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1),
                   gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
                   **res, prof={k: round(v, 2) for k, v in PT.items()} if cfg.prof else None, qk_clips=getattr(cfg, "qk_clips", 0), smax_peak=round(getattr(cfg, "smax_peak", 0.0), 2), config={k: v for k, v in c.items()}, last_train=hist[-1])
        print(json.dumps({k: v for k, v in row.items() if k != "config"}), flush=True)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")

    def snap():
        return dict(model=copy.deepcopy(model.state_dict()), opt=copy.deepcopy(opt.state_dict()),
                    mopt=copy.deepcopy(mopt.state_dict()) if mopt is not None else None, mt=mopt.t if mopt is not None else 0,
                    rng=copy.deepcopy(rng.bit_generator.state), trng=torch.get_rng_state(),
                    crng=torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                    frng=copy.deepcopy(fw.rng.bit_generator.state) if fw is not None else None,
                    last=fw.last_step.copy() if fw is not None else None,
                    seen=[list(x) for x in fw.seen] if fw is not None else None, bio=bio_tok, hl=len(hist))

    def restore(sv):
        nonlocal bio_tok
        model.load_state_dict(sv["model"]); opt.load_state_dict(sv["opt"])
        if mopt is not None:
            mopt.load_state_dict(sv["mopt"]); mopt.t = sv["mt"]
        rng.bit_generator.state = sv["rng"]; torch.set_rng_state(sv["trng"])
        if sv["crng"] is not None:
            torch.cuda.set_rng_state(sv["crng"])
        if fw is not None:
            fw.rng.bit_generator.state = sv["frng"]; fw.last_step[:] = sv["last"]; fw.seen = sv["seen"]
        bio_tok = sv["bio"]
        del hist[sv["hl"]:]
        model.train()

    br_start = {b - int(round(cfg.decay * b)): b for b in brs}
    s0, tkey = 0, None
    if cfg.gckpt and fw is not None:   # 2026-11-25 speed: all gap variants share steps 0..w1-1; train them once, save, branch the gap
        VK = ("ffkind", "ffmt", "gaplr", "fmarg", "gsplice", "gfreeze", "gckpt", "fdrift", "flog", "gstep", "gmom", "gmb", "gb1", "gbs", "gsgd", "gsgdm", "gnoclip", "glrx", "wlog", "wsig", "fsplit", "zcheap", "zfdil", "zrare", "rsp", "rsn", "rsm", "rsx", "rpfile", "rpadd", "pfscreen", "sm2file", "fref")
        tkey = dict({k: v for k, v in c.items() if k not in VK}, _steps=a.steps, _data=a.data, _split=a.split, _dev=str(dev))
        os.makedirs(os.path.dirname(os.path.abspath(cfg.gckpt)), exist_ok=True)
        lk = cfg.gckpt + ".lock"   # parallel queue: the first run trains the shared steps, the others wait for its checkpoint
        while not os.path.exists(cfg.gckpt):
            try:
                os.close(os.open(lk, os.O_CREAT | os.O_EXCL)); break
            except FileExistsError:
                if time.time() - os.path.getmtime(lk) > 2400:   # owner died: take over
                    os.utime(lk, None); break
                time.sleep(5)
        if os.path.exists(cfg.gckpt):
            if cfg.gckpt not in _CK:
                _CK[cfg.gckpt] = torch.load(cfg.gckpt, weights_only=False)
            sv = copy.deepcopy(_CK[cfg.gckpt])
            # 2026-11-26 spacing-learn r5 la/moe: batched.py sends non-softmax files here one job at a time; their jobs branch
            # lr/sched/decay after w1 from one trunk, so ignore the same branch fields as batched.py (and the device name)
            _BR = ("rpfile", "lr", "mlr", "sched", "min_lr_frac", "warmup", "decay", "rwp", "rww", "mwd", "gaplr", "gstep", "wlog", "wsig", "fsplit", "sm2file", "_dev")
            assert {k: v for k, v in sv["tkey"].items() if k not in _BR} == {k: v for k, v in tkey.items() if k not in _BR}, \
                "gckpt trunk differs from this run"
            restore(sv); hist[:] = sv["hist"]; s0 = fw.w1
            if sv.get("rb") is not None: _dr["rb"] = sv["rb"]
            del sv
    for step in range(s0, a.steps):
        if tkey is not None and s0 == 0 and step == fw.w1:
            sv = snap(); sv.update(tkey=tkey, hist=list(hist), rb=_dr.get("rb")); tmp = f"{cfg.gckpt}.{os.getpid()}"
            torch.save(sv, tmp); os.replace(tmp, cfg.gckpt)
            del sv
        if step in br_start:          # decayed copy that ends at b; the trunk then replays the same steps at flat lr
            b = br_start[step]
            sv = snap()
            for s2 in range(step, b):
                one_step(s2, b, trunk=False)
            finish(b, f"{a.tag}_br{b}")
            restore(sv)
            del sv
        if (cfg.gmb or cfg.gb1) and fw is not None and step == fw.w1 and "mb" not in _dr:   # 2026-11-26 phase-2 momentum: Muon beta, Adam beta1
            _dr["mb"] = 1
            if cfg.gmb: cfg.mom = float(cfg.gmb)
            if cfg.gb1:
                for pg in opt.param_groups: pg["betas"] = (float(cfg.gb1), pg["betas"][1])
        if cfg.gsgd and fw is not None and step == fw.w1 and "sgd" not in _dr:   # 2026-11-26 cause test: SGD + momentum on every weight in phase 2
            _dr["sgd"] = torch.optim.SGD(list(model.parameters()), lr=float(cfg.gsgd), momentum=float(cfg.gsgdm))
        if cfg.gmom and fw is not None and step == fw.w1 and "mz" not in _dr:   # 2026-11-25 momentum test: 1 zero Muon momentum, 2 also Adam first moments
            _dr["mz"] = 1
            for o_ in ([mopt] if mopt is not None else []) + ([opt] if cfg.gmom >= 2 else []):
                for st_ in o_.state.values():
                    for k_ in ("m", "exp_avg"):
                        if k_ in st_: st_[k_].zero_()
        if cfg.gsplice and fw is not None and step == fw.w1 and "sp" not in _dr:   # 2026-11-25 splice test: keep all weights at gap start
            _dr["sp"] = {n_: w.detach().clone() for n_, w in model.named_parameters()}
        if cfg.fdrift and fw is not None and step == fw.w1 and "w0" not in _dr:
            _dr.update(w0=_hid(), path=0.0, n=0)
            _dr["prev"] = _dr["w0"]
            _dr["g"] = fw.fact_grad(model, dev, list(_dr["w0"].keys()))   # fact-loss gradient at the start of the gap
            _dr["gL"] = fw.fact_grad(model, dev, list(_dr["w0"].keys()), kmin=max(fw.bins) // 2)   # learned facts only (k >= 64)
            _dr["L0"] = fw.last_fact_loss
        if cfg.pfscreen and fw is not None and step == fw.w1:   # 2026-11-25 practical-win screen: per-fact and per-relation cosF
            P_ = dict(model.named_parameters()); nm_ = list(P_.keys()); gr_ = np.random.default_rng(777)   # own rng: main rng untouched
            model.zero_grad(set_to_none=True)
            for _ in range(8):   # G = gradient of the web-text loss (the text that follows the window), 8 batches
                st_ = gr_.integers(0, len(train) - cfg.seq_len - 1, cfg.batch)
                xb_ = torch.from_numpy(np.stack([train[s_:s_ + cfg.seq_len + 1] for s_ in st_]).astype(np.int64)).to(dev)
                lg_ = model(xb_[:, :-1]); lg_ = lg_[0] if isinstance(lg_, tuple) else lg_
                (torch.nn.functional.cross_entropy(lg_.float().reshape(-1, lg_.shape[-1]), xb_[:, 1:].reshape(-1)) / 8).backward()
            G_ = {n_: P_[n_].grad.detach().float().clone() for n_ in nm_}; model.zero_grad(set_to_none=True)
            gn_ = sum(float(v.pow(2).sum()) for v in G_.values()) ** 0.5
            def pcos(only):   # paper sign: positive = a step on the web text raises the facts' loss
                g1 = fw.fact_grad(model, dev, nm_, only=only); n1 = sum(float(v.pow(2).sum()) for v in g1.values()) ** 0.5
                return round(-sum(float((G_[n_] * g1[n_]).sum()) for n_ in nm_) / max(1e-20, gn_ * n1), 6), round(fw.last_fact_loss, 4)
            lp_ = [q for q in range(fw.n_eval) if fw.k[q] >= max(fw.bins) // 2]
            outp = {"tag": a.tag, "step": step, "people": lp_, "fact": [[q, ai, *pcos([(q, ai)])] for q in lp_ for ai in range(5)],
                    "rel": [pcos([(q, ai) for q in lp_]) for ai in range(5)], "all": pcos([(q, ai) for q in lp_ for ai in range(5)])}
            json.dump(outp, open(cfg.pfscreen, "w")); print(f"pfscreen written {cfg.pfscreen} rel {outp['rel']}", flush=True)
            return
        if cfg.gstep and fw is not None and step == fw.w1 and "gs" not in _dr:   # 2026-11-25 early measure: learned-fact bits after every gap step
            _dr["gsp"] = [q for q in range(fw.n_eval) if fw.k[q] >= max(fw.bins) // 2]
            if cfg.gstep == 2:   # 2026-11-26 also track people seen <= 4 times (chance reference per step), appended after the learned
                _dr["gsp"] += [q for q in range(fw.n_eval) if fw.k[q] <= 4]
            _dr["gs"] = [_gbits()]
        if cfg.fgroups and fw is not None:   # fgroups: each group's learned facts (k >= max bin / 2), bits at its window end
            GT = _dr.setdefault("gt", {})   # and after every later step; mean bits over all learned facts (b) and over k = max bin (bmax)
            for j_, w_ in enumerate(fws):
                if step == w_.w1 and j_ not in GT:
                    GT[j_] = {"group": int(cfg.fgonly) if cfg.fgonly >= 0 else j_, "w0": w_.w0, "w1": w_.w1,
                              "people": [q for q in range(w_.n_eval) if w_.k[q] >= max(w_.bins) // 2], "b": [], "bmax": []}
                    _gtr(j_)
        if one_step(step, a.steps):
            break
        for j_ in _dr.get("gt", {}):
            if step >= fws[j_].w1: _gtr(j_)
        if "gs" in _dr:
            _dr["gs"].append(_gbits())
            if cfg.sm2file:   # 2026-11-25 schedule search: online SM-2 review rule (SRT baseline, Atreya et al. 2026)
                _sm2(step)
        if cfg.wlog and fw is not None and step + 1 == fw.w1 and "wl_w1" not in _dr:
            _dr["wl_w1"] = _hid()
        if cfg.wlog and ((step + 1) % cfg.wlog == 0 or (fw is not None and step + 1 == fw.w1)):
            _wlog(step + 1)
        if _dr.get("w0"):
            now = _hid()
            _dr["path"] += sum((now[k] - _dr["prev"][k]).pow(2).sum().item() for k in now) ** 0.5
            _dr["prev"], _dr["n"] = now, _dr["n"] + 1
    finish(step + 1, a.tag)


def sweep(argv):
    """--sweep JOBS.jsonl --out F [--threads N] [--device D] [--tag T]: run every job (lines {"tag", "args"} as in run_jobs.py)
    in this one process, skipping tags already in --out; imports and gap-start checkpoints load once. Rows are identical to
    one process per job (checked 2026-11-25 on 35 stored rows). Queue use (cpu.sh / gpu.sh): a job {"tag": "sweep_X",
    "args": ["--sweep", "sweeps/X.jsonl"]}; relative paths resolve against the search root, then arena/env. With --tag a
    marker row {"tag", "sweep_ok", ...} is appended at the end so the queue sees the job as done."""
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="comma-separated tag substrings")
    s, _ = ap.parse_known_args(argv)
    sf = s.sweep
    if not os.path.isabs(sf):
        for b in (os.getcwd(), os.path.join(HERE, "..", ".."), HERE):
            if os.path.exists(os.path.join(b, sf)):
                sf = os.path.join(b, sf); break
    done = set()
    if os.path.exists(s.out):
        for l in open(s.out):
            try: done.add(json.loads(l)["tag"])
            except Exception: pass
    jobs = [json.loads(l) for l in open(sf) if l.strip()]
    jobs = [j for j in jobs if j["tag"] not in done and (not s.only or any(x in j["tag"] for x in s.only.split(",")))]
    t_all, ok = time.time(), 0
    for j in jobs:
        t1 = time.time()
        try:
            main([*j["args"], "--tag", j["tag"], "--threads", str(s.threads), "--out", s.out, *(["--device", s.device] if s.device else [])]); ok += 1
        except Exception as e:   # one bad variant must not stop the sweep
            print(f"SWEEP FAIL {j['tag']}: {e!r}", flush=True)
        print(f"SWEEP {j['tag']} {time.time() - t1:.2f}s", flush=True)
    print(f"SWEEP done {ok}/{len(jobs)} in {time.time() - t_all:.1f}s", flush=True)
    if s.tag:
        with open(s.out, "a") as f:
            f.write(json.dumps(dict(tag=s.tag, sweep_file=s.sweep, sweep_ok=ok, sweep_jobs=len(jobs), sweep_seconds=round(time.time() - t_all, 1))) + "\n")


if __name__ == "__main__":
    if "--batched" in sys.argv:   # 2026-11-25 queue entry for env/batched.py: {"tag", "args": ["--batched", "--jobs", "sweeps/X.jsonl"]}
        import batched
        batched.main([x for x in sys.argv[1:] if x != "--batched"])
    elif "--sweep" in sys.argv:
        sweep(sys.argv[1:])
    else:
        main()
