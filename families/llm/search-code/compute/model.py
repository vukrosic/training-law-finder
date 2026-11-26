"""Decoder-only transformer with a dense or mixture-of-experts FFN in every block.

Routers (all share the same SwiGLU experts and the same token-choice dispatch loop):
  topk_aux       Switch / GShard / Mixtral top-k softmax router with the Switch load-balancing
                 loss computed per micro-batch (lbl_ranks virtual data-parallel ranks) + ST-MoE z-loss
  global_aux     same router, load-balancing loss over the global batch (Qiu et al. 2025)
  dsv3           DeepSeek-V3: sigmoid scores, auxiliary-loss-free expert bias (updated after each
                 step), tiny sequence-wise balance loss, fine-grained routed experts + shared expert
  expert_choice  Expert Choice (Zhou et al. 2022): each expert picks its top-capacity tokens
  remoe          ReMoE (Wang et al., ICLR 2025): ReLU router + L1 penalty whose coefficient adapts
                 to hit the target sparsity 1 - k/E
The routing, loss and bias formulas come from the pinned official code (env/official.py):
Megatron-Core moe_utils.py for top-k, capacity dropping, Switch / sequence / global balance
losses, z-loss and the expert-bias update; the ReMoE fork of Megatron for ReLU routing and the
L1 coefficient rule. Expert Choice has no public official code and is written from the paper.

Layout: tokens enter the router seq-first ([T, B] flattened), as in Megatron, so the official
sequence-wise loss reshape is used unchanged.
"""
import math
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from official import mg

ROUTERS = ["topk_aux", "global_aux", "dsv3", "expert_choice", "remoe", "gen"]

# "gen" router (2026 routers, written from their reports; Megatron-Core v0.16.1 has no
# Sqrt(Softplus) score and no quantile balancing):
#   score_fn  "sigmoid" (Kimi K3) or "sqrtsoftplus" (DeepSeek-V4: s = sqrt(softplus(logit)))
#   balance   "auxfree" (sign bias update at bias_rate, DeepSeek) or "quantile" (Kimi K3
#             Quantile Balancing: alpha_i = (k+1)-th routing score of token i,
#             b_hat_j = -quantile_{1-k/E}(s_:,j - alpha), b = b_hat - mean(b_hat), used next step)
#   hash_layers  first MoE layers that pick experts by a fixed hash of the token id (DeepSeek-V4)
#   n_shared  shared experts; the shared width shared_hidden is split evenly between them
# Selection on s + b; gates = selected s renormalized to sum 1, times route_scale.


def score_fn(logits, name):
    x = logits.float()
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "sqrtsoftplus":
        return torch.sqrt(F.softplus(x))
    raise ValueError(name)


class MultiSwiGLU(nn.Module):
    """n shared experts of width h // n each, outputs summed."""

    def __init__(self, d, h, n):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(d, h // n) for _ in range(n)])

    def forward(self, x):
        return sum(e(x) for e in self.experts)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cap=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cap is not None:
            cap["q"] = q.transpose(1, 2).reshape(B, T, D)               # pre-rope queries
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if cap is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                                                           # explicit probs for inventions
            s = (q @ k.transpose(-1, -2)) / self.dh ** 0.5
            s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            att = s.softmax(-1)
            y = att @ v
            cap["att"], cap["heads"] = att, y
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """One MoE FFN. forward(x [B,T,D]) -> y, and appends its auxiliary losses to self.aux."""

    def __init__(self, d, cfg, layer=0):
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        E, h = cfg.n_experts, cfg.expert_hidden
        self.E, self.k, self.router = E, cfg.topk, cfg.router
        # router input (step 3): "x" = token vector (DeepSeek-V3); "shared_h" = the shared expert's
        # gated hidden activations silu(W1 x)*(W3 x) (SHARED-ROUTE); "shared_h_sg" = same, no gradient
        # from the router into the shared expert; "mlp" = a separate learned SwiGLU hidden layer of
        # the same width (extra parameters and compute); "rand_mlp" = same, frozen random weights;
        # "shared_out" = the shared expert's output. route_norm: RMS-normalize the router input.
        self.route_input = getattr(cfg, "route_input", "x")
        feat = d if self.route_input in ("x", "shared_out") else cfg.shared_hidden
        self.gate = nn.Parameter(torch.randn(E, feat) * 0.02)          # Megatron router: no bias
        if getattr(cfg, "freeze_gate", False):                          # control: router never learns
            self.gate.requires_grad_(False)
        self.w1 = nn.Parameter(torch.randn(E, d, h) * 0.02)
        self.w3 = nn.Parameter(torch.randn(E, d, h) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, h, d) * 0.02 / math.sqrt(2 * cfg.n_layers))
        # ROUND 12 (compute search): how experts compute and combine. ex = idea name; params are
        # named k_* so GPT's normal_ init skips them; every gate starts at identity (scale 1, add 0)
        self.ex = getattr(cfg, "ex", "none")
        # ROUND 16 (compute search, standard init): a per-token-id branch added to the MoE layer output.
        # idb = r4 / r4x2 / r8 / r4e / r4y (per-id low-rank map), sw / swb (shared down, per-id up), vec, ple
        self.idb = getattr(cfg, "idb", "none")
        if self.idb != "none":
            V = cfg.vocab
            g = torch.Generator().manual_seed(1601 + layer)
            rk = 8 if self.idb == "r8" else 4
            R = {"r4": 128, "r4x2": 128, "r8": 64, "r4e": 128, "r4y": 128, "sw": 256, "swb": 256, "vec": 1024, "ple": 1024, "gain": 1024, "gain2": 1024}[self.idb]
            self.idR, self.idrk = R, rk
            self.register_buffer("id_t1", torch.randint(0, R, (V,), generator=g), persistent=False)
            self.register_buffer("id_t2", torch.randint(0, R, (V,), generator=g), persistent=False)
            self.register_buffer("id_t3", torch.randint(0, R, (V,), generator=g), persistent=False)
            if self.idb in ("r4", "r4x2", "r8", "r4e", "r4y"):
                self.k_iU = nn.Parameter(torch.zeros(R, rk, d))
                self.k_iW = nn.Parameter(torch.randn(R, d, rk) * 0.1)
            elif self.idb in ("sw", "swb"):
                self.k_iU = nn.Parameter(torch.zeros(R, rk, d))
                self.k_iWs = nn.Parameter(torch.randn(d, rk) * 0.1)
                if self.idb == "swb":
                    self.k_iWp = nn.Parameter(torch.randn(d, rk) * 0.1)
            elif self.idb == "vec":
                self.k_iV = nn.Parameter(torch.zeros(R, d))
            elif self.idb in ("gain", "gain2"):
                # ROUND 17: per-id channel gain on the MoE output, y * (1 + G[id]); zero init = baseline at step 0
                self.k_iG1 = nn.Parameter(torch.zeros(R, d))
                if self.idb == "gain2":
                    self.k_iG2 = nn.Parameter(torch.zeros(R, d))
            elif self.idb == "ple":
                self.k_iE = nn.Parameter(torch.randn(R, 64) * 0.1)
                self.k_iG = nn.Parameter(torch.randn(d, 64) * 0.02)
                self.k_iP = nn.Parameter(torch.zeros(64, d))
        if self.ex != "none":
            ex, r = self.ex, int(getattr(cfg, "exa", 0) or 16)
            z = lambda *sh: nn.Parameter(torch.zeros(*sh))
            if ex in ("act_pow", "act_slope"): self.k_p = z(E)
            if ex == "act_bias": self.k_bias = z(E, h)
            if ex == "act_learn": self.k_mix = z(E, 3)
            if ex == "hnorm": self.k_g = nn.Parameter(torch.full((E, 1), 0.1))
            if ex in ("bil", "bilw"):
                self.k_U = nn.Parameter(torch.randn(d, r) / math.sqrt(d)); self.k_V = z(r, d)
            if ex == "maxpool": self.k_c = z(1)
            if ex == "prodsum": self.k_c = z(d)
            if ex in ("outattn", "gatemul", "sh_gate"): self.k_W = z(d, d)
            if ex == "normsum": self.k_g = nn.Parameter(torch.full((d,), 1e-3))
            if ex in ("sh_agree", "sh_conf", "sh_split", "xl_gate", "sh_conf_r", "sh_agree_both", "sh_ent",
                      "sh_cos", "sh_conf_both"): self.k_wb = z(2)
            if ex in ("sh_agree_v", "sh_conf_v"): self.k_wb = z(2, d)
            if ex in ("chain_sc", "chain_rev_sc"): self.k_a = nn.Parameter(torch.full((1,), 4.0))
            if ex == "sh_film": self.k_A = nn.Parameter(torch.randn(d, 8) / math.sqrt(d)); self.k_B = z(8, d)
            if ex == "routed_gate_sh": self.k_w = z(d)
            if ex in ("xl_prev", "xl_prev_sh", "xl_in", "xl_sub"): self.k_a = z(1)
            if ex == "xl_dense": self.k_a = z(max(layer, 1))
        n_sh = getattr(cfg, "n_shared", 1)
        if cfg.shared_hidden <= 0:
            self.shared = None
        elif n_sh > 1:
            self.shared = MultiSwiGLU(d, cfg.shared_hidden, n_sh)
        else:
            self.shared = SwiGLU(d, cfg.shared_hidden)
        if self.route_input in ("mlp", "rand_mlp"):
            self.r1 = nn.Linear(d, cfg.shared_hidden, bias=False)
            self.r3 = nn.Linear(d, cfg.shared_hidden, bias=False)
            if self.route_input == "rand_mlp":
                self.r1.weight.requires_grad_(False)
                self.r3.weight.requires_grad_(False)
        if self.router == "dsv3":
            # STD-ROUTE: running per-expert mean / std of sigmoid scores (selection only)
            self.register_buffer("sc_mean", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("sc_std", torch.ones(E, dtype=torch.float32))
            self.register_buffer("sc_init", torch.zeros((), dtype=torch.float32))
            self.cur_ids = None
            self.register_buffer("expert_bias", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("local_tokens_per_expert", torch.zeros(E, dtype=torch.float32))
        if self.router == "gen":
            self.register_buffer("sc_mean", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("sc_std", torch.ones(E, dtype=torch.float32))
            self.register_buffer("sc_init", torch.zeros((), dtype=torch.float32))
            self.cur_ids = None
            self.register_buffer("expert_bias", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("local_tokens_per_expert", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("next_bias", torch.zeros(E, dtype=torch.float32))
            hp = getattr(cfg, "hash_pos", 0)
            self.hash = hp <= layer < hp + getattr(cfg, "hash_layers", 0)
            self.hash_mode = getattr(cfg, "hash_mode", "uni")
            if self.hash and self.hash_mode != "uni":
                # knob: key the table by (previous id, id) or by the previous id alone
                g2 = torch.Generator().manual_seed(4321 + layer)
                R = 1 << 16 if self.hash_mode == "bigram" else cfg.vocab
                self.register_buffer("hash_table2", torch.rand(R, E, generator=g2).argsort(1)[:, :self.k],
                                     persistent=False)
            if self.hash:
                # fixed hash: token id -> k distinct experts (seeded random table, never trained)
                g = torch.Generator().manual_seed(1234 + layer)
                tab = torch.stack([torch.randperm(E, generator=g)[:self.k] for _ in range(cfg.vocab)])
                M = getattr(cfg, "hash_sets", 0)
                if M:   # knob: only M distinct expert sets; each token id gets one of them at random
                    tab = tab[:M][torch.randint(0, M, (cfg.vocab,), generator=g)]
                self.register_buffer("hash_table", tab, persistent=False)
            # MIX-HASH (ours): in learned layers, hash_k of the k experts come from a fixed hash of
            # the token id, the other k - hash_k from the learned scores (hashed ones excluded)
            self.hash_k = 0 if self.hash else getattr(cfg, "hash_k", 0)
            if self.hash_k:
                g = torch.Generator().manual_seed(5678 + layer)
                tab = torch.stack([torch.randperm(E, generator=g)[:self.hash_k] for _ in range(cfg.vocab)])
                self.register_buffer("hk_table", tab, persistent=False)
            # ID-BIAS (ours): learned per-token-id router logits added in learned layers (zero init)
            self.id_bias = None
            idl = getattr(cfg, "id_bias_layers", 0)
            if getattr(cfg, "id_bias", False) and not self.hash and (idl == 0 or layer < idl):
                self.id_bias = nn.Embedding(cfg.vocab, E)
                nn.init.zeros_(self.id_bias.weight)
                c = getattr(cfg, "id_bias_init", 0.0)
                if c:   # SOFT-HASH: start at +c on the experts the DeepSeek-V4 hash table would pick
                    g = torch.Generator().manual_seed(1234 + layer)
                    tab = torch.stack([torch.randperm(E, generator=g)[:self.k] for _ in range(cfg.vocab)])
                    with torch.no_grad():
                        self.id_bias.weight.scatter_(1, tab, c)
        if self.router == "default_moe":
            # Default MoE (Panda et al. 2504.12463, vatsal0/default-moe 962381f moe.py):
            # one default vector per expert = EMA of that expert's outputs, beta = cfg.dm_beta
            self.register_buffer("default_vector", torch.zeros(E, d, dtype=torch.float32))
        self.aux = {}
        self.dm_scores = None
        self.last_map = None   # [N, E] bool routing map of the last forward (for load metrics)
        self.last_sparsity = None

    # ----- routing: returns probs [N, E] (0 where not routed) and routing_map [N, E] bool -----
    def route(self, xs, T, B, feat=None):
        cfg, E, k = self.cfg, self.E, self.k
        logits = F.linear(xs if feat is None else feat, self.gate)
        N = logits.shape[0]
        train = self.training and torch.is_grad_enabled()
        if self.router in ("topk_aux", "global_aux", "default_moe"):
            if cfg.z_coeff > 0 and train:
                self.aux["z_loss"] = mg.z_loss_func(logits, cfg.z_coeff)
            probs, rmap = mg.topk_routing_with_score_function(
                logits, k, use_pre_softmax=cfg.pre_softmax, score_function="softmax")
            if self.router == "default_moe":
                # official router.py: scores = softmax over all experts; weights = top-k scores,
                # not renormalized (Megatron use_pre_softmax=True gives the same)
                self.dm_scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            if cfg.capacity_factor > 0:
                probs, rmap = mg.apply_router_token_dropping(probs, rmap, k, cfg.capacity_factor)
            if train and cfg.aux_coeff > 0:
                rmap_a, scores_a = mg.compute_routing_scores_for_aux_loss(logits, k, "softmax")
                if self.router in ("global_aux", "default_moe"):
                    # Megatron _apply_global_aux_loss on one rank: counts over the whole batch
                    self.aux["lbl"] = mg.switch_load_balancing_loss_func(
                        scores_a, rmap_a.sum(0), N, k, E, cfg.aux_coeff)
                else:
                    # micro-batch loss on R virtual data-parallel ranks (B/R sequences each),
                    # averaged as data-parallel gradient averaging does
                    R = cfg.lbl_ranks
                    sa = scores_a.view(T, R, B // R, E).transpose(0, 1).reshape(R, -1, E)
                    ra = rmap_a.view(T, R, B // R, E).transpose(0, 1).reshape(R, -1, E)
                    self.aux["lbl"] = sum(mg.switch_load_balancing_loss_func(
                        sa[r], ra[r].sum(0), sa.shape[1], k, E, cfg.aux_coeff) for r in range(R)) / R
        elif self.router == "dsv3":
            sel_bias = self.expert_bias
            if getattr(cfg, "std_route", False):
                # select on (score + bias - mean_e) / std_e; passed to the unchanged Megatron function
                # as a per-token bias so that scores + sel_bias = the standardized selection score
                sc = torch.sigmoid(logits.detach().float())
                if train:
                    with torch.no_grad():
                        r = 1.0 if self.sc_init.item() == 0 else cfg.std_rate
                        self.sc_mean.lerp_(sc.mean(0), r)
                        self.sc_std.lerp_(sc.std(0) + 1e-6, r)
                        self.sc_init.fill_(1.0)
                sel_bias = (sc + self.expert_bias - self.sc_mean) / self.sc_std - sc
            probs, rmap = mg.topk_routing_with_score_function(
                logits, k, score_function="sigmoid", expert_bias=sel_bias,
                num_groups=cfg.n_groups or None, group_topk=cfg.group_topk or None,
                scaling_factor=cfg.route_scale)
            if train:
                if cfg.seq_aux_coeff > 0:
                    # Megatron _apply_seq_aux_loss with one rank (total_num_tokens = T)
                    rmap_a, scores_a = mg.compute_routing_scores_for_aux_loss(logits, k, "sigmoid")
                    self.aux["seq_lbl"] = mg.switch_load_balancing_loss_func(
                        scores_a.reshape(T, -1), rmap_a.reshape(T, -1).sum(0), T, k, E,
                        cfg.seq_aux_coeff) / B
                with torch.no_grad():
                    if getattr(cfg, "type_balance", False):
                        # TYPE-BALANCE: each token counts 1/frequency(token id), mean weight 1
                        w = cfg.tok_weight[self.cur_ids.transpose(0, 1).reshape(-1)]
                        self.local_tokens_per_expert += (rmap.float() * (w / w.mean())[:, None]).sum(0)
                    else:
                        self.local_tokens_per_expert += rmap.sum(0)
        elif self.router == "gen":
            sfn = cfg.score_fn
            ids = self.cur_ids.transpose(0, 1).reshape(-1)                  # seq-first token ids
            if self.id_bias is not None:
                logits = logits + self.id_bias(ids).to(logits.dtype)
            s_all = score_fn(logits, sfn)                                   # [N, E], float32
            if self.hash:
                if self.hash_mode == "uni":
                    top_idx = self.hash_table[ids]
                else:
                    c = self.cur_ids
                    pv = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], 1).transpose(0, 1).reshape(-1)
                    key = pv if self.hash_mode == "prev" else (pv * 8191 + ids * 131) % (1 << 16)
                    top_idx = self.hash_table2[key]
                rm = getattr(self, "rare_mask", None)
                if rm is not None and cfg.rare_mode == "learned":   # rare token ids route by learned scores
                    lt = torch.topk(s_all.detach() + self.expert_bias, k, dim=1).indices
                    top_idx = torch.where(rm[ids][:, None], lt, top_idx)
            else:
                sel = s_all.detach()
                if getattr(cfg, "std_route", False):
                    if train:
                        with torch.no_grad():
                            r = 1.0 if self.sc_init.item() == 0 else cfg.std_rate
                            self.sc_mean.lerp_(sel.mean(0), r)
                            self.sc_std.lerp_(sel.std(0) + 1e-6, r)
                            self.sc_init.fill_(1.0)
                    sel = (sel - self.sc_mean) / self.sc_std
                sel = sel + self.expert_bias
                if cfg.balance == "quantile" and train:
                    with torch.no_grad():
                        top_v, top_i = torch.topk(sel, k + 1, dim=1)
                        alpha = top_v[:, k:k + 1]                           # (k+1)-th routing score
                        base = sel - self.expert_bias                       # score before bias
                        b_hat = -torch.quantile(base - alpha, 1 - k / E, dim=0)
                        self.next_bias.copy_(b_hat - b_hat.mean())
                        top_idx = top_i[:, :k]
                elif self.hash_k:
                    hidx = self.hk_table[ids]
                    lidx = torch.topk(sel.scatter(1, hidx, float("-inf")), k - self.hash_k, dim=1).indices
                    top_idx = torch.cat([hidx, lidx], 1)
                else:
                    top_idx = torch.topk(sel, k, dim=1).indices
            g = torch.gather(s_all, 1, top_idx)
            if self.hash and getattr(cfg, "hash_gate", "learned") == "uniform":
                g = torch.ones_like(g)          # knob: equal weights on the hashed experts
            g = g / (g.sum(-1, keepdim=True) + 1e-20)
            if cfg.route_scale:
                g = g * cfg.route_scale
            if getattr(self, "rare_mask", None) is not None and cfg.rare_mode == "skip":
                rk = getattr(cfg, "rare_k", 0)          # rare token ids keep their first rare_k experts (0: shared only)
                keep = torch.ones_like(g)
                keep[self.rare_mask[ids], rk:] = 0
                g = g * keep
                if rk:
                    g = g / (g.sum(-1, keepdim=True) + 1e-20) * (cfg.route_scale or 1.0)
            probs = torch.zeros_like(s_all).scatter(1, top_idx, g).to(logits.dtype)
            rmap = torch.zeros_like(s_all, dtype=torch.bool).scatter(1, top_idx, True)
            if train:
                if cfg.seq_aux_coeff > 0 and not self.hash:
                    # sequence-wise loss as Megatron _apply_seq_aux_loss, scores normalized per token
                    sa = s_all / (s_all.sum(-1, keepdim=True) + 1e-20)
                    ra = torch.zeros_like(rmap).scatter(1, torch.topk(sa, k, dim=1).indices, True)
                    self.aux["seq_lbl"] = mg.switch_load_balancing_loss_func(
                        sa.reshape(T, -1), ra.reshape(T, -1).sum(0), T, k, E, cfg.seq_aux_coeff) / B
                with torch.no_grad():
                    self.local_tokens_per_expert += rmap.sum(0)
        elif self.router == "expert_choice":
            # Zhou et al. 2022, eq. 1-2: S = softmax(X W_g); each expert takes its top-c tokens
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            cap = min(N, mg.get_capacity(N, E, cfg.ec_capacity))
            g, idx = torch.topk(scores, k=cap, dim=0)                      # [cap, E]
            rmap = torch.zeros_like(scores, dtype=torch.bool).scatter(0, idx, True)
            probs = torch.zeros_like(scores).scatter(0, idx, g)
        elif self.router == "remoe":
            # ReMoE fork router.py ReLURouter.l1_reg_load_balancing / apply_l1_reg
            probs = torch.relu(logits)
            rmap = probs > 0
            if train:
                self.aux["l1_reg"] = mg.switch_load_balancing_loss_func(
                    probs, rmap.sum(0), N, k, E, cfg.l1_coeff_state[0])
                self.last_sparsity = (1 - rmap.sum().float() / rmap.numel()).item()
        else:
            raise ValueError(self.router)
        return probs, rmap

    def forward(self, x):
        y = self._forward_core(x)
        if self.idb == "none":
            return y
        return y + self._id_branch(x, y)

    def _id_branch(self, x, y):
        ids = self.cur_ids
        B, T, D = x.shape
        b, t1 = self.idb, self.id_t1[ids]
        if b in ("r4", "r4x2", "r8", "r4e", "r4y"):
            src = y if b == "r4y" else x
            rows = t1
            if b == "r4e":
                e1 = (self.last_probs * self.last_map.to(self.last_probs.dtype)).argmax(-1).view(T, B).transpose(0, 1)
                rows = (t1 * 31 + e1 * 7 + 3) % self.idR
            out = torch.einsum("btr,btrd->btd", torch.einsum("btd,btdr->btr", src, self.k_iW[rows]), self.k_iU[rows])
            if b == "r4x2":
                r2 = self.id_t2[ids]
                out = out + torch.einsum("btr,btrd->btd", torch.einsum("btd,btdr->btr", x, self.k_iW[r2]), self.k_iU[r2])
            return out
        if b in ("sw", "swb"):
            out = torch.einsum("btr,btrd->btd", x @ self.k_iWs, self.k_iU[t1])
            if b == "swb":
                pv = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1)
                xp = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
                out = out + torch.einsum("btr,btrd->btd", xp @ self.k_iWp, self.k_iU[self.id_t3[pv]])
            return out
        if b == "vec":
            return self.k_iV[t1]
        if b == "gain":
            return y * self.k_iG1[t1]
        if b == "gain2":
            return y * (self.k_iG1[t1] + self.k_iG2[self.id_t2[ids]])
        if b == "ple":
            return (F.gelu(x @ self.k_iG) * self.k_iE[t1]) @ self.k_iP
        raise ValueError(b)

    def _forward_core(self, x):
        B, T, D = x.shape
        xs = x.transpose(0, 1).reshape(T * B, D)       # seq-first, as Megatron
        self.aux = {}
        ri, sh_out, feat = self.route_input, None, None
        if ri in ("shared_h", "shared_h_sg", "shared_out"):
            hs = F.silu(self.shared.w1(xs)) * self.shared.w3(xs)
            sh_out = self.shared.w2(hs)                                    # reused below, no recompute
            feat = {"shared_h": hs, "shared_h_sg": hs.detach(), "shared_out": sh_out}[ri]
        elif ri in ("mlp", "rand_mlp"):
            feat = F.silu(self.r1(xs)) * self.r3(xs)
        if feat is not None and getattr(self.cfg, "route_norm", False):
            feat = feat * torch.rsqrt(feat.pow(2).mean(-1, keepdim=True) + 1e-6)
        probs, rmap = self.route(xs, T, B, feat)
        self.last_map = rmap.detach()
        self.last_probs = probs.detach()
        if self.ex != "none":
            return self._forward_ex(xs, probs, rmap, sh_out).view(T, B, D).transpose(0, 1)
        y = torch.zeros_like(xs)
        dm = self.router == "default_moe"
        dm_update = dm and self.training and torch.is_grad_enabled()
        if dm_update:
            new_mean = torch.zeros_like(self.default_vector)
        # manager 2026-11-22 18:20: on the GPU, run every expert on every token with weight 0 for unrouted
        # tokens. Same math as the loop (loss equal, grads within 6e-9), but no nonzero() CPU-GPU sync per
        # expert: XS 600 steps 274 s -> 36 s on the A4000. CPU runs keep the loop; large sizes keep it (memory).
        fast = (xs.is_cuda and not dm and getattr(self.cfg, "moe_fast", True)
                and xs.shape[0] * self.E * self.w1.shape[-1] < 2 ** 26)
        if fast:
            wgt = (probs * rmap.to(probs.dtype)).to(xs.dtype)
            hh = F.silu(torch.einsum("nd,edh->enh", xs, self.w1)) * torch.einsum("nd,edh->enh", xs, self.w3)
            y = torch.einsum("enh,ehd,ne->nd", hh, self.w2, wgt)
        for e in (range(0) if fast else range(self.E)):
            idx = rmap[:, e].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            xe = xs.index_select(0, idx)
            he = F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])
            oe = he @ self.w2[e]
            if dm_update:
                new_mean[e] = oe.detach().float().mean(0)
            y.index_add_(0, idx, oe * probs.index_select(0, idx)[:, e:e + 1].to(xs.dtype))
        if dm:
            if dm_update:
                # official default_vector.py: default = beta * default + (1 - beta) * mean output
                # (an expert with no tokens this step only decays, as in the kernel)
                self.default_vector.mul_(self.cfg.dm_beta).add_((1 - self.cfg.dm_beta) * new_mean)
            # unselected experts contribute p_e * default_e (selected experts get weight 0)
            masked = self.dm_scores.masked_fill(rmap, 0.0)
            y = y + (masked @ self.default_vector).to(xs.dtype)
        if getattr(self, "rare_mask", None) is not None and self.cfg.rare_mode == "detach":
            r = self.rare_mask[self.cur_ids.transpose(0, 1).reshape(-1)]   # rare ids: routed output kept, no gradient
            y = torch.where(r[:, None], y.detach(), y)
        if self.shared is not None:
            y = y + (sh_out if sh_out is not None else self.shared(xs))
        return y.view(T, B, D).transpose(0, 1)

    def _ex_hidden(self, e, xe):
        """expert e's gated hidden activations under the round-12 activation ideas."""
        ex = self.ex
        a, b = xe @ self.w1[e], xe @ self.w3[e]
        if ex == "act_mix":                     # expert e uses activation e % 4
            m = e % 4
            g = F.silu(a) if m == 0 else F.relu(a).square() if m == 1 else F.gelu(a) if m == 2 else a.abs()
        elif ex == "act_pow":                   # relu(a)^p_e, p_e learned, starts at 2
            g = (F.relu(a) + 1e-6).pow(1 + F.softplus(self.k_p[e] + 0.5413))
        elif ex == "act_slope":                 # silu(beta_e a), beta_e learned, starts at 1
            g = F.silu(a * self.k_p[e].exp())
        elif ex == "act_bias":                  # learned per-expert per-unit threshold
            g = F.silu(a + self.k_bias[e])
        elif ex == "bil_half":                  # odd experts bilinear (no nonlinearity)
            g = a if e % 2 else F.silu(a)
        elif ex == "act_learn":                 # learned mix of silu, relu^2, identity per expert
            w = self.k_mix[e].softmax(0)
            g = w[0] * F.silu(a) + w[1] * F.relu(a).square() + w[2] * a
        elif ex == "relu2":
            g = F.relu(a).square()
        else:
            g = F.silu(a)
        he = g * b
        if ex == "hnorm":                       # RMS-normalized hidden, learned per-expert gain
            he = he * torch.rsqrt(he.pow(2).mean(-1, keepdim=True) + 1e-6) * self.k_g[e]
        return he

    def _forward_ex(self, xs, probs, rmap, sh_out):
        """round-12 MoE forward: per-slot expert outputs O [N,k,D] with weights P [N,k], then the
        idea's combination, shared-expert coupling and cross-layer terms."""
        ex, (N, D), k, E = self.ex, xs.shape, self.k, self.E
        h = self.w1.shape[2]
        rn = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
        gctx = getattr(self, "gctx", None)
        xl = gctx.get("xl") if gctx is not None else None
        prev = xl[-1] if xl else None
        order = probs.masked_fill(~rmap, -1.0).argsort(1, descending=True)[:, :k]     # slot -> expert
        P = probs.gather(1, order).to(xs.dtype) * rmap.gather(1, order).to(xs.dtype)    # [N,k]
        slot = torch.full((N, E), -1, dtype=torch.long, device=xs.device)
        slot.scatter_(1, order, torch.arange(k, device=xs.device).expand(N, k))
        slot = slot.masked_fill(~rmap, -1)
        pn = P / P.sum(1, keepdim=True).clamp_min(1e-9)
        keep = torch.ones(N, k, dtype=torch.bool, device=xs.device)
        if ex in ("kent", "width_ent"):
            H = -(pn * (pn + 1e-9).log()).sum(1)
            low = H < H.median()
        if ex == "kthr":                        # drop slots whose weight < exb x top weight
            keep = P >= (self.cfg.exb or 0.5) * P[:, :1]
        elif ex == "kent":                      # confident (low-entropy) half of tokens: k-1 experts
            keep[:, k - 1] = ~low
        elif ex == "topp":                      # keep slots until normalized mass >= exb (top-p)
            keep = (pn.cumsum(1) - pn) < (self.cfg.exb or 0.7)
        sh = None
        if ex in ("chain_sh",):
            sh = self.shared(xs)
        base_in = xs
        if ex == "chain_sh":
            base_in = xs + sh
        elif ex == "xl_in" and prev is not None:
            base_in = xs + self.k_a * prev
        O = xs.new_zeros(N, k, D)
        chain = ex in ("chain", "chain_n", "chain_last", "chain_sc", "chain_mid", "chain_last_n",
                       "chain_sh_last", "chain_det", "chain_first2")
        if ex == "chain_sh_last":
            sh = self.shared(xs)
        last_only = ex in ("chain_last", "chain_last_n", "chain_sh_last", "chain_det")
        for s in range(k) if chain else [None]:
            if chain and s > 0 and (not last_only or s == k - 1) and (ex != "chain_first2" or s == 1):
                acc = (P[:, :1, None] * O[:, :1]).sum(1) if ex == "chain_mid" else (P[:, :s, None] * O[:, :s]).sum(1)
                if ex == "chain_det":
                    acc = acc.detach()
                if ex == "chain_sc":
                    acc = acc * self.k_a
                inp = xs + acc + (sh if ex == "chain_sh_last" else 0)
                if ex in ("chain_n", "chain_last_n"):
                    inp = rn(inp) * xs.pow(2).mean(-1, keepdim=True).sqrt()
            else:
                inp = base_in
            for e in range(E):
                sel = rmap[:, e] & keep.gather(1, slot[:, e].clamp_min(0)[:, None])[:, 0]
                if s is not None:
                    sel = sel & (slot[:, e] == s)
                idx = sel.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                sl = slot[idx, e]
                he = self._ex_hidden(e, inp.index_select(0, idx))
                if ex == "width_soft":          # soft width: unit j weighted by sigmoid(h k p - j)
                    kc = h * (pn[idx, sl] * k).clamp(max=1.0)
                    he = he * torch.sigmoid(kc[:, None] - torch.arange(h, device=xs.device)[None]).to(he.dtype)
                if ex in ("width_rank", "width_prob", "width_ent", "width_prob_f"):
                    if ex == "width_rank":      # slot s uses the first h >> s hidden units
                        kk = h >> sl
                    elif ex in ("width_prob", "width_prob_f"):   # hidden units in proportion to the slot's weight
                        kk = torch.ceil(h * (pn[idx, sl] * k).clamp(max=1.0).pow(self.cfg.exb or 1.0)).long()
                        if ex == "width_prob_f":
                            kk = kk.clamp(min=h // 4)
                    else:                       # confident tokens use half width
                        kk = torch.where(low[idx], h // 2, h)
                    he = he * (torch.arange(h, device=xs.device)[None] < kk[:, None]).to(he.dtype)
                O = O.index_put((idx, sl), he @ self.w2[e])
        Pk = P * keep.to(P.dtype)
        if ex == "ortho":                       # remove from each output its part along the others' mean
            m = (O.sum(1, keepdim=True) - O) / max(k - 1, 1)
            O = O - (O * m).sum(-1, keepdim=True) / ((m * m).sum(-1, keepdim=True) + 1e-12) * m
        if ex == "outattn":                     # combination weights: log p + <rn(o_s), x W>
            q = xs @ self.k_W
            lg = (Pk + 1e-9).log() + (rn(O) * q[:, None]).sum(-1) / math.sqrt(D)
            lg = lg.masked_fill(Pk <= 0, -1e9)
            y = ((Pk.sum(1, keepdim=True) * lg.softmax(1))[..., None] * O).sum(1)
        else:
            y = (Pk[..., None] * O).sum(1)
        if ex in ("bil", "bilw"):               # low-rank pairwise products of the routed outputs
            u = rn(O) @ self.k_U
            if ex == "bilw":
                u = u * Pk[..., None]
            y = y + (0.5 * (u.sum(1).square() - u.square().sum(1))) @ self.k_V
        elif ex == "maxpool":
            y = y + self.k_c * (Pk[..., None] * O).amax(1)
        elif ex == "gatemul":                   # top-1 output gated by a projection of the top-2 output
            y = y + (Pk[:, :1] * O[:, 0]) * (rn(O[:, 1]) @ self.k_W)
        elif ex == "prodsum":
            y = y + self.k_c * rn(O[:, 0]) * rn(O[:, 1])
        elif ex == "normsum":
            y = rn(y) * self.k_g
        y_pure = y
        if prev is not None:
            if ex == "xl_prev":
                y = y + self.k_a * prev
            elif ex == "xl_gate":
                c = F.cosine_similarity(y, prev, dim=-1, eps=1e-12)[:, None]
                y = y * 2 * torch.sigmoid(self.k_wb[0] * c + self.k_wb[1])
            elif ex == "xl_sub":
                y = y - self.k_a * (y * prev).sum(-1, keepdim=True) / ((prev * prev).sum(-1, keepdim=True) + 1e-12) * prev
            elif ex == "xl_dense":
                y = y + sum(self.k_a[j] * xl[j] for j in range(len(xl)))
        if ex == "xl_prev_sh" and gctx is not None and gctx.get("xl_sh"):
            y = y + self.k_a * gctx["xl_sh"][-1]
        if self.shared is not None:
            if sh is None:
                if ex == "chain_rev":
                    sh = self.shared(xs + y)
                elif ex == "chain_rev_sc":
                    sh = self.shared(xs + self.k_a * y)
                elif ex == "sh_split":
                    hs = F.silu(self.shared.w1(xs)) * self.shared.w3(xs)
                    H2 = hs.shape[1] // 2
                    On = F.normalize(O, dim=-1)
                    A = ((On @ On.transpose(1, 2)).sum((1, 2)) - k) / (k * (k - 1))
                    gt = 2 * torch.sigmoid(self.k_wb[0] * A + self.k_wb[1])[:, None]
                    sh = self.shared.w2(torch.cat([hs[:, :H2], hs[:, H2:] * gt], 1))
                else:
                    sh = sh_out if sh_out is not None else self.shared(xs)
            if ex == "sh_agree":                # shared output scaled by routed-expert agreement
                On = F.normalize(O, dim=-1)
                A = ((On @ On.transpose(1, 2)).sum((1, 2)) - k) / (k * (k - 1))
                sh = sh * 2 * torch.sigmoid(self.k_wb[0] * A + self.k_wb[1])[:, None]
            elif ex == "sh_conf":               # shared output scaled by router confidence
                sh = sh * 2 * torch.sigmoid(self.k_wb[0] * pn[:, :1] + self.k_wb[1])
            elif ex in ("sh_agree_v", "sh_agree_both"):
                On = F.normalize(O, dim=-1)
                A = (((On @ On.transpose(1, 2)).sum((1, 2)) - k) / (k * (k - 1)))[:, None]
                gt = 2 * torch.sigmoid(self.k_wb[0] * A + self.k_wb[1])
                sh = sh * gt
                if ex == "sh_agree_both":
                    y = y * (2 - gt)
            elif ex in ("sh_conf_v", "sh_conf_r", "sh_conf_both"):
                gt = 2 * torch.sigmoid(self.k_wb[0] * pn[:, :1] + self.k_wb[1])
                if ex == "sh_conf_r":
                    y = y * gt
                else:
                    sh = sh * gt
                    if ex == "sh_conf_both":
                        y = y * (2 - gt)
            elif ex == "sh_ent":                # shared output scaled by router entropy
                He = -(pn * (pn + 1e-9).log()).sum(1, keepdim=True)
                sh = sh * 2 * torch.sigmoid(self.k_wb[0] * He + self.k_wb[1])
            elif ex == "sh_cos":                # shared output scaled by its agreement with the routed sum
                c = F.cosine_similarity(sh, y, dim=-1, eps=1e-12)[:, None]
                sh = sh * 2 * torch.sigmoid(self.k_wb[0] * c + self.k_wb[1])
            elif ex == "sh_gate":
                sh = sh * 2 * torch.sigmoid(rn(y) @ self.k_W)
            elif ex == "sh_proj":
                sh = sh - (sh * y).sum(-1, keepdim=True) / ((y * y).sum(-1, keepdim=True) + 1e-12) * y
            elif ex == "sh_film":
                y = y * (1 + torch.tanh(rn(sh) @ self.k_A @ self.k_B))
            elif ex == "routed_gate_sh":
                y = y * 2 * torch.sigmoid(rn(sh) @ self.k_w)[:, None]
            if gctx is not None and "xl_sh" in gctx:
                gctx["xl_sh"].append(sh)
            y = y + sh
        if xl is not None:
            xl.append(y_pure)
        return y

    @torch.no_grad()
    def update_expert_bias(self):
        """Megatron finalize_model_grads -> get_updated_expert_bias, once per optimizer step."""
        if self.router == "dsv3":
            self.expert_bias.copy_(mg.get_updated_expert_bias(
                self.local_tokens_per_expert, self.expert_bias, self.cfg.bias_rate))
            self.local_tokens_per_expert.zero_()
        elif self.router == "gen" and not self.hash:
            if self.cfg.balance == "quantile":
                self.expert_bias.copy_(self.next_bias)               # computed last step, no rate
            else:
                self.expert_bias.copy_(mg.get_updated_expert_bias(
                    self.local_tokens_per_expert, self.expert_bias, self.cfg.bias_rate))
            self.local_tokens_per_expert.zero_()


class Block(nn.Module):
    def __init__(self, cfg, layer=0, ctx=None):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, cfg.n_heads)
        self.inv = cfg.router == "inv"
        if self.inv:
            import inventions
            self.ctx = ctx
            self.ffn = inventions.make(cfg.inv, d, cfg, layer, ctx)
        else:
            self.ffn = SwiGLU(d, cfg.dense_hidden) if cfg.router == "dense" else MoE(d, cfg, layer)

    def forward(self, x, cos, sin):
        if self.inv:
            c = self.ctx
            c["delta"] = x if c.get("prev_in") is None else x - c["prev_in"]
            c["prev_in"] = x
            a = self.attn(self.n1(x), cos, sin, cap=c if c.get("need_att") else None)
            c["attn"] = a
            x = x + a
            c["resid"] = x
            return x + self.ffn(self.n2(x))
        x = x + self.attn(self.n1(x), cos, sin)
        if getattr(self.ffn, "cfg", None) is not None and self.ffn.cfg.ckpt_ffn and torch.is_grad_enabled():
            # memory only: recompute the FFN in backward (same values); used for ReMoE at size M,
            # whose early dense routing doubles FFN activations. Not for dsv3 (its load counter
            # would count twice).
            return x + torch.utils.checkpoint.checkpoint(lambda h: self.ffn(self.n2(h)), x, use_reentrant=False)
        return x + self.ffn(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.ctx = {}                                                    # invention side inputs
        self.blocks = nn.ModuleList([Block(cfg, i, self.ctx) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "gate" not in n and "id_bias" not in n and not n.split(".")[-1].startswith("k_"):
                nn.init.normal_(p, std=0.02)
        if getattr(cfg, "pe", "none") != "none":                       # previous-token id embedding (train.py inits it)
            self.emb_prev = nn.Embedding(cfg.vocab, cfg.d_model)
            nn.init.zeros_(self.emb_prev.weight)
        dh = cfg.d_model // cfg.n_heads
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        ang = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(self, idx, targets=None):
        for m in self.moe_layers():
            m.cur_ids = idx
            m.gctx = self.ctx
        self.ctx.update(ids=idx, prev_in=None, prev_logits=None, prev_map=None, stale=None, xl=[], xl_sh=[])
        x = self.emb(idx)
        if hasattr(self, "emb_prev"):                                  # x_t += E_prev[x_{t-1}], nothing at t = 0
            pv = F.pad(idx[:, :-1], (1, 0))
            x = x + self.emb_prev(pv) * (torch.arange(idx.shape[1], device=idx.device) > 0)[None, :, None]
        self.ctx["emb"] = x
        T = idx.shape[1]
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        return logits, loss.view(targets.shape)

    def aux_loss(self):
        tot, parts = 0.0, {}
        for m in self.moe_layers():
            for k, v in m.aux.items():
                tot = tot + v
                parts[k] = parts.get(k, 0.0) + float(v)
        return tot, parts

    def count_params(self):
        """total and active parameters (active = per-token: k routed experts + shared + rest)."""
        tot = sum(p.numel() for p in self.parameters())
        cfg = self.cfg
        if cfg.router in ("dense", "inv"):
            return tot, tot
        per_expert = 3 * cfg.d_model * cfg.expert_hidden
        k_eff = cfg.topk if cfg.router != "expert_choice" else cfg.ec_capacity
        inactive = (cfg.n_experts - k_eff) * per_expert * cfg.n_layers
        return tot, tot - inactive
