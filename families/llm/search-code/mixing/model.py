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
        if getattr(cfg, "mix", "attn") == "attn":
            self.attn = Attention(d, cfg.n_heads)
        else:                                                           # mixing search: mixers.py
            import mixers
            self.attn = mixers.make(cfg, layer)
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
            if p.dim() == 2 and "gate" not in n and "id_bias" not in n:
                nn.init.normal_(p, std=0.02)
        for m in self.modules():
            if hasattr(m, "post_init"):
                m.post_init()
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
        self.ctx.update(ids=idx, prev_in=None, prev_logits=None, prev_map=None, stale=None)
        if getattr(self.cfg, "mix", "attn") != "attn":
            import mixers
            mixers.CTX["ids"] = idx
        x = self.emb(idx)
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
