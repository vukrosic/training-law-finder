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
        g = lambda k, dv=0: getattr(cfg, k, dv)
        d, V = cfg.d_model, cfg.vocab
        self.emb = FacEmb(V, d, int(g("hd_efac"))) if g("hd_efac") else nn.Embedding(V, d)
        self.ctx = {}                                                    # invention side inputs
        self.blocks = nn.ModuleList([Block(cfg, i, self.ctx) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.hdon = any(g(k, "none" if k == "hd_tsg" else 0) not in (0, "none") for k in HD_KEYS) or g("hd_hmul", 1.0) != 1.0 or g("hd_emul", 1.0) != 1.0 or g("hd_ex0", -1.0) >= 0
        if g("hd_tie") or g("hd_tadapt") or g("hd_tres") or g("hd_tln") or g("hd_ptie") or g("hd_thalf") or g("hd_tsg", "none") != "none":
            self.head = TiedHead(self)
        elif g("hd_rank"):
            self.head = LowRankHead(d, V, int(g("hd_rank")))
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.hd = HDExtras(cfg) if self.hdon else None
        # TARGET SEARCH: training-only modules (DeepSeek-V3 MTP modules, extra heads); never used at eval
        import targets
        self.tmods = targets.make_modules(cfg, self.ctx)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "gate" not in n and "id_bias" not in n:
                nn.init.normal_(p, std=0.02)
        if self.hd is not None:
            self.hd.special_init(self)
        dh = cfg.d_model // cfg.n_heads
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        ang = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def moe_layers(self, train=False):
        bl = list(self.blocks)
        if train and self.tmods is not None and hasattr(self.tmods, "blocks"):
            bl += list(self.tmods.blocks)
        return [b.ffn for b in bl if isinstance(b.ffn, MoE)]

    def forward(self, idx, targets=None):
        for m in self.moe_layers():
            m.cur_ids = idx
        self.ctx.update(ids=idx, prev_in=None, prev_logits=None, prev_map=None, stale=None)
        x = self.emb(idx) if self.hd is None else self.hd.embed(self, idx)
        self.ctx["emb"] = x
        x0 = x
        T = idx.shape[1]
        hs = []
        for i, b in enumerate(self.blocks):
            if self.hd is not None and self.hd.ex0 is not None and i > 0:
                x = x + self.hd.ex0[i] * x0
            x = b(x, self.cos[:T], self.sin[:T])
            hs.append(x)
        self.ctx["hs"] = hs
        self.ctx["h"] = x
        logits = self.head(self.norm(x)) if self.hd is None else self.hd.out(self, x, hs, x0, idx)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        return logits, loss.view(targets.shape)

    def aux_loss(self):
        tot, parts = 0.0, {}
        for m in self.moe_layers(train=True):
            for k, v in m.aux.items():
                tot = tot + v
                parts[k] = parts.get(k, 0.0) + float(v)
        return tot, parts

    def count_params(self):
        """total and active parameters (active = per-token: k routed experts + shared + rest)."""
        tot = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tmods."))
        cfg = self.cfg
        if cfg.router in ("dense", "inv"):
            return tot, tot
        per_expert = 3 * cfg.d_model * cfg.expert_hidden
        k_eff = cfg.topk if cfg.router != "expert_choice" else cfg.ec_capacity
        inactive = (cfg.n_experts - k_eff) * per_expert * cfg.n_layers
        return tot, tot - inactive


# ---------------- TARGET SEARCH round 12+: output-head and input-embedding designs (all off by default) ----------------
HD_KEYS = ("hd_tie", "hd_cap", "hd_rank", "hd_mos", "hd_gate", "hd_lmix", "hd_skip", "hd_cos", "hd_temp", "hd_vscale",
           "hd_nonorm", "hd_mlp", "hd_moeh", "hd_dual", "hd_capl", "hd_hnorm", "hd_bil", "hd_film", "hd_ctx", "hd_eln",
           "hd_escale", "hd_efac", "hd_ehash", "hd_eglu", "hd_eout", "hd_ebias", "hd_ecos", "hd_ptie", "hd_tadapt",
           "hd_tres", "hd_tsg", "hd_tln", "hd_tT", "hd_eproj", "hd_thalf")


class FacEmb(nn.Module):
    """ALBERT factorized embedding: V x r table then r -> d."""
    def __init__(self, V, d, r):
        super().__init__()
        self.e = nn.Embedding(V, r)
        self.up = nn.Linear(r, d, bias=False)

    @property
    def weight(self):
        return self.up(self.e.weight)

    def forward(self, idx):
        return self.up(self.e(idx))


class TiedHead(nn.Module):
    """head that reads the input embedding matrix (no own V x d weight)."""
    def __init__(self, gpt):
        super().__init__()
        self.__dict__["_g"] = gpt

    @property
    def weight(self):
        g = self._g
        return g.hd.head_weight(g) if g.hd is not None else g.emb.weight

    def forward(self, x):
        return F.linear(x, self.weight)


class LowRankHead(nn.Module):
    def __init__(self, d, V, r):
        super().__init__()
        self.a = nn.Linear(d, r, bias=False)
        self.b = nn.Linear(r, V, bias=False)

    @property
    def weight(self):
        return self.b.weight @ self.a.weight

    def forward(self, x):
        return self.b(self.a(x))


def _rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class HDExtras(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g = lambda k, dv=0: getattr(cfg, k, dv)
        self.c = cfg
        d, V, L = cfg.d_model, cfg.vocab, cfg.n_layers
        P = lambda *sh: nn.Parameter(torch.empty(*sh))
        self.ex0 = nn.Parameter(torch.full((L,), float(g("hd_ex0", -1.0)))) if g("hd_ex0", -1.0) >= 0 else None
        if g("hd_ehash"): self.hash = nn.Embedding(int(g("hd_ehash")), d)
        if g("hd_eglu"): r = int(g("hd_eglu")); self.eg_a = P(V, r); self.eg_b = P(r, d)        # gate = 2 sigmoid(a b), b zero
        if g("hd_escale"): self.es = nn.Parameter(torch.ones(V))
        if g("hd_ecos"): self.ecg = nn.Parameter(torch.tensor(0.02 * d ** 0.5))
        if g("hd_eproj"): self.ep = P(d, d)
        if g("hd_tln"): self.g_in = nn.Parameter(torch.ones(d)); self.g_out = nn.Parameter(torch.ones(d))
        if g("hd_eln"): self.eln = RMSNorm(d)
        if g("hd_ptie"): self.k = int(round(g("hd_ptie") * d)); self.pw = P(V, d - self.k)
        if g("hd_tres"): r = int(g("hd_tres")); self.ta = P(V, r); self.tb = P(r, d)
        if g("hd_tadapt"): self.tm = P(d, d)
        if g("hd_thalf"): self.hw = P(V, d)
        if g("hd_tT"): self.tT = nn.Parameter(torch.zeros(()))
        if g("hd_lmix"): self.lm = nn.Parameter(torch.zeros(L))
        if g("hd_eout"): self.eo = nn.Parameter(torch.zeros(()))
        if g("hd_ebias"): self.eb = P(V, d)
        if g("hd_nonorm") == 2: pass
        if g("hd_mlp"): h = int(g("hd_mlp") * d); self.m1 = P(h, d); self.m2 = P(d, h)
        if g("hd_film"): self.fg = P(V, d)
        if g("hd_bil"): self.bp = P(d, d)
        if g("hd_cos"): self.ct = nn.Parameter(torch.tensor(math.log(g("hd_cos"))))
        if g("hd_hnorm"): self.rg = nn.Parameter(torch.ones(V))
        if g("hd_moeh"): E = int(g("hd_moeh")); self.mw = P(E - 1, V, d); self.mr = P(E, d)
        if g("hd_dual"): self.dn = RMSNorm(d); self.dw = P(V, d)
        if g("hd_skip"): r = int(g("hd_skip")); self.sa = P(d, r); self.sb = P(V, r)
        if g("hd_ctx"): r = int(g("hd_ctx")); self.xa = P(d, r); self.xb = P(V, r)
        if g("hd_gate"): self.gw = P(d); self.gb = nn.Parameter(torch.zeros(()))
        if g("hd_temp"): self.tw = P(d)
        if g("hd_vscale"): self.vs = nn.Parameter(torch.ones(V)); self.vb = nn.Parameter(torch.zeros(V)) if g("hd_vscale") >= 2 else None
        if g("hd_capl"): self.lc = nn.Parameter(torch.tensor(math.log(g("hd_capl"))))
        if g("hd_mos"): K = int(g("hd_mos")); self.mp = P(K, d, d); self.mpi = P(K, d)

    @torch.no_grad()
    def special_init(self, gpt):
        """after the global normal(0.02) init: zero the residual/added paths, identity for maps, exact weight-norm."""
        for n in ("eg_b", "tb", "eb", "m2", "fg", "bp", "sb", "xb"):
            if hasattr(self, n): getattr(self, n).zero_()
        for n in ("gw", "tw"):
            if hasattr(self, n): getattr(self, n).zero_()
        for n in ("ep", "tm"):
            if hasattr(self, n): getattr(self, n).copy_(torch.eye(self.c.d_model))
        if hasattr(self, "lm"): self.lm[-1] = 4.0
        if hasattr(self, "mr"): nn.init.normal_(self.mr, std=0.02)
        if hasattr(self, "mpi"): nn.init.normal_(self.mpi, std=0.02)
        if hasattr(self, "mp"): nn.init.normal_(self.mp, std=self.c.d_model ** -0.5)
        if hasattr(self, "rg"): self.rg.copy_(gpt.head.weight.norm(dim=1))

    def embed(self, gpt, idx):
        c = self.c
        g = lambda k, dv=0: getattr(c, k, dv)
        e = F.embedding(idx, gpt.emb.weight.detach()) if g("hd_tsg", "none") == "emb" else gpt.emb(idx)
        if hasattr(self, "hash"): e = e + self.hash((idx * 40503 + 7) % self.hash.num_embeddings)
        if hasattr(self, "eg_a"): e = e * 2 * torch.sigmoid(self.eg_a[idx] @ self.eg_b)
        if hasattr(self, "es"): e = e * self.es[idx].unsqueeze(-1)
        if hasattr(self, "ecg"): e = F.normalize(e, dim=-1) * self.ecg
        if hasattr(self, "ep"): e = e @ self.ep.T
        if hasattr(self, "g_in"): e = e * self.g_in
        if g("hd_emul", 1.0) != 1.0: e = e * g("hd_emul", 1.0)
        if hasattr(self, "eln"): e = self.eln(e)
        return e

    def head_weight(self, gpt):
        c = self.c
        g = lambda k, dv=0: getattr(c, k, dv)
        E = gpt.emb.weight
        if g("hd_tsg", "none") == "head": E = E.detach()
        if hasattr(self, "pw"): return torch.cat([E[:, :self.k], self.pw], 1)
        W = E
        if hasattr(self, "tb"): W = W + self.ta @ self.tb
        if hasattr(self, "tm"): W = W @ self.tm.T
        if hasattr(self, "g_out"): W = W * self.g_out
        if hasattr(self, "hw"): W = 0.5 * (W + self.hw)
        return W

    def out(self, gpt, x, hs, x0, idx):
        c = self.c
        g = lambda k, dv=0: getattr(c, k, dv)
        if hasattr(self, "lm"): x = sum(w * h for w, h in zip(torch.softmax(self.lm, 0), hs))
        if hasattr(self, "eo"): x = x + self.eo * x0
        if hasattr(self, "eb"): x = x + self.eb[idx]
        nn_ = g("hd_nonorm")
        n = x if nn_ == 1 else (_rms(x) if nn_ == 2 else gpt.norm(x))
        if hasattr(self, "m1"): n = n + F.relu(n @ self.m1.T).square() @ self.m2.T
        if hasattr(self, "fg"): n = n * (1 + self.fg[idx])
        if hasattr(self, "bp"): n = n * (1 + x0 @ self.bp.T)
        if g("hd_hmul", 1.0) != 1.0: n = n * g("hd_hmul", 1.0)
        W = gpt.head.weight
        if hasattr(self, "mp"):                                          # mixture of softmaxes: returns log-probs
            pi = torch.log_softmax(n @ self.mpi.T, -1)                   # [B,T,K]
            lps = [torch.log_softmax(F.linear(torch.tanh(n @ self.mp[k].T), W).float(), -1) + pi[..., k:k + 1].float()
                   for k in range(self.mp.shape[0])]
            return torch.logsumexp(torch.stack(lps, 0), 0)
        if hasattr(self, "ct"):
            lg = F.linear(F.normalize(n, dim=-1), F.normalize(W, dim=-1)) * self.ct.exp()
        elif hasattr(self, "rg"):
            lg = F.linear(n, F.normalize(W, dim=-1) * self.rg.unsqueeze(1))
        elif isinstance(gpt.head, LowRankHead):
            lg = gpt.head(n)
        else:
            lg = F.linear(n, W)
        if hasattr(self, "mr"):                                          # MoE head: top-1 of E full-vocab heads, straight-through gate
            p = torch.softmax(n @ self.mr.T, -1)
            top = p.argmax(-1, keepdim=True)
            pg = p.gather(-1, top)
            out = lg
            for e in range(1, self.mr.shape[0]):
                out = torch.where(top == e, F.linear(n, self.mw[e - 1]), out)
            lg = out * (pg / pg.detach())
        if hasattr(self, "dw"): lg = lg + F.linear(self.dn(hs[-2]), self.dw)
        if hasattr(self, "sa"): lg = lg + F.linear(x0 @ self.sa, self.sb)
        if hasattr(self, "xa"):
            cm = n.cumsum(1) / torch.arange(1, n.shape[1] + 1, device=n.device).view(1, -1, 1)
            lg = lg + F.linear(cm @ self.xa, self.xb)
        if hasattr(self, "gw"):
            gg = torch.sigmoid(n @ self.gw + self.gb).unsqueeze(-1)
            lg = gg * lg + (1 - gg) * F.linear(n, gpt.emb.weight)
        if hasattr(self, "tw"): lg = lg * torch.exp(n @ self.tw).unsqueeze(-1)
        if hasattr(self, "vs"):
            lg = lg * self.vs
            if self.vb is not None: lg = lg + self.vb
        if hasattr(self, "tT"): lg = lg * self.tT.exp()
        if g("hd_cap"): cp = g("hd_cap"); lg = cp * torch.tanh(lg / cp)
        if hasattr(self, "lc"): cp = self.lc.exp(); lg = cp * torch.tanh(lg / cp)
        return lg
