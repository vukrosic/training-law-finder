"""MOE-PRETRAIN benchmark scoring.

Language modelling: validation loss (nats per token) and bits per byte on the fixed FineWeb-Edu
validation split (data/fineweb_edu/val.bin, docs 0..511 of the pinned shard). Sequence i covers
val[i*256 : i*256+257] (input 256 tokens, next-token targets). The laptop eval uses the first
--eval_seqs sequences (default 64 = 16,384 target tokens); eval_seqs=0 scores all 2,050.
Bits per byte = sum of target-token losses / ln 2 / UTF-8 bytes of those targets, so it does not
depend on the tokenizer.

Load balance (the metrics MoE papers report), per MoE layer and per eval batch of 16 sequences
(4,096 tokens), then averaged over batches and layers:
  load_cv        std / mean of tokens per expert (Switch / ST-MoE style coefficient of variation)
  maxvio         (max - mean) / mean of tokens per expert (MaxVio_batch, Wang et al. 2024 = DeepSeek
                 auxiliary-loss-free paper)
  drop_at_1.25   share of token->expert assignments over capacity ceil(1.25 * N * k / E), i.e. the
                 tokens a capacity-factor-1.25 (Switch / GShard) system would drop
  zero_expert    share of tokens that get no routed expert (Expert Choice / ReMoE can do this)
  experts_per_token  mean routed experts per token
"""
import json, math, os
import numpy as np
import torch

TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tokenizer", "bpe8192.json")


def token_bytes(path=TOK):
    """UTF-8 byte length of each token id. Byte-level BPE: one vocab character = one byte;
    the special token <|endoftext|> has 0 bytes."""
    vocab = json.load(open(path))["model"]["vocab"]
    tb = np.zeros(len(vocab), dtype=np.int64)
    for s, i in vocab.items():
        tb[i] = len(s)
    tb[0] = 0
    return torch.from_numpy(tb)


@torch.no_grad()
def evaluate(model, val, tb, cfg, n_seqs=64, device="cpu"):
    T, B = cfg.seq_len, cfg.batch
    total = (len(val) - 1) // T
    n = total if n_seqs == 0 else min(n_seqs, total)
    n = (n // B) * B
    model.eval()
    loss_sum, byte_sum, tok_sum = 0.0, 0, 0
    lb = {k: [] for k in ["load_cv", "maxvio", "drop_at_1.25", "zero_expert", "experts_per_token"]}
    moes = model.moe_layers()
    for i in range(0, n, B):
        xy = torch.from_numpy(np.stack([val[j * T:j * T + T + 1] for j in range(i, i + B)]).astype(np.int64)).to(device)
        _, loss = model(xy[:, :-1], xy[:, 1:])
        loss_sum += loss.sum().item()
        byte_sum += tb[xy[:, 1:].cpu()].sum().item()
        tok_sum += loss.numel()
        for m in moes:
            rm = m.last_map
            cnt = rm.sum(0).float()
            mean = cnt.mean()
            lb["load_cv"].append((cnt.std(unbiased=False) / mean).item())
            lb["maxvio"].append(((cnt.max() - mean) / mean).item())
            k_eff = cnt.sum().item() / rm.shape[0]
            cap = math.ceil(1.25 * rm.shape[0] * k_eff / m.E)
            lb["drop_at_1.25"].append((torch.clamp(cnt - cap, min=0).sum() / cnt.sum()).item())
            lb["zero_expert"].append((rm.sum(1) == 0).float().mean().item())
            lb["experts_per_token"].append(k_eff)
    model.train()
    out = dict(val_loss=round(loss_sum / tok_sum, 4),
               val_bpb=round(loss_sum / math.log(2) / byte_sum, 4),
               eval_tokens=tok_sum, eval_bytes=byte_sum)
    for k, v in lb.items():
        out[k] = round(float(np.mean(v)), 4) if v else None
    return out
