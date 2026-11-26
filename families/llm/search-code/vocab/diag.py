"""VOCAB search diagnostics (analysis only, never changes training). Called by train.py when --set diag=1.

Measures where an embedding/head design wins or loses:
  - loss by target frequency bucket (8 log-spaced buckets of the train-count rank) and by how many times the
    target id was a training target in this run (seen-count buckets)
  - head and embedding row norms by bucket, emb-head row cosine (untied only), effective rank of head, embedding
    and final hidden states
  - logit stats: logsumexp, max logit, entropy, correct-token logit, top-1 confidence vs accuracy
  - frequency calibration: predicted probability mass per bucket vs the bucket's true target share
"""
import numpy as np
import torch
import torch.nn.functional as F

RANK_EDGES = [16, 64, 256, 512, 1024, 2048, 4096]          # frequency rank buckets: 0-15, 16-63, ..., 4096-8191
SEEN_EDGES = [1, 4, 16, 64, 256, 1024]                    # times seen as a target in this run: 0, 1-3, 4-15, ...


def erank(M):
    s = torch.linalg.svdvals(M.float())
    p = s / s.sum()
    return round(float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())), 2)


@torch.no_grad()
def run(model, val, train, cfg, seen_cnt, dev, n_seqs=512):
    V, T = cfg.vocab, cfg.seq_len
    fr = np.bincount(train[:4_000_000], minlength=V)
    rk = np.empty(V, dtype=np.int64); rk[np.argsort(-fr)] = np.arange(V)
    rb = torch.from_numpy(np.digitize(rk, RANK_EDGES)).to(dev)          # token id -> rank bucket
    sb = torch.from_numpy(np.digitize(seen_cnt.cpu().numpy(), SEEN_EDGES)).to(dev)
    n = min(n_seqs, (len(val) - 1) // T)
    xy = torch.from_numpy(val[:n * T + 1].astype(np.int64))
    x, y = xy[:-1].view(n, T).to(dev), xy[1:].view(n, T).to(dev)
    model.eval()
    L, LSE, MX, ENT, CL, CONF, ACC, HS = [], [], [], [], [], [], [], []
    mass = torch.zeros(len(RANK_EDGES) + 1, device=dev)
    for i in range(0, n, 32):
        lg, l = model(x[i:i + 32], y[i:i + 32])
        lg = lg.float()
        lp = F.log_softmax(lg, -1)
        p = lp.exp()
        L.append(l.reshape(-1)); LSE.append(torch.logsumexp(lg, -1).reshape(-1)); MX.append(lg.max(-1).values.reshape(-1))
        ENT.append(-(p * lp).sum(-1).reshape(-1)); CL.append(lg.gather(-1, y[i:i + 32].unsqueeze(-1)).reshape(-1))
        CONF.append(p.max(-1).values.reshape(-1)); ACC.append((lg.argmax(-1) == y[i:i + 32]).float().reshape(-1))
        mass += torch.zeros_like(mass).index_add_(0, rb, p.reshape(-1, V).sum(0))
        HS.append(model.ctx["h"].reshape(-1, model.ctx["h"].shape[-1]))
    L, LSE, MX, ENT, CL, CONF, ACC = (torch.cat(t) for t in (L, LSE, MX, ENT, CL, CONF, ACC))
    yf = y.reshape(-1)
    out = {}
    for nm, b, k in (("rk", rb[yf], len(RANK_EDGES) + 1), ("seen", sb[yf], len(SEEN_EDGES) + 1)):
        out[nm + "_loss"] = [round(L[b == j].mean().item(), 4) if (b == j).any() else None for j in range(k)]
        out[nm + "_frac"] = [round((b == j).float().mean().item(), 4) for j in range(k)]
    true_share = torch.zeros_like(mass).index_add_(0, rb[yf], torch.ones_like(L))
    out["mass_pred"] = [round(v, 4) for v in (mass / mass.sum()).tolist()]
    out["mass_true"] = [round(v, 4) for v in (true_share / true_share.sum()).tolist()]
    out["lse"] = round(LSE.mean().item(), 3); out["maxlogit"] = round(MX.mean().item(), 3)
    out["ent"] = round(ENT.mean().item(), 3); out["corr_logit"] = round(CL.mean().item(), 3)
    out["conf"] = round(CONF.mean().item(), 4); out["acc"] = round(ACC.mean().item(), 4)
    cb = torch.clamp((CONF * 10).long(), max=9)                         # expected calibration error, 10 bins
    out["ece"] = round(sum(((cb == j).float().mean() * (CONF[cb == j].mean() - ACC[cb == j].mean()).abs()).item()
                           for j in range(10) if (cb == j).any()), 4)
    W = model.head.weight.detach().float()
    E = model.emb.weight.detach().float()
    nb = len(RANK_EDGES) + 1
    out["head_norm"] = [round(W[rb == j].norm(dim=1).mean().item(), 4) for j in range(nb)]
    out["emb_norm"] = [round(E[rb == j].norm(dim=1).mean().item(), 4) for j in range(nb)]
    if W.shape == E.shape:
        cs = F.cosine_similarity(W, E, dim=1)
        out["emb_head_cos"] = [round(cs[rb == j].mean().item(), 4) for j in range(nb)]
    out["erank_head"] = erank(W); out["erank_emb"] = erank(E)
    # direct token path: logits from the input embedding alone through the same head (bigram capacity of emb->head)
    xs, ys = x[:64], y[:64]
    x0 = model.emb(xs) if model.hd is None else model.hd.embed(model, xs)
    lg0 = model.head(model.norm(x0)) if model.hd is None else model.hd.out(model, x0, [x0] * len(model.blocks), x0, xs)
    out["direct_loss"] = round(F.cross_entropy(lg0.float().reshape(-1, V), ys.reshape(-1)).item(), 4)
    model(xs, ys); h = model.ctx["h"]
    out["emb_share"] = round((x0.norm(dim=-1) / h.norm(dim=-1)).mean().item(), 4)
    out["cos_h_emb"] = round(F.cosine_similarity(h, x0, dim=-1).mean().item(), 4)
    H = torch.cat(HS)[:8192]
    out["erank_h"] = erank(H - H.mean(0)); out["h_norm"] = round(H.norm(dim=1).mean().item(), 3)
    return out
