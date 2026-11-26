"""Muon for hidden matrices, Adam for embeddings, output head, norms and scalars.

Muon update for a matrix W with gradient G:
    m  = mom * m + G
    u  = G + mom * m                       (Nesterov)
    W -= lr * 0.2 * sqrt(max(rows, cols)) * NS5(u)
    W *= 1 - lr * wd                       (decoupled weight decay)
NS5 is five Newton-Schulz steps that push the update toward the nearest
orthogonal matrix. The 0.2 * sqrt(max dim) factor makes the update RMS close
to Adam's, so one learning rate works for both groups.
"""
import torch

NS_COEF = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def ns5(G, steps=5, eps=1e-7):
    a, b, c = NS_COEF
    X = G.bfloat16() if G.is_cuda else G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X.to(G.dtype)


def is_muon_param(name, p):
    return p.dim() >= 2 and not name.startswith(("emb", "head"))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, wd=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                m.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(m, alpha=g["momentum"])
                scale = 0.2 * max(p.shape[-2:]) ** 0.5
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(ns5(u, g["ns_steps"]), alpha=-g["lr"] * scale)


def build_optimizers(model, lr, emb_lr=None, head_lr=None, muon_wd=0.0, adam_betas=(0.9, 0.95)):
    """Return a list of optimizers. Every param group stores its base lr in
    `base_lr`; the training loop multiplies it by the schedule each step."""
    emb_lr = lr if emb_lr is None else emb_lr
    head_lr = lr if head_lr is None else head_lr
    mats, emb, head, other = [], [], [], []
    for n, p in model.named_parameters():
        if is_muon_param(n, p):
            mats.append(p)
        elif n.startswith("emb"):
            emb.append(p)
        elif n.startswith("head"):
            head.append(p)
        else:
            other.append(p)
    muon = Muon([dict(params=mats, lr=lr, base_lr=lr)], wd=muon_wd)
    groups = [dict(params=emb, lr=emb_lr, base_lr=emb_lr),
              dict(params=head, lr=head_lr, base_lr=head_lr),
              dict(params=other, lr=lr, base_lr=lr)]
    adam = torch.optim.Adam([g for g in groups if g["params"]], betas=adam_betas, eps=1e-10,
                            weight_decay=0.0)
    return [muon, adam]
