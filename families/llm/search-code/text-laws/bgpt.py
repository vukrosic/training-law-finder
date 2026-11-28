"""K independent copies of the testbed GPT trained in one batched pass.

Same architecture as testbed/model.py (RMSNorm, RoPE, QK-norm, SwiGLU, untied emb/head,
init N(0, 0.02)), but every weight has a leading model axis K, so K models with the same
shape run as one set of batched matmuls. Each model has its own learning rate, data range,
weight decay and init. Muon (NS5, scale 0.2*sqrt(max dim)) and Adam are written for the
batched layout; the loss head is a fused chunked cross-entropy that never stores the full
logits tensor.
"""
import math
import torch
import torch.nn.functional as F

NS = (3.4445, -4.7750, 2.0315)


def ff_hidden(d, mult=8 / 3, multiple=32):
    return int(multiple * math.ceil(d * mult / multiple))


def rope(T, dh, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=dev).float() / dh))
    f = torch.outer(torch.arange(T, device=dev).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, c, s):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class FusedCE(torch.autograd.Function):
    """Per-model mean cross-entropy of logits = x @ H^T, computed in token chunks.
    Gradients are formed in the forward pass, so the logits are never kept."""

    @staticmethod
    def forward(ctx, x, H, y, chunk):
        K, N, _ = x.shape
        gx = torch.empty_like(x)
        gH = torch.zeros_like(H)
        loss = torch.zeros(K, device=x.device)
        for i in range(0, N, chunk):
            xc, yc = x[:, i:i + chunk], y[:, i:i + chunk]
            lg = torch.bmm(xc, H.mT)
            lse = torch.logsumexp(lg, -1)
            loss += (lse - lg.gather(-1, yc[..., None]).squeeze(-1)).sum(1)
            p = torch.exp(lg - lse[..., None])
            p.scatter_add_(-1, yc[..., None], torch.full_like(yc[..., None], -1.0, dtype=p.dtype))
            gx[:, i:i + chunk] = torch.bmm(p, H)
            gH += torch.bmm(p.mT, xc)
        ctx.save_for_backward(gx / N, gH / N)
        return loss / N

    @staticmethod
    def backward(ctx, gl):
        gx, gH = ctx.saved_tensors
        return gx * gl[:, None, None], gH * gl[:, None, None], None, None


class BGPT:
    def __init__(self, K, vocab, d, L, heads, seq, ffm=8 / 3, dev="cpu", seeds=None):
        self.K, self.V, self.d, self.L, self.h, self.seq = K, vocab, d, L, heads, seq
        self.dh = d // heads
        hid = ff_hidden(d, ffm)
        self.dev = dev
        gens = [torch.Generator().manual_seed(int(s)) for s in (seeds or range(K))]

        def mat(o, i):
            return torch.stack([torch.randn(o, i, generator=g) * 0.02 for g in gens]).to(dev).requires_grad_()

        def ones(n):
            return torch.ones(K, n, device=dev).requires_grad_()

        self.emb = mat(vocab, d)
        self.head = mat(vocab, d)
        self.fnorm = ones(d)
        self.blocks = []
        for _ in range(L):
            self.blocks.append(dict(qkv=mat(3 * d, d), proj=mat(d, d), w1=mat(hid, d), w3=mat(hid, d),
                                    w2=mat(d, hid), n1=ones(d), n2=ones(d), qn=ones(self.dh), kn=ones(self.dh)))
        self.cos, self.sin = rope(seq, self.dh, dev)
        self.kidx = torch.arange(K, device=dev)[:, None, None]

    def matrices(self):  # Muon-eligible hidden matrices
        return [b[n] for b in self.blocks for n in ("qkv", "proj", "w1", "w3", "w2")]

    def others(self):
        return [self.emb, self.head, self.fnorm] + [b[n] for b in self.blocks for n in ("n1", "n2", "qn", "kn")]

    def params(self):
        return self.matrices() + self.others()

    def n_params(self, non_emb=True):
        n = sum(p[0].numel() for p in self.params())
        return n - 2 * self.V * self.d if non_emb else n

    def hidden(self, idx):  # idx [K, B, T] -> normed final hidden [K, B*T, d]
        K, B, T = idx.shape
        d, h, dh = self.d, self.h, self.dh
        x = self.emb[self.kidx, idx].view(K, B * T, d)
        c, s = self.cos[:T], self.sin[:T]
        for bl in self.blocks:
            a = rms(x, bl["n1"][:, None])
            qkv = torch.bmm(a, bl["qkv"].mT).view(K, B, T, 3, h, dh).permute(3, 0, 1, 4, 2, 5)
            q = apply_rope(rms(qkv[0], bl["qn"][:, None, None, None]), c, s)
            k = apply_rope(rms(qkv[1], bl["kn"][:, None, None, None]), c, s)
            y = F.scaled_dot_product_attention(q.reshape(K * B, h, T, dh), k.reshape(K * B, h, T, dh),
                                               qkv[2].reshape(K * B, h, T, dh), is_causal=True)
            y = y.view(K, B, h, T, dh).permute(0, 1, 3, 2, 4).reshape(K, B * T, d)
            x = x + torch.bmm(y, bl["proj"].mT)
            a = rms(x, bl["n2"][:, None])
            x = x + torch.bmm(F.silu(torch.bmm(a, bl["w1"].mT)) * torch.bmm(a, bl["w3"].mT), bl["w2"].mT)
        return rms(x, self.fnorm[:, None])

    def loss(self, idx, chunk=1024):  # idx [K, B, T+1] -> per-model mean loss [K]
        hx = self.hidden(idx[:, :, :-1])
        return FusedCE.apply(hx, self.head, idx[:, :, 1:].reshape(self.K, -1), chunk)

    @torch.no_grad()
    def token_losses(self, idx, chunk=None):
        chunk = chunk or max(256, 8192 // self.K)  # [K, B, T+1] -> [K, B*T]
        out = []
        y = idx[:, :, 1:].reshape(self.K, -1)
        hx = torch.cat([self.hidden(idx[:, i:i + 64, :-1]) for i in range(0, idx.shape[1], 64)], 1)
        for i in range(0, hx.shape[1], chunk):
            lg = torch.bmm(hx[:, i:i + chunk], self.head.mT)
            out.append(torch.logsumexp(lg, -1) - lg.gather(-1, y[:, i:i + chunk, None]).squeeze(-1))
        return torch.cat(out, 1)


# Polar Express style per-iteration coefficients (Amsel et al. 2025), safety-scaled; 5 steps.
PE = [(8.28721201814563, -23.595886519098837, 17.300387312530933),
      (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
      (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
      (3.3184196573706015, -2.488488024314874, 0.51004894012372),
      (2.300652019954817, -1.6689039845747493, 0.4188073119525673)]


@torch.no_grad()
def ns5(G, steps=5, eps=1e-7, coef="ns"):
    X = G
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if coef == "pe":
        X = X / 1.01
    for i in range(steps):
        a, b, c = PE[min(i, len(PE) - 1)] if coef == "pe" else NS
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


class Opt:
    """Muon on hidden matrices + Adam elsewhere (opt='muon'), or Adam everywhere (opt='adam').
    lr, wd: tensors [K]. Adam betas (0.9, b2), eps 1e-10, bias corrected."""

    def __init__(self, model, kind, lr, wd=None, b2=0.95, mom=0.95, ns_steps=5, emb_mult=None,
                 nest=True, coef="ns", mscale="r02", pexp=None, pnorm="spec", pend=None, total=None):
        self.kind, self.lr, self.b2, self.mom, self.ns = kind, lr, b2, mom, ns_steps
        self.nest, self.coef, self.mscale = nest, coef, mscale
        self.pnorm = pnorm  # "spec": top singular value 1 for every p (default); "fro": Frobenius sqrt(min dim)
        self.p0, self.pend, self.total = pexp, pend, total  # optional linear p schedule pexp -> pend over total steps
        self.pexp = pexp  # spectral power: update = U S^p V^T rescaled to Frobenius sqrt(min dim); 0 = exact Muon
        self.wd = wd if wd is not None else torch.zeros_like(lr)
        self.muon = model.matrices() if kind == "muon" else []
        self.adam = model.others() + ([] if kind == "muon" else model.matrices())
        self.st = {id(p): dict(m=torch.zeros_like(p), v=torch.zeros_like(p)) for p in self.adam}
        for p in self.muon:
            self.st[id(p)] = dict(m=torch.zeros_like(p))
        self.t = 0
        self.emb_ids = {id(model.emb), id(model.head)}
        self.emb_mult = emb_mult  # optional [K] multiplier for emb/head lr

    @torch.no_grad()
    def step(self, mult):
        self.t += 1
        lr = self.lr * mult
        if self.pend is not None:
            self.pexp = self.p0 + (self.pend - self.p0) * min(1.0, (self.t - 1) / max(1, self.total - 1))
        for p in self.muon:
            st = self.st[id(p)]
            st["m"].mul_(self.mom).add_(p.grad)
            u = p.grad.add(st["m"], alpha=self.mom) if self.nest else st["m"]
            r, c = p.shape[-2:]
            # r02: 0.2*sqrt(max dim) (Moonlight RMS match); ar: sqrt(max(1, rows/cols)) (Keller) x 0.2*sqrt(64)
            scale = 0.2 * max(r, c) ** 0.5 if self.mscale == "r02" else 1.6 * max(1, r / c) ** 0.5
            p.mul_((1 - lr * self.wd)[:, None, None])
            if self.pexp is None:
                X = ns5(u, self.ns, coef=self.coef)
            else:
                uc = u.float().cpu() if u.device.type == "mps" else u
                try:
                    Uu, Su, Vh = torch.linalg.svd(uc, full_matrices=False)
                except torch._C._LinAlgError:  # ill-conditioned: retry in float64
                    Uu, Su, Vh = (t.to(uc.dtype) for t in torch.linalg.svd(uc.double(), full_matrices=False))
                rel = Su / Su.amax(-1, keepdim=True).clamp_min(1e-30)
                if self.pexp < 0:  # negative power boosts weak directions; floor the ratio so null directions stay finite
                    rel = rel.clamp_min(1e-3)
                X = (Uu * (rel ** self.pexp)[..., None, :]) @ Vh
                if self.pnorm == "fro":
                    X = X * (min(r, c) ** 0.5 / X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30))
                X = X.to(u.device)
            p.sub_(X * (lr * scale)[:, None, None])
            p.grad = None
        bc1, bc2 = 1 - 0.9 ** self.t, 1 - self.b2 ** self.t
        for p in self.adam:
            st = self.st[id(p)]
            st["m"].mul_(0.9).add_(p.grad, alpha=0.1)
            st["v"].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            l = lr if (self.emb_mult is None or id(p) not in self.emb_ids) else lr * self.emb_mult
            shape = (-1,) + (1,) * (p.dim() - 1)
            upd = (st["m"] / bc1) / ((st["v"] / bc2).sqrt() + 1e-10)
            p.sub_(upd * l.view(shape))
            p.grad = None
