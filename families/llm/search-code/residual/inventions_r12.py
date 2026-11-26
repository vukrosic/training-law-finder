"""Round 12 (residual): 40 parameter-free stream transforms at a fixed strength rk, stacked on NAG (rs=nag).
Question r11 cannot answer: is the code cost of stream ideas (rule 2: 21/22 worse on code, mean +0.0175) caused by
their extra learned parameters? These ideas add no parameters at all; rk = 0 is exactly NAG (wrapper-null).
Read ideas:  sublayer reads x + rk * RMS(x) * g(rms(x), ctx)
Write ideas: write y + rk * RMS(y) * h(rms(y), ctx)
"""
import math
import torch
import torch.nn.functional as F
from inventions_r4 import Base, rms, rmsv


def _hadamard(d):
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def _dct(d):
    n = torch.arange(d).float()
    C = torch.cos(math.pi / d * (n.view(1, -1) + 0.5) * n.view(-1, 1)) * math.sqrt(2 / d)
    C[0] /= math.sqrt(2)
    return C                                                # orthonormal, rows = frequencies


def _smooth(z):
    return (z.roll(1, -1) + z + z.roll(-1, -1)) / 3


def _topk(z, k):
    t = z.abs().topk(k, -1).values[..., -1:]
    return z * (z.abs() >= t)


def _wta(z, g=8):
    v = z.unflatten(-1, (-1, g))
    m = v.abs().amax(-1, keepdim=True)
    return (v * (v.abs() >= m)).flatten(-2)


def _ste_round(z, q=4):
    return z + ((z * q).round() / q - z).detach()


# g / h maps: f(z, s, ctx) -> feature of unit RMS-ish; z = rms(input), s = module (buffers), ctx = dict
MAPS = dict(
    hadamard=lambda z, s, c: z @ s.H,                                  # fixed orthogonal mixing
    dctlow=lambda z, s, c: (z @ s.C.t() * s.lo) @ s.C - z,             # spectral low-pass (residual form)
    dcthigh=lambda z, s, c: z - (z @ s.C.t() * s.lo) @ s.C,            # spectral high-pass (sharpen)
    power=lambda z, s, c: torch.sign(z) * (z.abs() + 1e-6).sqrt() - z,  # magnitude compression
    cubic=lambda z, s, c: z.pow(3) / 3,                                # odd polynomial expansion
    softshrink=lambda z, s, c: F.softshrink(z, 0.5) - z,               # denoise small channels
    topk=lambda z, s, c: _topk(z, 8) - z,                              # sparse read (top 8 of 64)
    wta=lambda z, s, c: _wta(z) - z,                                   # winner-take-all per 8-group
    roll=lambda z, s, c: z.roll(1, -1),                                # channel-index shift
    flip=lambda z, s, c: z.flip(-1),                                   # channel-index reversal
    smooth=lambda z, s, c: _smooth(z) - z,                             # channel smoothing
    laplace=lambda z, s, c: z - _smooth(z),                            # channel high-pass
    groupstd=lambda z, s, c: z.unflatten(-1, (-1, 8)).std(-1, keepdim=True).expand(*z.shape[:-1], -1, 8).flatten(-2) - 1,
    absfeat=lambda z, s, c: z.abs() - z.abs().mean(-1, keepdim=True),  # rectified magnitude
    sinfeat=lambda z, s, c: torch.sin(z),                              # periodic feature (no random basis)
    gelufeat=lambda z, s, c: F.gelu(z) - 0.5 * z,                      # even part of a gelu
    quant=lambda z, s, c: _ste_round(z) - z,                           # quantisation residual (STE)
    signfeat=lambda z, s, c: torch.sign(z) - z,                        # sign pull
    x0gate=lambda z, s, c: z * torch.tanh(rms(c["x0"])),               # embedding-gated
    tanhfeat=lambda z, s, c: torch.tanh(z) - z,                        # soft cap pull
)
# depth maps use the stored stream states / writes
DEPTH = dict(
    stale=lambda c: c["S"][-2] if len(c["S"]) > 1 else None,           # read one sublayer stale
    depthmax=lambda c: torch.stack(c["S"], 0).amax(0) if len(c["S"]) > 1 else None,
    depthmean=lambda c: torch.stack(c["S"], 0).mean(0) if len(c["S"]) > 1 else None,
    lastwrite=lambda c: c["Y"][-1] if c["Y"] else None,                # re-read the previous write
    diffwrite=lambda c: c["Y"][-1] - c["Y"][-2] if len(c["Y"]) > 1 else None,
)


class _PF(Base):
    side, fn = "read", "hadamard"

    def __init__(s, cfg, L):
        super().__init__(cfg, L)
        d = cfg.d_model
        s.register_buffer("H", _hadamard(d))
        s.register_buffer("C", _dct(d))
        s.register_buffer("lo", (torch.arange(d) < d // 4).float())

    def forward(s, x, subs):
        c = dict(x0=x, S=[], Y=[])
        for j, f in enumerate(subs):
            c["S"].append(x)
            r = x
            if s.side == "read" and s.k:
                r = x + s.k * rmsv(x) * MAPS[s.fn](rms(x), s, c)
            elif s.side == "dread" and s.k:
                v = DEPTH[s.fn](c)
                if v is not None:
                    r = x + s.k * (v - x) if s.fn in ("stale", "depthmax", "depthmean") else x + s.k * v
            y = f(r)
            c["Y"].append(y)
            if s.side == "write" and s.k:
                y = y + s.k * rmsv(y) * MAPS[s.fn](rms(y), s, c)
            elif s.side == "dwrite" and s.k:
                v = DEPTH[s.fn](c)
                if v is not None and len(c["Y"]) > 1:
                    v = c["Y"][-2] if s.fn == "lastwrite" else (c["Y"][-2] - c["Y"][-3] if s.fn == "diffwrite" and len(c["Y"]) > 2 else None)
                    if v is not None:
                        y = y + s.k * v
            x = x + y
        return x


IDEAS, FAMILY = {}, {}
_FAM = dict(hadamard="fixed orthogonal mixing", dctlow="spectral filter", dcthigh="spectral filter",
            power="magnitude compression", cubic="odd polynomial", softshrink="sparsification",
            topk="sparsification", wta="sparsification", roll="channel permutation", flip="channel permutation",
            smooth="channel filter", laplace="channel filter", groupstd="group statistics", absfeat="rectified magnitude",
            sinfeat="periodic feature", gelufeat="even nonlinearity", quant="quantisation", signfeat="sign pull",
            x0gate="embedding gate", tanhfeat="soft cap pull")
for side in ("read", "write"):
    for fn in MAPS:
        n = f"pf{side[0]}_{fn}"
        IDEAS[n] = type(n, (_PF,), dict(side=side, fn=fn))
        FAMILY[n] = f"parameter-free {side}: {_FAM[fn]}"
for fn in ("stale", "depthmax", "depthmean", "lastwrite", "diffwrite"):
    n = f"pfd_{fn}"
    IDEAS[n] = type(n, (_PF,), dict(side="dread", fn=fn))
    FAMILY[n] = "parameter-free depth read"
for fn in ("lastwrite", "diffwrite"):
    n = f"pfdw_{fn}"
    IDEAS[n] = type(n, (_PF,), dict(side="dwrite", fn=fn))
    FAMILY[n] = "parameter-free depth write (echo)"


def make(cfg, L):
    return IDEAS[cfg.res](cfg, L)
