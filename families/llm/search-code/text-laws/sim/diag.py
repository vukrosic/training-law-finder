"""Diagonal model of the spectral-power dial: n independent directions, curvature lam_j = j^-a,
error e_j (start 1), gradient g_j = lam_j e_j + noise. Update per direction: sign(m_j)|m_j|^p / max|m|^p
(the dial acting on singular values when everything is diagonal). Cosine lr, tuned per p. Prints best p per T."""
import numpy as np, sys
def run(p, T, lr, a=1.5, n=64, sig=0.0, B=16, rng=None):
    lam = np.arange(1, n + 1) ** -a; e = np.ones((len(lr), n)); m = np.zeros_like(e)
    for t in range(T):
        f = min(1, (t + 1) / max(1, int(.05 * T))) * 0.5 * (1 + np.cos(np.pi * t / T))
        g = lam * e + (sig / np.sqrt(B)) * (np.sqrt(lam) if ISO == 0 else 1.0) * rng.standard_normal(e.shape)
        m = 0.95 * m + g; u = g + 0.95 * m
        r = np.abs(u) / np.abs(u).max(1, keepdims=True)
        e -= (lr[:, None] * f) * np.sign(u) * r ** p
    return 0.5 * (lam * e ** 2).sum(1)
ISO = 1
lrs = 2.0 ** np.arange(-14, 0)
for sig in map(float, sys.argv[1:] or ["0"]):
    for B in (4, 16, 64):
        row = []
        for T in (64, 256, 1024, 4096):
            L = {p: run(p, T, lrs, sig=sig, B=B, rng=np.random.default_rng(0)).min() for p in (0, .25, .5, .75, 1)}
            bp = min(L, key=L.get); row.append(f"T{T}:p{bp}({L[0]/L[bp]:.2f}x)")
        print(f"sig {sig} B{B:3d}", "  ".join(row), flush=True)
        if sig == 0: break
