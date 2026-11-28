"""nqm.py: matrix linear regression trained with the spectral-power dial, the same optimizer as bgpt.Opt (Muon family).
y = W* x + noise, x ~ N(0, diag(j^-a)), W* columns scaled j^-b/2. Minibatch gradient G = (W X - Y) X^T / B.
Update: Nesterov momentum 0.95, X = U (S/Smax)^p V^T, step lr * 0.2 sqrt(max dim), cosine schedule, 5% warmup.
Prints / appends JSON rows: best population loss over lr per (T, B, p).
usage: python3 nqm.py --T 64 256 1024 --B 4 16 64 --out sim/nqm.jsonl [--a 1.5 --b 1.0 --sig 0.5 --m 32 --n 64]"""
import argparse, json, math, time, torch
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, nargs="+", default=[64, 256, 1024]); ap.add_argument("--B", type=int, nargs="+", default=[16])
ap.add_argument("--P", type=float, nargs="+", default=[0, .25, .5, .75, 1]); ap.add_argument("--a", type=float, default=1.5)
ap.add_argument("--b", type=float, default=1.0); ap.add_argument("--sig", type=float, default=0.5)
ap.add_argument("--m", type=int, default=32); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--lrs", type=float, nargs="+", default=[2 ** k for k in range(-9, 0)]); ap.add_argument("--out", default="")
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tag", default="")
A = ap.parse_args(); torch.set_num_threads(A.threads)
m, n = A.m, A.n
lam = torch.arange(1, n + 1, dtype=torch.float64) ** -A.a
g = torch.Generator().manual_seed(0)
Wst = torch.randn(m, n, generator=g, dtype=torch.float64) * (torch.arange(1, n + 1, dtype=torch.float64) ** (-A.b / 2)) / math.sqrt(m)
def poploss(W): return 0.5 * (((W - Wst) ** 2) * lam).sum((-2, -1))  # excess loss, noise floor removed
for T in A.T:
    for B in A.B:
        t0 = time.time()
        PL = [(p, lr) for p in A.P for lr in A.lrs]; K = len(PL)
        pex = torch.tensor([x[0] for x in PL], dtype=torch.float64)[:, None]; lr0 = torch.tensor([x[1] for x in PL], dtype=torch.float64)
        W = torch.zeros(K, m, n, dtype=torch.float64); M = torch.zeros_like(W)
        gen = torch.Generator().manual_seed(1)
        sc = 0.2 * max(m, n) ** 0.5; warm = max(1, int(0.05 * T)); L0 = poploss(W[:1])[0].item()
        for t in range(T):
            mult = (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, T - warm)))
            X = torch.randn(B, n, generator=gen, dtype=torch.float64) * lam.sqrt()  # same batch for every run
            Y = X @ Wst.T + A.sig * torch.randn(B, m, generator=gen, dtype=torch.float64)
            R = torch.einsum("kmn,bn->kbm", W, X) - Y
            G = torch.einsum("kbm,bn->kmn", R, X) / B
            M.mul_(0.95).add_(G); u = G + 0.95 * M
            U, S, Vh = torch.linalg.svd(u, full_matrices=False)
            S = (S / S.amax(-1, keepdim=True).clamp_min(1e-300)) ** pex
            W -= (U * S[:, None, :]) @ Vh * (lr0 * mult * sc)[:, None, None]
        L = poploss(W); L = torch.where(torch.isfinite(L), L, torch.full_like(L, 1e9))
        rows = []
        for i, p in enumerate(A.P):
            Lp = L[i * len(A.lrs):(i + 1) * len(A.lrs)]; j = int(Lp.argmin())
            rows.append(dict(tag=A.tag, T=T, B=B, p=p, a=A.a, b=A.b, sig=A.sig, m=m, n=n, loss=Lp[j].item(), L0=L0, lr=A.lrs[j],
                             edge=("hi" if j == len(A.lrs) - 1 else "lo" if j == 0 else "")))
        best = min(rows, key=lambda r: r["loss"])
        print(f"T{T:5d} B{B:4d} best p={best['p']:<5} " + " ".join(f"p{r['p']}:{r['loss']:.4g}{'^' if r['edge']=='hi' else 'v' if r['edge']=='lo' else ''}" for r in rows) + f"  {time.time()-t0:.1f}s", flush=True)
        if A.out:
            with open(A.out, "a") as f: f.write("".join(json.dumps(r) + "\n" for r in rows))
