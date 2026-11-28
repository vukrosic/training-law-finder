"""Checks the FLOP counter against hand-worked numbers.  python3 test_flops.py  (a few seconds, cpu only)

Each case runs one op under flops.Meter and compares the count with the formula in flops.py. The last cases are
the shortcuts the old matmul-only counter missed: FFT mixing, eigh, sort, cumsum, softmax and n-gram counting
with scatter_add all cost FLOPs now.
"""
import math

import torch

import flops


def cost(fn):
    m = flops.Meter()
    n = m.count(fn)
    return n, m


def check(name, fn, want):
    got, m = cost(fn)
    assert got == want, f"{name}: counted {got}, expected {want} ({dict(m.by_op)})"
    print(f"  {name:28s} {got:>12,}")


def main():
    a, b = torch.randn(64, 128), torch.randn(128, 32)
    x = torch.randn(8, 1024)
    s = torch.randn(4, 64, 64)
    s = s @ s.transpose(-1, -2)
    table = torch.randn(1000, 32)
    idx = torch.randint(0, 1000, (8, 16))
    lin = torch.nn.Linear(128, 32, bias=False)

    check("matmul", lambda: a @ b, 2 * 64 * 128 * 32)
    check("linear", lambda: lin(a), 2 * 64 * 128 * 32)
    check("view and transpose", lambda: a.view(128, 64).t(), 0)
    check("elementwise add", lambda: a + a, 64 * 128)
    check("sum (charged per input)", lambda: a.sum(), 64 * 128)
    check("softmax", lambda: x.softmax(-1), 8 * 1024)
    check("cumsum", lambda: x.cumsum(-1), 8 * 1024)
    check("rfft", lambda: torch.fft.rfft(x, dim=-1), 5 * 8 * 1024 * 10)
    check("sort", lambda: x.sort(dim=-1), 8 * 1024 * 10)
    check("sort dim 0, descending", lambda: x.sort(0, True), 8 * 1024 * 3)
    check("eigh", lambda: torch.linalg.eigh(s), 4 * 64 ** 3)
    check("embedding (per element read)", lambda: torch.nn.functional.embedding(idx, table), 8 * 16 * 32)
    counts = torch.zeros(1 << 20)
    keys, ones = torch.randint(0, 1 << 20, (4096,)), torch.ones(4096)
    check("n-gram count, scatter_add", lambda: counts.scatter_add_(0, keys, ones), 4096)
    check("bincount", lambda: torch.bincount(keys, minlength=1 << 20), 1 << 20)

    # a whole training step: backward and optimizer are counted
    def step():
        w = torch.randn(128, 32, requires_grad=True)
        opt = torch.optim.SGD([w], lr=0.1)
        (a @ w).square().sum().backward()
        opt.step()
    n, m = cost(step)
    assert n >= 2 * 2 * 64 * 128 * 32, n  # forward and the weight-gradient matmul at least
    assert not m.unknown_heavy and not m.nonstandard, (m.unknown_heavy, m.nonstandard)
    print(f"  {'tiny training step':28s} {n:>12,}")

    # ops a standard model does not use are listed, so the entry is told to run --full-count
    for name, fn in [("rfft", lambda: torch.fft.rfft(x)), ("eigh", lambda: torch.linalg.eigh(s)),
                     ("sort", lambda: x.sort()), ("cumsum", lambda: x.cumsum(-1)),
                     ("scatter_add", lambda: counts.scatter_add_(0, keys, ones))]:
        _, m = cost(fn)
        assert m.nonstandard, f"{name} not listed as nonstandard"
    _, m = cost(lambda: (x.softmax(-1) * 2).sum())
    assert not m.nonstandard, m.nonstandard

    # heavy op without a formula: listed
    _, m = cost(lambda: torch.linalg.lstsq(s, s))
    assert m.unknown_heavy or m.by_op.get("linalg_lstsq"), dict(m.by_op)
    print("flops tests pass")


if __name__ == "__main__":
    main()
