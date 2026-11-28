"""python3 test_flops.py: a matrix multiply costs the same however it is written."""
import torch

import flops


def test_hidden_matmul():
    W = torch.randn(64, 64, requires_grad=True)
    x = torch.randn(8, 32, 64)
    mm = flops.count(lambda: (x @ W).sum().backward())
    ew = flops.count(lambda: (x[..., None] * W).sum(-2).sum().backward())
    assert abs(ew - mm) / mm < 0.05, (mm, ew)


def test_embedding_counts():
    E = torch.nn.Embedding(1 << 16, 64)
    assert flops.count(lambda: E(torch.randint(0, 1 << 16, (8, 32))).sum().backward()) >= E.weight.numel()


if __name__ == "__main__":
    test_hidden_matmul()
    test_embedding_counts()
    print("flops tests pass")
