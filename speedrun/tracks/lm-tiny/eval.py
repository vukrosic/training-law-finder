"""Validation bits per byte, the same for every tokenizer.

    bpb = total nats over every predicted validation token / (ln 2 * UTF-8 bytes of the raw validation text)

The byte count comes from the raw text in data/val_docs.jsonl, never from tokens. The token stream is
BOS doc1 BOS doc2 ...; every token after the first is predicted exactly once, so each tokenizer also pays for
the document breaks. A tokenizer must give back the exact text: check_lossless() tests that.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def val_docs(path=os.path.join(DATA, "val_docs.jsonl")):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def val_bytes(docs):
    return sum(len(d.encode("utf-8")) for d in docs)


def bpb(total_nats, n_bytes):
    return total_nats / (math.log(2) * n_bytes)


def check_lossless(tokens, decode, docs, bos):
    """tokens: the validation stream; decode: list of ids -> str. True if it gives back every document exactly."""
    t = np.asarray(tokens)
    starts = np.flatnonzero(t == bos)
    pieces = np.split(t, starts[1:]) if len(starts) else []
    return len(pieces) == len(docs) and all(decode(p[1:].tolist()) == d for p, d in zip(pieces, docs))


@torch.no_grad()
def total_nats(model, tokens, seq_len, device, batch=16):
    """Sum of -ln p(next token) over the stream, in windows of seq_len that do not overlap."""
    t = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    n = len(t) - 1
    starts = list(range(0, n, seq_len))
    full = [s for s in starts if s + seq_len <= n]
    nats = 0.0
    for i in range(0, len(full), batch):
        s = full[i:i + batch]
        x = torch.stack([t[j:j + seq_len] for j in s]).to(device)
        y = torch.stack([t[j + 1:j + seq_len + 1] for j in s]).to(device)
        nats += F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), reduction="sum").item()
    if len(full) < len(starts):  # last short window
        s = starts[-1]
        x, y = t[s:n][None].to(device), t[s + 1:n + 1][None].to(device)
        nats += F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), reduction="sum").item()
    return nats
