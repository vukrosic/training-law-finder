"""Validation bits per byte, the same for every tokenizer.

    bpb = total nats over every predicted validation token / (ln 2 * UTF-8 bytes of the raw validation text)

The byte count comes from the raw text in data/val_docs.jsonl, never from tokens. The token stream is
BOS doc1 BOS doc2 ...; every token after the first is predicted exactly once, so each tokenizer also pays for
the document breaks. A tokenizer must give back the exact text: check_lossless() tests that.

The scored model is the one saved in the artifact: one lzma file with the training code, the tokenizer and the
weights, at most SIZE_LIMIT bytes. save_artifact() writes it and load_artifact() reads the weights back, so
whatever the file loses (rounding, quantization) shows in the score.
"""
import io
import json
import lzma
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")  # artifacts, never in git
SIZE_LIMIT = 16_000_000  # bytes: code + tokenizer + weights, compressed


def save_artifact(path, state_dict, code_paths, tokenizer_path):
    """Write code, tokenizer and weights as one lzma file; return its size in bytes."""
    blob = dict(state=state_dict,
                code={os.path.basename(p): open(p, encoding="utf-8").read() for p in code_paths},
                tokenizer=open(tokenizer_path, "rb").read())
    buf = io.BytesIO()
    torch.save(blob, buf)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(lzma.compress(buf.getvalue(), preset=6))
    return os.path.getsize(path)


def load_artifact(path):
    """The dict save_artifact() wrote: state (weights), code, tokenizer."""
    with open(path, "rb") as f:
        return torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu")


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
    """Sum of -ln p(next token) over the stream, in windows of seq_len that do not overlap: one forward pass per
    token, no learning (rule 4)."""
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
