"""Checks that bpb is counted the same way for any tokenizer.  python3 test_bpb.py

1. Bytes come from the raw text, so two tokenizers of the same text get the same byte count.
2. A model that gives every token the same probability has a bpb we can work out by hand, for a byte
   tokenizer and for the default tokenizer: the code divides by bytes, not by tokens.
3. The default tokenizer gives back every validation document exactly (if the data is downloaded).
"""
import math
import os

import numpy as np
import torch

import eval as ev

DOCS = ["Hello, world!\n\nA second line.", "  leading spaces, tabs\tand unicode: café, 数据, 🙂", "x"]
BOS = 256


def byte_tokens(docs):
    return np.array([t for d in docs for t in [BOS] + list(d.encode("utf-8"))], dtype=np.int64)


def byte_decode(ids):
    return bytes(ids).decode("utf-8")


class Uniform(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, x):
        return torch.zeros(*x.shape, self.vocab)


def test_bytes_from_text():
    n = ev.val_bytes(DOCS)
    assert n == sum(len(d.encode("utf-8")) for d in DOCS) and n != sum(len(d) for d in DOCS)


def test_uniform_byte_model():
    toks = byte_tokens(DOCS)
    assert ev.check_lossless(toks, byte_decode, DOCS, BOS)
    for seq_len in (4, 7, 1000):  # the window size must not change the total
        nats = ev.total_nats(Uniform(257), toks, seq_len, "cpu")
        n_targets = len(toks) - 1
        assert abs(nats - n_targets * math.log(257)) < 1e-6 * nats
        n_bytes = ev.val_bytes(DOCS)
        expect = n_targets * math.log2(257) / n_bytes
        assert abs(ev.bpb(nats, n_bytes) / expect - 1) < 1e-6
    # a lossy tokenizer fails the check
    assert not ev.check_lossless(toks, lambda ids: byte_decode(ids).strip(), DOCS, BOS)


def test_default_tokenizer():
    model = os.path.join(ev.DATA, "sp1024.model")
    if not os.path.exists(model):
        print("skip: run python3 prepare_data.py first to check the default tokenizer")
        return
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model)
    toks = np.array([t for d in DOCS for t in [1] + sp.encode(d)])
    assert ev.check_lossless(toks, sp.decode, DOCS, 1)
    nats = ev.total_nats(Uniform(1024), toks, 16, "cpu")
    # fewer tokens than bytes, same byte count: the uniform bpb is (tokens - 1) * 10 bits / bytes
    assert abs(ev.bpb(nats, ev.val_bytes(DOCS)) / ((len(toks) - 1) * 10 / ev.val_bytes(DOCS)) - 1) < 1e-6
    docs = ev.val_docs()
    val = np.fromfile(os.path.join(ev.DATA, "val.bin"), dtype=np.uint16)
    assert ev.check_lossless(val, sp.decode, docs, 1), "val.bin does not decode to val_docs.jsonl"
    print(f"default tokenizer: {len(val):,} tokens for {ev.val_bytes(docs):,} bytes, lossless")


if __name__ == "__main__":
    test_bytes_from_text()
    test_uniform_byte_model()
    test_default_tokenizer()
    print("bpb tests pass")
