"""Download one FineWeb-Edu shard, train a byte-level BPE tokenizer and write token files.

    python3 -I testbed/prepare_data.py                 # vocab 8192, all docs (~95 MB download)
    python3 -I testbed/prepare_data.py --vocab 4096    # smaller vocab, faster CPU runs
    python3 -I testbed/prepare_data.py --shard /path/to/shard_00000.parquet

Source: shard_00000.parquet of karpathy/fineweb-edu-100b-shuffle on Hugging Face
(FineWeb-Edu, ODC-By license), pinned to one revision so every user gets the same text.

Split, fixed by document index:
    docs 0..511    -> validation (val.bin)
    docs 512..end  -> train      (train.bin)
The tokenizer is trained on train docs 512..12799 only. Documents are joined with
<|endoftext|> (id 0) placed before each document. Tokens are stored as uint16.
"""
import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "karpathy/fineweb-edu-100b-shuffle"
REV = "4c8f30d6756da75362432a4d5569e1b229263b71"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/shard_00000.parquet"
N_VAL_DOCS, TOK_TRAIN_END = 512, 12800
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download(dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("downloading", URL)
    tmp = dst + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--shard", default=None, help="local parquet instead of downloading")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all docs in the shard")
    a = ap.parse_args()
    assert a.vocab <= 65535, "uint16 token files"

    shard = a.shard or download(os.path.join(ROOT, "raw", "shard_00000.parquet"))
    h = hashlib.sha256(open(shard, "rb").read()).hexdigest()
    texts = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    if a.max_docs:
        texts = texts[:a.max_docs]
    print(f"{len(texts)} docs, {sum(map(len, texts)):,} chars, sha256 {h[:16]}")

    out = os.path.join(ROOT, f"fineweb_v{a.vocab}")
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tok.train_from_iterator(texts[N_VAL_DOCS:TOK_TRAIN_END], trainer=tr)
    tok.save(os.path.join(out, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    assert eot == 0

    meta = dict(source=REPO, revision=REV, shard_sha256=h, vocab=a.vocab)
    for name, docs in [("val", texts[:N_VAL_DOCS]), ("train", texts[N_VAL_DOCS:])]:
        parts = []
        for i in range(0, len(docs), 2048):
            for e in tok.encode_batch(docs[i:i + 2048]):
                parts.append(np.array([eot] + e.ids, dtype=np.uint16))
        arr = np.concatenate(parts)
        arr.tofile(os.path.join(out, f"{name}.bin"))
        nbytes = sum(len(d.encode("utf-8")) for d in docs)
        meta[name] = dict(tokens=int(len(arr)), docs=len(docs), utf8_bytes=nbytes,
                          bytes_per_token=round(nbytes / len(arr), 4))
        print(name, meta[name])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
