"""Build the MOE-PRETRAIN token files from the pinned FineWeb-Edu shard.

Run with python3 -I (the shard is an untrusted download):
    python3 -I data/prepare.py            # uses data/tokenizer/bpe8192.json (committed)
    python3 -I data/prepare.py --retrain  # rebuild the tokenizer first (same settings)

Split (fixed by document index in shard_00000.parquet of karpathy/fineweb-edu-100b-shuffle):
    docs 0..511     -> validation   (data/fineweb_edu/val.bin, uint16)
    docs 512..end   -> train        (data/fineweb_edu/train.bin, uint16)
Tokenizer: byte-level BPE, vocab 8192, trained on docs 512..12799 (train docs only).
Documents are joined with the <|endoftext|> token (id 0) placed before each document.
"""
import os, sys, json, argparse
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.path.join(HERE, "fineweb_edu", "shard_00000.parquet")
TOK = os.path.join(HERE, "tokenizer", "bpe8192.json")
N_VAL_DOCS, TOK_TRAIN_END = 512, 12800

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    texts = pq.read_table(SHARD, columns=["text"]).column("text").to_pylist()
    print(f"{len(texts)} docs, {sum(map(len, texts)):,} chars")
    if a.retrain or not os.path.exists(TOK):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<|endoftext|>"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(texts[N_VAL_DOCS:TOK_TRAIN_END], trainer=tr)
        tok.save(TOK)
        print("tokenizer written", TOK)
    tok = Tokenizer.from_file(TOK)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot == 0 and tok.get_vocab_size() == 8192
    def encode(docs, path):
        out, nbytes = [], 0
        for i in range(0, len(docs), 2048):
            for e in tok.encode_batch(docs[i:i + 2048]):
                out.append(np.array([eot] + e.ids, dtype=np.uint16))
        arr = np.concatenate(out)
        nbytes = sum(len(d.encode("utf-8")) for d in docs)
        arr.tofile(path)
        return len(arr), nbytes
    meta = {}
    for name, docs in [("val", texts[:N_VAL_DOCS]), ("train", texts[N_VAL_DOCS:])]:
        n, nb = encode(docs, os.path.join(HERE, "fineweb_edu", f"{name}.bin"))
        meta[name] = {"tokens": n, "utf8_bytes": nb, "bytes_per_token": nb / n, "docs": len(docs)}
        print(name, meta[name])
    json.dump(meta, open(os.path.join(HERE, "fineweb_edu", "meta.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
