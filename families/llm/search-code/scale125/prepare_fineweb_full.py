"""Token files for the GPU scale runs (SCALE-PLAN.md): more FineWeb-Edu, same tokenizer, same validation set.

Run with python3 -I after `sh arena/data/download.sh full 20` (parquet shards are untrusted downloads):
    python3 -I arena/env/prepare_fineweb_full.py --shards 20
train.bin = shard 0 docs 512..end (the laptop train set) + shards 1..N, appended shard by shard.
val.bin   = copy of arena/data/fineweb_edu/val.bin (shard 0 docs 0..511), so validation and test match the laptop runs.
Output: <video folder>/data/fineweb_full/ (never writes into arena/data).
"""
import os, argparse, shutil
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "fineweb_edu")
TOK = os.path.join(HERE, "..", "data", "tokenizer", "bpe8192.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "data", "fineweb_full"))
    ap.add_argument("--src", default=os.path.join(HERE, "..", "..", "data", "raw"), help="folder with shard_*.parquet (scale125: data/raw, never the product env)")
    ap.add_argument("--max_docs", type=int, default=0, help="per shard; 0 = all (small values only for the local check)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(TOK)
    assert tok.token_to_id("<|endoftext|>") == 0 and tok.get_vocab_size() == 8192
    shutil.copyfile(os.path.join(SRC, "val.bin"), os.path.join(a.out, "val.bin"))
    path, total = os.path.join(a.out, "train.bin"), 0
    with open(path, "wb") as f:
        for i in range(a.shards + 1):
            texts = pq.read_table(os.path.join(a.src, f"shard_{i:05d}.parquet"), columns=["text"]).column("text").to_pylist()
            texts = texts[512:] if i == 0 else texts
            if a.max_docs:
                texts = texts[:a.max_docs]
            for j in range(0, len(texts), 2048):
                for e in tok.encode_batch(texts[j:j + 2048]):
                    arr = np.array([0] + e.ids, dtype=np.uint16)
                    arr.tofile(f)
                    total += len(arr)
            print(f"shard {i}: {total:,} train tokens so far", flush=True)
    print("done", path, total)

if __name__ == "__main__":
    main()
