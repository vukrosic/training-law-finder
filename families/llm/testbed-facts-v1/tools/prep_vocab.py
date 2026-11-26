"""Small-vocab FineWeb-Edu copies for the fast bench and the vocab-size test (2026-11-24).
python3 -I tools/prep_vocab.py 1024   -> data/fineweb_v1024/{tokenizer.json,train.bin,val.bin,meta.json}
Same docs and split as data/prepare.py (val = docs 0..511); tokenizer trained on docs 512..3511;
train.bin = docs 512..12511 only (enough for runs of a few million tokens)."""
import os, sys, json
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
V = int(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
OUT = os.path.join(D, f"fineweb_v{V}"); os.makedirs(OUT, exist_ok=True)
texts = pq.read_table(os.path.join(D, "fineweb_edu", "shard_00000.parquet"), columns=["text"]).slice(0, 12512).column("text").to_pylist()
tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=V, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
tok.train_from_iterator(texts[512:3512], trainer=tr)
tok.save(os.path.join(OUT, "tokenizer.json"))
assert tok.token_to_id("<|endoftext|>") == 0 and tok.get_vocab_size() == V
meta = {}
for name, docs in [("val", texts[:512]), ("train", texts[512:12512])]:
    arr = np.concatenate([np.array([0] + e.ids, dtype=np.uint16) for i in range(0, len(docs), 500) for e in tok.encode_batch(docs[i:i + 500])])
    arr.tofile(os.path.join(OUT, f"{name}.bin"))
    nb = sum(len(d.encode("utf-8")) for d in docs)
    meta[name] = {"tokens": len(arr), "utf8_bytes": nb, "bytes_per_token": nb / len(arr), "docs": len(docs)}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(V, meta)
