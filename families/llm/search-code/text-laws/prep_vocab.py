"""Rebuild FineWeb-edu token files at a smaller vocab from the local fineweb_v4096 files
(decode the 4096-vocab documents, train byte-level BPE the same way as testbed/prepare_data.py,
re-encode). Usage: python3 -I prep_vocab.py 1024"""
import json, os, sys
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
src = os.path.join(ROOT, "data", "fineweb_v4096")
V = int(sys.argv[1])
out = os.path.join(ROOT, "data", f"fineweb_v{V}")
os.makedirs(out, exist_ok=True)
old = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
def docs(name):
    a = np.fromfile(os.path.join(src, f"{name}.bin"), dtype=np.uint16)
    cut = np.flatnonzero(a == 0).tolist() + [len(a)]
    seqs = [a[cut[i] + 1:cut[i + 1]].tolist() for i in range(len(cut) - 1)]
    return old.decode_batch(seqs)
val, train = docs("val"), docs("train")
print(len(val), len(train), "docs", flush=True)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=V, special_tokens=["<|endoftext|>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
tok.train_from_iterator(train[:12288], trainer=tr)
tok.save(os.path.join(out, "tokenizer.json"))
assert tok.token_to_id("<|endoftext|>") == 0
meta = dict(source="decoded from fineweb_v4096", vocab=V)
for name, ds in [("val", val), ("train", train)]:
    parts = []
    for i in range(0, len(ds), 2048):
        for e in tok.encode_batch(ds[i:i + 2048]):
            parts.append(np.array([0] + e.ids, dtype=np.uint16))
    arr = np.concatenate(parts)
    arr.tofile(os.path.join(out, f"{name}.bin"))
    nb = sum(len(d.encode("utf-8")) for d in ds)
    meta[name] = dict(tokens=int(arr.size), docs=len(ds), utf8_bytes=nb, bytes_per_token=round(nb / arr.size, 4))
    print(name, meta[name], flush=True)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
