"""Second benchmark: Python source code, the CPython 3.12 standard library on this machine
(PSF license; every .py file under sysconfig stdlib, site-packages excluded).

    python3 -I env/prepare_pystd.py

Split by file, fixed by md5(relative path) % 20: 0 -> val (tuning), 1 -> test (never used for
tuning), 2..19 -> train. Same BPE-8192 tokenizer as FineWeb-Edu (trained on FineWeb only, so
code costs more tokens per byte; bits per byte does not depend on the tokenizer).
Writes ../data/pystd/ (the video folder, not the shared arena data link)/{train,val,test}.bin (uint16, <|endoftext|> id 0 before each file) and meta.json.
"""
import glob, hashlib, json, os, sysconfig
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")   # arena/data (tokenizer)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pystd")  # video folder


def main():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(f for f in glob.glob(root + "/**/*.py", recursive=True) if "site-packages" not in f)
    tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer", "bpe8192.json"))
    splits = {"train": [], "val": [], "test": []}
    for f in files:
        rel = os.path.relpath(f, root)
        h = int(hashlib.md5(rel.encode()).hexdigest(), 16) % 20
        try:
            txt = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        splits["val" if h == 0 else "test" if h == 1 else "train"].append(txt)
    os.makedirs(OUT, exist_ok=True)
    meta = {"source": root, "python": sysconfig.get_python_version(), "files": len(files)}
    for name, docs in splits.items():
        ids = [np.array([0] + e.ids, dtype=np.uint16) for e in tok.encode_batch(docs)]
        arr = np.concatenate(ids)
        arr.tofile(os.path.join(OUT, f"{name}.bin"))
        nb = sum(len(d.encode("utf-8")) for d in docs)
        meta[name] = {"docs": len(docs), "tokens": int(len(arr)), "utf8_bytes": nb, "bytes_per_token": nb / len(arr)}
        print(name, meta[name])
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
