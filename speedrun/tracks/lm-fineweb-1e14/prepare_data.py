"""Get the speedrun data into speedrun/tracks/lm-fineweb-1e14/data/.

    python3 prepare_data.py            # download the prepared data from Hugging Face and check sha256
    python3 prepare_data.py --build    # rebuild it from FineWeb-Edu (maintainers; same files)
    python3 prepare_data.py --upload   # upload speedrun/tracks/lm-fineweb-1e14/data/ to Hugging Face (maintainers)

Files (all in speedrun/tracks/lm-fineweb-1e14/data/, never in git):
    train_docs.jsonl.gz   fixed training text: the first documents of FineWeb-Edu sample-10BT shard 000
    val_docs.jsonl        fixed validation text: the first VAL_DOCS documents of shard 013 (never train on it)
    sp1024.model          default tokenizer: SentencePiece BPE, 1024 tokens, lossless (byte fallback)
    train_000.bin ...     default tokens of the training text, uint16, each document starts with BOS
    val.bin               default tokens of the validation text, same format
    SHA256SUMS            sha256 of every file above
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HF_REPO = "vukrosic/training-law-speedrun"
SOURCE = "HuggingFaceFW/fineweb-edu"
TRAIN_SHARD = "sample/10BT/000_00000.parquet"
VAL_SHARD = "sample/10BT/013_00000.parquet"
TRAIN_BYTES = 150_000_000  # take whole documents until the training text reaches this many UTF-8 bytes
VAL_DOCS = 1000
VOCAB = 1024
BOS = 1
SHARD_TOKENS = 50_000_000


def read_docs(shard, stop):
    """Yield document texts from one FineWeb-Edu parquet shard, reading only the row groups needed."""
    import fsspec
    import pyarrow.parquet as pq
    fs = fsspec.filesystem("http", client_kwargs={"trust_env": True})  # trust_env: use HTTPS_PROXY if set
    f = pq.ParquetFile(fs.open(f"https://huggingface.co/datasets/{SOURCE}/resolve/main/{shard}", "rb",
                               block_size=4 << 20))
    n = nbytes = 0
    for batch in f.iter_batches(columns=["text"], batch_size=1024):
        for text in batch.column(0).to_pylist():
            yield text
            n += 1
            nbytes += len(text.encode("utf-8"))
            if stop(n, nbytes):
                return


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def write_jsonl(path, docs):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")


def train_tokenizer(docs, path):
    import sentencepiece as spm
    lines = (line for d in docs[:15000] for line in d.split("\n") if line)  # ~70 MB is plenty for 1024 tokens
    spm.SentencePieceTrainer.train(
        sentence_iterator=lines, model_prefix=path[:-len(".model")],
        model_type="bpe", vocab_size=VOCAB, byte_fallback=True, character_coverage=0.9995,
        normalization_rule_name="identity", add_dummy_prefix=False, remove_extra_whitespaces=False,
        split_digits=True, unk_id=0, bos_id=BOS, eos_id=-1, pad_id=-1, input_sentence_size=2_000_000,
        shuffle_input_sentence=True, max_sentence_length=16384, num_threads=4, minloglevel=2)


def encode(docs, model_path):
    """Token stream: BOS doc1 BOS doc2 ...  Checks that decoding gives back the exact text."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    out = []
    for i in range(0, len(docs), 4096):
        chunk = docs[i:i + 4096]
        ids = sp.encode(chunk, num_threads=4)
        assert sp.decode(ids) == chunk, "tokenizer is not lossless on this text"
        out.append(np.array([x for t in ids for x in [BOS] + t], dtype=np.uint16))
    return np.concatenate(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build():
    os.makedirs(DATA, exist_ok=True)
    val = list(read_docs(VAL_SHARD, lambda n, b: n >= VAL_DOCS))
    write_jsonl(os.path.join(DATA, "val_docs.jsonl"), val)
    print(f"val: {len(val)} documents, {sum(len(d.encode()) for d in val):,} bytes")
    train = list(read_docs(TRAIN_SHARD, lambda n, b: b >= TRAIN_BYTES))
    write_jsonl(os.path.join(DATA, "train_docs.jsonl.gz"), train)
    print(f"train: {len(train)} documents, {sum(len(d.encode()) for d in train):,} bytes")
    model = os.path.join(DATA, "sp1024.model")
    train_tokenizer(train, model)
    encode(val, model).tofile(os.path.join(DATA, "val.bin"))
    toks = encode(train, model)
    print(f"tokens: train {len(toks):,}, val {os.path.getsize(os.path.join(DATA, 'val.bin')) // 2:,}")
    for i in range(0, len(toks), SHARD_TOKENS):
        toks[i:i + SHARD_TOKENS].tofile(os.path.join(DATA, f"train_{i // SHARD_TOKENS:03d}.bin"))
    names = sorted(f for f in os.listdir(DATA) if f != "SHA256SUMS" and not f.endswith(".vocab"))
    with open(os.path.join(DATA, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write(f"{sha256(os.path.join(DATA, n))}  {n}\n")


def verify():
    bad = []
    for line in open(os.path.join(DATA, "SHA256SUMS")):
        h, n = line.split()
        if sha256(os.path.join(DATA, n)) != h:
            bad.append(n)
    print("sha256 ok" if not bad else f"sha256 MISMATCH: {bad}")
    return not bad


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=DATA)
    assert verify()


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=DATA,
                      allow_patterns=["*.bin", "*.jsonl", "*.jsonl.gz", "*.model", "SHA256SUMS", "README.md"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.upload:
        upload()
    else:
        download()
