"""Train a 4,096-token tokenizer on the training text and tokenize train and validation with it (not yet run).

    python3 search/make_vocab.py            # writes data/sp4096/: sp4096.model, train_000.bin ..., val.bin
    python3 search/make_vocab.py --vocab 2048

Same settings as prepare_data.py's sp1024 (SentencePiece BPE, byte fallback, lossless, BOS = 1), trained on the
training text only, never on validation. The bpb score counts raw validation bytes, so it compares across vocabularies.
Tokens are uint16 like the default files. Takes a few minutes of cpu; the files stay out of git (data/ is ignored).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import prepare_data as pd  # noqa: E402
from prepare_data import BOS, DATA, SHARD_TOKENS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--train-docs", type=int, default=40000, help="documents used to train the tokenizer")
    a = ap.parse_args()
    import sentencepiece as spm
    out = os.path.join(DATA, f"sp{a.vocab}")
    os.makedirs(out, exist_ok=True)
    model = os.path.join(out, f"sp{a.vocab}.model")
    train = pd.read_jsonl(os.path.join(DATA, "train_docs.jsonl.gz"))
    val = pd.read_jsonl(os.path.join(DATA, "val_docs.jsonl"))
    lines = (line for d in train[:a.train_docs] for line in d.split("\n") if line)
    spm.SentencePieceTrainer.train(
        sentence_iterator=lines, model_prefix=model[:-len(".model")],
        model_type="bpe", vocab_size=a.vocab, byte_fallback=True, character_coverage=0.9995,
        normalization_rule_name="identity", add_dummy_prefix=False, remove_extra_whitespaces=False,
        split_digits=True, unk_id=0, bos_id=BOS, eos_id=-1, pad_id=-1, input_sentence_size=4_000_000,
        shuffle_input_sentence=True, max_sentence_length=16384, num_threads=4, minloglevel=2)
    pd.encode(val, model).tofile(os.path.join(out, "val.bin"))  # encode() checks the text comes back exactly
    toks = pd.encode(train, model)
    for i in range(0, len(toks), SHARD_TOKENS):
        toks[i:i + SHARD_TOKENS].tofile(os.path.join(out, f"train_{i // SHARD_TOKENS:03d}.bin"))
    n_val = os.path.getsize(os.path.join(out, "val.bin")) // 2
    print(f"sp{a.vocab}: train {len(toks):,} tokens, val {n_val:,} tokens "
          f"(lossless check passed); files in {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
