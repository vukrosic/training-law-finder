# Track lm-fineweb-1e14

Train the best language model you can with 1e14 training FLOPs, scored by validation bits per byte on FineWeb-Edu.

| model | data | metric | constraint | hardware | baseline |
|---|---|---|---|---|---|
| language model | FineWeb-Edu ([vukrosic/training-law-speedrun](https://huggingface.co/datasets/vukrosic/training-law-speedrun)) | validation bits per byte, lower is better | 1e14 training FLOPs | MacBook (M4, mps); the count is the same on any machine | 1.932 bpb |

The same rules are in `track.json`, which [common/check_submission.py](../../common/check_submission.py) reads. All tracks: [speedrun/](../../).

It is a small nanoGPT-style speedrun that fits on a laptop or one GPU. The score depends only on the FLOPs
counted, not on the machine or on what else runs on it. The rules follow
[OpenAI Parameter Golf](https://github.com/openai/parameter-golf), with a FLOP budget in place of
10 minutes on 8xH100 and no model size cap.

## Leaderboard

| bpb | FLOPs | date | name | who | what changed |
|---|---|---|---|---|---|
| 1.9320 | 9.989e13 | 2026-11-27 | [baseline](records/2026-11-27_baseline/) | vukrosic | 4 layers, width 192, Muon + Adam, sp1024 tokenizer |

## Rules

1. **Budget.** At most 1e14 training FLOPs, about 5 minutes on a MacBook (M4, mps). [common/flops.py](../../common/flops.py) counts them with
   `torch.utils.flop_counter.FlopCounterMode` on one full training step (forward, backward and optimizer step,
   so Muon's Newton-Schulz is in it), times the number of steps. The step runs your real code at the real shapes
   on the `meta` device, so the count is the same on every machine (`flops.py` says why). If your steps differ
   (for example a batch or sequence length schedule), count each kind of step. If your step does work the
   counter cannot see, such as a custom kernel, count it and say how. Evaluation is not part of the budget.
2. **Score.** Bits per byte on the fixed validation set, computed by `eval.py`:
   total nats over every predicted validation token / (ln 2 x UTF-8 bytes of the raw validation text).
   Lower is better. Use `eval.py` and `common/flops.py` as they are.
3. **Data.** Train only on the fixed training stream. No validation text in training, in the tokenizer or in
   the model, and no network calls during training or evaluation.
4. **Tokenizer is free.** The default is a 1024-token SentencePiece BPE (`sp1024.model`). A new tokenizer must
   give back the exact validation text (`eval.check_lossless`); show that check in your README. The byte count
   always comes from the raw text, so every tokenizer is scored the same way.
5. **Anything else is free:** model size, architecture, optimizer, sequence length, batch size, precision.
6. **New record.** It must beat the current best by at least 0.005 bpb. A maintainer reruns it before it goes
   on the board.

## Data

| file | what |
|---|---|
| `train_docs.jsonl.gz` | training text: the first 30,491 documents (150 MB) of FineWeb-Edu `sample-10BT` shard 000 |
| `val_docs.jsonl` | validation text: the first 1,000 documents (4.65 MB) of shard 013 |
| `sp1024.model` | default tokenizer, lossless (byte fallback, no text normalization) |
| `train_*.bin`, `val.bin` | default tokens (61.5M train, 1.88M validation), uint16, each document starts with BOS (id 1) |

All of it is on Hugging Face: [vukrosic/training-law-speedrun](https://huggingface.co/datasets/vukrosic/training-law-speedrun),
with a sha256 for each file. Nothing of it is in git.

## Run

```bash
cd speedrun/tracks/lm-fineweb-1e14
pip install torch numpy sentencepiece huggingface_hub
python3 prepare_data.py          # downloads the data into data/ here and checks sha256
python3 test_bpb.py              # checks the bpb accounting
python3 train.py | tee train.log # cuda, then mps, then cpu
```

The last line of the log is `result {...}` with `val_bpb`, `flops`, the device and the wall time.

## Submit

Open a pull request that adds one folder, `records/<date>_<name>/`, with:

1. `README.md`: what you changed and why, in a few lines.
2. `submission.json`: `name`, `github`, `date`, `val_bpb`, `flops`, `device`, `train_seconds`, `tokenizer`, `params`.
3. `train.log`: the log your run printed, trimmed to under 50 KB.
4. `train.py`: the exact script. It must run with `python3 records/<date>_<name>/train.py` from this folder.
   If it needs other packages, add `requirements.txt`.

Add your row to the leaderboard above in the same pull request. `python3 tools/check.py` (from the repository root)
must pass; it runs `speedrun/common/check_submission.py` on every record.
