# Track lm-tiny

Train the best language model you can with 1e14 training FLOPs, scored by validation bits per byte on FineWeb-Edu.

| model | data | metric | constraint | hardware | baseline |
|---|---|---|---|---|---|
| language model | FineWeb-Edu ([vukrosic/training-law-speedrun](https://huggingface.co/datasets/vukrosic/training-law-speedrun)) | validation bits per byte, lower is better | 1e14 training FLOPs | MacBook (M4, mps), or any machine | 1.9319 bpb |

The name says only the model and its size (see [speedrun/](../../)); the dataset, budget, hardware and metric are the ones above.
The same rules are in `track.json`, which [common/check_submission.py](../../common/check_submission.py) reads. All tracks: [speedrun/](../../).

It is a small nanoGPT-style speedrun that fits on a laptop or one GPU. The score depends on the FLOPs
counted, not on how fast the machine is or what else runs on it. The rules follow
[OpenAI Parameter Golf](https://github.com/openai/parameter-golf): a 16 MB artifact limit as there, and a
FLOP budget in place of 10 minutes on 8xH100.

## Leaderboard

| bpb | FLOPs | date | name | who | what changed |
|---|---|---|---|---|---|
| 1.8466 | 9.993e13 | 2026-11-28 | [bigram-narrow](records/2026-11-28_bigram-narrow/) | vukrosic | hashed bigram embedding, 6 layers of width 128, batch 32 |
| 1.9319 | 9.986e13 | 2026-11-27 | [baseline](records/2026-11-27_baseline/) | vukrosic | 4 layers, width 192, Muon + Adam, sp1024 tokenizer |

## Rules

1. **Budget.** At most 1e14 training FLOPs, about 5 minutes on a MacBook (M4, mps). Every op counts, not only
   matrix multiplies. [common/flops.py](../../common/flops.py) (counter `all-ops-v1`) charges each op by its kind:

   | op | FLOPs |
   |---|---|
   | matrix multiply, attention, convolution | 2 per multiply-add |
   | FFT | 5 n log2(m), n elements, transform length m |
   | sort, topk, median | n log2(m), m = length of the sorted dim |
   | eigh, svd, qr, solve, inverse, cholesky | m x n x min(m, n) per matrix |
   | views (view, transpose, slice) | 0 |
   | gathers (embedding, index, gather) | 1 per element read |
   | scatters (index_put, scatter_add) | 1 per element written |
   | every other op (elementwise, softmax, norms, sums, cumsum, casts, optimizer updates) | 1 per element of its largest input or output |

   `train.py` counts one step on the meta device before training (the fast count, under 0.5 s, cached) and runs
   budget / step FLOPs steps. Example: the baseline step is 2.407e11 FLOPs of matrix multiplies and attention
   plus 1.7e9 of everything else. Evaluation is not part of the budget.
2. **Score.** Bits per byte on the fixed validation set, computed by `eval.py`:
   total nats over every predicted validation token / (ln 2 x UTF-8 bytes of the raw validation text).
   Lower is better. Use `eval.py` and `common/flops.py` as they are.
3. **Tokenizer is free.** The default is a 1024-token SentencePiece BPE (`sp1024.model`). A new tokenizer must
   be trained only on the training text and give back the exact validation text (`eval.check_lossless`); show
   that check in your README. The byte count always comes from the raw text, so every tokenizer is scored the
   same way.
4. **Anything else is free:** model size, architecture, optimizer, sequence length, batch size, precision,
   within the fair-play rules below.
5. **New record.** It must beat the current best by at least 0.005 bpb. A maintainer reruns it before it goes
   on the board.

## Fair play

Each rule blocks one shortcut. `common/check_submission.py` checks what it can automatically.

1. **Every op counts.** Example it blocks: mixing tokens with an FFT, which the old matrix-multiply count
   charged 0 FLOPs.
2. **Records are counted on a real device.** `train.py --record` counts every step on the machine that trains;
   if that full count differs from the fast count by more than 1%, the full count is the score. Example it
   blocks: code that checks `x.is_meta` and does less work while the meta count runs.
3. **16 MB.** Code, tokenizer and weights go into one compressed file (`--record` writes `out/model.xz`), at most
   16,000,000 bytes, and the score is for the model loaded back from that file. Example it blocks: a 1 GB
   n-gram table that memorizes the training text.
4. **Training text reaches the model only through counted PyTorch steps.** Outside a step, code may only cut
   the fixed stream into batches and move them to the device; training the tokenizer is the one exception.
   Example it blocks: counting bigrams with numpy before training and loading the table into the model.
5. **Evaluation is one forward pass per token, with no learning.** Example it blocks: updating the weights on
   validation text during evaluation (test-time training).
6. **No outside weights or data.** Weights start random or from a simple formula (zeros, identity). Example it
   blocks: starting from a pretrained model or distilling from one.
7. **No torch.compile or custom kernels in a step.** They hide ops from the counter.

You don't need the full counter unless you use FFT, eigendecompositions, sorts, scans, scatters or custom
kernels. For standard models the fast count is exact (the baseline's full count on mps is 0.2% higher, from
bf16 casts). `train.py` prints `standard ops, fast count is exact`, or names the non-standard ops and asks for
`--full-count`.

## Research tips

- **Search small.** The baseline gets only 4 steps at 1e12, so search runs need a smaller model or batch, for
  example width 64 and batch 8 at 6e11 FLOPs: 113 steps, about 10 s of training on an M4.
- **Score search runs on part of the validation set:** `--val-docs 100` takes 2 s instead of 33 s.
- **Check an idea's cost before training.** The fast count prints FLOPs per step in under a second.
- **Keep step shapes fixed during a search.** Then the fast count is cached and costs 0 s per run; changing only
  learning rates or initialization does not change the cache key.
- **Compress and full-count only finalists:** `python3 train.py --record`.

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
cd speedrun/tracks/lm-tiny
pip install torch numpy sentencepiece huggingface_hub
python3 prepare_data.py          # downloads the data into data/ here and checks sha256
python3 test_bpb.py              # checks the bpb accounting
python3 ../../common/test_flops.py  # checks the FLOP counter
python3 train.py --budget 1e12 --val-docs 100   # quick check, about 10 s, not a record
python3 train.py --record | tee train.log       # record run: cuda, then mps, then cpu
```

The last line of the log is `result {...}` with `val_bpb`, `flops`, the device and the wall time; a `--record`
run adds `full_count` and `artifact_bytes`, and the line before it lists the full count by op. The artifact is
saved in `out/` (not in git). To check a record before you open the pull request:

```bash
python3 ../../common/check_submission.py --count records/<date>_<name>                      # 3 real steps vs the fast count
python3 ../../common/check_submission.py --artifact out/model.xz records/<date>_<name>      # 16 MB and the score from the file
```

## Submit

Open a pull request that adds one folder, `records/<date>_<name>/`, with:

1. `README.md`: what you changed and why, in a few lines.
2. `submission.json`: `name`, `github`, `date`, `val_bpb`, `flops`, `counter` (`all-ops-v1`), `artifact_bytes`,
   `device`, `train_seconds`, `tokenizer`, `params`.
3. `train.log`: the log of `train.py --record`, trimmed to under 50 KB.
4. `train.py`: the exact script. It must run with `python3 records/<date>_<name>/train.py` from this folder.
   If it needs other packages, add `requirements.txt`.

Add your row to the leaderboard above in the same pull request. `python3 tools/check.py` (from the repository root)
must pass; it runs `speedrun/common/check_submission.py` on every record.
