# Hashed n-gram tables: 1.7682 bpb

0.164 bpb below the baseline (1.9319), at 9.994e13 of the 1e14 FLOPs (counter all-ops-v1).

- **Hashed bigram and trigram embeddings.** Two 131,072 x 128 tables, indexed by hashes of
  (previous token, token) and (two back, previous, token), added to the token embedding. They start at zero
  and train with Adam (lr 0.15) inside the counted steps. A lookup is a gather, charged 1 FLOP per element read.
- **3-bit tables in the artifact.** Each table row is stored as levels -3..3 in int8 with one float16 scale,
  and lzma packs them; every other weight is bf16. `--record` starts at 4 bits and steps down until the file
  fits: 4 bits gave 16,624,532 bytes, 3 bits 12,211,900. The score is for the model loaded back from that file.
- **Narrower and deeper:** 6 layers, width 128, 2 heads of 64, SwiGLU hidden 352. 35.0M parameters, of which
  33.6M are the two tables.
- **Training:** sequence length 128, batch 64 (8,192 tokens per step, as in the baseline), 1,242 steps,
  10.2M tokens. Muon momentum 0.8, cooldown over the last 60% of steps, embedding lr 0.3, head lr 0.008.

Full count (`--record`, every step on cpu): 9.976e13, 0.19% below the fast count.

`train.py` is the sweep script used for every run below; the settings of this record are its defaults, and
each can be overridden by an env var of the same name (unset options are off). The log is from that code run
with these settings passed as env vars.

## How it was found

1. Screens at 1e13 FLOPs scored on the first 100 validation documents, 2 to 7 min each on a 4-core cpu.
   Three seeds of one config gave 2.1627, 2.1604 and 2.1594, so gaps under about 0.006 are noise.
2. Winning changes were confirmed at 1e14 on the whole validation set.

| step (screens at 1e13) | bpb |
|---|---|
| baseline settings, batch 64 | 2.519 |
| batch 16 | 2.413 |
| + bigram table 8,192 rows | 2.241 |
| + width 128, 6 layers, bigram 16,384 | 2.155 |
| + trigram 16,384 | 2.139 |
| + Muon momentum 0.9, seq 128 x batch 32, bigram 32,768 | 2.104 |
| + Muon momentum 0.85, trigram 32,768, cooldown 0.6 | 2.085 |
| + table lr 0.15 | 2.076 |
| + Muon momentum 0.8, head lr 0.008 | 2.069 |
| tables 49,152 / 65,536 / 98,304 / 131,072 rows | 2.060 / 2.058 / 2.050 / 2.048 |

Storing the tables in 8, 6, 5 or 4 bits gave 2.0599, 2.0599, 2.0601 and 2.0603 (49,152 rows), and bf16 for
the other weights changed nothing, so the size limit, not quantization, decides the table size.

Did not help at 1e13: ReLU squared (+0.006), Muon lr 0.08 (+0.022), a 4-gram table (+0.016), table lr 1.0
(+0.042), narrower tables with a projection (+0.005 to +0.019), adding the tables before every block or before
the head (within noise). Logit softcap, Muon lr 0.02/0.03 and embedding lr 0.6 were within noise.

| confirmation at 1e14 (whole validation set) | bpb |
|---|---|
| baseline | 1.9319 |
| width 192, 4 layers, bigram 8,192, batch 32 | 1.8741 |
| width 96, 8 layers, bigram 16,384, batch 32 | 1.8523 |
| width 128, 6 layers, bigram 16,384, batch 32 | 1.8466 |
| + trigram, both 32,768, seq 128, momentum 0.85, cooldown 0.6, int8 tables | 1.7829 |
| + table lr 0.15, momentum 0.8, head lr 0.008 | 1.7824 |
| + both tables 131,072 rows, 3-bit tables, bf16 weights (this record) | 1.7682 |

The narrowest model (width 96) won at 1e13 but not at 1e14, so screens at 1e13 favor small models; sizes
need the full budget. The last three 1e13 gains (table lr, momentum 0.8, head lr) did not show at 1e14.

Run from `speedrun/tracks/lm-tiny/`: `python3 records/2026-11-28_ngram-tables/train.py --record`.
Training took 1,972 s on 4 cpu cores (Hetzner, Neoverse-N1) in float32; evaluation 84 s. Not yet run on mps (bf16).
