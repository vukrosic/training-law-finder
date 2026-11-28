# lm-tiny record search, 2026-11-28

Search size: width 64, 1 head, mlp 176, 4 layers, batch 8, 6e11 FLOPs (113 steps), scored on the first 100
validation documents. Each line is one run of `search/queue.txt`; no repeats. Baseline lrs: muon 0.05, embed 0.6.

## r0: does the search size rank changes like a 4x budget?

| change | 6e11 | 2.4e12 |
|---|---|---|
| muon 0.025 | 2.6279 | 2.4026 |
| base | 2.6379 | 2.4205 |
| half batch | 2.6421 | 2.4276 |
| muon 0.1 | 2.6800 | 2.4967 |

Takeaway: same order at both budgets, so 6e11 is the search size, and muon 0.025 is the new reference (2.6279).

## r1: value embeddings and bigram hash table (best lr of each)

| idea | muon | table lr | val_bpb |
|---|---|---|---|
| bigram 16384 x 64 | 0.025 | 0.3 | 2.4094 |
| bigram 8192 x 64 (width) | 0.025 | 0.3 | 2.4158 |
| value embeddings, 4 layers | 0.025 | 0.3 | 2.5957 |
| value embeddings, 2 layers | 0.025 | 0.6 | 2.6022 |

Takeaway: the bigram table gives -0.22 bpb, value embeddings -0.03; muon 0.05 was worse in all 16 runs.

## r2: half and quarter batch

| batch | best muon | steps | val_bpb |
|---|---|---|---|
| half (4) | 0.0125 | 213 | 2.6262 |
| quarter (2) | 0.0125 | 380 | 2.6540 |

Takeaway: half batch equals base (-0.002), quarter batch is worse (+0.026).

## r3: smaller model on more tokens (mini widths: w128 -> 42, w160 -> 54)

| model | best lrs | steps | val_bpb |
|---|---|---|---|
| 3 layers | muon 0.05, embed 0.6 | 143 | 2.6054 |
| width 160 | muon 0.05, embed 0.6 | 146 | 2.6135 |
| width 128 | muon 0.05, embed 1.2 | 216 | 2.6138 |

Takeaway: 3 layers is the best smaller model (-0.022).

## r4: combinations (all with the bigram table 16384 x 64 unless named)

| run | val_bpb |
|---|---|
| bigram + value embeddings 3 + 3 layers, muon 0.025 | **2.3564** |
| bigram + 3 layers, muon 0.025 | 2.3700 |
| bigram + value embeddings 3 + 3 layers, muon 0.05 | 2.3742 |
| bigram + 3 layers, muon 0.05 | 2.3826 |
| bigram 32768 x 32 | 2.4072 |
| bigram + half batch, muon 0.0125 | 2.4100 |
| bigram, table lr 0.15 | 2.4113 |
| bigram + value embeddings 4 (4 layers) | refused: 16.35 MB at full size |

Takeaway: the gains add up: bigram -0.22, 3 layers -0.04, value embeddings -0.014.

## r5, r6: 4,096 vocabulary (tokenizer from search/make_vocab.py, lossless)

| run | steps | val_bpb |
|---|---|---|
| v4096, muon 0.025, embed 0.6 | 77 | 2.4287 |
| v4096 + 3 layers + bigram 16384 x 32 | 90 | 2.3779 |
| v4096 + 3 layers + value embeddings 1 | 90 | 2.3924 |
| v4096 + 3 layers | 90 | 2.3942 |
| v4096 + bigram 8192 x 32 | 77 | 2.4190 |
| v4096 + value embeddings 1 | | refused: 16.82 MB at full size |

Takeaway: vocab 4096 alone gives -0.20, but with it the tables no longer fit 16 MB, and the best v4096 run
(2.3779) loses to the best sp1024 run (2.3564).

## Confirm at full size, 1e14, whole validation set

| run | val_bpb | steps | artifact | full count |
|---|---|---|---|---|
| baseline | 1.9319 | 412 | 7,896,828 | +0.19% |
| bigram 16384 x 64 + value embeddings 3 + 3 layers, muon 0.025, table 0.3 | **1.8273** | 532 | 12,053,604 | +0.20% |

Takeaway: -0.105 bpb, a record by the 0.005 rule: [records/2026-11-28_bigram-ve-3layers](../records/2026-11-28_bigram-ve-3layers/).
Open: lookups cost 1 FLOP per element read, so the tables are almost free in FLOPs; the run took 2197 s
(4.1 s per step vs 0.39 s for the baseline), cause unknown.
