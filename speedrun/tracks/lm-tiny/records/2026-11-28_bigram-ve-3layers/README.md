# Bigram hash table + value embeddings + 3 layers: 1.8273 bpb

1.8273 bpb at 9.998e13 FLOPs, 0.105 below the baseline's 1.9319.

What changed against the [baseline](../2026-11-27_baseline/):

1. **Bigram hash table.** Row `(previous token x 1031 + token) mod 16384` of a 16384 x 64 table, mapped up to width
   192 and added to the input embedding. Alone it gave -0.22 bpb in the search (width 64), the largest of all ideas.
2. **Value embeddings** in all 3 blocks: a per-token vector (zero at the start) added to the attention values.
3. **3 layers instead of 4**, so the budget buys 532 steps instead of 412.
4. **Muon lr 0.025** (was 0.05); the tables train with Adam at lr 0.3.
5. Muon orthogonalizes same-shape matrices in one batched call: same update, fewer kernel launches.

Lookups cost 1 FLOP per element read under counter `all-ops-v1`, so the two tables (2.2M of the 3.37M
parameters) add almost no FLOPs; the 16 MB limit caps them. Artifact 12,053,604 bytes, scored from the file.
Full count 1.00179e14, +0.20% vs the fast count.

How it was found: 54 search runs at width 64, batch 8, 6e11 FLOPs (113 steps, 5-10 s each), scored on 100
validation documents; the search size ranks changes the same as a 4x budget. Tables: [search/NOTES.md](../../search/NOTES.md).

| search run (width 64, 6e11) | val_bpb |
|---|---|
| base, muon 0.025 | 2.6279 |
| + bigram 16384 x 64 | 2.4094 |
| + bigram, 3 layers | 2.3700 |
| + bigram, 3 layers, value embeddings in 3 | 2.3564 |

`train.log` has the result line only: the run's stdout was not kept. Training took 2197 s (4.1 s per step),
10x slower per step than the baseline; the cause is not known.
