# Bigram embedding, narrower and deeper: 1.8466 bpb

The baseline with three changes, 0.085 bpb below it, at 9.993e13 of the 1e14 FLOPs (counter all-ops-v1).

- **Hashed bigram embedding.** A 16,384 x 128 table, indexed by `(previous token * 1031 + token) % 16384`,
  added to the token embedding. It starts at zero and trains with Adam in the counted steps, like the token
  embedding. A lookup is a gather, charged 1 FLOP per element read.
- **Narrower and deeper:** 6 layers, width 128, 2 heads of 64, SwiGLU hidden 352. 3.57M parameters, of which
  2.10M are the bigram table.
- **Smaller batch, lower Adam rates:** batch 32 x 256 tokens, 1,113 steps, 9.1M tokens; embedding lr 0.3,
  head lr 0.004.

Muon, the schedule, the tokenizer, RoPE, QK-norm and the zero-init outputs are the same as the baseline.

Full count (`--record`, every step on cpu): 9.979e13, 0.14% below the fast count. Artifact (code + tokenizer +
float32 weights, lzma): 13,113,792 bytes of the 16,000,000 allowed.

## How it was found

Screens at 1e13 FLOPs scored on the first 100 validation documents, about 2 min each on a 4-core cpu:

| change (at 1e13) | bpb |
|---|---|
| baseline settings, batch 64 | 2.519 |
| batch 32 | 2.445 |
| batch 16 | 2.413 |
| batch 8 | 2.444 |
| batch 16, embedding lr 0.3, head lr 0.004 | 2.398 |
| batch 16, bigram 8,192 rows | 2.241 |
| batch 16, width 128, 6 layers | 2.333 |
| batch 16, width 128, 6 layers, bigram 16,384 | 2.155 |

Muon lr 0.03, 0.05 and 0.08 gave 2.432, 2.413 and 2.412; cooldown 0.6 gave 2.410; width 256 with 3 layers
gave 2.494. The best batch at 1e13 was 16; the record uses 32 because the best batch grows with the budget.
Batch 16 at 1e14 is not tested.

Run from `speedrun/tracks/lm-tiny/`: `python3 records/2026-11-28_bigram-narrow/train.py --record`.
Training took 1,627 s on 4 cpu cores in float32; evaluation 84 s. Not yet run on mps (bf16).
