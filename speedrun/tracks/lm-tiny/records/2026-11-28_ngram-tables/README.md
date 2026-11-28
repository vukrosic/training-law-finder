# Hashed n-gram tables: 1.7829 bpb

0.149 bpb below the baseline (1.9319), at 9.997e13 of the 1e14 FLOPs (counter all-ops-v1).

- **Hashed bigram and trigram embeddings.** Two 32,768 x 128 tables, indexed by hashes of
  (previous token, token) and (two back, previous, token), added to the token embedding. They start at zero
  and train with Adam (lr 0.3) inside the counted steps. A lookup is a gather, charged 1 FLOP per element read.
- **int8 tables in the artifact.** Each table row is stored as int8 with one float16 scale; the rest of the
  weights are float32. The score is for the model loaded back from that file. Artifact: 13,507,696 bytes.
- **Narrower and deeper:** 6 layers, width 128, 2 heads of 64, SwiGLU hidden 352. 9.86M parameters, of which
  8.39M are the two tables.
- **Training:** sequence length 128, batch 64 (8,192 tokens per step, as in the baseline), 1,247 steps,
  10.2M tokens. Muon momentum 0.85, cooldown over the last 60% of steps, embedding lr 0.3, head lr 0.004.

Full count (`--record`, every step on cpu): 9.984e13, 0.12% below the fast count.

`train.py` is the sweep script used for every run below; the settings of this record are its defaults, and
each can be overridden by an env var of the same name (unset options are off). The log is from that code run
with these settings passed as env vars.

## How it was found

1. Screens at 1e13 FLOPs scored on the first 100 validation documents, about 2 min each on a 4-core cpu.
   Three seeds of one config gave 2.1627, 2.1604 and 2.1594, so gaps under about 0.006 are noise.
2. Each change that won a screen was confirmed at 1e14 on the whole validation set.

| step (screens at 1e13, batch 16 unless noted) | bpb |
|---|---|
| baseline settings, batch 64 | 2.519 |
| batch 16 | 2.413 |
| + bigram table 8,192 rows | 2.241 |
| + width 128, 6 layers, bigram 16,384 | 2.155 |
| + trigram 16,384 | 2.139 |
| + Muon momentum 0.9, seq 128 x batch 32, bigram 32,768 | 2.104 |
| + Muon momentum 0.85 | 2.096 |
| + trigram 32,768 | 2.093 |

Did not help at 1e13: ReLU squared (+0.006), Muon lr 0.08 (+0.022), width 256 with 3 layers. Logit softcap 15
and embedding lr 0.6 were within noise.

| confirmation at 1e14 (whole validation set) | bpb |
|---|---|
| baseline | 1.9319 |
| width 192, 4 layers, bigram 8,192, batch 32 | 1.8741 |
| width 96, 8 layers, bigram 16,384, batch 32 | 1.8523 |
| width 128, 6 layers, bigram 16,384, batch 32 | 1.8466 |
| this record | 1.7829 |

The narrowest model (width 96) won at 1e13 but not at 1e14, so screens at 1e13 favor small models; sizes
need the full budget.

Run from `speedrun/tracks/lm-tiny/`: `python3 records/2026-11-28_ngram-tables/train.py --record`.
Training took 1,927 s on 4 cpu cores (Hetzner, Neoverse-N1) in float32; evaluation 83 s. Not yet run on mps (bf16).
