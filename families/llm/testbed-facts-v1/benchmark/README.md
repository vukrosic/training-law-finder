# Benchmark: FineWeb-Edu bits per byte + load balance

Every model is trained from scratch on shard 0 of `karpathy/fineweb-edu-100b-shuffle` (rev 4c8f30d, sha256 in `data/download.sh`), tokenized with the environment's own byte-level BPE (vocabulary 8,192, file and checksum in `data/`).

- Split: documents 0..511 of shard 0 are validation (525,030 tokens, 2.06M bytes), documents 512..53,247 train (65.0M tokens; `data/prepare.py`). The tokenizer was trained on train documents 512..12,799 only. More shards for long runs: `sh data/download.sh full N`.
- Quality score: validation bits per byte = summed next-token cross-entropy (nats) / ln 2 / UTF-8 bytes of the scored tokens. Bytes, not tokens, so a different tokenizer can be compared. Lower is better.
- Laptop eval items: the first 64 validation windows of 256 tokens (16,384 tokens), fixed, no randomness. Full eval: `--eval_seqs 0` scores all validation windows.
- Load-balance scores, measured on the same eval windows in batches of 16 x 256 = 4,096 tokens, averaged over MoE layers and batches:
  - `load_cv`: standard deviation / mean of tokens per expert (0 = perfect balance).
  - `maxvio`: (max load - mean load) / mean load, the MaxVio of Wang et al. 2024 (2408.15664).
  - `drop_at_1.25`: share of token-expert assignments that a capacity factor of 1.25 would drop (the Switch default), for routers that are trained dropless.
  - `zero_expert`: share of tokens that get no routed expert (possible for Expert Choice and ReMoE; 0 for top-k).
  - `experts_per_token`: mean experts per token (fixed k for token choice; variable for Expert Choice and ReMoE).
- Compute is reported beside every score: steps, tokens, total and active parameters, train seconds. Compare methods at equal tokens and equal active parameters.
- Table: `python3 -I benchmark/table.py experiments/runs/*.jsonl`.

Published reference points (larger scale, different data; context, not targets):
- Loss-Free Balancing (2408.15664, 1B MoE on 100B tokens): validation perplexity 9.50 with loss-free balancing vs 9.56 with the auxiliary loss; MaxVio_global 0.04 vs 0.72.
- Global-batch balance (2501.11873, 3.4B MoE on 120B tokens): average downstream benchmark improves and loss drops by about 0.02-0.03 vs micro-batch balancing.
- ReMoE (2412.14711, 182M-978M active on the Pile): ReMoE beats top-k MoE with auxiliary loss at every size reported.
- Expert Choice (2202.09368): 2x faster convergence than Switch and GShard top-2 at 8B/64E in their setup.
