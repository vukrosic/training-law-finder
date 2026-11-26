# LLM testbed with facts (v1)

The training code behind `data/llm/runs.jsonl.gz`: small decoder-only language models (RMSNorm,
RoPE, QK-norm, SwiGLU, no biases), Muon on block matrices and Adam on the rest, trained on
FineWeb-Edu, with optional synthetic facts mixed into training and a probe that measures how many
bits of each fact the model knows. Older rows in the run table came from earlier versions of this
code; this folder is the current version.

## Set up

```bash
sh data/download.sh                 # pinned official code + FineWeb-Edu shard 0 (94.5 MB)
python3 -I data/prepare.py          # 8,192-token vocabulary (MoE benchmark)
python3 -I tools/prep_vocab.py 1024 # 1,024-token vocabulary (most rows in the run table)
```

## One run

```bash
python3 -I env/train.py --method dense --data fineweb_v1024 --steps 1024 \
  --set vocab=1024 --set facts=64 --set fkmax=128 --set fwlo=0.2 --set fwhi=0.45 \
  --out experiments/runs/mine.jsonl
```

`--set key=value` overrides any config field (`env/train.py` lists them: width `d_model`,
`n_layers`, `lr`, `sched`, `min_lr_frac`, `mwd`, ...). Each run appends one JSON line with the
config, validation loss, bits per byte and the fact probe.

Facts: 64 synthetic people, each shown k times (k = 1, 2, 4, ... 128) inside the window
`fwlo`-`fwhi` (share of training steps). After the window the model trains on plain text only,
which is what `problems/forgetting` measures.

The MoE side (`--method topk_aux | global_aux | dsv3 | expert_choice | remoe`) and its benchmark
are described in `benchmark/README.md`.
