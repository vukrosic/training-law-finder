# LLM testbed with facts (v1)

The training code behind `data/llm/runs.jsonl.gz`: small decoder-only language models (RMSNorm,
RoPE, QK-norm, SwiGLU, no biases), Muon on block matrices and Adam on the rest, trained on
FineWeb-Edu, with optional synthetic facts mixed into training and a probe that measures how many
bits of each fact the model knows. Older rows in the run table came from earlier versions of this
code; this folder is the current version.

## Set up

Run every command from this folder (`cd families/llm/testbed-facts-v1`).

```bash
sh data/download.sh                 # pinned official code (needed for dense runs too) + FineWeb-Edu shard 0 (94.5 MB)
python3 -I tools/prep_vocab.py 1024 # 1,024-token vocabulary (most rows in the run table)
python3 -I data/prepare.py          # only for the 8,192-token vocabulary (MoE benchmark)
```

## One run

```bash
python3 -I env/train.py --method dense --data fineweb_v1024 --steps 1024 \
  --set vocab=1024 facts=64 fkmax=128 fwlo=0.2 fwhi=0.45 \
  --out experiments/runs/mine.jsonl
```

Give all overrides after one `--set`: a second `--set` replaces the first. This run takes about
160 s on a laptop CPU. `--device auto` uses an NVIDIA GPU, then a Mac GPU (mps), then the CPU. If a
run stops with "too many bios per step", the window cannot fit 64 facts x up to 128 repeats: use more
steps or a larger batch (512 steps works with batch 64). To match a row of a problem's frozen data,
use the command in that problem's README (`problems/final-loss/README.md`, `problems/forgetting/README.md`).

`--set key=value` overrides any config field (`env/train.py` lists them: width `d_model`,
`n_layers`, `lr`, `sched`, `min_lr_frac`, `mwd`, ...). Each run appends one JSON line with the
config, validation loss, bits per byte and the fact probe.

Facts: 64 synthetic people, each shown k times (k = 1, 2, 4, ... 128) inside the window
`fwlo`-`fwhi` (share of training steps). After the window the model trains on plain text only,
which is what `problems/forgetting` measures.

The MoE side (`--method topk_aux | global_aux | dsv3 | expert_choice | remoe`) and its benchmark
are described in `benchmark/README.md`.
