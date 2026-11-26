#!/bin/sh
# Fetch everything MOE-PRETRAIN uses into data/ (gitignored). Downloads are untrusted: Python reading them runs with -I.
#   sh data/download.sh          official code at pinned commits + FineWeb-Edu shard 0 (94.5 MB) -> token files (131 MB)
#   sh data/download.sh full 20  also shards 1..20 of the same dataset (about 1.9 GB of parquet) for members with GPUs
set -e
cd "$(dirname "$0")"
mkdir -p ext fineweb_edu
REV=4c8f30d6756da75362432a4d5569e1b229263b71   # karpathy/fineweb-edu-100b-shuffle (ODC-By), pinned revision
HF=https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/$REV
fetch_git() {  # name url commit [paths...]: blobless clone, checkout of the pinned commit
  name=$1; url=$2; commit=$3; shift 3
  if [ ! -d "ext/$name/.git" ]; then git clone --quiet --filter=blob:none --no-checkout "$url" "ext/$name"; fi
  git -C "ext/$name" fetch --quiet origin "$commit" 2>/dev/null || true
  if [ $# -gt 0 ]; then git -C "ext/$name" checkout --quiet "$commit" -- "$@"; else git -C "ext/$name" checkout --quiet "$commit"; fi
  echo "ok $name $commit"
}
fetch_raw() {  # dir repo commit path...: single files from GitHub at a pinned commit
  dir=$1; repo=$2; commit=$3; shift 3
  for f in "$@"; do mkdir -p "ext/$dir/$(dirname "$f")"; [ -f "ext/$dir/$f" ] || curl -sSL -o "ext/$dir/$f" "https://raw.githubusercontent.com/$repo/$commit/$f"; done
  echo "ok $dir $commit"
}
fetch_raw megatron-core NVIDIA/Megatron-LM 55ac7082517c3878ae653c07c09c534b8aed49f6 \
  megatron/core/transformer/moe/moe_utils.py megatron/core/transformer/moe/router.py LICENSE   # core_v0.16.1, BSD-3-Clause
fetch_git remoe https://github.com/thu-ml/ReMoE 463c6af38ffaa841941ea834470ee09f8d515652 \
  megatron/core/transformer/moe/router.py megatron/core/transformer/moe/moe_utils.py megatron/training/training.py \
  megatron/core/transformer/transformer_config.py LICENSE README.md                            # ReMoE (ICLR 2025), Megatron fork
fetch_git deepseek-v3 https://github.com/deepseek-ai/DeepSeek-V3 9b4e9788e4a3a731f7567338ed15d3ec549ce03b   # MIT (code)
if [ ! -f fineweb_edu/shard_00000.parquet ]; then curl -sSL -o fineweb_edu/shard_00000.parquet "$HF/shard_00000.parquet"; fi
echo "6a3450ea745ee6b6830c09819c0f2c72682eba443022de595bcb50ceb3092c22  fineweb_edu/shard_00000.parquet" | shasum -a 256 -c
[ -f fineweb_edu/val.bin ] || python3 -I prepare.py
shasum -a 256 -c checksums.sha256
if [ "$1" = "full" ]; then
  n=${2:-20}; i=1
  while [ $i -le $n ]; do f=$(printf "shard_%05d.parquet" $i); [ -f "fineweb_edu/$f" ] || curl -sSL -o "fineweb_edu/$f" "$HF/$f"; i=$((i+1)); done
  echo "full: shards 1..$n in data/fineweb_edu/ (tokenize with the same bpe8192.json; not used by the laptop presets)"
fi
