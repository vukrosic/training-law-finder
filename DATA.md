# Data

GitHub holds code, problems with their frozen data, formulas and leaderboards, and stays small (a pushed file stays in history forever). Hugging Face [vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research) holds the bulk: the tables below, every raw result line (`raw/`), every job file with the exact config of each run (`jobs/`), and the code snapshot of each upload (`code/`). The training code of every search is also in [families/llm/search-code/](families/llm/search-code/); both are refreshed together, so any row traces to its config and code.

## The tables

| file | rows | what |
|---|---|---|
| [runs.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/runs.jsonl.gz) | 24,012 | small decoder-only LMs: settings, final loss, fact-memory probe; full forgetting curve for 7,412 runs |
| [other_searches.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/other_searches.jsonl.gz) | 15,586 | other searches (compute, data mix, init, learning rate, mixture of experts, residual, loss target, vocabulary): settings and final loss |
| [sim.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/sim.jsonl.gz) | 4,944 | a fast fact-memory simulator |

Widths 24-2048, 1-12 layers, 160-32,000 steps, mostly Muon, constant / cosine / WSD schedules, dense and mixture-of-experts. Columns: [data/llm/SCHEMA.md](data/llm/SCHEMA.md). Code: [families/llm/testbed-facts-v1](families/llm/testbed-facts-v1/). Each row keeps its `device` (GPU model or CPU), since hardware shifts losses slightly.

`core.rows.load()` downloads a table into `data/llm/` on first use. Or:

```python
from datasets import load_dataset
runs = load_dataset("vukrosic/training-laws-research", "summary", split="train")
```

Reading needs no account. Uploading to Hugging Face needs a free account and a write token (`hf auth login`, or `huggingface-cli login` on older versions).

Training text: FineWeb-Edu (ODC-By), `shard_00000.parquet` of `karpathy/fineweb-edu-100b-shuffle`, pinned to one revision. Documents 0-511 are validation, the rest training.

## Studies

One-off studies keep their results, job files and a code snapshot under `studies/<name>/` of the dataset, outside `raw/`, `jobs/` and `code/`, which every refresh replaces.

| folder | runs | what |
|---|---|---|
| `studies/text-laws/` | about 800 | optimizer dial p from Muon (p = 0) to normalized momentum (p = 1); code in [families/llm/search-code/text-laws/](families/llm/search-code/text-laws/) |

## Checkpoints

[vukrosic/training-laws-checkpoints](https://huggingface.co/vukrosic/training-laws-checkpoints)

- Small runs keep none; they rerun in minutes.
- Runs over about 1 GPU-hour or 100M parameters, and runs a paper cites, upload only their final checkpoint, then delete the local copy.
- Mid-training checkpoints are never uploaded unless a paper studies the training curve.

## How new data flows in

1. **Up to 1 MB gzipped (about 5,000 rows):** GitHub PR with the raw output lines as `data/llm/contrib/<github-name>-<date>.jsonl.gz`. `tools/check.py` blocks files over 5 MB.
2. **Bigger:** upload to `contrib/<github-name>-<date>/` of the dataset with `huggingface_hub.upload_folder(..., repo_type="dataset", create_pr=True)`, and link it from a GitHub PR with the code and exact commands.
3. **After merge:** a maintainer exports the rows to the schema, adds them to the tables with contributor, machine and GitHub commit, and removes the file from the GitHub tree.
4. **Never send:** checkpoints (`*.pt`), logs, videos, archives, local paths, e-mails or tokens.
5. **If repo history passes 50 MB,** all result sets go to Hugging Face and GitHub takes code only.
