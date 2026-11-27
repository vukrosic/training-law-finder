# Data

## The tables

| file | rows | what |
|---|---|---|
| [runs.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/runs.jsonl.gz) | 24,012 | real training runs of small decoder-only language models: settings, final loss, a fact-memory probe, and for 7,412 runs the full forgetting curve |
| [other_searches.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/other_searches.jsonl.gz) | 15,586 | real runs from the other searches (compute, data mix, init, learning rate, mixture of experts, residual, loss target, vocabulary): settings and final loss |
| [sim.jsonl.gz](https://huggingface.co/datasets/vukrosic/training-laws-research/blob/main/sim.jsonl.gz) | 4,944 | runs of a fast fact-memory simulator |

Widths 24-2048, 1-12 layers, 160-32,000 steps, mostly Muon, constant / cosine / WSD
schedules, dense and mixture-of-experts. Every column: [data/llm/SCHEMA.md](data/llm/SCHEMA.md).
The code that produced them: [families/llm/testbed-facts-v1](families/llm/testbed-facts-v1/).

**Where everything is kept.** GitHub holds code, the problems with their frozen data, formulas and
leaderboards; it stays small. Hugging Face holds the bulk data:
[vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research) has the
tables above, every raw result line (`raw/`), every job file with the exact config each run was queued
with (`jobs/`), and the code snapshot that goes with each upload (`code/`). `core.rows.load()` downloads
a table into `data/llm/` the first time you ask for it. The training code of every search is also here in
[families/llm/search-code/](families/llm/search-code/). Both are refreshed together, so any row can be
traced to its config and code.

```python
from datasets import load_dataset
runs = load_dataset("vukrosic/training-laws-research", "summary", split="train")
```


## Checkpoints

Repo: [vukrosic/training-laws-checkpoints](https://huggingface.co/vukrosic/training-laws-checkpoints)

- Small runs keep no checkpoint. Their settings, results and code rerun them in minutes.
- Runs over about 1 GPU-hour or over 100M parameters, and runs a paper cites, keep only their final
  checkpoint. It goes to the Hugging Face model repo above, and then the local copy is deleted.
- Checkpoints from the middle of training are never uploaded, unless a paper studies the training curve.


## Training text

FineWeb-Edu (ODC-By), `shard_00000.parquet` of `karpathy/fineweb-edu-100b-shuffle` on Hugging Face, pinned to one revision. Documents 0-511 are validation, the rest training.


## How new data flows in (and keeps scaling)

GitHub holds code and small frozen problem data. Hugging Face holds every run. The repo must stay small
for everyone's clone, because a file pushed to GitHub stays in its history forever.

1. **Small result sets (up to 1 MB gzipped, about 5,000 rows): a GitHub pull request.** Add the training
   script's output lines, unchanged, as `data/llm/contrib/<github-name>-<date>.jsonl.gz`.
   `tools/check.py` blocks any file over 5 MB; aim for 1 MB or less.
2. **Bigger result sets: a pull request on the Hugging Face dataset.** Upload to
   `contrib/<github-name>-<date>/` of [vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research)
   with `huggingface_hub.upload_folder(..., repo_type="dataset", create_pr=True)`, and link that pull
   request from a GitHub pull request that has the code and the exact commands.
3. **After a merge:** a maintainer runs the export, which turns the rows into the schema of
   [data/llm/SCHEMA.md](data/llm/SCHEMA.md), adds them to the Hugging Face tables with the contributor's
   name, the machine and the GitHub commit, and removes the file from the GitHub tree.
4. **Never send:** checkpoints (`*.pt`), logs, videos, archives, local paths, e-mails or tokens.
5. **If the repo history passes 50 MB,** every result set goes to Hugging Face (step 2) and GitHub takes
   code only.

Each row keeps its machine (`device`: GPU model or CPU), because the same settings can give slightly
different losses on different hardware; formulas are checked for that.
