# How to research here

A result counts by its two scores ([METHOD.md](METHOD.md#the-two-scores); for open problems, [problems/README.md](problems/README.md#two-scores)), and each added constant must earn its place ([rule 2](problems/README.md#rules)).

## Try it in 2 minutes

```bash
pip install -r requirements-train.txt
python3 -I testbed/prepare_data.py --vocab 4096        # downloads a ~95 MB FineWeb-Edu shard, ~1 min
python3 testbed/train.py --data data/fineweb_v4096 --d 64 --layers 2 --heads 2 --seq 128 --steps 100
```

The last command trains a small language model in about 10 seconds on a laptop CPU and prints one JSON row: settings, parameter count, tokens seen, validation loss. If the download fails, get `shard_00000.parquet` from [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) and pass `--shard /path/to/shard_00000.parquet`.

## The worked example

`sh examples/loss-law/run.sh` (20 minutes on a laptop): final loss from model size and training length. It refits on all of `results.jsonl` and overwrites `fits/`; to reproduce from scratch, move `results.jsonl`, `fits/` and `predictions_*.jsonl` aside first.

20 runs (widths 48-128, 100-700 steps), then 6 runs predicted before they ran, 3 at width 160, wider than any fit run:

| Formula | Score 1, R² held out | Score 2, R² predicted |
|---|---|---|
| Two power laws: $L = E + A N^{-a} + B D^{-b}$ | 0.964 | 0.939 |
| Log-linear with interaction: $\log L = c + \alpha \log N + \beta \log D + \gamma \log N \log D$ | 0.963 | **0.985** |

$N$ = non-embedding parameters, $D$ = training tokens, the rest are fitted constants. Both fit old runs equally; only advance predictions separate them. The $\log N \log D$ term captures that wider models gain more from longer training. Details: [examples/loss-law](examples/loss-law/README.md).

## Use it for your own problem

Only `testbed/` is specific to language models. Any training script that takes settings as flags and appends one JSON row per run works (RL agents, image models, robotics simulators):

```json
{"tag": "lr0.003_w256", "config": {"lr": 0.003, "width": 256}, "lr": 0.003, "width": 256, "reward": 412.5}
```

```json
{"tag": "lr0.003_w256", "prio": 1, "cmd": "python3 my_rl_train.py --lr 0.003 --width 256 --tag lr0.003_w256 --out results.jsonl"}
```

```bash
python3 queue/run_jobs.py jobs.jsonl --results results.jsonl --par 2
python3 fit/laws.py results.jsonl --y reward --vars LR=lr W=width \
    --terms "log(LR)" "log(W)" "log(LR)**2" --holdout config.width
```

## What is inside

| Folder | What it holds |
|---|---|
| [problems/](problems/README.md) | one folder per predicted number |
| [core/](core/) | `score.py` (score 1), `round_score.py` (score 2), `leaderboard.py`, `formula_template.py`, `rows.py` (load data), `columns.yaml` |
| [data/llm/](data/llm/SCHEMA.md) | column guide for the run tables |
| [tools/](tools/) | `check.py` (every PR), `check_paper.py` (paper drafts), `build_problem_data.py` (freeze a problem's data) |
| [rounds/](rounds/README.md) | predictions committed before the runs |
| [families/llm/testbed-facts-v1/](families/llm/testbed-facts-v1/README.md) | training code behind the run table: dense and MoE transformers, Muon, facts probe |
| [families/llm/search-code/](families/llm/search-code/) | training code of each search |
| [papers/](papers/README.md) | papers checked and written by people; PDFs on GitHub releases |
| [drafts/](drafts/README.md) | AI-written drafts, not checked by a person |
| [agents/](agents/README.md) | agent prompts and the stage-by-stage [WORKFLOW.md](agents/WORKFLOW.md) |
| [testbed/](testbed/) | minimal single-file trainer |
| [queue/](queue/) | `run_jobs.py` (run a job file in parallel), `status.py` |
| [fit/](fit/) | `laws.py`, `prereg.py`: fit and score formulas on your own results |
| [examples/loss-law/](examples/loss-law/README.md) | the worked example |
