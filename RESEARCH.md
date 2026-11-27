# How to research here

The method: find a rule of training, write it as a formula, and test it on runs it never saw. The full method is in [METHOD.md](METHOD.md); the paper-writing skill is in [skill/writing-conference-paper](skill/writing-conference-paper/SKILL.md).

## How a result counts

- **Score 1, held out.** Fit without one model width, predict that width, repeat for every width, average the error. A formula that only memorizes the table fails.
- **Score 2, predicted before the runs exist.** Commit predictions for new settings, then train them. The git commit time proves the predictions came first. This is what makes a paper convincing. See [rounds/](rounds/README.md).
- **Fewer constants win ties.** Each added constant must lower the error by at least 0.005.


## Write a paper with an AI agent

Open Claude Code, Codex or another coding agent in an empty folder and paste:

```text
Clone https://github.com/vukrosic/training-law-finder (if it is already cloned, run `git pull --rebase --autostash` in it first) and read README.md, RESEARCH.md, IDEAS.md, problems/README.md, METHOD.md, agents/README.md and agents/manager.md. Keep all project files in `work/<topic>/`, which git ignores, and pull again at every 15-minute check.

You are the manager of a research project whose goal is a paper with one new formula about AI training, tested on runs it never saw (score 1) and on runs predicted before they existed (score 2).

Do not run anything yet. First propose 8-10 options: beating the leaderboard on one of the open problems in problems/, a topic from IDEAS.md, or a new idea of your own (a rule the full run tables can already answer, or one that needs new runs; `core.rows.load()` downloads those tables from Hugging Face into data/llm/ on first use). For each give one line each on: the number the formula would predict, why a formula might exist, the fastest test, and the closest published work and how this differs. Then recommend one and ask me which ideas I like, what I would change or combine, and whether I have an idea of my own. We pick together, then you follow agents/manager.md.

Talk to me short and simple the whole time: plain words, short sentences, no jargon. Keep each option to 4 short lines and the whole list under 500 words. Give details only when I ask.
```

Check in every 15 minutes. The agents run the search; you steer and decide what goes in the paper.
The paper-writing method is in [skill/writing-conference-paper](skill/writing-conference-paper/SKILL.md).


## Try it in 2 minutes

```bash
pip install -r requirements.txt
python3 -I testbed/prepare_data.py --vocab 4096        # downloads a ~95 MB FineWeb-Edu shard, ~1 min
python3 testbed/train.py --data data/fineweb_v4096 --d 64 --layers 2 --heads 2 --seq 128 --steps 100
```

The last command trains a tiny language model for about 10 seconds on a laptop CPU and prints one JSON row: its settings, parameter count, tokens seen and validation loss.

If the download fails, download `shard_00000.parquet` from [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) yourself and pass it with `--shard /path/to/shard_00000.parquet`.

## The worked example

**Example:** final loss from model size and training length, a small known case that runs end to end on a laptop in 20 minutes: `sh examples/loss-law/run.sh`. The script refits on every row in `results.jsonl` and overwrites the committed `fits/`, so to reproduce the table below from scratch, move `results.jsonl`, `fits/` and `predictions_*.jsonl` aside first.

20 training runs (widths 48-128, 100-700 steps), then 6 new runs predicted before they ran, 3 of them at width 160, wider than any fit run.

| Formula | Score 1, R² (held out, higher is better) | Score 2, R² (predicted in advance, higher is better) |
|---|---|---|
| Sum of two power laws: $L = E + A N^{-a} + B D^{-b}$ | 0.964 | 0.939 |
| Log-linear with interaction: $\log L = c + \alpha \log N + \beta \log D + \gamma \log N \log D$ | 0.963 | **0.985** |

$N$ is non-embedding parameters, $D$ is training tokens, and $E, A, a, B, b, c, \alpha, \beta, \gamma$ are fitted constants. Both formulas fit the old runs equally well, and only the predictions made in advance separate them. The second formula's $\log N \log D$ term captures that wider models gain more from longer training. Details: [examples/loss-law](examples/loss-law/README.md).

## Use it for your own problem

Nothing outside `testbed/` is specific to language models. Give the agents a training script that takes settings as flags and appends one JSON row per run, and they can search for formulas in it: RL agents, image models, robotics simulators.

A result row:

```json
{"tag": "lr0.003_w256", "config": {"lr": 0.003, "width": 256}, "lr": 0.003, "width": 256, "reward": 412.5}
```

A job that runs your script:

```json
{"tag": "lr0.003_w256", "prio": 1, "cmd": "python3 my_rl_train.py --lr 0.003 --width 256 --tag lr0.003_w256 --out results.jsonl"}
```

Run the jobs, then fit and score a formula:

```bash
python3 queue/run_jobs.py jobs.jsonl --results results.jsonl --par 2
python3 fit/laws.py results.jsonl --y reward --vars LR=lr W=width \
    --terms "log(LR)" "log(W)" "log(LR)**2" --holdout config.width
```

## What is inside

| Folder | What it holds |
|---|---|
| [problems/](problems/README.md) | One folder per predicted number: `data/<model family>.jsonl.gz`, `problem.py`, baseline formulas, scores and leaderboard |
| [core/](core/) | `score.py` (score 1), `leaderboard.py`, `round_score.py` (score 2), `formula_template.py`, `rows.py` (load problem data and the run tables), `columns.yaml` (one meaning per column across model families) |
| [data/llm/](data/llm/SCHEMA.md) | Column guide for the full run tables; the tables themselves download from Hugging Face |
| [tools/](tools/) | `check.py` (the checks every pull request must pass) and `build_problem_data.py` (freeze a problem's data) |
| [rounds/](rounds/README.md) | Prediction rounds: predictions committed before the runs |
| [families/llm/testbed-facts-v1/](families/llm/testbed-facts-v1/README.md) | The training code behind the run table: dense and MoE transformers, Muon, facts probe |
| [families/llm/search-code/](families/llm/search-code/) | Current training code of each search (compute, data, init, ...), copied with every data refresh |
| [papers/](papers/README.md) | Finished papers checked and written by people; PDFs are attached to GitHub releases |
| [drafts/](drafts/README.md) | Paper drafts written by AI agents, not checked by a person |
| [agents/](agents/README.md) | Prompts for the manager, ops, theory, sim and writer sessions |
| [testbed/](testbed/) | A minimal single-file trainer for your first runs (used by the worked example) |
| [queue/](queue/) | `run_jobs.py` runs a job file a few jobs at a time; `status.py` prints a one-screen status |
| [fit/](fit/) | `laws.py` and `prereg.py`: fit and score formulas on your own result files (R² version of the two scores) |
| [skill/](skill/writing-conference-paper/SKILL.md) | A paper-writing skill for coding agents: one finding, simple writing, controls, gates |
| [examples/loss-law/](examples/loss-law/README.md) | A worked example end to end on a laptop in 20 minutes |

> [!NOTE]
> All runs are small scale: widths up to 2048 and at most 32,000 steps, on laptops and one GPU. A formula found here holds for that range; state the range in any paper.
