# Training Law Finder

Write papers for top AI conferences & help solve intelligence.

<sub>Funded by Vuk Rosić's community, [Become AI Researcher](https://www.skool.com/become-ai-researcher-2669).</sub>

We are discovering all rules of LLM training, soon expanding to world models, JEPA and everything else, with a goal to solve intelligence.

Find a rule of training, give it a formula that predicts runs it never saw, and you have the core of a
research paper. Pick one of the open problems or topics below, or bring your own idea. The open problems
come with all of our run data, one scoring command and a leaderboard.

- **Main goal: your paper.** Write an AI research paper in days, submit it to a conference, and use it to get a research job or a PhD place.
- **Bigger goal: find every rule of training.** Each problem is one rule, for example how much a model forgets or what loss a run will reach. Many people searching in parallel, with checking automated, can find them far faster than one lab.

## Write a paper with an AI agent

Open Claude Code, Codex or another coding agent in an empty folder and paste:

```text
Clone https://github.com/vukrosic/training-law-finder (if it is already cloned, run `git pull --rebase --autostash` in it first) and read README.md, problems/README.md, METHOD.md, agents/README.md and agents/manager.md. Keep all project files in `work/<topic>/`, which git ignores, and pull again at every 15-minute check.

You are the manager of a research project whose goal is a paper with one new formula about AI training, tested on runs it never saw (score 1) and on runs predicted before they existed (score 2).

Do not run anything yet. First propose 8-10 options: beating the leaderboard on one of the open problems in problems/, a topic from the list in README.md, or a new idea of your own (a rule the data in data/llm/ can already answer, or one that needs new runs). For each give one line each on: the number the formula would predict, why a formula might exist, the fastest test, and the closest published work and how this differs. Then recommend one and ask me which ideas I like, what I would change or combine, and whether I have an idea of my own. We pick together, then you follow agents/manager.md.

Talk to me short and simple the whole time: plain words, short sentences, no jargon. Keep each option to 4 short lines and the whole list under 500 words. Give details only when I ask.
```

Check in every 15 minutes. The agents run the search; you steer and decide what goes in the paper.
The paper-writing method is in [skill/writing-conference-paper](skill/writing-conference-paper/SKILL.md).

## Open problems

| problem | predict | runs | best so far (score 1, lower is better) | noise floor |
|---|---|---|---|---|
| [forgetting](problems/forgetting/) | share of facts a model still knows at the end of training | 375 | 0.0441, 10 constants | 0.0126 |
| [final-loss](problems/final-loss/) | final validation loss from the training settings | 22,231 | 0.1476, 8 constants | 0.0001 |

runs = runs that measure that number. Forgetting needs a special run (64 facts shown in one window, then plain text), so only 775 of the 24,012 runs can measure it; 375 of those are the frozen fit table, and the newer ones join it at the next update. Another 6,700 runs track forgetting under different setups (replayed facts, a changed optimizer), so they are not mixed in. The rest record final loss.

Both are far from their floor. Rules, scores and how to add a problem: [problems/README.md](problems/README.md).

## Score a formula in one minute

```bash
git clone https://github.com/vukrosic/training-law-finder && cd training-law-finder
pip install numpy scipy
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py
cp core/formula_template.py problems/forgetting/formulas/my_idea.py   # edit fit() and predict()
python3 core/score.py forgetting problems/forgetting/formulas/my_idea.py
```

No GPU and no training needed. Each problem's data is frozen in the repo (`problems/<name>/data/`), so a score never changes when new runs arrive. To score on one model family only, add `--family llm`.

For a light clone, use `git clone --depth 1`. To update a clone, run `git pull --rebase --autostash`; if that fails because the history was rewritten (once, on 2026-11-26), see "Keep your copy of the repo current" in [agents/README.md](agents/README.md). Before a pull request, run `python3 tools/check.py` (it needs `pip install pyyaml`); GitHub runs the same checks.

## The data

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

## How a result counts

- **Score 1, held out.** Fit without one model width, predict that width, repeat for every width, average the error. A formula that only memorizes the table fails.
- **Score 2, predicted before the runs exist.** Commit predictions for new settings, then train them. The git commit time proves the predictions came first. This is what makes a paper convincing. See [rounds/](rounds/README.md).
- **Fewer constants win ties.** Each added constant must lower the error by at least 0.005.

## Contribute

Three kinds of pull request: a better formula, new runs, or a new problem. Every one comes with the
command that checks it. See [CONTRIBUTING.md](CONTRIBUTING.md), including how credit and co-authorship work.

## Where this goes

| family | status |
|---|---|
| language models (`families/llm/`) | open now: 2 problems, 39,598 runs |
| world models | planned |
| reinforcement learning | planned |

Problems that the current data can already start on, waiting for a folder: fact memorization
(bits stored per repeat count, `bits_by_k`), best learning rate per width, schedule shape at a fixed
learning-rate sum, mixture-of-experts load balance. Pull requests welcome.

<details>
<summary>60 more topics a formula could answer</summary>

| Topic | The formula predicts |
|---|---|
| **Loss and scaling** | |
| Final loss | validation loss at the end of training |
| Loss curve | validation loss at every step of training |
| Early stopping guess | final loss, read from the first 10% of the loss curve |
| Compute-optimal size | the model size with the lowest loss for a fixed compute budget |
| Depth vs width | the best ratio of layers to width for a given parameter count |
| Repeated data | loss when training on the same data several times |
| Diminishing returns | the point where more training tokens stop lowering loss |
| Small-to-large transfer | a large model's loss, predicted from small models' runs |
| **Learning rate and schedule** | |
| Best learning rate | the peak learning rate that gives the lowest loss |
| Learning-rate schedule | final loss under any schedule shape |
| Warmup length | the shortest warmup that still trains stably |
| Cooldown length | the best share of training spent decaying the learning rate |
| Schedule switch | loss after the learning rate changes mid-run |
| Learning-rate sensitivity | how much loss rises when the learning rate is off by 2× |
| Restart cost | loss lost when training resumes with a new schedule |
| **Optimizer** | |
| Muon weight decay | the best weight decay for the Muon optimizer |
| Muon vs Adam gap | how much lower Muon's loss is than Adam's at the same compute |
| Momentum | the best momentum value |
| Gradient clipping | the best clipping threshold |
| Update size | how large each weight update is at every step |
| Newton-Schulz steps | how many orthogonalization steps Muon needs |
| Embedding learning rate | the best learning rate for embeddings, relative to hidden layers |
| **Batch size and data** | |
| Critical batch size | the largest batch that still trains efficiently |
| Best batch size | the batch size with the lowest loss for a fixed token budget |
| Batch-size ramp | loss when the batch size grows during training |
| Data mixture | loss on each domain for a given mix of training data |
| Data order | how much the order of training data changes final loss |
| Sequence length | the best training context length for a given budget |
| Noisy data | how much loss rises as a share of the data is corrupted |
| **Memory and facts** | |
| Fact memorization | the share of facts a model can recall |
| Forgetting | how much a model forgets after it stops seeing a fact |
| Memory capacity | how many facts a model can store per parameter |
| Repetitions needed | how many times a fact must be seen before it is recalled |
| Timing of facts | how recall depends on when in training a fact was seen |
| Interference | how much learning new facts erases old ones |
| **Stability** | |
| Loss spikes | the learning rate at which training breaks |
| Divergence time | the step at which an unstable run blows up |
| Attention logit growth | how large attention logits grow during training |
| Weight norm growth | how large the weights grow during training |
| Low-precision safety | whether a run stays stable in 8-bit or 16-bit training |
| **Architecture** | |
| Attention scale | the best attention temperature |
| RoPE base | the best rotary-position base for a given context length |
| Number of heads | the best number of attention heads for a given width |
| MLP width | the best hidden size of the feed-forward layer |
| Residual scale | the best scale for each layer's residual branch |
| Init scale | the best initial weight size per layer type |
| **Mixture of experts** | |
| Number of experts | the best number of experts for a given compute budget |
| Expert load balance | how unevenly tokens spread across experts during training |
| Top-k routing | the best number of experts per token |
| Dense vs MoE gap | how much lower an MoE model's loss is than a dense model's at the same compute |
| **Fine-tuning and RL** | |
| Fine-tuning forgetting | how much pretraining skill is lost during fine-tuning |
| LoRA rank | the smallest LoRA rank that matches full fine-tuning |
| RL reward curve | reward at every step of reinforcement learning |
| KL penalty | the best KL coefficient in RL fine-tuning |
| Distillation gap | how close a small student gets to its teacher |
| **Compression and inference** | |
| Quantization loss | how much loss rises after 4-bit or 8-bit quantization |
| Pruning loss | how much loss rises when a share of weights is removed |
| KV-cache eviction | accuracy when the cache keeps only some tokens |
| Speculative decoding | how many draft tokens get accepted |

</details>

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
> All runs are small scale: widths up to 1024 and a few thousand steps, on laptops and one GPU. A formula found here holds for that range; state the range in any paper.

## Data

FineWeb-Edu (ODC-By), `shard_00000.parquet` of `karpathy/fineweb-edu-100b-shuffle` on Hugging Face, pinned to one revision. Documents 0-511 are validation, the rest training.

## License

MIT
