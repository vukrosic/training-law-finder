# Training Law Finder

A method and toolkit for a few-day research project that finds a formula predicting some behaviour of AI training, such as final loss, the best learning rate, or how many facts a model remembers, from the knobs a lab sets. A small team of AI coding-agent sessions runs the experiments, fits candidate formulas and writes the paper, while a human checks in every 15 minutes. Formulas are judged by two numbers: the worst R² on held-out slices of real runs (score 1), and the R² of predictions written before their runs existed (score 2). The repo holds the agent prompts, a small modern transformer testbed with optional planted facts, a job queue, the fitting and pre-registration tools, a paper-writing skill, and a worked example that runs on a laptop CPU.

> [!TIP]
> Start with [METHOD.md](METHOD.md) for the method, then run the example: `sh examples/loss-law/run.sh` (about 20 minutes on a laptop CPU).

## Quick start

```bash
pip install -r requirements.txt
python3 -I testbed/prepare_data.py --vocab 4096        # ~95 MB FineWeb-Edu shard, ~1 min
python3 testbed/train.py --data data/fineweb_v4096 --d 64 --layers 2 --heads 2 --seq 128 --steps 100
```

That prints one JSON row: config, parameter counts, tokens, validation loss and bits per byte.

## The worked example

Final validation loss from width and steps, 20 fit runs and 6 pre-registered runs on CPU ([examples/loss-law](examples/loss-law/README.md)):

| Candidate | Score 1 (held-out) | Score 2 (pre-registered) |
|---|---|---|
| E + A·N^-a + B·D^-b | 0.964 | 0.939 |
| log L = c + α log N + β log D + γ log N · log D | 0.963 | **0.985** |

The two candidates tie on held-out slices; predictions written before the runs, half of them at a width wider than any fit run, pick the second one.

## What is inside

| Folder | What it holds |
|---|---|
| [agents/](agents/README.md) | Prompts for the manager, ops, theory, sim and writer sessions, the orders template, and the 15-minute check |
| [testbed/](testbed/) | `train.py`: dense transformer (RMSNorm, RoPE, QK-norm, SwiGLU), Muon for hidden matrices and Adam for the rest, cosine / constant / warmup-stable-decay schedules, every knob a flag, one JSON row out. `facts.py`: made-up people with five facts each, mixed into training at chosen sighting counts, with per-person recall. `prepare_data.py`: FineWeb-Edu shard and BPE tokenizer |
| [queue/](queue/) | `run_jobs.py` runs a job file a few at a time and skips finished tags; `status.py` prints a one-screen status |
| [fit/](fit/) | `laws.py` fits linear-in-terms or any curve formula and scores held-out slices, with a gradient-boosted-tree ceiling; `prereg.py` writes and scores pre-registered predictions |
| [skill/](skill/writing-conference-paper/SKILL.md) | A paper-writing skill for coding agents: one finding, simple writing, controls, gates |
| [examples/loss-law/](examples/loss-law/README.md) | The worked example with its results |

## How a project runs

1. Write the target in one sentence: what number, from which knobs, over what range.
2. Open one agent session per role with its prompt from `agents/`. The manager gives numbered orders; the others answer with "done" and numbers.
3. Ops keeps the queues at about one hour of work. Theory fits candidates and writes predictions before each new batch.
4. Every 15 minutes the human reads the log, the queue status and the scores, and steps in when the scores stop moving.
5. When score 1 is high and score 2 agrees with it, the writer drafts the paper.

> [!NOTE]
> The testbed is small on purpose: widths of tens to a few hundred and a few thousand steps run on a laptop or one GPU. A formula found here is a claim about that range; state the range in the paper.

## Data

FineWeb-Edu (ODC-By), `shard_00000.parquet` of `karpathy/fineweb-edu-100b-shuffle` on Hugging Face, pinned to one revision. Documents 0-511 are validation, the rest training.

## License

MIT
