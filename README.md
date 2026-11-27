# Training Law Finder

Write papers for top AI conferences & help solve intelligence.

Find one rule of LLM training, prove it on runs it never saw, and write it up as your paper.

## Copy this prompt to your AI agent

Open Claude Code, Codex or another coding agent in an empty folder and paste:

```text
Clone https://github.com/vukrosic/training-law-finder (if it is already cloned, run `git pull --rebase --autostash` in it first) and read README.md, RESEARCH.md, IDEAS.md, problems/README.md, METHOD.md, agents/README.md, agents/manager.md and agents/WORKFLOW.md. Keep all project files in `work/<topic>/`, which git ignores, and pull again at every 15-minute check.

You are the manager of a research project whose goal is a paper with one new formula about AI training, tested on runs it never saw (score 1) and on runs predicted before they existed (score 2).

First ask me how I want to work and how much you should talk to me (the start questions in agents/manager.md), and keep to my answers. Do not run anything yet. Then propose 8-10 options: beating the leaderboard on one of the open problems in problems/, a topic from IDEAS.md, or a new idea of your own (a rule the full run tables can already answer, or one that needs new runs; `core.rows.load()` downloads those tables from Hugging Face into data/llm/ on first use). For each give one line each on: the number the formula would predict, why a formula might exist, the fastest test, and the closest published work and how this differs. Then recommend one and ask me which ideas I like, what I would change or combine, and whether I have an idea of my own. We pick together (in fully automatic mode you pick), then you follow agents/WORKFLOW.md.

Talk to me short and simple the whole time: plain words, short sentences, no jargon. Keep each option to 4 short lines and the whole list under 500 words. Give details only when I ask.
```

Check in every 15 minutes. The agents run the search; you steer and decide what goes in the paper.
The paper-writing method is in [skill/writing-conference-paper](skill/writing-conference-paper/SKILL.md).

## Open problems

| problem | predict | runs | best so far (lower is better) | noise floor |
|---|---|---|---|---|
| [forgetting](problems/forgetting/) | share of facts a model still knows at the end of training | 375 | 0.0441, 10 constants | 0.0126 |
| [final-loss](problems/final-loss/) | final validation loss from the training settings | 1,163 | 0.1350, 8 constants | 0.0019 |

Both are far from their floor. Rules and scoring: [problems/README.md](problems/README.md).

Install: `pip install -r requirements.txt` is enough to score formulas (no torch). Training new runs
needs `pip install -r requirements-train.txt`.

## Contribute back

Send a better formula, new runs or a new problem as a pull request. Every one comes with the command that
checks it, and the leaderboard records who found what first. How: [CONTRIBUTING.md](CONTRIBUTING.md).

## Read next

| file | what |
|---|---|
| [RESEARCH.md](RESEARCH.md) | how a result counts, your first training run, the worked example |
| [IDEAS.md](IDEAS.md) | problems waiting for a folder and 60 topics a formula could answer |
| [DATA.md](DATA.md) | the 39,598 real runs on Hugging Face, and how new runs flow in from GitHub to Hugging Face |
| [METHOD.md](METHOD.md) | the full method |

All runs are small scale. The range differs per problem: each problem README has a "Tested range:" line
(final-loss: widths 24 to 1024, 512 to 16,384 steps), and a paper states the range of its own data.

## License

MIT: use the code, change it and keep your changes private if you like. Sending them back is welcome, not required.

Discuss results and get help picking an idea: [AI Research School](https://www.skool.com/become-ai-researcher-2669).
