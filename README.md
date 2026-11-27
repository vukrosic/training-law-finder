# Training Law Finder

> If you are an AI agent and were given only this link: read [agents/START.md](agents/START.md) and follow it.

Write papers for top AI conferences & help solve intelligence.

Find one rule of LLM training, prove it on runs it never saw, and write it up as your paper.

## Copy this prompt to your AI agent

Open Claude Code, Codex or another coding agent in an empty folder and paste:

```text
Clone https://github.com/vukrosic/training-law-finder (if it is already cloned, run `git pull --rebase --autostash` in it first), then read agents/START.md and follow it.
```

New here? Read [GUIDE.md](GUIDE.md) first: how to pick a target, how to work with the AI and how to keep it busy.

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
| [GUIDE.md](GUIDE.md) | for students: pick one target, choose how to work, keep the AI busy |
| [RESEARCH.md](RESEARCH.md) | how a result counts, your first training run, the worked example |
| [IDEAS.md](IDEAS.md) | problems waiting for a folder and 60 topics a formula could answer |
| [DATA.md](DATA.md) | the 39,598 real runs on Hugging Face, and how new runs flow in from GitHub to Hugging Face |
| [METHOD.md](METHOD.md) | the full method |

All runs are small scale. The range differs per problem: each problem README has a "Tested range:" line
(final-loss: widths 24 to 1024, 512 to 16,384 steps), and a paper states the range of its own data.

## License

MIT: use the code, change it and keep your changes private if you like. Sending them back is welcome, not required.

## Community

Discussion on Discord: https://discord.gg/6AbXGpKTwN

Funded by my [AI Research School](https://www.skool.com/become-ai-researcher-2669), coaching school for writing AI conference papers.
