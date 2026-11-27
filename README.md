# Training Law Finder

> AI agent given only this link: read [agents/START.md](agents/START.md) and follow it.

Find one rule of LLM training, prove it on runs it never saw, and write it up as a paper for a top AI conference.

## Copy this prompt to your AI agent

Open Claude Code, Codex or another coding agent in an empty folder and paste:

```text
Clone https://github.com/vukrosic/training-law-finder (if it is already cloned, run `git pull --rebase --autostash` in it first), then read agents/START.md and follow it.
```

The agents run the search; you check in every 15 minutes, steer, and decide what goes in the paper. Start with [GUIDE.md](GUIDE.md).

Install: `pip install -r requirements.txt` to score formulas (no torch); `pip install -r requirements-train.txt` to train runs.

## Open problems

| problem | predict | runs | best so far | noise floor |
|---|---|---|---|---|
| [forgetting](problems/forgetting/) | share of facts a model still knows at the end of training | 375 | ★ in the [leaderboard](problems/forgetting/LEADERBOARD.md) | 0.0126 |
| [final-loss](problems/final-loss/) | final validation loss from the training settings | 1,163 | ★ in the [leaderboard](problems/final-loss/LEADERBOARD.md) | 0.0019 |

Both are far from their floor. Rules and scoring: [problems/README.md](problems/README.md).

All runs are small scale. Each problem README has a "Tested range:" line (final-loss: widths 24 to 1024, 512 to 16,384 steps); a paper states the range of its own data.

## Draft paper

**Rules for How Language Models Forget Facts During Training** (Vuk Rosić, November 2026), AI-written, not yet checked by a person.
[PDF on Zenodo](https://zenodo.org/records/24575890) · DOI [10.5281/zenodo.24575889](https://doi.org/10.5281/zenodo.24575889) ·
[source and reproduce script](drafts/forgetting-rules/) · open directions, from no training to one GPU: [NEXT-STEPS.md](drafts/forgetting-rules/NEXT-STEPS.md).

```bibtex
@misc{rosic2026forgetting,
  author    = {Rosi{\'c}, Vuk},
  title     = {Rules for How Language Models Forget Facts During Training},
  year      = {2026},
  month     = nov,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24575889},
  url       = {https://doi.org/10.5281/zenodo.24575889}
}
```

## Read next

| file | what |
|---|---|
| [GUIDE.md](GUIDE.md) | for students: pick a target, choose how to work, keep the AI busy |
| [RESEARCH.md](RESEARCH.md) | first training run, worked example, repo map |
| [METHOD.md](METHOD.md) | the two scores and the full method |
| [IDEAS.md](IDEAS.md) | problems waiting for a folder, 60 topics |
| [DATA.md](DATA.md) | the 39,598 real runs on Hugging Face, how new runs flow in |
| [CONTRIBUTING.md](CONTRIBUTING.md) | send a formula, runs, a problem or a paper; credit: [CREDIT.md](CREDIT.md) |
| [speedrun/](speedrun/) | fixed-rule competitions; first track lm-tiny: best small GPT with 1e14 FLOPs, bits per byte on FineWeb-Edu |
| [skill/writing-conference-paper](skill/writing-conference-paper/SKILL.md) | the paper-writing method |

## License and community

MIT: use and change the code, keep changes private if you like; sending them back is welcome.

Discord: https://discord.gg/6AbXGpKTwN. Funded by my [AI Research School](https://www.skool.com/become-ai-researcher-2669), coaching school for writing AI conference papers.
