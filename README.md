# Training Law Finder

Find one rule of LLM training, prove it on runs it never saw, and write it up as your paper.

## Start in 3 commands

```bash
git clone --depth 1 https://github.com/vukrosic/training-law-finder && cd training-law-finder
pip install numpy scipy
python3 core/score.py forgetting problems/forgetting/formulas/end_z2.py
```

No GPU and no training needed: each problem's data is frozen in the repo. To try your own formula, copy
`core/formula_template.py` into `problems/forgetting/formulas/my_idea.py`, edit `fit()` and `predict()`, and
score it the same way. To update your clone later: `git pull --rebase --autostash`.

## Open problems

| problem | predict | runs | best so far (lower is better) | noise floor |
|---|---|---|---|---|
| [forgetting](problems/forgetting/) | share of facts a model still knows at the end of training | 375 | 0.0441, 10 constants | 0.0126 |
| [final-loss](problems/final-loss/) | final validation loss from the training settings | 22,221 | 0.1979, 8 constants | 0.0001 |

Both are far from their floor. Rules and scoring: [problems/README.md](problems/README.md).

## Contribute back

Send a better formula, new runs or a new problem as a pull request. Every one comes with the command that
checks it, and the leaderboard records who found what first. How: [CONTRIBUTING.md](CONTRIBUTING.md).

## Read next

| file | what |
|---|---|
| [RESEARCH.md](RESEARCH.md) | how a result counts, a copy-paste prompt to research with a coding agent, your first training run |
| [IDEAS.md](IDEAS.md) | problems waiting for a folder and 60 topics a formula could answer |
| [DATA.md](DATA.md) | the 39,598 real runs on Hugging Face, and how new runs flow in from GitHub to Hugging Face |
| [METHOD.md](METHOD.md) | the full method |

All runs are small scale (widths up to 2048, at most 32,000 steps), so state that range in any paper.

## License

MIT: use the code, change it and keep your changes private if you like. Sending them back is welcome, not required.

Discuss results and get help picking an idea: [AI Research School](https://www.skool.com/become-ai-researcher-2669).
