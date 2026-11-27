# Leaderboard: final-loss

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py final-loss`.
"all" scores every model family combined; other tables score one family (`--family`).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | [chinchilla_lr.py](../../problems/final-loss/formulas/chinchilla_lr.py) | 8 | 0.1350 | 0.2899 (1024) | 0.1093 | 1163 | vukrosic | 2026-11-27 |
| 2 | [chinchilla.py](../../problems/final-loss/formulas/chinchilla.py) | 5 | 0.1471 | 0.3165 (1024) | 0.1171 | 1163 | vukrosic | 2026-11-27 |
| 3 | [mean.py](../../problems/final-loss/formulas/mean.py) | 1 | 0.2994 | 0.5153 (24) | 0.2581 | 1163 | vukrosic | 2026-11-27 |
