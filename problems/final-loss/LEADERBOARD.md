# Leaderboard: final-loss

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py final-loss`.
"all" scores every model family combined; other tables score one family (`--family`).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | [chinchilla_lr.py](../../problems/final-loss/formulas/chinchilla_lr.py) | 8 | 0.1476 | 0.5059 (1024) | 0.1493 | 22231 | vukrosic | 2026-11-26 |
| 2 | [chinchilla.py](../../problems/final-loss/formulas/chinchilla.py) | 5 | 0.2552 | 0.5871 (1024) | 0.1907 | 22231 | vukrosic | 2026-11-26 |
| 3 | [mean.py](../../problems/final-loss/formulas/mean.py) | 1 | 0.4398 | 0.8018 (160) | 0.2796 | 22231 | vukrosic | 2026-11-26 |
