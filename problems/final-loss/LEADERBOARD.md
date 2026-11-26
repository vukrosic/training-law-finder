# Leaderboard: final-loss

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py final-loss`.
"all" scores every model family combined; other tables score one family (`--family`).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | [chinchilla_lr.py](../../problems/final-loss/formulas/chinchilla_lr.py) | 8 | 0.1979 | 0.5062 (1024) | 0.1493 | 22221 | vukrosic | 2026-11-26 |
| 2 | [chinchilla.py](../../problems/final-loss/formulas/chinchilla.py) | 5 | 0.2598 | 0.5869 (1024) | 0.1906 | 22221 | vukrosic | 2026-11-26 |
| 3 | [mean.py](../../problems/final-loss/formulas/mean.py) | 1 | 0.2990 | 0.4745 (384) | 0.2795 | 22221 | vukrosic | 2026-11-26 |
