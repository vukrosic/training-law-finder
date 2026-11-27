# Leaderboard: forgetting

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py forgetting`.
"all" scores every model family combined; other tables score one family (`--family`).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | [end_z3.py](../../problems/forgetting/formulas/end_z3.py) | 12 | 0.0366 | 0.0488 (384) | 0.0300 | 375 | vukrosic | 2026-11-27 |
| 2 | [end_z2.py](../../problems/forgetting/formulas/end_z2.py) | 10 | 0.0441 | 0.0625 (384) | 0.0375 | 375 | vukrosic | 2026-11-26 |
| 3 | [lr_clock.py](../../problems/forgetting/formulas/lr_clock.py) | 2 | 0.1507 | 0.2384 (384) | 0.1364 | 375 | vukrosic | 2026-11-26 |
| 4 | [mean.py](../../problems/forgetting/formulas/mean.py) | 1 | 0.2177 | 0.3237 (512) | 0.1894 | 375 | vukrosic | 2026-11-26 |
