# Leaderboard: final-loss

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py final-loss`.
"all" scores every model family combined; other tables score one family (`--family`).
★ = the formula to beat: the best score among formulas whose constants each earn their place (rule 2 in [problems/README.md](../README.md#rules)).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | ★ [lr_area.py](../../problems/final-loss/formulas/lr_area.py) | 12 | 0.0839 | 0.1536 (1024) | 0.0749 | 1163 | vukrosic | 2026-11-27 |
| 2 | [width_bowl.py](../../problems/final-loss/formulas/width_bowl.py) | 10 | 0.1041 | 0.1995 (1024) | 0.0860 | 1163 | vukrosic | 2026-11-27 |
| 3 | [chinchilla_lr.py](../../problems/final-loss/formulas/chinchilla_lr.py) | 8 | 0.1350 | 0.2899 (1024) | 0.1093 | 1163 | vukrosic | 2026-11-27 |
| 4 | [chinchilla.py](../../problems/final-loss/formulas/chinchilla.py) | 5 | 0.1471 | 0.3165 (1024) | 0.1171 | 1163 | vukrosic | 2026-11-27 |
| 5 | [mean.py](../../problems/final-loss/formulas/mean.py) | 1 | 0.2994 | 0.5153 (24) | 0.2581 | 1163 | vukrosic | 2026-11-27 |
