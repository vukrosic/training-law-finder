# Leaderboard: loss-curve

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py loss-curve`.
"all" scores every model family combined; other tables score one family (`--family`).
★ = the formula to beat: the best score among formulas whose constants each earn their place (rule 2 in [problems/README.md](../README.md#rules)).

## llm

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | ★ [clock_lr.py](../../problems/loss-curve/formulas/clock_lr.py) | 8 | 0.1219 | 0.1514 (96) | 0.1121 | 6240 | maintainers | 2026-11-27 |
| 2 | [clock_s2.py](../../problems/loss-curve/formulas/clock_s2.py) | 9 | 0.1499 | 0.1914 (96) | 0.1328 | 6240 | maintainers | 2026-11-27 |
| 3 | [multi_power.py](../../problems/loss-curve/formulas/multi_power.py) | 9 | 0.1869 | 0.2418 (96) | 0.1758 | 6240 | maintainers | 2026-11-27 |
| 4 | [tissue_anneal.py](../../problems/loss-curve/formulas/tissue_anneal.py) | 7 | 0.2360 | 0.3051 (96) | 0.2246 | 6240 | maintainers | 2026-11-27 |
| 5 | [mean.py](../../problems/loss-curve/formulas/mean.py) | 1 | 0.5357 | 0.5983 (96) | 0.5217 | 6240 | maintainers | 2026-11-27 |

## synthetic

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | ★ [multi_power.py](../../problems/loss-curve/formulas/multi_power.py) | 9 | 0.0418 | 0.0735 (32) | 0.0400 | 2880 | maintainers | 2026-11-27 |
| 2 | [tissue_anneal.py](../../problems/loss-curve/formulas/tissue_anneal.py) | 7 | 0.0540 | 0.0883 (32) | 0.0497 | 2880 | maintainers | 2026-11-27 |
| 3 | [mean.py](../../problems/loss-curve/formulas/mean.py) | 1 | 0.4111 | 0.7289 (32) | 0.3772 | 2880 | maintainers | 2026-11-27 |
