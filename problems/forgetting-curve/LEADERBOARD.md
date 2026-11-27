# Leaderboard: forgetting-curve

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py forgetting-curve`.
"all" scores every model family combined; other tables score one family (`--family`).
★ = the formula to beat: the best score among formulas whose constants each earn their place (rule 2 in [problems/README.md](../README.md#rules)).

## all

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | ★ [clock_zb.py](../../problems/forgetting-curve/formulas/clock_zb.py) | 12 | 0.0612 | 0.1010 (2048) | 0.0525 | 10264 | maintainers | 2026-11-27 |
| 2 | [clock_z.py](../../problems/forgetting-curve/formulas/clock_z.py) | 10 | 0.0765 | 0.1637 (2048) | 0.0633 | 10264 | maintainers | 2026-11-27 |
| 3 | [lr_clock.py](../../problems/forgetting-curve/formulas/lr_clock.py) | 2 | 0.1297 | 0.2275 (384) | 0.1286 | 10264 | maintainers | 2026-11-27 |
| 4 | [chang_power.py](../../problems/forgetting-curve/formulas/chang_power.py) | 2 | 0.1588 | 0.2817 (512) | 0.1465 | 10264 | maintainers | 2026-11-27 |
| 5 | [mean.py](../../problems/forgetting-curve/formulas/mean.py) | 1 | 0.2077 | 0.3320 (2048) | 0.1883 | 10264 | maintainers | 2026-11-27 |
