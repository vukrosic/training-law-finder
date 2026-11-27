# Leaderboard: loss-curve

Score 1 = mean held-out RMSE per group (lower is better). Rebuilt by `python3 core/leaderboard.py loss-curve`.
"all" scores every model family combined; other tables score one family (`--family`).
★ = the formula to beat: the best score among formulas whose constants each earn their place (rule 2 in [problems/README.md](../README.md#rules)).

## synthetic

| # | Formula | Constants | Score 1 | Worst group | In-sample | Rows | Author | Date |
|---|---|---|---|---|---|---|---|---|
| 1 | ★ [multi_power.py](../../problems/loss-curve/formulas/multi_power.py) | 9 | 0.0418 | 0.0735 (32) | 0.0400 | 2880 | maintainers | 2026-11-27 |
| 2 | [tissue_anneal.py](../../problems/loss-curve/formulas/tissue_anneal.py) | 7 | 0.0540 | 0.0883 (32) | 0.0497 | 2880 | maintainers | 2026-11-27 |
| 3 | [mean.py](../../problems/loss-curve/formulas/mean.py) | 1 | 0.4111 | 0.7289 (32) | 0.3772 | 2880 | maintainers | 2026-11-27 |
