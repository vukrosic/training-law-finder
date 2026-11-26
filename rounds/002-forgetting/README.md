# Round 002: forgetting at 503M parameters

Three runs, all with the facts in window A (20-45% of steps):

| run | width | layers | non-embedding params | batch | steps | peak lr | wd |
|---|---|---|---|---|---|---|---|
| o112_w2048L12_b32_p0.008 | 2048 | 12 | 503M | 32 | 1024 | 8e-3 | 0 |
| o112_w2048L12_b32_p0.004 | 2048 | 12 | 503M | 32 | 1024 | 4e-3 | 0 |
| o112_w768L10_b96_p0.0086_T2048_wd0.3_A | 768 | 10 | 59M | 96 | 2048 | 8.6e-3 | 0.3 |

The widest model in the fit table is 1024 wide with 12 layers (126M non-embedding parameters), so the
two 2048-wide runs are four times larger than anything the formulas were fit on.

`predictions.jsonl` was made with `core/predict.py` from end_z2.py, hA.py and a_hA.py, each fit on the 375 repo rows.

The 768-wide run finished before these predictions were written: measured kept = 0.104. Every formula
overpredicts it by about +0.19, so it is in this round as a known miss, not as proof that predictions came first.

Predictions from the same three formulas fit on our larger internal table (900 rows, batch 16-1024), frozen
before the runs on 2026-11-26/27:

| run | end+z2 | hA | a+hA |
|---|---|---|---|
| o112_w2048L12_b32_p0.008 | 0.241 | 0.180 | 0.157 |
| o112_w2048L12_b32_p0.004 | 0.388 | 0.304 | 0.288 |
