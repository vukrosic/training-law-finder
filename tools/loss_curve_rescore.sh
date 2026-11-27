#!/bin/sh
# Score the loss-curve formulas that have no real-data score yet, in order, then rebuild the leaderboard
# and run the repo check. Fitting only, no training. About 1-5 minutes per formula on a MacBook CPU.
#
#     sh tools/loss_curve_rescore.sh                  # multi_power, clock_lr, clock_s2
#     sh tools/loss_curve_rescore.sh clock_lr         # only the named formulas
#
# Each score is appended to problems/loss-curve/scores.jsonl (--record, family llm). Then copy the numbers
# into the table in problems/loss-curve/RESULTS.md.
set -e
cd "$(dirname "$0")/.."
FORMULAS="${*:-multi_power clock_lr clock_s2}"
for f in $FORMULAS; do
    echo "== $f"
    nice -n 10 python3 core/score.py loss-curve "problems/loss-curve/formulas/$f.py" --family llm --record --author maintainers
done
python3 core/leaderboard.py loss-curve
nice -n 10 python3 tools/check.py
