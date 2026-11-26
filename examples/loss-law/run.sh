#!/bin/sh
# Loss-law example, end to end on a laptop CPU (about 15-20 minutes with 2 runs at a time).
set -e
cd "$(dirname "$0")/../.."
E=examples/loss-law
D='{"data": "data/fineweb_v4096", "layers": 3, "seq": 128, "batch": 16, "lr": 0.02}'

[ -f data/fineweb_v4096/train.bin ] || python3 -I testbed/prepare_data.py --vocab 4096

# 1. fit data: 20 random draws of width and steps
python3 $E/make_jobs.py fit > $E/jobs_fit.jsonl
python3 queue/run_jobs.py $E/jobs_fit.jsonl --results $E/results.jsonl --par 2 --threads 2 --defaults "$D"

# 2. two candidate formulas, scored on held-out widths and run lengths (score 1)
mkdir -p $E/fits
python3 fit/laws.py $E/results.jsonl --y val_loss --vars N=nonemb_params D=tokens \
    --curve "E + A*N**(-a) + B*D**(-b)" --p0 E=2 A=20 a=0.3 B=20 b=0.3 --bounds E=0:10 a=0.05:2 b=0.05:2 \
    --holdout config.d config.steps --ceiling --save $E/fits/two_power.json
python3 fit/laws.py $E/results.jsonl --y val_loss --log-y --vars N=nonemb_params D=tokens \
    --terms "log(N)" "log(D)" "log(N)*log(D)" --holdout config.d config.steps --save $E/fits/loglinear.json

# 3. pre-register predictions for 6 new runs, half wider than any fit run
python3 $E/make_jobs.py predict > $E/jobs_predict.jsonl
for f in two_power loglinear; do
  python3 fit/prereg.py write --fit $E/fits/$f.json --jobs $E/jobs_predict.jsonl \
      --out $E/predictions_$f.jsonl --results $E/results.jsonl --defaults "$D"
done

# 4. run them, then score the predictions (score 2)
python3 queue/run_jobs.py $E/jobs_predict.jsonl --results $E/results.jsonl --par 2 --threads 2 --defaults "$D"
for f in two_power loglinear; do
  echo "== $f"; python3 fit/prereg.py score --pred $E/predictions_$f.jsonl --results $E/results.jsonl
done
