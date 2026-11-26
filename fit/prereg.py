"""Score 2: write predictions before the runs exist, then score them.

    # 1. predict every job in a new jobs file from a saved fit (before running it)
    python3 fit/prereg.py write --fit fit.json --jobs new_jobs.jsonl \
        --data data/fineweb_v4096 --out predictions.jsonl

    # 2. run the jobs (queue/run_jobs.py), then score
    python3 fit/prereg.py score --pred predictions.jsonl --results results.jsonl

`write` refuses any tag that already has a result row, stamps each prediction
with the time and a hash of the fit, and appends; it never overwrites. `score`
uses only predictions written before their run finished, and prints R^2 and
the mean absolute error. That R^2 is score 2.

The predicted columns are built from the job's flags the same way train.py
would: config.* from the merged flags, params and nonemb_params from building
the model, tokens = steps * batch * seq.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "testbed"))
sys.path.insert(0, os.path.join(ROOT, "queue"))
from laws import Law, flatten, r2  # noqa: E402
from run_jobs import read_jsonl  # noqa: E402


def virtual_row(job, defaults, data):
    import train
    from model import GPT
    flags = dict(defaults)
    flags.update(job.get("args", {}))
    flags.setdefault("data", data)
    argv = []
    for k, v in flags.items():
        argv += [f"--{k}", str(v)]
    a = train.get_args(argv)
    from tokenizers import Tokenizer
    vocab = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json")).get_vocab_size()
    m = GPT(vocab, a.d, a.layers, a.heads, a.seq, a.ff_mult, bool(a.qk_norm))
    row = dict(tag=job["tag"], config=vars(a), vocab=vocab, params=m.n_params(False),
               nonemb_params=m.n_params(True), tokens=a.steps * a.batch * a.seq)
    return flatten(row)


def cmd_write(a):
    fit = json.load(open(a.fit))
    law = Law.from_json(fit)
    fit_hash = hashlib.sha256(open(a.fit, "rb").read()).hexdigest()[:12]
    have = {r.get("tag") for r in read_jsonl(a.results)} if a.results else set()
    done_pred = {p["tag"] for p in read_jsonl(a.out)}
    defaults = json.loads(a.defaults)
    n = 0
    with open(a.out, "a") as f:
        for job in read_jsonl(a.jobs):
            if job.get("withdrawn") or job["tag"] in done_pred:
                continue
            if job["tag"] in have:
                print(f"skip {job['tag']}: a result already exists, too late to predict")
                continue
            row = virtual_row(job, defaults, a.data)
            pred = float(law.predict([row])[0])
            f.write(json.dumps(dict(tag=job["tag"], y=law.spec["y"], pred=round(pred, 5),
                                    written=round(time.time(), 1), fit=a.fit,
                                    fit_sha=fit_hash)) + "\n")
            n += 1
    print(f"wrote {n} predictions to {a.out}")


def cmd_score(a):
    preds = {p["tag"]: p for p in read_jsonl(a.pred)}
    rows = {r["tag"]: r for r in read_jsonl(a.results)}
    ys, ps, late = [], [], 0
    print(f"{'tag':<28} {'pred':>9} {'actual':>9} {'error':>8}")
    for tag, p in preds.items():
        r = rows.get(tag)
        if r is None:
            continue
        if r.get("finished") and r["finished"] < p["written"]:
            late += 1
            continue
        y = flatten(r).get(p["y"])
        if y is None:
            continue
        ys.append(float(y))
        ps.append(p["pred"])
        print(f"{tag:<28} {p['pred']:>9.4f} {float(y):>9.4f} {float(y) - p['pred']:>+8.4f}")
    if late:
        print(f"ignored {late} predictions written after their run finished")
    if len(ys) < 3:
        print(f"{len(ys)} scored predictions; need 3+ for R^2")
        return
    ys, ps = np.array(ys), np.array(ps)
    print(f"SCORE 2 (pre-registered R^2) = {r2(ys, ps):.4f}   "
          f"MAE {np.abs(ys - ps).mean():.4f}   n={len(ys)}   pending {len(preds) - len(ys) - late}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--fit", required=True)
    w.add_argument("--jobs", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--results", default="", help="refuse tags already in this file")
    w.add_argument("--data", default=os.path.join(ROOT, "data", "fineweb_v8192"))
    w.add_argument("--defaults", default="{}", help="same JSON as run_jobs.py --defaults")
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--results", required=True)
    a = ap.parse_args()
    cmd_write(a) if a.cmd == "write" else cmd_score(a)


if __name__ == "__main__":
    main()
