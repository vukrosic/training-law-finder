"""Build the tables behind CLAIMS.md into work/claims/ (gitignored). No training.

    python3 tools/claims_data.py            # reads data/llm/runs.jsonl.gz (downloads it once)

Writes:
  work/claims/forget_new/settings.jsonl, results.jsonl   the clean-protocol runs (protocol lf) that are
      not in problems/forgetting/data, in the round format, so core/predict.py and core/round_score.py
      score the frozen formulas on them
  work/claims/lf.jsonl          every clean-protocol run with derived columns, for fit/laws.py
  work/claims/curve.jsonl       one row per reading of the clean-protocol curves (every 4th reading)
  work/claims/other_<search>.jsonl   other_searches rows per search, flattened, for fit/laws.py
"""
import gzip
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.rows import OTHER, RUNS, load, read_jsonl  # noqa: E402

OUT = os.path.join(ROOT, "work", "claims")
SETTINGS = ["width", "layers", "heads", "hidden", "params", "nonemb_params", "batch", "seq", "steps",
            "tokens", "sched", "peak", "mlr", "wd", "min_lr_frac", "warmup", "decay", "momentum", "window",
            "fwlo", "fwhi", "lr"]


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows):6d} rows -> {os.path.relpath(path, ROOT)}")


def derived(r):
    lr = r["lr"]
    peak = r["peak"]
    S = sum(lr)
    half = sum(1 for x in lr if x < 0.5 * peak) / len(lr)
    d = {k: r.get(k) for k in SETTINGS if k != "lr"}
    d.update(tag=r["tag"], family=r["family"], machine=r["machine"], kept=r["kept"][-1], S=S,
             S_peak=S / peak, n_after=len(lr), lr_mean_after=S / len(lr), lr_last=lr[-1],
             share_below_half=half, tokens_after=len(lr) * r["batch"] * r["seq"],
             b0=r.get("b0"), C=r.get("C"), val_loss=r.get("val_loss"), wnorm=r.get("wnorm"),
             rms_w=r.get("rms_w"), rms_w_end=r.get("rms_w_end"), ref0=r.get("ref0"),
             ref_end=r.get("ref_end"), b_end=r.get("b_end"), kept_max=max(r["kept"]),
             bits_by_k=r.get("bits_by_k"))
    k = min(max(d["kept"], 1e-3), 1 - 1e-3)
    d["logit"] = math.log(k / (1 - k))
    return d


def main():
    runs = load(RUNS)
    lf = [r for r in runs if r.get("protocol") == "lf" and r.get("kept") and r.get("lr")]
    frozen = {r["tag"] for r in read_jsonl(os.path.join(ROOT, "problems/forgetting/data/llm.jsonl.gz"))}
    new = [r for r in lf if r["tag"] not in frozen and r.get("opt") == "muon" and r.get("sched") in ("cos", "wsd", "const")]
    write(os.path.join(OUT, "forget_new", "settings.jsonl"),
          [{"tag": r["tag"], "settings": {k: r.get(k) for k in SETTINGS}} for r in new])
    write(os.path.join(OUT, "forget_new", "results.jsonl"), [{"tag": r["tag"], "actual": r["kept"][-1]} for r in new])
    lfd = []
    for r in lf:
        d = derived(r)
        d["in_frozen"] = int(r["tag"] in frozen)
        lfd.append(d)
    write(os.path.join(OUT, "lf.jsonl"), lfd)
    curve = []
    for r in lf:
        if r.get("opt") != "muon" or r.get("sched") not in ("cos", "wsd", "const"):
            continue
        S = r["S"] if isinstance(r.get("S"), list) else None
        if not S:
            acc, S = 0.0, []
            for x in r["lr"]:
                acc += x
                S.append(acc)
        n = min(len(S), len(r["kept"]))
        for i in range(3, n, 4):
            curve.append({"tag": r["tag"], "width": r["width"], "layers": r["layers"], "peak": r["peak"],
                          "wd": r["wd"], "steps": r["steps"], "batch": r["batch"], "seq": r["seq"],
                          "sched": r["sched"], "window": r["window"], "i": i, "frac": (i + 1) / n,
                          "S": S[i], "S_peak": S[i] / r["peak"], "kept": r["kept"][i]})
    write(os.path.join(OUT, "curve.jsonl"), curve)
    by = {}
    for r in load(OTHER):
        by.setdefault(r.get("search", "none"), []).append(r)
    for s, rows in sorted(by.items()):
        write(os.path.join(OUT, f"other_{s}.jsonl"), rows)


if __name__ == "__main__":
    main()
