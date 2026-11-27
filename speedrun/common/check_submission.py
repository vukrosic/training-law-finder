"""Check a speedrun record folder against its track's rules.

    python3 check_submission.py <record> ...                 # files and fair-play rules, no torch, a second
    python3 check_submission.py --count <record>             # + full count of 3 real steps on cpu vs the fast count
    python3 check_submission.py --artifact <file> <record>   # + 16 MB limit and the score of the model in the file

A record lives in speedrun/tracks/<track>/records/<date>_<name>/. The track folder has track.json with the metric,
the constraint, the budget and the fair-play rules. The plain check (run by tools/check.py on every pull request):
1. README.md, submission.json, train.py and train.log are there, and each file is under 50 KB.
2. submission.json has name, github, date, device, the track's metric and the track's constraint.
3. The constraint value is within the budget, and the record was counted with the track's counter.
4. The artifact size is within the track's limit.
5. Fair play: train.py has none of the forbidden patterns (torch.compile, custom kernels, checks for the meta
   device or the counter, outside weights, network) and calls what it must (the fast count, eval.py's scoring,
   the artifact loader).
6. The last `result` line of train.log matches submission.json, was scored on the whole validation set, and
   comes from a full count (--record) that agrees with the fast count within the track's tolerance.
--count and --artifact run the model; a maintainer also reruns a new record before it goes on the board.
"""
import importlib.util
import json
import os
import re
import sys

FILES = ("README.md", "submission.json", "train.py", "train.log")
FIELDS = ("name", "github", "date", "device")
MAX_BYTES = 50_000


def load_rules(record):
    track = os.path.dirname(os.path.dirname(os.path.abspath(record)))
    path = os.path.join(track, "track.json")
    return (json.load(open(path)) if os.path.exists(path) else None), track


def last_result(log_path):
    result = None
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if line.startswith("result "):
            try:
                result = json.loads(line[len("result "):])
            except json.JSONDecodeError:
                pass
    return result


def check(record):
    """Return a list of problems with one record folder (empty if it passes). No torch needed."""
    record = os.path.abspath(record)
    rules, track = load_rules(record)
    where = os.path.relpath(record, os.path.dirname(os.path.dirname(track)))
    if rules is None:
        return [f"{where}: no track.json in the track folder"]
    errs = []
    for f in FILES:
        if not os.path.exists(os.path.join(record, f)):
            errs.append(f"{where}: {f} is missing")
    for f in sorted(os.listdir(record)):
        n = os.path.getsize(os.path.join(record, f))
        if n > MAX_BYTES:
            errs.append(f"{where}/{f} is {n / 1000:.0f} KB (limit 50 KB; trim the log)")
    sub_path = os.path.join(record, "submission.json")
    sub = {}
    if os.path.exists(sub_path):
        try:
            sub = json.load(open(sub_path))
        except json.JSONDecodeError as e:
            return errs + [f"{where}: submission.json is not valid JSON ({e})"]
        for k in FIELDS + (rules["metric"], rules["constraint"]):
            if k not in sub:
                errs.append(f"{where}: submission.json has no '{k}'")
        used = sub.get(rules["constraint"])
        if isinstance(used, (int, float)) and used > rules["budget"]:
            errs.append(f"{where}: {rules['constraint']} {used:g} is over the budget {rules['budget']:g}")
        if "counter" in rules and sub.get("counter") != rules["counter"]:
            errs.append(f"{where}: submission.json counter is {sub.get('counter')!r}, the track counts with "
                        f"{rules['counter']!r}; rerun with the current common/flops.py")
        if "size_field" in rules:
            size = sub.get(rules["size_field"])
            if not isinstance(size, (int, float)):
                errs.append(f"{where}: submission.json has no '{rules['size_field']}' (run train.py --record)")
            elif size > rules["size_limit"]:
                errs.append(f"{where}: {rules['size_field']} {size:,} is over the limit {rules['size_limit']:,}")

    fair = rules.get("fair_play", {})
    train_py = os.path.join(record, "train.py")
    if os.path.exists(train_py):
        code = open(train_py, encoding="utf-8").read()
        for pattern, why in fair.get("forbidden_in_train_py", {}).items():
            if re.search(pattern, code):
                errs.append(f"{where}/train.py matches '{pattern}': {why}")
        for pattern, why in fair.get("required_in_train_py", {}).items():
            if not re.search(pattern, code):
                errs.append(f"{where}/train.py does not call '{pattern}': {why}")

    log = os.path.join(record, "train.log")
    if fair and os.path.exists(log):
        res = last_result(log)
        if res is None:
            errs.append(f"{where}/train.log has no 'result {{...}}' line")
        else:
            for k in fair.get("log_must_match", []):
                if k in sub and res.get(k) != sub[k]:
                    errs.append(f"{where}: train.log says {k} {res.get(k)}, submission.json says {sub[k]}")
            if res.get("device") == "meta":
                errs.append(f"{where}: the run was on the meta device; train on a real device")
            if "val_docs" in fair and res.get("val_docs", fair["val_docs"]) != fair["val_docs"]:
                errs.append(f"{where}: scored on {res.get('val_docs')} validation documents, "
                            f"not all {fair['val_docs']}")
            if "full_count_tolerance" in fair:
                full = res.get("full_count")
                if full is None:
                    errs.append(f"{where}: train.log has no full count; run train.py --record")
                elif isinstance(res.get("flops"), (int, float)):
                    gap = full / res["flops"] - 1
                    if abs(gap) > fair["full_count_tolerance"] and res["flops"] != full:
                        errs.append(f"{where}: full count {full:.4g} differs from the scored {res['flops']:.4g} "
                                    f"by {gap:+.1%}; past {fair['full_count_tolerance']:.0%} the full count is the score")
                    if max(full, res["flops"]) > rules["budget"] * (1 + fair["full_count_tolerance"]):
                        errs.append(f"{where}: counted {max(full, res['flops']):.4g}, over the budget")
    return errs


def import_train(record):
    spec = importlib.util.spec_from_file_location("record_train", os.path.join(record, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_check(record, steps=3):
    """Full count of `steps` real training steps on cpu (random tokens) against the fast count on meta."""
    import torch
    rules, _ = load_rules(record)
    tol = rules.get("fair_play", {}).get("full_count_tolerance", 0.01)
    mod = import_train(record)
    flops = mod.flops
    fast, rep = flops.fast_count(mod.build, mod.train_step, (mod.BATCH, mod.SEQ_LEN))
    torch.manual_seed(0)
    model, opts = mod.build()
    meter = flops.Meter()
    x = torch.randint(0, mod.VOCAB, (mod.BATCH, mod.SEQ_LEN + 1))
    for _ in range(steps):
        with meter.step():
            mod.train_step(model, opts, x[:, :-1], x[:, 1:], torch.float32)
    real = meter.steps[-1]
    gap = real / fast - 1
    print(f"fast count {fast:.6g}, full count of step {steps} on cpu {real:.6g} ({gap:+.2%}); "
          f"non-standard ops {sorted(rep['nonstandard']) or 'none'}")
    return [] if abs(gap) <= tol else [f"full count differs from the fast count by {gap:+.2%} (tolerance "
                                       f"{tol:.0%}): the full count is the budget"]


def artifact_check(record, path):
    """Size of the artifact and the score of the model loaded from it."""
    import numpy as np
    import torch
    rules, track = load_rules(record)
    errs = []
    size = os.path.getsize(path)
    if size > rules.get("size_limit", float("inf")):
        errs.append(f"artifact {size:,} bytes, over the limit {rules['size_limit']:,}")
    spec = importlib.util.spec_from_file_location("track_eval", os.path.join(track, "eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    mod = import_train(record)
    model = mod.build()[0]
    model.load_state_dict(ev.load_artifact(path)["state"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).float().eval()
    docs = ev.val_docs()
    val = np.fromfile(os.path.join(ev.DATA, "val.bin"), dtype=np.uint16)
    got = ev.bpb(ev.total_nats(model, val, mod.SEQ_LEN, device), ev.val_bytes(docs))
    want = json.load(open(os.path.join(record, "submission.json")))[rules["metric"]]
    print(f"artifact {size:,} bytes, {rules['metric']} from the file {got:.4f}, submitted {want}")
    if abs(got - want) > 0.002:
        errs.append(f"score from the artifact {got:.4f} differs from the submitted {want}")
    return errs


def main():
    args = sys.argv[1:]
    count = "--count" in args
    artifact = None
    if "--artifact" in args:
        i = args.index("--artifact")
        artifact = args[i + 1]
        del args[i:i + 2]
    records = [a for a in args if a != "--count"]
    errs = []
    for d in records:
        errs += check(d)
        if count:
            errs += count_check(d)
        if artifact:
            errs += artifact_check(d, artifact)
    for e in errs:
        print("FAIL  " + e)
    print(f"{len(records)} record(s), {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
