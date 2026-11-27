"""Check a speedrun record folder against its track's rules.  python3 check_submission.py <record folder> ...

A record lives in speedrun/tracks/<track>/records/<date>_<name>/. The track folder has track.json, which names
the metric (a key in submission.json), the constraint (another key) and the budget. This checks that:
1. README.md, submission.json, train.py and train.log are there, and each file is under 50 KB.
2. submission.json has name, github, date, device, the track's metric, the track's constraint and any extra
   fields the track lists in track.json under "fields" (for example flops on a track whose constraint is
   environment steps).
3. The constraint value is a number within the budget, and "version" matches the track's version.
4. On a track with "eval_items" in track.json: per_item.json holds that many per-item scores whose mean is the
   metric, and the record beats the record before it by more than 2 standard errors of the paired difference
   (common/stats.py); a smaller gain is "tie, not a record".
It does not rerun training; a maintainer reruns a new record before it goes on the board.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats  # noqa: E402

FILES = ("README.md", "submission.json", "train.py", "train.log")
FIELDS = ("name", "github", "date", "device")
MAX_BYTES = 50_000


def check(record):
    """Return a list of problems with one record folder (empty if it passes)."""
    record = os.path.abspath(record)
    track = os.path.dirname(os.path.dirname(record))
    where = os.path.relpath(record, os.path.dirname(os.path.dirname(track)))
    rules_path = os.path.join(track, "track.json")
    if not os.path.exists(rules_path):
        return [f"{where}: no track.json in the track folder"]
    rules = json.load(open(rules_path))
    errs = []
    for f in FILES:
        if not os.path.exists(os.path.join(record, f)):
            errs.append(f"{where}: {f} is missing")
    for f in sorted(os.listdir(record)):
        n = os.path.getsize(os.path.join(record, f))
        if n > MAX_BYTES:
            errs.append(f"{where}/{f} is {n / 1000:.0f} KB (limit 50 KB; trim the log)")
    sub_path = os.path.join(record, "submission.json")
    if os.path.exists(sub_path):
        try:
            sub = json.load(open(sub_path))
        except json.JSONDecodeError as e:
            return errs + [f"{where}: submission.json is not valid JSON ({e})"]
        for k in FIELDS + (rules["metric"], rules["constraint"]) + tuple(rules.get("fields", ())):
            if k not in sub:
                errs.append(f"{where}: submission.json has no '{k}'")
        used = sub.get(rules["constraint"])
        if used is not None and not isinstance(used, (int, float)):
            errs.append(f"{where}: {rules['constraint']} is not a number")
        elif isinstance(used, (int, float)) and used > rules["budget"]:
            errs.append(f"{where}: {rules['constraint']} {used:g} is over the budget {rules['budget']:g}")
        if sub.get("version") != rules.get("version"):
            errs.append(f"{where}: scored under track version {sub.get('version')}, the track is {rules.get('version')}")
        if "eval_items" in rules:
            errs += record_rule(record, where, rules, sub)
    return errs


def per_item(record):
    p = os.path.join(record, "per_item.json")
    return json.load(open(p)) if os.path.exists(p) else None


def record_rule(record, where, rules, sub):
    """Per-item scores must match the metric, and the record must beat the current record by > 2 paired SE."""
    v, d, metric = per_item(record), rules["decimals"], rules["metric"]
    if v is None or len(v) != rules["eval_items"]:
        return [f"{where}: per_item.json needs {rules['eval_items']} per-item scores"]
    if not isinstance(sub.get(metric), (int, float)) or abs(sum(v) / len(v) - sub[metric]) > 10 ** -d:
        return [f"{where}: the mean of per_item.json is not {metric} {sub.get(metric)}"]
    best = None  # walk the earlier records in date order; each accepted one becomes the current record
    for r in sorted(os.listdir(os.path.dirname(record))):
        path = os.path.join(os.path.dirname(record), r)
        if path == record:
            break
        w = per_item(path)
        if w is not None and len(w) == len(v):
            if best is None or stats.record_test(w, best, rules["lower_is_better"])[2]:
                best = w
    if best is None:
        return []
    gain, needs, ok = stats.record_test(v, best, rules["lower_is_better"])
    return [] if ok else [f"{where}: tie, not a record: gain {gain:.{d}f}, needs more than {needs:.{d}f}"]


def main():
    errs = [e for d in sys.argv[1:] for e in check(d)]
    for e in errs:
        print("FAIL  " + e)
    print(f"{len(sys.argv) - 1} record(s), {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
