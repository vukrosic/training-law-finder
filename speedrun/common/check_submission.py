"""Check a speedrun record folder against its track's rules.  python3 check_submission.py <record folder> ...

A record lives in speedrun/tracks/<track>/records/<date>_<name>/. The track folder has track.json, which names
the metric (a key in submission.json), the constraint (another key) and the budget. This checks that:
1. README.md, submission.json, train.py and train.log are there, and each file is under 50 KB.
2. submission.json has name, github, date, device, the track's metric, the track's constraint and any extra
   fields the track lists in track.json under "fields" (for example flops on a track whose constraint is
   environment steps).
3. The constraint value is a number within the budget.
It does not rerun training; a maintainer reruns a new record before it goes on the board.
"""
import json
import os
import sys

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
    return errs


def main():
    errs = [e for d in sys.argv[1:] for e in check(d)]
    for e in errs:
        print("FAIL  " + e)
    print(f"{len(sys.argv) - 1} record(s), {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
