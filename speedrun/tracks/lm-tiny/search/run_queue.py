"""Run the record search in search/queue.txt, one run at a time, and print a table per round.

    python3 search/run_queue.py              # run every queued line whose tag is not in out/search/results.jsonl
    python3 search/run_queue.py --dry        # print the commands only
    python3 search/run_queue.py --only r1    # only tags starting with r1
    python3 search/run_queue.py --table      # print the tables from results.jsonl, run nothing

Each queue line is `TAG  flags...`; # starts a comment. A finished tag is skipped, so a stopped queue resumes
where it stopped. Runs use nice 10 and one Python process. This script does not import torch.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
RESULTS = os.path.join(TRACK, "out", "search", "results.jsonl")


def queue():
    for line in open(os.path.join(HERE, "queue.txt")):
        line = line.split("#")[0].strip()
        if line:
            tag, *flags = shlex.split(line)
            yield tag, flags


def results():
    out = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line)
            out[r["tag"]] = r
    return out


def tables(only=""):
    res = results()
    rounds = {}
    for tag, flags in queue():
        if tag in res and tag.startswith(only):
            rounds.setdefault(tag.split("-")[0], []).append(res[tag])
    for rnd, rows in rounds.items():
        print(f"\n{rnd}: {len(rows)} runs, best first")
        print(f"| tag | val_bpb | steps | flags |\n|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["val_bpb"]):
            print(f"| {r['tag']} | {r['val_bpb']:.4f} | {r['steps']} | {r['flags']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return tables(a.only)
    done = results()
    todo = [(t, f) for t, f in queue() if t.startswith(a.only) and t not in done]
    print(f"{len(todo)} runs to do, {len(done)} already in {os.path.relpath(RESULTS, TRACK)}")
    for i, (tag, flags) in enumerate(todo):
        cmd = ["nice", "-n", "10", sys.executable, os.path.join(HERE, "search.py"), *flags, "--tag", tag]
        print(f"[{i + 1}/{len(todo)}] {tag}: {' '.join(flags)}", flush=True)
        if a.dry:
            continue
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        r = results().get(tag)
        if p.returncode or r is None:
            print("  FAILED\n" + (p.stdout + p.stderr)[-1500:])
            continue
        print(f"  val_bpb {r['val_bpb']:.4f}, {r['steps']} steps, {time.time() - t:.0f} s", flush=True)
    if not a.dry:
        tables(a.only)


if __name__ == "__main__":
    main()
