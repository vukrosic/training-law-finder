"""Run the record search in search/queue.txt, one run at a time, and print a table per round.

    python3 search/run_queue.py              # run every queued line whose tag is not in out/search/results.jsonl
    python3 search/run_queue.py --dry        # print the commands only
    python3 search/run_queue.py --only r1    # only tags starting with r1
    python3 search/run_queue.py --table      # print the tables from results.jsonl, run nothing

Each queue line is `TAG  flags...`; # starts a comment. A finished tag is skipped, so a stopped queue resumes
where it stopped. Each run's output goes to out/search/logs/<tag>.log. Runs use nice 10 and one Python process. This script does not import torch.
Guard: while any blaw.py job runs (Vuk's ICML training), nothing starts, and a running search is killed and
queued again; the queue checks every 3 minutes and goes on when the Mac is free.
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
LOGS = os.path.join(os.path.dirname(RESULTS), "logs")


def busy():
    """True while another heavy job (blaw.py) is running."""
    # a python process whose script is blaw.py (not a shell line that only mentions it)
    return subprocess.run(["pgrep", "-f", r"^[^ ]*[Pp]ython[0-9.]*( -[A-Za-z]+)* [^ ]*blaw\.py"],
                          capture_output=True).returncode == 0


def wait_free():
    while busy():
        print("  blaw.py is running: waiting 3 min", flush=True)
        time.sleep(180)


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
    os.makedirs(LOGS, exist_ok=True)
    done = results()
    todo = [(t, f) for t, f in queue() if t.startswith(a.only) and t not in done]
    print(f"{len(todo)} runs to do, {len(done)} already in {os.path.relpath(RESULTS, TRACK)}")
    for i, (tag, flags) in enumerate(todo):
        cmd = ["nice", "-n", "10", sys.executable, os.path.join(HERE, "search.py"), *flags, "--tag", tag]
        print(f"[{i + 1}/{len(todo)}] {tag}: {' '.join(flags)}", flush=True)
        if a.dry:
            continue
        while True:
            wait_free()
            t = time.time()
            log_path = os.path.join(LOGS, tag + ".log")
            log = open(log_path, "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            while p.poll() is None and not busy():
                time.sleep(1)
            if p.poll() is None:  # blaw.py started: stop this run and run it again later
                p.kill()
                p.wait()
                log.close()
                print("  killed: blaw.py started", flush=True)
                continue
            log.close()
            break
        out = open(log_path).read()
        r = results().get(tag)
        if p.returncode or r is None:
            print("  FAILED\n" + out[-1500:])
            continue
        print(f"  val_bpb {r['val_bpb']:.4f}, {r['steps']} steps, {time.time() - t:.0f} s", flush=True)
    if not a.dry:
        tables(a.only)


if __name__ == "__main__":
    main()
