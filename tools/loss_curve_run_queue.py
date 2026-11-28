"""Run prereg/loss-curve-small.jsonl in a fixed order, at most 2 at once, for one time budget.

    python3 tools/loss_curve_run_queue.py 540     # launch new runs for 540 s, then wait for the running ones
    python3 problems/loss-curve/build.py work/loss-curve/small.jsonl     # then rebuild data/llm.jsonl.gz

Results go to work/loss-curve/small.jsonl (git-ignored), logs to work/loss-curve/logs/. Tags already in that
file are skipped, so a second call runs only what is missing (33 runs on 2026-11-27: P 17, F 7, D 6, E 3,
about 4 minutes with 2 at once). Before each launch: if a Remotion render.mjs is running, cap = 1.
Each run: nice -n 10, --threads 2, CPU. A run over 2 GB or 35 s is killed by PID and logged.
After build.py the row count changes, so every llm score must be redone: sh tools/loss_curve_rescore.sh
mean tissue_anneal multi_power clock_lr clock_s2.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "work", "loss-curve")
RES = os.path.join(HERE, "small.jsonl")
LIMIT_KB = 2_000_000
MAX_S = 35


def order():
    L = [json.loads(l) for l in open(os.path.join(ROOT, "prereg", "loss-curve-small.jsonl"))]
    sched = {"const": 0, "wsd": 1, "cos": 2}

    def key(g):
        d = g["draw"]
        # kill-line triples first: block S at 1x peak, 256 then 512 then 128 steps, every width
        first = g["block"] == "S" and d["peak_x"] == 1.0
        tier = 0 if first else (1 if g["block"] in "STL" else 2)
        return (tier, {256: 0, 512: 1, 128: 2}.get(d["steps"], 3) if first else 0, d["width"], g["block"],
                d["steps"], sched[d["sched"]], g["tag"])
    return sorted(L, key=key)


def done():
    if not os.path.exists(RES):
        return set()
    return {json.loads(l)["tag"] for l in open(RES) if l.strip()}


def rendering():
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return any("render.mjs" in l and "grep" not in l for l in out.splitlines())


def rss_kb(pid):
    """RSS of the nice wrapper's child tree (python)."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    rows = [tuple(int(x) for x in l.split()) for l in out.splitlines() if l.strip()]
    kids, tot, frontier = {pid}, 0, [pid]
    while frontier:
        p = frontier.pop()
        for a, b, r in rows:
            if b == p and a not in kids:
                kids.add(a)
                frontier.append(a)
    return sum(r for a, b, r in rows if a in kids)


def main(budget):
    t0 = time.time()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    have = done()
    queue = [g for g in order() if g["tag"] not in have]
    print(f"{len(have)} done, {len(queue)} to go", flush=True)
    running = {}
    while queue or running:
        for tag, (p, ts) in list(running.items()):
            if p.poll() is not None:
                print(f"end  {tag}  rc={p.returncode}  {time.time() - ts:.0f}s", flush=True)
                del running[tag]
            elif time.time() - ts > MAX_S:
                p.kill()
                print(f"KILL {tag}: over {MAX_S} s", flush=True)
                del running[tag]
            elif rss_kb(p.pid) > LIMIT_KB:
                p.kill()
                print(f"KILL {tag}: over 2 GB", flush=True)
                del running[tag]
        cap = 1 if rendering() else 2
        while queue and len(running) < cap and time.time() - t0 < budget:
            g = queue.pop(0)
            log = open(os.path.join(HERE, "logs", g["tag"] + ".log"), "w")
            p = subprocess.Popen(["nice", "-n", "10", sys.executable, "testbed/train.py", *g["args"], "--threads", "2",
                                  "--out", RES], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running[g["tag"]] = (p, time.time())
            print(f"run  {g['tag']}  (cap {cap})", flush=True)
        if time.time() - t0 >= budget and not running:
            break
        time.sleep(2)
    print(f"{len(done())} done, {len(queue)} left, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]))
