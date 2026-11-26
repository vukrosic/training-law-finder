"""One-screen status of a queue: what is done, running, waiting and failed.

    python3 queue/status.py examples/loss-law/jobs.jsonl examples/loss-law/results.jsonl

This is what the human reads at each 15-minute check, next to the open orders.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_jobs import read_jsonl  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    jobs_path, results_path = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "val_loss"
    jobs = [j for j in read_jsonl(jobs_path) if not j.get("withdrawn")]
    rows = read_jsonl(results_path)
    have = {r.get("tag"): r for r in rows}
    st = {}
    if os.path.exists(results_path + ".state.json"):
        st = json.load(open(results_path + ".state.json"))
    running = st.get("running", {})
    now = time.time()

    by_prio = {}
    for j in jobs:
        p = j.get("prio", 0)
        d = by_prio.setdefault(p, [0, 0, 0])
        d[0] += 1
        if j["tag"] in have:
            d[1] += 1
        elif j["tag"] in running:
            d[2] += 1
    print(f"queue {jobs_path}")
    for p in sorted(by_prio, reverse=True):
        n, done, run = by_prio[p]
        print(f"  prio {p}: {done}/{n} done, {run} running, {n - done - run} waiting")
    secs = [r["seconds"] for r in rows if "seconds" in r]
    waiting = sum(1 for j in jobs if j["tag"] not in have and j["tag"] not in running)
    if secs and st.get("par"):
        eta = (sum(secs) / len(secs)) * waiting / st["par"] / 60
        print(f"  mean run {sum(secs) / len(secs):.0f}s; about {eta:.0f} min of waiting work")
    for tag, t0 in running.items():
        print(f"  running {tag} for {now - t0:.0f}s")
    for f in st.get("failed", []):
        print(f"  FAILED {f['tag']} rc={f['rc']} (see logs/{f['tag']}.log)")
    if st.get("finished"):
        print("  runner finished")
    elif st and not running:
        print("  runner idle or stopped")
    recent = rows[-5:]
    if recent:
        print(f"last {len(recent)} results ({metric}):")
        for r in recent:
            print(f"  {r.get('tag'):<28} {r.get(metric)}  {r.get('seconds', '')}s")


if __name__ == "__main__":
    main()
