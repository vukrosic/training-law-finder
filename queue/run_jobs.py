"""Run a file of jobs, a few at a time, and append each result row to one results file.

    python3 queue/run_jobs.py examples/loss-law/jobs.jsonl --results examples/loss-law/results.jsonl --par 2 --threads 2

A job is one JSON line:
    {"tag": "d64_s200", "prio": 1, "args": {"d": 64, "steps": 200}}
`args` become testbed/train.py flags. Instead of `args` a job may give `cmd`, any
shell command (for example a simulator), which must append its own row with the
same tag to the results file.

Rules the runner enforces:
  - a tag that already has a row in the results file is skipped, so a stopped
    queue can simply be started again;
  - higher prio runs first (2 = deciding checks, 1 = baselines, 0 = screens);
  - a job with "withdrawn": true is skipped; never delete rows from results.
Logs go to <results dir>/logs/<tag>.log. Live state goes to <results>.state.json
for queue/status.py.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "testbed", "train.py")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def done_tags(results):
    return {r.get("tag") for r in read_jsonl(results)}


def job_command(job, a):
    if "cmd" in job:
        return job["cmd"]
    args = dict(a.defaults)
    args.update(job.get("args", {}))
    parts = [sys.executable, TRAIN, "--tag", job["tag"], "--out", a.results]
    if a.threads:
        parts += ["--threads", str(a.threads)]
    for k, v in args.items():
        parts += [f"--{k}", str(v)]
    cmd = " ".join(shlex.quote(p) for p in parts)
    return f"nice -n {a.nice} {cmd}" if a.nice else cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--results", required=True)
    ap.add_argument("--par", type=int, default=2, help="jobs at once")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per job")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--defaults", default="{}", help="JSON of flags shared by every job")
    ap.add_argument("--dry", action="store_true", help="print commands only")
    a = ap.parse_args()
    a.defaults = json.loads(a.defaults)

    jobs = [j for j in read_jsonl(a.jobs) if not j.get("withdrawn")]
    tags = [j["tag"] for j in jobs]
    dup = {t for t in tags if tags.count(t) > 1}
    if dup:
        sys.exit(f"duplicate tags in {a.jobs}: {sorted(dup)}")
    have = done_tags(a.results)
    todo = sorted((j for j in jobs if j["tag"] not in have), key=lambda j: -j.get("prio", 0))
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run, par {a.par}")
    if a.dry:
        for j in todo:
            print(job_command(j, a))
        return

    logdir = os.path.join(os.path.dirname(os.path.abspath(a.results)), "logs")
    os.makedirs(logdir, exist_ok=True)
    state_path = a.results + ".state.json"
    lock = threading.Lock()
    state = dict(started=time.time(), jobs_file=a.jobs, par=a.par, running={}, failed=[])

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    def run(job):
        tag = job["tag"]
        with lock:
            if tag in done_tags(a.results):
                return
            state["running"][tag] = time.time()
            save_state()
        t0 = time.time()
        with open(os.path.join(logdir, f"{tag}.log"), "w") as log:
            rc = subprocess.call(job_command(job, a), shell=True, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            state["running"].pop(tag, None)
            ok = rc == 0 and tag in done_tags(a.results)
            if not ok:
                state["failed"].append(dict(tag=tag, rc=rc))
            save_state()
            print(f"{'ok  ' if ok else 'FAIL'} {tag} {time.time() - t0:.0f}s", flush=True)

    save_state()
    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(run, todo))
    state["finished"] = time.time()
    save_state()
    print(f"finished; failed: {[f['tag'] for f in state['failed']]}")


if __name__ == "__main__":
    main()
