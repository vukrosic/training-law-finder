"""Freeze a problem's data from the full run table (maintainers only).

    python3 tools/build_problem_data.py final-loss

Reads SOURCE, COLUMNS and select() from problems/<problem>/problem.py, downloads the full table
from Hugging Face if needed, and writes problems/<problem>/data/<model_family>.jsonl.gz.
Optional in problem.py: project(r) returns the row to keep (for nested keys; default: the COLUMNS of r),
and MIN_GROUP drops groups with fewer rows than that.
Rebuilding changes the benchmark: python3 tools/check.py then lists every score that no longer
reproduces; score each of those formulas again with --record and rebuild the leaderboard.
Every tag in the frozen data must also be in the Hugging Face table, so upload the table first
(projects/zero-to-icml/search/export_hf.py --upload for the maintainers' runs).
"""
import gzip
import json
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.rows import HF, load  # noqa: E402
from core.score import load_problem  # noqa: E402


def main(name):
    p = load_problem(name)
    for fam, table in p.SOURCE.items():
        project = getattr(p, "project", lambda r: {k: r.get(k) for k in p.COLUMNS})
        rows = [project(r) for r in load(os.path.join(ROOT, "data", fam, table)) if p.select(r)]
        if getattr(p, "MIN_GROUP", 0):
            n = {}
            for r in rows:
                n[p.group(r)] = n.get(p.group(r), 0) + 1
            dropped = sorted(g for g, c in n.items() if c < p.MIN_GROUP)
            rows = [r for r in rows if n[p.group(r)] >= p.MIN_GROUP]
            print(f"{fam}: left out {len(dropped)} groups with fewer than {p.MIN_GROUP} rows: {dropped}")
        rows.sort(key=lambda r: (str(r.get("tag")), str(r.get("machine"))))
        out = os.path.join(ROOT, "problems", name, "data", fam + ".jsonl.gz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with gzip.GzipFile(out, "wb", mtime=0) as f:
            for r in rows:
                f.write((json.dumps(r, sort_keys=True) + "\n").encode())
        print(f"{out}: {len(rows)} rows, {os.path.getsize(out) / 1e6:.2f} MB")
        check_on_hf(table, rows)


def check_on_hf(table, rows):
    """Every frozen run must be in the public table, so anyone can trace it back."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, table)
        urllib.request.urlretrieve(HF + table, path)
        public = {(r.get("tag"), r.get("machine")) for r in load(path)}
    missing = [r["tag"] for r in rows if (r.get("tag"), r.get("machine")) not in public]
    if missing:
        print(f"NOT ON HUGGING FACE: {len(missing)} frozen runs, for example {missing[:5]}. "
              "Upload the run table first, then rebuild.")
    else:
        print("all frozen runs are in the Hugging Face table")


if __name__ == "__main__":
    main(sys.argv[1])
