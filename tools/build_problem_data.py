"""Freeze a problem's data from the full run table (maintainers only).

    python3 tools/build_problem_data.py final-loss

Reads SOURCE, COLUMNS and select() from problems/<problem>/problem.py, downloads the full table
from Hugging Face if needed, and writes problems/<problem>/data/<model_family>.jsonl.gz.
Rebuilding changes the benchmark: rescore every formula afterwards (python3 tools/check.py --rescore).
"""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.rows import load  # noqa: E402
from core.score import load_problem  # noqa: E402


def main(name):
    p = load_problem(name)
    for fam, table in p.SOURCE.items():
        rows = [{k: r.get(k) for k in p.COLUMNS} for r in load(os.path.join(ROOT, "data", fam, table)) if p.select(r)]
        rows.sort(key=lambda r: (str(r.get("tag")), str(r.get("machine"))))
        out = os.path.join(ROOT, "problems", name, "data", fam + ".jsonl.gz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with gzip.GzipFile(out, "wb", mtime=0) as f:
            for r in rows:
                f.write((json.dumps(r, sort_keys=True) + "\n").encode())
        print(f"{out}: {len(rows)} rows, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
