"""Checks a paper draft must pass before it is shared. Run it after agents/reviewer.md:

    python3 tools/check_paper.py work/<topic>/PAPER.md
    python3 tools/check_paper.py PAPER.md --scores work/<topic>/score1.txt        # own project: saved fit/laws.py output
    python3 tools/check_paper.py PAPER.md --rescore final-loss problems/final-loss/formulas/my.py

1. abstract   the Abstract section has 180 words or fewer
2. score 1    every "score 1 ... <number>" matches a score in problems/*/scores.jsonl, a number in a
              --scores file, or what core/score.py gives for --rescore
3. score 2    a score-2 number needs a line "predictions: <file>, sha256 <64 hex>, frozen <YYYY-MM-DD...>";
              if the file is found, its sha256 must match
4. closest    a section titled "Closest work", "Related work" or "Prior work"
5. range      one line states the small-scale range: width up to 2048 and at most 32,000 steps
6. privacy    no local paths, e-mail addresses or tokens (same patterns as tools/check.py)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check import PRIVATE  # noqa: E402

NUM = r"(-?\d+\.\d+)"
HEAD = re.compile(r"^\s*(#{1,6}\s|\*\*)")  # a markdown heading or a bold label such as "**In simple words.**"


def section(lines, title):
    """Text from the heading named `title` up to the next heading."""
    start = re.compile(r"^\s*(#{1,6}\s*|\*\*)?\s*" + title + r"\b[.:]?(\*\*)?[.:]?", re.I)
    out, inside = [], False
    for ln in lines:
        if inside and HEAD.match(ln):
            break
        if not inside and start.match(ln):
            inside, ln = True, start.sub("", ln)
        if inside:
            out.append(ln)
    return "\n".join(out) if inside else None


def known_scores(args):
    vals = []
    for f in glob.glob(os.path.join(ROOT, "problems", "*", "scores.jsonl")):
        for line in open(f):
            r = json.loads(line)
            vals += [r["score"], r["worst"]]
    for f in args.scores or []:
        vals += [float(x) for x in re.findall(NUM, open(f).read())]
    if args.rescore:
        from core.score import load_module, load_problem, score
        p = load_problem(args.rescore[0])
        res = score(p, load_module(args.rescore[1], "formula"))
        vals += [res["score"], res["worst"]]
        print(f"rescored  {args.rescore[1]}: score 1 {res['score']:.4f}")
    return vals


def matches(text_num, vals):
    d = len(text_num.split(".")[1])
    return any(f"{v:.{d}f}" == text_num for v in vals)


def main():
    ap = argparse.ArgumentParser(description="Check a paper draft before sharing it (see agents/WORKFLOW.md).")
    ap.add_argument("paper", help="the draft, for example work/<topic>/PAPER.md")
    ap.add_argument("--scores", nargs="*", help="saved score output (core/score.py or fit/laws.py) to match score 1 against")
    ap.add_argument("--rescore", nargs=2, metavar=("PROBLEM", "FORMULA"), help="rerun core/score.py and match its score 1")
    a = ap.parse_args()
    text = open(a.paper).read()
    lines = text.splitlines()
    here = os.path.dirname(os.path.abspath(a.paper))
    fails = []

    abstract = section(lines, "abstract")
    if abstract is None:
        fails.append("abstract: no section titled Abstract")
    elif len(abstract.split()) > 180:
        fails.append(f"abstract: {len(abstract.split())} words (limit 180)")

    vals = known_scores(a)
    s1 = re.findall(r"score 1\b[^.\n]{0,60}?" + NUM, text, re.I)
    if not s1:
        fails.append("score 1: no 'score 1 ... <number>' in the paper")
    for n in s1:
        if not matches(n, vals):
            fails.append(f"score 1: {n} is not in problems/*/scores.jsonl or the given --scores / --rescore output")

    if re.search(r"score 2\b[^.\n]{0,60}?" + NUM, text, re.I):
        m = re.search(r"predictions?:?\s*`?([^\s,`]+)`?,?\s*sha256:?\s*`?([0-9a-f]{64})`?,?\s*frozen:?\s*(\d{4}-\d\d-\d\d[T \d:Z+-]*)", text, re.I)
        if not m:
            fails.append("score 2: a score-2 number needs 'predictions: <file>, sha256 <64 hex>, frozen <date time>'")
        else:
            path = next((p for p in (m.group(1), os.path.join(here, m.group(1)), os.path.join(ROOT, m.group(1)))
                         if os.path.isfile(p)), None)
            if path is None:
                print(f"note      score 2: {m.group(1)} not found here, sha256 not recomputed")
            elif hashlib.sha256(open(path, "rb").read()).hexdigest() != m.group(2):
                fails.append(f"score 2: sha256 of {m.group(1)} does not match the paper; the predictions changed after freezing")

    if not any(re.match(r"^\s*(#{1,6}\s*|\*\*)?\s*(\d+\.?\s*)?(closest|related|prior) (published )?work", ln, re.I) for ln in lines):
        fails.append("closest: no section titled 'Closest work' or 'Related work'")

    if not any("2048" in ln and re.search(r"32,000|32000|32k", ln, re.I) for ln in lines):
        fails.append("range: no line states the small-scale range (width up to 2048, at most 32,000 steps)")

    for rx, what in PRIVATE:
        hit = rx.search(text)
        if hit:
            fails.append(f"privacy: {what}: {hit.group(0)[:40]!r}")

    for f in fails:
        print("FAIL      " + f)
    print(f"{len(fails)} failures in {os.path.basename(a.paper)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
