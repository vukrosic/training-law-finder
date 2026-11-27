"""Checks a paper draft must pass before it is shared. Run it after agents/reviewer.md:

    python3 tools/check_paper.py work/<topic>/PAPER.md --scores work/<topic>/score1.txt
    python3 tools/check_paper.py PAPER.md --problem final-loss --scores work/<topic>/score1.txt
    python3 tools/check_paper.py PAPER.md --rescore final-loss problems/final-loss/formulas/my.py

The paper's problem is --problem, else the problem of --rescore, else the one problem the paper names
(as problems/<name>/ or a hyphenated name such as final-loss). No problem means your own project.

1. abstract   the Abstract section has 180 words or fewer (counted up to the next heading or the
              "In simple words" paragraph; a bold label at the start of the abstract counts too)
2. score 1    every "score 1 ... <number>" in the text, and every number in a table column headed
              "score 1", matches a score 1 or worst-group score of the paper's own problem
              (problems/<name>/scores.jsonl), a "score 1" or "worst" line of a --scores file, or
              what core/score.py gives for --rescore. A number right after "by" is a difference
              ("lowers score 1 by 0.0172") and is not checked.
3. score 2    any score-2 claim (a sentence with a number and "score 2", "pre-registered" or
              "predicted in advance / before the runs existed") needs a line
              "predictions: <file>, sha256 <64 hex>, frozen <YYYY-MM-DD...>"; if the file is found,
              its sha256 must match
4. closest    a section titled "Closest work", "Related work" or "Prior work"
5. range      one line holds every number of the problem's "Tested range:" line (problems/<name>/README.md),
              or, for your own project, of the "Tested range:" line in TARGET.md next to the paper
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
HEAD = re.compile(r"^\s*#{1,6}\s")  # a markdown heading
SIMPLE = re.compile(r"^\s*\*\*\s*in simple words", re.I)  # the paragraph after the abstract (skill/writing-conference-paper)
SCORE2 = re.compile(r"score 2\b|pre-?regist|predicted (in advance|before)|before (the |any )?(new )?runs? (existed|were trained)"
                    r"|frozen predictions?|predictions? (were )?(frozen|written|made) before", re.I)


def section(lines, title):
    """Text from the heading named `title` up to the next heading (or the "In simple words" paragraph)."""
    start = re.compile(r"^\s*(#{1,6}\s*|\*\*)?\s*" + title + r"\b[.:]?(\*\*)?[.:]?", re.I)
    out, inside = [], False
    for ln in lines:
        if inside and (HEAD.match(ln) or SIMPLE.match(ln)):
            break
        if not inside and start.match(ln):
            inside, ln = True, start.sub("", ln)
        if inside:
            out.append(ln)
    return "\n".join(out) if inside else None


def problem_names():
    d = os.path.join(ROOT, "problems")
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "problem.py")))


def paper_problem(args, text):
    """The problem the paper is about, or None for an own project. Raises ValueError if unclear."""
    if args.problem:
        if args.problem not in problem_names():
            raise ValueError(f"--problem {args.problem} is not a folder in problems/")
        return args.problem
    if args.rescore:
        return args.rescore[0]
    named = {n for n in problem_names()
             if re.search(r"problems/" + re.escape(n) + r"\b", text) or ("-" in n and re.search(r"\b" + re.escape(n) + r"\b", text))}
    if len(named) > 1:
        raise ValueError(f"the paper names several problems ({', '.join(sorted(named))}); pass --problem")
    return named.pop() if named else None


def scores_of(name):
    vals = []
    for line in open(os.path.join(ROOT, "problems", name, "scores.jsonl")):
        r = json.loads(line)
        vals += [r["score"], r["worst"]]
    return vals


def known_scores(args, problem):
    vals = scores_of(problem) if problem else []
    for f in args.scores or []:
        for ln in open(f):
            if re.match(r"^\s*(score 1|worst)\b", ln, re.I):  # core/score.py and fit/laws.py print these lines
                vals += [float(x) for x in re.findall(NUM, ln)]
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


def score1_in_text(text):
    """Numbers claimed as score 1 in sentences: the first number after "score 1", unless it follows "by"."""
    out = []
    for m in re.finditer(r"score 1\b([^.\n]|\.\d){0,60}", text, re.I):
        for n in re.finditer(NUM, m.group(0)):
            if re.search(r"\bby\s*(about\s*)?$", m.group(0)[:n.start()], re.I):
                continue
            out.append(n.group(1))
            break
    return out


def score1_in_tables(lines):
    """Numbers in markdown table columns whose header says "score 1"."""
    out = []
    for i in range(len(lines) - 1):
        if not (lines[i].strip().startswith("|") and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1])):
            continue
        cols = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        want = [j for j, c in enumerate(cols) if re.search(r"score 1\b", c, re.I)]
        for ln in lines[i + 2:]:
            if not ln.strip().startswith("|"):
                break
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            for j in want:
                n = re.search(NUM, cells[j]) if j < len(cells) else None
                if n:
                    out.append(n.group(1))
    return out


def tested_range(problem, here):
    """Numbers of the "Tested range:" line, from the problem README or from TARGET.md next to the paper."""
    src = os.path.join(ROOT, "problems", problem, "README.md") if problem else os.path.join(here, "TARGET.md")
    if not os.path.isfile(src):
        return src, None
    for ln in open(src):
        m = re.search(r"tested range:(.*)", ln, re.I)
        if m:
            return src, [x.replace(",", "") for x in re.findall(r"\d[\d,]*\d|\d", m.group(1))]
    return src, None


def main():
    ap = argparse.ArgumentParser(description="Check a paper draft before sharing it (see agents/WORKFLOW.md).")
    ap.add_argument("paper", help="the draft, for example work/<topic>/PAPER.md")
    ap.add_argument("--problem", help="the open problem the paper is about (default: the one it names)")
    ap.add_argument("--scores", nargs="*", help="saved score output (core/score.py or fit/laws.py) to match score 1 against")
    ap.add_argument("--rescore", nargs=2, metavar=("PROBLEM", "FORMULA"), help="rerun core/score.py and match its score 1")
    a = ap.parse_args()
    text = open(a.paper).read()
    lines = text.splitlines()
    here = os.path.dirname(os.path.abspath(a.paper))
    fails = []

    try:
        problem = paper_problem(a, text)
    except ValueError as e:
        print(f"FAIL      problem: {e}")
        sys.exit(1)
    print(f"problem   {problem or 'none named: own project'}")

    abstract = section(lines, "abstract")
    if abstract is None:
        fails.append("abstract: no section titled Abstract")
    elif len(abstract.split()) > 180:
        fails.append(f"abstract: {len(abstract.split())} words (limit 180)")

    vals = known_scores(a, problem)
    s1 = score1_in_text(text)
    if not s1:
        fails.append("score 1: no 'score 1 ... <number>' in the paper")
    where = f"problems/{problem}/scores.jsonl" if problem else "the --scores / --rescore output"
    for n in s1 + score1_in_tables(lines):
        if matches(n, vals):
            continue
        other = [o for o in problem_names() if o != problem and matches(n, scores_of(o))]
        hint = f" (it is a score of {other[0]}, not of this paper's problem)" if other else ""
        fails.append(f"score 1: {n} is not in {where} or the given --scores / --rescore output{hint}")

    claims = [sn for sn in re.split(r"(?<=[.!?])\s+|\n", text)
              if SCORE2.search(sn) and re.search(NUM, sn) and not re.search(r"sha256", sn, re.I)]
    if claims:
        m = re.search(r"predictions?:?\s*`?([^\s,`]+)`?,?\s*sha256:?\s*`?([0-9a-f]{64})`?,?\s*frozen:?\s*(\d{4}-\d\d-\d\d[T \d:Z+-]*)", text, re.I)
        if not m:
            fails.append(f"score 2: a score-2 claim ({claims[0].strip()[:60]!r}) needs 'predictions: <file>, sha256 <64 hex>, frozen <date time>'")
        else:
            path = next((p for p in (m.group(1), os.path.join(here, m.group(1)), os.path.join(ROOT, m.group(1)))
                         if os.path.isfile(p)), None)
            if path is None:
                print(f"note      score 2: {m.group(1)} not found here, sha256 not recomputed")
            elif hashlib.sha256(open(path, "rb").read()).hexdigest() != m.group(2):
                fails.append(f"score 2: sha256 of {m.group(1)} does not match the paper; the predictions changed after freezing")

    if not any(re.match(r"^\s*(#{1,6}\s*|\*\*)?\s*(\d+\.?\s*)?(closest|related|prior) (published )?work", ln, re.I) for ln in lines):
        fails.append("closest: no section titled 'Closest work' or 'Related work'")

    src, rng = tested_range(problem, here)
    if rng is None:
        fails.append(f"range: no 'Tested range:' line in {os.path.relpath(src, ROOT) if problem else 'TARGET.md next to the paper'}")
    elif not any(all(re.search(r"(?<![\d.])" + x + r"(?!\d)", ln.replace(",", "")) for x in rng) for ln in lines):
        fails.append(f"range: no line states the tested range ({', '.join(rng)} from the 'Tested range:' line of "
                     f"{os.path.relpath(src, ROOT) if problem else 'TARGET.md'})")

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
