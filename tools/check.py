"""Checks every pull request must pass. GitHub runs this on each PR; run it yourself first:

    python3 tools/check.py            # all checks
    python3 tools/check.py --fast     # skip rescoring formulas

1. size      no tracked file over 5 MB, no problem data file over 3 MB, all tracked files under 100 MB
2. privacy   no local paths, e-mail addresses or access tokens in tracked files or problem data
3. problems  each problems/<name>/ has README.md, problem.py, data/<family>.jsonl.gz, formulas/mean.py,
             scores.jsonl and LEADERBOARD.md
4. columns   every column in problem data is listed in core/columns.yaml
5. formulas  a formula imports only math, numpy, scipy, statistics, itertools, functools, collections;
             calls no open/exec/eval and no numpy/scipy file reader (load*, save*, genfromtxt, ...);
             reads no id or target column (a measured column is a warning). core/score.py also hides
             those columns from the rows a formula gets.
6. scores    every line in scores.jsonl reproduces to 4 decimals on the current data and file
7. tables    LEADERBOARD.md equals what core/leaderboard.py builds
8. papers    no .pdf anywhere; in papers/ and drafts/, images under 500 KB and each paper folder under 5 MB
"""
import ast
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import leaderboard  # noqa: E402
from core.rows import families  # noqa: E402
from core.score import load_module, load_problem, score  # noqa: E402

MB = 1e6
IMPORTS = {"math", "numpy", "scipy", "statistics", "itertools", "functools", "collections"}
BANNED = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
          "delattr", "input", "breakpoint", "memoryview"}
# file readers and writers in numpy / scipy: a formula must not read the data files or write anything
BANNED_ATTR = {"genfromtxt", "fromfile", "tofile", "memmap", "DataSource", "fromregex", "lib", "io"}
BANNED_PREFIX = ("load", "save")
PRIVATE = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "local macOS path"),
    (re.compile(r"/home/[A-Za-z0-9._-]+/"), "local Linux path"),
    (re.compile(r"(?<![\w.])/root/(?!\.)"), "server path"),
    (re.compile(r"/workspace/[A-Za-z]"), "server path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com|anthropic\.com|example\.)[A-Za-z0-9-]+\.[A-Za-z]{2,}"), "e-mail address"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"(ghp|gho|ghs)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"), "GitHub token"),
    (re.compile(r"sk-[A-Za-z0-9_-]{30,}"), "API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS key"),
]
TEXT = (".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".log", ".csv")

errors, warnings = [], []


def tracked():
    # tracked files plus new files not yet added (but not gitignored), so the check runs before git add
    out = subprocess.run(["git", "ls-files", "-z", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout
    return [f for f in out.split("\0") if f and os.path.isfile(os.path.join(ROOT, f))]


def check_size(files):
    total = 0
    for f in files:
        n = os.path.getsize(os.path.join(ROOT, f))
        total += n
        if n > 5 * MB:
            errors.append(f"size: {f} is {n / MB:.1f} MB (limit 5 MB; large data goes to Hugging Face)")
        elif f.startswith("problems/") and "/data/" in f and n > 3 * MB:
            errors.append(f"size: {f} is {n / MB:.1f} MB (problem data limit 3 MB)")
    if total > 100 * MB:
        errors.append(f"size: tracked files total {total / MB:.0f} MB (limit 100 MB)")


def check_papers(files):
    folders = {}
    for f in files:
        n = os.path.getsize(os.path.join(ROOT, f))
        if f.lower().endswith(".pdf"):
            errors.append(f"papers: {f} is a PDF; attach PDFs to a GitHub release instead (papers/README.md)")
        parts = f.split("/")
        if parts[0] in ("papers", "drafts") and len(parts) > 2:
            folders[parts[0] + "/" + parts[1]] = folders.get(parts[0] + "/" + parts[1], 0) + n
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) and n > 0.5 * MB:
                errors.append(f"papers: {f} is {n / MB:.1f} MB (image limit 0.5 MB)")
    for d, n in sorted(folders.items()):
        if n > 5 * MB:
            errors.append(f"papers: {d}/ is {n / MB:.1f} MB (paper folder limit 5 MB)")


def scan(name, text):
    for rx, what in PRIVATE:
        m = rx.search(text)
        if m:
            errors.append(f"privacy: {what} in {name}: {m.group(0)[:40]!r}")


def check_privacy(files):
    for f in files:
        p = os.path.join(ROOT, f)
        if f == "tools/check.py":  # holds the patterns themselves
            continue
        if f.endswith(TEXT) or os.path.basename(f) in ("LICENSE", ".gitignore"):
            scan(f, open(p, errors="replace").read())
        elif f.endswith(".jsonl.gz"):
            scan(f, gzip.open(p, "rt", errors="replace").read())


def columns():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "core", "columns.yaml")))


def problem_names():
    d = os.path.join(ROOT, "problems")
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)) and not n.startswith((".", "_")))


def check_problem(name, cols):
    d = os.path.join(ROOT, "problems", name)
    for need in ("README.md", "problem.py", "formulas/mean.py", "scores.jsonl", "LEADERBOARD.md"):
        if not os.path.exists(os.path.join(d, need)):
            errors.append(f"problems: problems/{name}/{need} is missing")
    if not families(name):
        errors.append(f"problems: problems/{name}/data/ has no <family>.jsonl.gz file")
        return
    try:
        p = load_problem(name)
    except Exception as e:  # noqa: BLE001
        errors.append(f"problems: problems/{name}/problem.py does not load: {e}")
        return
    for attr in ("rows", "target", "group", "GROUP_BY", "FLOOR", "TARGET"):
        if not hasattr(p, attr):
            errors.append(f"problems: problems/{name}/problem.py has no {attr}")
    for fam in families(name):
        seen = set()
        for r in p.rows(fam):
            seen.update(r)
        for c in sorted(seen - set(cols)):
            errors.append(f"columns: '{c}' in problems/{name}/data/{fam}.jsonl.gz is not in core/columns.yaml")


def check_formula(path, cols, target_cols):
    rel = os.path.relpath(path, ROOT)
    tree = ast.parse(open(path).read(), rel)
    ok = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in IMPORTS or any(p in BANNED_ATTR for p in m.split(".")[1:]):
                errors.append(f"formulas: {rel} imports {m} (allowed: {', '.join(sorted(IMPORTS))}, no file modules)")
                ok = False
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in BANNED_ATTR or a.name.startswith(BANNED_PREFIX):
                    errors.append(f"formulas: {rel} imports {a.name} (file readers and writers are not allowed)")
                    ok = False
        if isinstance(node, ast.Name) and node.id in BANNED:
            errors.append(f"formulas: {rel} uses {node.id}()")
            ok = False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"formulas: {rel} reads {node.attr}")
            ok = False
        if isinstance(node, ast.Attribute) and (node.attr in BANNED_ATTR or node.attr.startswith(BANNED_PREFIX)):
            errors.append(f"formulas: {rel} uses .{node.attr} (file readers and writers are not allowed)")
            ok = False
        key = None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            key = node.args[0].value
        if key and key in cols:
            kind = cols[key]["kind"]
            if kind == "id" or key in target_cols:
                errors.append(f"formulas: {rel} reads '{key}' ({'the target' if key in target_cols else 'a run id'})")
                ok = False
            elif kind in ("measured", "target"):
                warnings.append(f"formulas: {rel} reads measured column '{key}': listed separately (rule 4)")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} | \
            {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    for need in ("CONSTANTS", "fit", "predict"):
        if need not in names:
            errors.append(f"formulas: {rel} does not define {need}")
            ok = False
    return ok


def target_columns(name):
    try:
        return {load_problem(name).TARGET}
    except Exception:  # noqa: BLE001  (check_problem reports a broken problem.py)
        return set()


def check_scores(name, safe):
    p = load_problem(name)
    cache = {}
    for i, line in enumerate(open(os.path.join(ROOT, "problems", name, "scores.jsonl")), 1):
        r = json.loads(line)
        where = f"problems/{name}/scores.jsonl line {i} ({r['name']})"
        path = os.path.join(ROOT, r["formula"])
        if not os.path.exists(path):
            errors.append(f"scores: {where}: {r['formula']} does not exist")
            continue
        if hashlib.sha256(open(path, "rb").read()).hexdigest()[:12] != r["sha"]:
            errors.append(f"scores: {where}: the formula file changed after scoring; score it again with --record")
            continue
        if path not in safe:
            continue
        fam = r.get("family", "all")
        if fam not in cache:
            cache[fam] = p.rows(None if fam == "all" else fam)
        if len(cache[fam]) != r["n_rows"]:
            errors.append(f"scores: {where}: scored on {r['n_rows']} rows, the data now has {len(cache[fam])}; score it again")
            continue
        got = score(p, load_module(path, "formula_check"), cache[fam], r.get("measured", False))["score"]
        if abs(got - r["score"]) > 5e-5:
            errors.append(f"scores: {where}: recorded {r['score']:.5f}, rescored {got:.5f}")


def main():
    fast = "--fast" in sys.argv
    files = tracked()
    check_size(files)
    check_privacy(files)
    check_papers(files)
    cols = columns()
    safe = set()
    for name in problem_names():
        check_problem(name, cols)
        fdir = os.path.join(ROOT, "problems", name, "formulas")
        if not os.path.isdir(fdir) or not os.path.exists(os.path.join(ROOT, "problems", name, "scores.jsonl")) \
                or not os.path.exists(os.path.join(ROOT, "problems", name, "LEADERBOARD.md")):
            continue  # check_problem already reported what is missing
        for f in sorted(os.listdir(fdir)):
            if f.endswith(".py"):
                path = os.path.join(ROOT, "problems", name, "formulas", f)
                if check_formula(path, cols, target_columns(name)):
                    safe.add(path)
        if not fast:
            check_scores(name, safe)
        if open(os.path.join(ROOT, "problems", name, "LEADERBOARD.md")).read() != leaderboard.build(name):
            errors.append(f"tables: problems/{name}/LEADERBOARD.md is out of date; run python3 core/leaderboard.py {name}")
    for w in warnings:
        print("warning  " + w)
    for e in errors:
        print("FAIL     " + e)
    print(f"{len(errors)} failures, {len(warnings)} warnings, {len(files)} tracked files, {len(problem_names())} problems")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
