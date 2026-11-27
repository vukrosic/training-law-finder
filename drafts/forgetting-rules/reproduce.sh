#!/bin/sh
# Rebuilds every number and figure of the paper from the public table in this repo. No training.
# Run from anywhere:  sh drafts/forgetting-rules/reproduce.sh
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export TLF_REPO="$ROOT"
PY="${PYTHON:-python3}"   # needs numpy, scipy and matplotlib; set PYTHON=/path/to/python to choose one

# 1. the held-out score of the paper's formula: prints 0.0441, which is 4.4 points
"$PY" "$ROOT/core/score.py" forgetting "$ROOT/problems/forgetting/formulas/end_z2.py" --family llm

# 2. the prediction committed before the 503M run, and the known miss
"$PY" "$ROOT/core/round_score.py" "$ROOT/rounds/002-forgetting"

# 3. every number in the text (numbers.json, not kept in git) and the six figures (figures/)
"$PY" paper_numbers.py > /dev/null
"$PY" make_figs.py

# 4. optional: the PDF, if LaTeX is installed
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null && echo "wrote paper.pdf"
fi
