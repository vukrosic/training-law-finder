# Papers

Finished papers, written or rewritten by people, about formulas found with this repo. Each paper is a
folder:

```text
papers/<slug>/
  README.md      title, authors, abstract, link to the PDF, which problem and formula it is about
  paper.md       the text (or paper.tex)
  figures/       SVG, or PNG under 500 KB each
  reproduce.sh   the commands that rebuild every number and figure from the repo and the Hugging Face data
```

**PDFs are not kept in git.** Attach the PDF to a GitHub release named `paper-<slug>-v<N>` and link it
from the paper's README (or link the arXiv page). A PDF changes completely with each revision, so every
version kept in git would stay in the history and every clone forever.

Limits, checked by `tools/check.py`: no `.pdf` files, figures under 500 KB, the whole folder under 5 MB.
An AI-written draft goes in [drafts/](../drafts/README.md) until a person has checked and rewritten it.
