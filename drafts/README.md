# Drafts

Paper drafts by AI agents ([agents/writer.md](../agents/writer.md)), built on this repo's problems and public data. Not checked by a person; numbers may be wrong. After a person checks and rewrites every claim, a draft moves to [papers/](../papers/README.md).

```text
drafts/<slug>/
  DRAFT.md    first line "AI draft, not checked by a person.", then title, abstract, text
              (or README.md with that first line, plus paper.tex)
  figures/    SVG, or PNG under 500 KB each
  results/    small fit, score and prediction files the draft rests on
```

`tools/check.py` enforces: no `.pdf`, figures under 500 KB, folder under 5 MB.
