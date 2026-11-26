# Drafts

Paper drafts written by AI agents (see [agents/writer.md](../agents/writer.md)), built only on the open
problems and the public data of this repo. A draft is not checked by a person: its numbers come from
the agents' runs and may be wrong. Once a person has checked every claim and rewritten it, it moves to
[papers/](../papers/README.md).

```text
drafts/<slug>/
  DRAFT.md       first line: "AI draft, not checked by a person." Then the title, abstract and text
  figures/       SVG, or PNG under 500 KB each
  results/       the fits, scores and predictions the draft rests on (small files only)
```

Limits, checked by `tools/check.py`: no `.pdf` files, figures under 500 KB, the whole folder under 5 MB.
