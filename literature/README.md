# Literature

`papers.jsonl` lists every paper an agent or a person has read for this repo. One JSON object per line:

| field | meaning |
|---|---|
| `id` | `arXiv:2406.11813`, or `doi:...`, or `s2:<Semantic Scholar id>` when there is no arXiv id |
| `url` | the abstract page |
| `title` | the paper's title |
| `year` | year of first posting |
| `one_line` | what it shows, in your own words (never a copied abstract) |
| `formula` | its main formula in plain text, or `""` |
| `topics` | short tags, for example `forgetting`, `schedule`, `batch`, `muon` |
| `claims` | the `CLAIMS.md` ids it bears on, for example `["C04"]` |

## Rules for agents

- Search the file with grep before you search the web:
  `grep -i "forgetting" literature/papers.jsonl`, `grep '"C04"' literature/papers.jsonl`,
  `grep 'arXiv:2406.11813' literature/papers.jsonl`.
- Add every new paper you read, one line each, even when it turns out not to match.
- Check for the same `id` before you add a line. If the paper is there, add your claim id to its `claims` list instead.
- No PDFs in the repo and no copied abstracts.
- Keep the file valid: `python3 -c "import json; [json.loads(l) for l in open('literature/papers.jsonl')]"`.
