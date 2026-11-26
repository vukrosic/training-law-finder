# LLM run data

Three gzipped JSON-lines files, one run per line. They live on Hugging Face
([vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research)), not in git;
`core.rows.load()` downloads them into this folder the first time. Row counts are from the 2026-11-26
refresh; `MANIFEST.json` on Hugging Face has the counts of the latest one.

| file | rows | what |
|---|---|---|
| `runs.jsonl.gz` | 24,012 | real training runs of small decoder-only language models (CPU, Mac, Kaggle and one RTX A4000) |
| `other_searches.jsonl.gz` | 15,586 | real runs from the other searches (compute, data mix, init, learning rate, mixture of experts, Muon scaling, residual, the 125M check of a DeepSeek-V3 style baseline, loss target, vocabulary): settings and final loss, no fact probe |
| `sim.jsonl.gz` | 4,944 | runs of a fast fact-memory simulator; never mix with real rows without saying so |

All real runs train on FineWeb text with a few synthetic facts mixed in (a "fact" is a random
name-to-number pair shown k times). Most runs use Muon on the block matrices and Adam on the rest,
a 1,024-token vocabulary (1,062 rows use 8,192), RMSNorm, RoPE, QK-norm and SwiGLU.
Older rows came from earlier versions of the training code; the current code is
`families/llm/testbed-facts-v1/`.

## Columns (every real row)

| column | meaning |
|---|---|
| tag, family, file, line, machine, device, prec | run id; family = tag up to the first `_` (a group of related runs, not the model family); file = the results file in the search folder, which is `raw/<search>/<file>.gz` on Hugging Face; line = 1-based line number in that file; machine = gpu / mac / cpu / kaggle |
| width, layers, heads, hidden, n_experts, batch, seq, steps | model and data size |
| method | `dense` or `dsv3` (DeepSeek-V3 style mixture of experts, 56 rows) |
| sched, peak, mlr, wd, min_lr_frac, warmup, decay | learning-rate schedule (const / cos / wsd / crw / cosrw; crw and cosrw drop the rate and then warm it back up at share rwp of training, over rww steps); min_lr_frac 1 = no decay |
| window, fwlo, fwhi | where in training the facts are shown, as a share of steps; A = 0.20-0.45, B = 0.45-0.70 |
| k, F | [min, max] repeats per fact; number of facts |
| amp, clip, momentum, eval_seqs, opt, gaplr, glrx, gstep, fref, wlog | other training arguments |
| params, nonemb_params, tokens, val_loss, val_bpb, train_seconds | end of run (val_loss in nats per token) |
| C, bits_by_k, acc_all, stored_bits, wnorm | end-of-run fact probe: bits of fact knowledge per repeat count k; C = mean over k 1, 2, 4 |
| cfg | training arguments that differ from the defaults |
| cfg_all | the full training config |
| protocol | `lf`, `other` or `end_only` (below) |
| dev | for `other` rows, the knobs that differ from `lf` |
| n_same_tag | how many files hold this tag (the same run on two machines gives 2) |

Rows with a per-step trace after the fact window also have:

| column | meaning |
|---|---|
| w1, n_read | step where the facts stop; number of readings after it |
| b0, b_end, b | fact bits at the window end, at the end, and at each reading |
| ref0, ref_end | bits on a reference set of fref facts the model never saw (the chance level), at the window end and at the end |
| kept | kept[i] = share of the learned fact bits still there at reading i |
| end_kept | kept[-1] |
| lr, S, S_end | learning rate at each step after the window; its running sum; the final sum |
| rms_w, rms_w_end, rms_w_trace | weight RMS at the window end, at the end, and over training |

## protocol

- `lf` (775 rows at this refresh): the clean forgetting protocol (64 facts, k 1-128, window A or B, plain text after the window, Muon). Comparable with each other. `problems/forgetting/data/llm.jsonl.gz` is 375 of these, frozen.
- `other` (6,700 rows): a per-step trace under a different protocol (branch from a checkpoint, replayed facts, changed optimizer after the window). `dev` says what differs.
- `end_only` (16,537 rows): fact probe at the end only, no curve. 14,242 are from a learning-rate schedule search.

## other_searches rows

Each row is the training script's own output line, unchanged, plus `search` (which search it came from) and `file` (its results file). Columns differ between searches; most rows have `method`, `tag`, `size`, `steps`, `params`, `tokens`, `val_loss`, `val_bpb`, `train_seconds` and `config` (the full training arguments). `device` is `cpu` or `cuda` where recorded. Identical lines are kept once.

## Simulator rows

Same column names where they exist (width = d, layers = L, batch = B, steps = T1), plus the simulator's own traces. protocol = `sim`.
