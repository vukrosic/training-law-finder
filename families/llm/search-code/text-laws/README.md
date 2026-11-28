# text-laws: Muon vs normalized momentum (spectral power p)

Code of the study behind "spectral power horizon", 2026-11-28: about 800 small 2-layer GPTs trained with one optimizer dial.

The dial is X_p = U (S / S_max)^p Vᵀ, the SVD of the momentum matrix with its singular values raised to the power p. p = 0 is Muon; p = 1 is normalized momentum. It is set in the `Opt` class in `bgpt.py`.

- `bgpt.py`: batched GPT and the optimizer with the p dial (plus an optional schedule that moves p during a run).
- `blaw.py`: trains a pack of K models in one batched pass from a job file; one JSON line per model.
- `gen/`: job-file generators. `spec3.py` and `spec4.py` make the main grid (p from 0 to 1, 64-1024 steps, batch 16 or 64, width 64 or 128, both learning rates). `spneg.py` covers negative p, `long4k.py` 4,096-step runs, `sched.py` p schedules and `edge.py` the edge cells.
- `fits/`: analysis scripts; `spec_dial.py` picks the best tuned run per p in each cell.
- `sim/`: the noisy quadratic model used as a curvature check (`diag.py`, `nqm.py`).
- `runpair.sh`, `run2.sh`, `run2b.sh`: launch helpers (two jobs at once, one on MPS and one on CPU).

Data: FineWeb-Edu, 1,024-token vocabulary (`prep_vocab.py`). Results, job files and a code snapshot are in the Hugging Face dataset [vukrosic/training-laws-research](https://huggingface.co/datasets/vukrosic/training-laws-research) under `studies/text-laws/`.
