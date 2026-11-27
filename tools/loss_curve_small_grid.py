"""Write the small-run grid for problems/loss-curve: runs of about 3-12 s each on a MacBook CPU.

    python3 tools/loss_curve_small_grid.py

Writes prereg/loss-curve-small.jsonl, one run per line (block, tag, args, machine, draw), like
tools/loss_curve_grid.py. That 300-run grid (512-16,384 steps, widths up to 1,024) is parked: it needs
minutes to hours per run. This grid keeps its design (3 schedules, 3 peaks, held-out widths, the same kill
line) at a smaller size.

Settings: vocabulary 1,024, Muon (same peak for the Adam part), batch 32, sequence 64, 2 layers, MLP hidden
2 x width, warmup 20 steps, weight decay 0, min_lr_frac 0, WSD decay share 0.2. Each run logs val loss and
4 token-band losses at 32 evenly spaced steps (--curve_points 32) on 16,384 fixed val tokens.

Blocks (228 runs):
  S  6 widths x 3 step counts (128, 256, 512) x 3 schedules x peak 0.5x, 1x, 2x the best,
     except 512 steps at width 96 and at width 64 off 1x (over 12 s)                     147
  T  widths 16-64 x 384 steps x 3 schedules at 1x                                          15
  P  widths 16-64 x 256 steps x 3 schedules x peak 0.25x and 4x                            30
  L  widths 16, 24 x 1,024 steps x 3 schedules at 1x                                         6
  D  6 widths x WSD decay share 0.1 and 0.5, 256 steps                                      12
  E  6 widths x cosine to 0.1 of peak, 256 steps                                             6
  F  widths 32, 64 x 1 and 4 layers x 3 schedules, 256 steps                                12
Best peak per width = e^-1.708 * width^-0.64 (the final-loss rule; the 0.5x-2x spread covers a shift).

Time model, measured on this MacBook (M-series, 2026-11-27, 2 threads, 2 layers, batch 32): about 3 s fixed
(imports, setup, 33 evals) plus per step 6 ms (width 16), 10 (24), 14 (32), 17 (48), 21 (64), 31 (96);
scaled by layers / 2 for the per-step part.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "prereg", "loss-curve-small.jsonl")
WIDTHS = [16, 24, 32, 48, 64, 96]
SCHEDS = ["const", "wsd", "cos"]
VOCAB, BATCH, SEQ, WARM, EVAL_TOK, POINTS, LAYERS = 1024, 32, 64, 20, 16384, 32, 2
STEP_MS = {16: 6, 24: 10, 32: 14, 48: 17, 64: 21, 96: 31}


def best_peak(width):
    return math.exp(-1.708) * width ** -0.64


def ff_hidden(d, mult=2.0, multiple=32):
    return int(multiple * math.ceil(d * mult / multiple))


def counts(width, layers):
    h = ff_hidden(width)
    nonemb = layers * (4 * width * width + 3 * width * h)
    return nonemb + 2 * VOCAB * width, nonemb


def run(width, steps, sched, f=1.0, layers=LAYERS, decay=0.2, mlf=0.0, extra=""):
    peak = float(f"{best_peak(width) * f:.3g}")
    mlf = mlf if sched != "const" else 0.0
    heads = max(1, width // 32)
    tag = f"lcs_w{width}_L{layers}_T{steps}_{sched}_p{peak:g}{extra}"
    est = 3 + steps * STEP_MS[width] / 1000 * layers / LAYERS
    args = ["--data", "data/fineweb_v1024", "--d", str(width), "--layers", str(layers), "--heads", str(heads),
            "--ff_mult", "2", "--seq", str(SEQ), "--batch", str(BATCH), "--steps", str(steps), "--lr", f"{peak:g}",
            "--wd", "0", "--sched", sched, "--warmup", f"{WARM / steps:.10g}", "--decay", str(decay),
            "--min_lr_frac", str(mlf), "--eval_tokens", str(EVAL_TOK), "--curve_points", str(POINTS),
            "--band_cache", "work/cache", "--tag", tag, "--device", "cpu"]
    n, nonemb = counts(width, layers)
    return {"tag": tag, "args": args, "machine": "mac",
            "draw": {"width": width, "layers": layers, "heads": heads, "steps": steps, "sched": sched, "peak": peak,
                     "peak_x": f, "decay": decay, "min_lr_frac": mlf, "params": n, "nonemb_params": nonemb,
                     "tokens": steps * BATCH * SEQ, "est_mac_s": round(est, 1)}}


def grid():
    out = []
    for w in WIDTHS:
        for T in (128, 256, 512):
            for s in SCHEDS:
                for f in (0.5, 1.0, 2.0):
                    if T == 512 and (w == 96 or (w == 64 and f != 1.0)):
                        continue          # over 12 s
                    out.append(("S", run(w, T, s, f)))
        if w < 96:
            for s in SCHEDS:
                out.append(("T", run(w, 384, s)))
                for f in (0.25, 4.0):
                    out.append(("P", run(w, 256, s, f)))
        for d in (0.1, 0.5):
            out.append(("D", run(w, 256, "wsd", decay=d, extra=f"_d{d:g}")))
        out.append(("E", run(w, 256, "cos", mlf=0.1, extra="_m0.1")))
    for w in (16, 24):
        for s in SCHEDS:
            out.append(("L", run(w, 1024, s)))
    for w in (32, 64):
        for L in (1, 4):
            for s in SCHEDS:
                out.append(("F", run(w, 256, s, layers=L)))
    return out


def main():
    rows = grid()
    tags = [r["tag"] for _, r in rows]
    assert len(tags) == len(set(tags)), "duplicate tag: every setting must appear once"
    with open(OUT, "w") as f:
        for block, r in rows:
            f.write(json.dumps({"block": block, **r}) + "\n")
    est = [r["draw"]["est_mac_s"] for _, r in rows]
    print(f"{len(rows)} runs -> {os.path.relpath(OUT, ROOT)}")
    print(f"est. {sum(est) / 60:.0f} CPU minutes, median {sorted(est)[len(est) // 2]} s, max {max(est)} s, "
          f"{sum(e > 10 for e in est)} runs over 10 s, {sum(e > 20 for e in est)} over 20 s")


if __name__ == "__main__":
    main()
