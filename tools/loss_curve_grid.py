"""Write the run grid for problems/loss-curve. Nothing here trains anything.

    python3 tools/loss_curve_grid.py

Writes prereg/loss-curve-grid.jsonl, one run per line:
  block    which part of the grid (A-F below)
  tag      lcg_w<width>_L<layers>_T<steps>_<sched>_p<peak>[...]; a queue skips tags that already have a result
  args     flags for testbed/train.py, run from the repo root (data from
           `python3 -I testbed/prepare_data.py --vocab 1024`)
  machine  "mac" if the run fits a MacBook (est. memory under 2 GB and est. CPU time under 20 min), else "gpu"
  draw     the settings, parameter counts, tokens, est. memory and est. seconds

Settings follow problems/final-loss: vocabulary 1,024, Muon (same peak for the Adam part), batch 64,
sequence 64, warmup 20 steps, MLP hidden 2 x width, 3 layers, weight decay 0, min_lr_frac 0, WSD decay
share 0.2. Each run sets --curve_points 32, so it logs val loss and 4 token-frequency band losses at 32
evenly spaced steps. Every setting appears once.

Blocks (300 runs):
  A  12 widths x 3 schedules x 4 step counts (512-4,096) at the best peak for the width     144
  B  12 widths x 3 schedules x peak 0.5x and 2x the best, 1,024 steps                        72
  C  widths 32, 64, 128, 256 x 3 schedules x 8,192 and 16,384 steps                          24
  D  12 widths x WSD decay share 0.1 and 0.5, 1,024 steps                                    24
  E  12 widths x cosine to 0.1 of peak, 1,024 steps                                          12
  F  widths 64, 128, 256, 512 x 1 and 6 layers x 3 schedules, 1,024 steps                    24
Best peak per width = e^-1.708 * width^-0.64 (c26_line in prereg/README.md, fitted on 60 cosine sweeps
of the final-loss data at 1,024 steps).

Cost model (rough): Mac CPU seconds = max(60, (6 N tokens + 2 N * 131,072 * 33) / 2e10); the second term
is the 32 curve evaluations plus the final one. The 2e10 rate is the one the tlf-setups grids use
(w64 L3 512 steps = 171 s). A4000 seconds = max(40, same flops / 2.5e12).
Memory = 16 bytes per weight (weights, grads, two optimizer states) + 3 copies of the logits
(batch x seq x vocab floats) + 12 floats per activation per layer + 0.5 GB for Python and torch.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "prereg", "loss-curve-grid.jsonl")
WIDTHS = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SCHEDS = ["cos", "wsd", "const"]
VOCAB, BATCH, SEQ, WARM, EVAL_TOK, POINTS = 1024, 64, 64, 20, 131072, 32
MAC_MAX_S, MAC_MAX_GB = 1200, 2.0


def best_peak(width):
    return float(f"{math.exp(-1.708) * width ** -0.64:.3g}")


def ff_hidden(d, mult=2.0, multiple=32):
    """Same rule as ff_hidden in testbed/model.py."""
    return int(multiple * math.ceil(d * mult / multiple))


def counts(width, layers):
    """Parameter counts of the testbed GPT: attention 4 d^2 + SwiGLU 3 d h per block (norms left out);
    embedding and output tables V d each."""
    h = ff_hidden(width)
    nonemb = layers * (4 * width * width + 3 * width * h)
    return nonemb + 2 * VOCAB * width, nonemb


def cost(width, layers, steps):
    n, _ = counts(width, layers)
    flops = 6 * n * steps * BATCH * SEQ + 2 * n * EVAL_TOK * (POINTS + 1)
    mem = (16 * n + 3 * 4 * BATCH * SEQ * VOCAB + 12 * 4 * BATCH * SEQ * width * layers) / 1e9 + 0.5
    return max(60.0, flops / 2e10), max(40.0, flops / 2.5e12), mem


def run(width, steps, sched, peak=None, layers=3, decay=0.2, mlf=0.0, extra=""):
    peak = peak or best_peak(width)
    mlf = mlf if sched != "const" else 0.0
    heads = max(1, width // 64)
    tag = f"lcg_w{width}_L{layers}_T{steps}_{sched}_p{peak:g}{extra}"
    mac_s, gpu_s, mem = cost(width, layers, steps)
    machine = "mac" if mem < MAC_MAX_GB and mac_s <= MAC_MAX_S else "gpu"
    args = ["--data", "data/fineweb_v1024", "--d", str(width), "--layers", str(layers), "--heads", str(heads),
            "--ff_mult", "2", "--seq", str(SEQ), "--batch", str(BATCH), "--steps", str(steps), "--lr", f"{peak:g}",
            "--wd", "0", "--sched", sched, "--warmup", f"{WARM / steps:.10g}", "--decay", str(decay),
            "--min_lr_frac", str(mlf), "--eval_tokens", str(EVAL_TOK), "--curve_points", str(POINTS),
            "--tag", tag, "--device", "cpu" if machine == "mac" else "cuda"]
    n, nonemb = counts(width, layers)
    return {"tag": tag, "args": args, "machine": machine,
            "draw": {"width": width, "layers": layers, "heads": heads, "steps": steps, "sched": sched, "peak": peak,
                     "decay": decay, "min_lr_frac": mlf, "params": n, "nonemb_params": nonemb,
                     "tokens": steps * BATCH * SEQ, "est_mem_gb": round(mem, 2), "est_mac_s": round(mac_s),
                     "est_gpu_s": round(gpu_s)}}


def grid():
    out = []
    for w in WIDTHS:                                               # A
        for s in SCHEDS:
            for T in (512, 1024, 2048, 4096):
                out.append(("A", run(w, T, s)))
    for w in WIDTHS:                                               # B
        for s in SCHEDS:
            for f in (0.5, 2.0):
                out.append(("B", run(w, 1024, s, peak=float(f"{best_peak(w) * f:.3g}"))))
    for w in (32, 64, 128, 256):                                   # C
        for s in SCHEDS:
            for T in (8192, 16384):
                out.append(("C", run(w, T, s)))
    for w in WIDTHS:                                               # D
        for d in (0.1, 0.5):
            out.append(("D", run(w, 1024, "wsd", decay=d, extra=f"_d{d:g}")))
    for w in WIDTHS:                                               # E
        out.append(("E", run(w, 1024, "cos", mlf=0.1, extra="_m0.1")))
    for w in (64, 128, 256, 512):                                  # F
        for L in (1, 6):
            for s in SCHEDS:
                out.append(("F", run(w, 1024, s, layers=L)))
    return out


def main():
    rows = grid()
    tags = [r["tag"] for _, r in rows]
    assert len(tags) == len(set(tags)), "duplicate tag: every setting must appear once"
    with open(OUT, "w") as f:
        for block, r in rows:
            f.write(json.dumps({"block": block, **r}) + "\n")
    mac = [r for _, r in rows if r["machine"] == "mac"]
    gpu = [r for _, r in rows if r["machine"] == "gpu"]
    print(f"{len(rows)} runs -> {os.path.relpath(OUT, ROOT)}")
    print(f"mac {len(mac)} runs, {sum(r['draw']['est_mac_s'] for r in mac) / 3600:.1f} CPU hours")
    print(f"gpu {len(gpu)} runs, {sum(r['draw']['est_gpu_s'] for r in gpu) / 3600:.1f} A4000 hours")
    for b in "ABCDEF":
        sel = [r for k, r in rows if k == b]
        print(f"  block {b}: {len(sel)} runs, {sum(r['machine'] == 'mac' for r in sel)} mac")


if __name__ == "__main__":
    main()
