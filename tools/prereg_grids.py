"""Pre-registered predictions for tlf grids 1-3, written before their results exist (2026-11-27).

    python3 tools/prereg_grids.py --results <run table .jsonl> --grids <tlf-setups grids folder>

The run table is read for two things only: which grid tags already have a result (they are skipped),
and the older grid-3-testbed runs used to fit C28. No grid result is read by any fit.
Writes prereg/grids-2026-11-27.jsonl (one prediction per line) and prereg/frozen-2026-11-27.json
(every frozen constant). Each line: tag, claim, quantity, predicted, formula (and k for bits).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.rows import problem_rows  # noqa: E402

FL_KEYS = ["width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched", "wd", "min_lr_frac", "warmup",
           "decay", "momentum", "device"]   # a final-loss sweep: all of these equal, only the peak differs
GRIDS = {"tlfg1": "window-lr", "tlfg2": "lr-width-depth", "tlfg3": "repeats-width-depth"}
PEAKS2 = [0.002 * 2 ** (i / 2) for i in range(8)]   # grid 2 peaks, sqrt(2) apart: 2x = two grid steps
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def job_cfg(job):
    """Turn a job's command line into one settings dict."""
    a, cfg, i = job["args"], {}, 0
    while i < len(a):
        if a[i] == "--set":
            i += 1
            while i < len(a) and not a[i].startswith("--"):
                k, v = a[i].split("=", 1)
                cfg[k] = v
                i += 1
            continue
        if a[i].startswith("--") and i + 1 < len(a):
            cfg[a[i][2:]] = a[i + 1]
            i += 2
            continue
        i += 1
    return cfg


def lr_after(cfg):
    """Learning rate at every step after the fact window, with the trainer's lr_at() and glrx rules."""
    T, lr = int(cfg["steps"]), float(cfg["lr"])
    warm, sched, mn = int(cfg.get("warmup", 20)), cfg.get("sched", "cos"), float(cfg.get("min_lr_frac", 0))
    lo, hi = float(cfg.get("fwlo", 0.2)), float(cfg.get("fwhi", 0.45))
    w1 = max(int(lo * T) + 1, int(hi * T))
    g = float(cfg.get("glrx", 0)) or 1.0
    out = []
    for s in range(w1, T):
        if s < warm:
            v = lr * (s + 1) / warm
        elif sched == "const":
            v = lr
        else:
            p = (s - warm) / max(1, T - warm)
            v = lr * (mn + (1 - mn) * 0.5 * (1 + math.cos(math.pi * p)))
        out.append(v * g)
    return out


def z2_row(cfg, d):
    """The row end_z2 reads. A constant schedule is min_lr_frac 1, as in the forgetting data;
    in grid 1 the peak is the lr while the facts are shown (window_lr)."""
    const = cfg.get("sched") == "const"
    return {"peak": float(d.get("window_lr", d["peak"])), "wd": float(cfg.get("mwd", 0)),
            "min_lr_frac": 1.0 if const else float(cfg.get("min_lr_frac", 0)), "steps": int(cfg["steps"]),
            "width": int(cfg["d_model"]), "layers": int(cfg["n_layers"]), "lr": lr_after(cfg)}


def vertex(v):
    """Best peak of one sweep {peak: val_loss}: vertex of a parabola in ln peak through the grid minimum
    and up to 2 neighbours each side (4+ points). None if the minimum is at either end of the grid."""
    P = sorted(v)
    i = min(range(len(P)), key=lambda j: v[P[j]])
    if i == 0 or i == len(P) - 1:
        return None
    s = P[max(0, i - 2): i + 3]
    if len(s) < 3:
        return None
    c2, c1, _ = np.polyfit([math.log(p) for p in s], [v[p] for p in s], 2)
    if c2 <= 0:
        return P[i]
    return float(min(max(math.exp(-c1 / (2 * c2)), s[0]), s[-1]))


def nonemb(width, layers):
    return 10 * layers * width * width   # attention 4w^2 + gated MLP with hidden 2w 6w^2, per layer


def fl_row(cfg, peak=None):
    return {"nonemb_params": nonemb(int(cfg["d_model"]), int(cfg["n_layers"])),
            "tokens": int(cfg["steps"]) * int(cfg["batch"]) * int(cfg["seq_len"]),
            "peak": float(cfg["lr"]) if peak is None else peak, "sched": cfg["sched"], "steps": int(cfg["steps"]),
            "warmup": int(cfg["warmup"]), "decay": float(cfg["decay"]), "min_lr_frac": float(cfg["min_lr_frac"]),
            "width": int(cfg["d_model"])}


def c28_rows(path):
    """Older runs of the grid-3 testbed: size S, FineWeb vocab 8192, Muon, facts shown in all of
    training, fkmax 256, default schedule, batch 16, seq 256, no filler facts."""
    out = []
    for line in open(path):
        r = json.loads(line)
        c = r.get("config") or {}
        if r["tag"].startswith("tlfg") or r.get("size") != "S" or r.get("data") != "fineweb":
            continue
        if (c.get("vocab"), c.get("fkmax"), c.get("opt"), c.get("batch"), c.get("seq_len")) != (8192, 256, "muon", 16, 256):
            continue
        if c.get("fwlo") not in (None, 0.0) or c.get("sched") is not None or c.get("ffill") not in (None, 0):
            continue
        if c.get("ffrand") not in (None, 0) or c.get("mom") not in (None, 0.95):
            continue
        bk = (r.get("facts_eval") or {}).get("bits_by_k")
        if bk and all(str(k) in bk for k in KS):
            out.append(r)
    return out


def c28_x(k, size, facts, steps, lr, inter=False):
    x = [1.0, math.log2(k), math.log2(size), math.log2(facts), math.log2(steps), math.log2(lr)]
    return x + [math.log2(k) * math.log2(size)] if inter else x


def c28_fit(rows, size_key, inter=False):
    X, y = [], []
    for r in rows:
        c = r["config"]
        for k in KS:
            X.append(c28_x(k, r[size_key], c["facts"], r["steps"], c["lr"], inter))
            y.append(r["facts_eval"]["bits_by_k"][str(k)])
    return np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]


def c28_err(rows, theta, size_key, inter=False):
    e = [r["facts_eval"]["bits_by_k"][str(k)] - np.dot(theta, c28_x(k, r[size_key], r["config"]["facts"], r["steps"], r["config"]["lr"], inter))
         for r in rows for k in KS]
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--grids", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "prereg"))
    a = ap.parse_args()

    done = set()
    for line in open(a.results):
        t = json.loads(line)["tag"]
        if t.startswith("tlfg"):
            done.add(t)
    jobs = {p: [json.loads(l) for l in open(os.path.join(a.grids, g, "a4000.jsonl"))] for p, g in GRIDS.items()}
    pend = {p: [j for j in js if j["tag"] not in done] for p, js in jobs.items()}
    frozen, out = {}, []

    def put(tag, claim, quantity, value, formula, **kw):
        out.append(dict(tag=tag, claim=claim, quantity=quantity, predicted=round(float(value), 6), formula=formula, **kw))

    # end_z2 (frozen 2026-11-26) and its depth-free variant for C01
    z2 = load(os.path.join(ROOT, "problems/forgetting/formulas/end_z2.py"))
    th = list(z2.FROZEN)
    th_l0 = th[:6] + [0.0] + th[7:]
    frozen["end_z2"] = th
    frozen["end_z2_L0"] = th_l0
    for p in ("tlfg1", "tlfg2"):
        for j in pend[p]:
            cfg, row = job_cfg(j), z2_row(job_cfg(j), j["draw"])
            put(j["tag"], "C06", "kept", z2.kept(th, row), "end_z2")
            z = float(z2.x(row) @ np.asarray(th[:9]))
            put(j["tag"], "C10", "s0", math.exp(-z), "end_z2_curve")
            if p == "tlfg2":
                put(j["tag"], "C01", "kept", z2.kept(th_l0, row), "end_z2_L0")

    for p in ("tlfg1", "tlfg2"):
        put(p + "_all", "C10", "q", math.exp(th[9]), "end_z2_curve")

    # lr_area fitted on the whole final-loss problem (1,163 runs, no grid run in it), then frozen
    la = load(os.path.join(ROOT, "problems/final-loss/formulas/lr_area.py"))
    fr = problem_rows("final-loss")
    assert not any(r["tag"].startswith("tlfg") for r in fr)
    P = la.fit(fr, [r["val_loss"] for r in fr])
    frozen["lr_area"] = [float(v) for v in P]
    frozen["lr_area_rmse_in_sample"] = float(np.sqrt(np.mean([(la.predict(P, r) - r["val_loss"]) ** 2 for r in fr])))
    cl = load(os.path.join(ROOT, "problems/final-loss/formulas/chinchilla_lr.py"))
    PC = cl.fit(fr, [r["val_loss"] for r in fr])
    frozen["chinchilla_lr"] = [float(v) for v in PC]
    for j in pend["tlfg2"]:
        put(j["tag"], "C09", "val_loss", la.predict(P, fl_row(job_cfg(j))), "lr_area")
        put(j["tag"], "C09", "val_loss", cl.predict(PC, fl_row(job_cfg(j))), "chinchilla_lr")

    # C26 line: best peak = exp(a) * width^-0.64, depth-free. a is fitted now on the cosine sweeps of the
    # final-loss data (best peak = vertex of a parabola in ln peak through the grid minimum and up to 2
    # neighbours each side, sweeps with the minimum inside the grid), with the exponent held at -0.64.
    sw = {}
    for r in fr:
        sw.setdefault(tuple(str(r.get(k)) for k in FL_KEYS), {})[r["peak"]] = r["val_loss"]
    va = []
    for k_, v in sw.items():
        if k_[7] == "cos" and len(v) >= 4:
            b = vertex(v)
            if b is not None:
                va.append((int(k_[0]), b))
    a26 = float(np.mean([math.log(b) + 0.64 * math.log(w) for w, b in va]))
    free = np.polyfit([math.log(w) for w, b in va], [math.log(b) for w, b in va], 1)
    frozen["c26_line"] = {"ln_a": a26, "exponent": -0.64, "fit_sweeps": len(va),
                          "free_exponent_on_same_sweeps": float(free[0])}

    # per sweep (width, depth) of grid 2: best peak (C26), 2x costs and curvature (C11)
    sweeps = {}
    for j in jobs["tlfg2"]:
        sweeps.setdefault((j["draw"]["width"], j["draw"]["layers"]), job_cfg(j))
    lnw, lnl, lnp = [], [], []
    for (w, L), cfg in sorted(sweeps.items()):
        f = lambda lp: la.predict(P, fl_row(cfg, math.exp(lp)))  # noqa: E731
        best = minimize_scalar(f, bounds=(math.log(1e-4), 0.0), method="bounded").x
        sid = "tlfg2_w%d_L%d_sweep" % (w, L)
        nd = sum(j["tag"] in done for j in jobs["tlfg2"] if (j["draw"]["width"], j["draw"]["layers"]) == (w, L))
        put(sid, "C26", "best_peak", math.exp(a26) * w ** -0.64, "c26_line", runs_done_before=nd)
        put(sid, "C26", "best_peak", math.exp(best), "lr_area_argmin", runs_done_before=nd)
        lnw.append(math.log(w)); lnl.append(math.log(L)); lnp.append(best)
        put(sid, "C11", "cost_2x_high", f(best + math.log(2)) - f(best), "lr_area_bowl", runs_done_before=nd)
        put(sid, "C11", "cost_2x_low", f(best - math.log(2)) - f(best), "lr_area_bowl", runs_done_before=nd)
        h = 0.05
        put(sid, "C11", "curvature", (f(best + h) - 2 * f(best) + f(best - h)) / h ** 2 / 2, "lr_area_bowl", runs_done_before=nd)
    A = np.column_stack([np.ones(len(lnw)), lnw, lnl])
    co = np.linalg.lstsq(A, np.array(lnp), rcond=None)[0]
    put("tlfg2_all", "C26", "width_exponent", -0.64, "c26_line")
    put("tlfg2_all", "C26", "depth_exponent", 0.0, "c26_line")
    put("tlfg2_all", "C26", "width_exponent", co[1], "lr_area_argmin")
    put("tlfg2_all", "C26", "depth_exponent", co[2], "lr_area_argmin")

    # C28: recall log-linear in repeats and size, fitted now on older runs of the grid-3 testbed
    cr = c28_rows(a.results)
    half = [int(hashlib.md5(r["tag"].encode()).hexdigest(), 16) % 2 for r in cr]
    tr = [r for r, h_ in zip(cr, half) if h_ == 0]
    te = [r for r, h_ in zip(cr, half) if h_ == 1]
    pick = {}
    for key in ("nonemb_params", "params"):
        pick[key] = c28_err(te, c28_fit(tr, key), key)
    key = min(pick, key=pick.get)
    th28 = c28_fit(cr, key)
    th28i = c28_fit(tr, key, True)
    frozen["c28_loglin"] = {"theta": [float(v) for v in th28],
                            "x": "[1, log2 k, log2 %s, log2 facts, log2 steps, log2 lr]" % key,
                            "size_column": key, "fit_runs": len(cr), "heldout_rmse_by_size_column": pick,
                            "heldout_rmse_interaction": c28_err(te, th28i, key, True),
                            "in_sample_rmse": c28_err(cr, th28, key),
                            "fit_range": {"facts": sorted({r["config"]["facts"] for r in cr}),
                                          "steps": sorted({r["steps"] for r in cr}),
                                          "width": sorted({r["config"]["d_model"] for r in cr}),
                                          "layers": sorted({r["config"]["n_layers"] for r in cr})}}
    for j in pend["tlfg3"]:
        cfg = job_cfg(j)
        w, L = int(cfg["d_model"]), int(cfg["n_layers"])
        size = nonemb(w, L) if key == "nonemb_params" else nonemb(w, L) + 2 * 8192 * w
        for k in KS:
            put(j["tag"], "C28", "bits", np.dot(th28, c28_x(k, size, int(cfg["facts"]), int(cfg["steps"]), float(cfg["lr"]))), "c28_loglin", k=k)

    # C01, loss half: a deeper model at the same width (and peak, or number of facts) has lower val_loss
    for p, other in (("tlfg2", "peak"), ("tlfg3", "n_facts")):
        cells = {}
        for j in jobs[p]:
            d = j["draw"]
            cells.setdefault((d["width"], d[other]), {})[d["layers"]] = j["tag"]
        for (w, o), byL in sorted(cells.items()):
            Ls = sorted(byL)
            for L0, L1 in zip(Ls, Ls[1:]):
                if byL[L0] in done and byL[L1] in done:
                    continue
                put("%s|%s" % (byL[L0], byL[L1]), "C01", "sign_val_loss_change", -1, "c01_deeper_lower_loss")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "grids-2026-11-27.jsonl"), "w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    with open(os.path.join(a.out, "frozen-2026-11-27.json"), "w") as fh:
        json.dump(frozen, fh, indent=1)
    cnt = {}
    for o in out:
        g = o["tag"][:5]
        cnt.setdefault(g, {}).setdefault(o["claim"], 0)
        cnt[g][o["claim"]] += 1
    print(json.dumps({"pending": {p: len(v) for p, v in pend.items()}, "lines": cnt, "c28_pick": pick,
                      "c26_exponents": [float(co[1]), float(co[2])]}, indent=1))


if __name__ == "__main__":
    main()
