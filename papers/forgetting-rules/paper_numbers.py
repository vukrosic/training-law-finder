"""All numbers of the paper, from the public table (problems/forgetting/data/llm.jsonl.gz).
    python3 paper_numbers.py      -> numbers.json
No training. Fits take about a minute on a laptop CPU.
"""
import gzip, json, math, os, collections
import numpy as np
from scipy.optimize import least_squares

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TLF_REPO", os.path.abspath(os.path.join(H, "..", "..")))   # the repo this folder sits in
rows = [json.loads(l) for l in gzip.open(os.path.join(REPO, "problems/forgetting/data/llm.jsonl.gz"), "rt")]
FROZEN = [0.6090888304082575, -0.5929143343302276, 0.8482437722622462, 0.5477102470699494,
          -1.1231337626121998, -0.3217546827755414, -0.12444372936842882, 0.17762544026882962,
          0.23118595535279032, 0.24836649590480928]
NAMES = ["const", "ln peak", "weight decay", "final-lr fraction", "ln steps", "ln width", "ln layers",
         "ln peak x ln width", "(ln width)^2"]

def feats(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return [1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld, math.log(r["layers"] / 8), lp * ld, ld * ld]

X = np.array([feats(r) for r in rows]); lnS = np.array([math.log(sum(r["lr"])) for r in rows])
y = np.array([r["kept"][-1] for r in rows]); W = np.array([r["width"] for r in rows])

def pred(p, X, lnS, cols):
    z = X[:, cols] @ p[:-1]
    return 1 / (1 + np.exp(np.minimum(50, np.exp(p[-1]) * (z + lnS))))

def fit(idx, cols):
    p0 = np.array([FROZEN[c] for c in cols] + [FROZEN[9]])
    return least_squares(lambda p: pred(p, X[idx], lnS[idx], cols) - y[idx], p0).x

def rmse(a): return float(np.sqrt(np.mean(np.square(a))))

def heldout(groups, cols):
    """leave one group out; returns mean per-group RMSE, per-group RMSE and the out-of-fold predictions"""
    oof = np.zeros(len(y)); per = {}
    for g in sorted(set(groups), key=str):
        te = np.array([x == g for x in groups]); p = fit(~te, cols)
        oof[te] = pred(p, X[te], lnS[te], cols); per[str(g)] = rmse(oof[te] - y[te])
    return float(np.mean(list(per.values()))), per, oof

ALL = list(range(9)); out = {"n": len(rows)}
s, per, oof = heldout(list(W), ALL)
out["score1"] = {"end_z2": s, "per_width": per, "n_per_width": {str(k): int(v) for k, v in collections.Counter(W.tolist()).items()}}
p_all = fit(np.ones(len(y), bool), ALL)
out["in_sample"] = rmse(pred(p_all, X, lnS, ALL) - y); out["theta"] = dict(zip(NAMES + ["ln q"], p_all.tolist())); out["q"] = float(np.exp(p_all[-1]))
out["oof"] = [{"tag": r["tag"], "width": r["width"], "layers": r["layers"], "y": float(a), "pred": float(b)} for r, a, b in zip(rows, y, oof)]
out["oof_R2"] = float(1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2))
out["oof_abs_err"] = {"median": float(np.median(np.abs(oof - y))), "share_within_0.05": float(np.mean(np.abs(oof - y) < 0.05)), "share_within_0.10": float(np.mean(np.abs(oof - y) < 0.10))}
# baselines
s2, per2, oof2 = heldout(list(W), [0]); out["score1"]["lr_clock"] = s2; out["score1"]["lr_clock_per_width"] = per2
perm = {str(g): rmse(y[W == g] - y[W != g].mean()) for g in sorted(set(W.tolist()))}
out["score1"]["mean"] = float(np.mean(list(perm.values()))); out["score1"]["mean_per_width"] = perm
# other ways to hold out
out["other_holdouts"] = {}
for name, key in [("steps", lambda r: r["steps"]), ("peak", lambda r: r["peak"]), ("window", lambda r: r["window"] if r["window"] != "const" else "A"),
                  ("schedule", lambda r: r["sname"]), ("weight decay", lambda r: r["wd"]), ("layers", lambda r: r["layers"])]:
    g = [key(r) for r in rows]; s_, per_, _ = heldout(g, ALL); s0, per0, _ = heldout(g, [0])
    out["other_holdouts"][name] = {"end_z2": s_, "lr_clock": s0, "per": per_, "n": {str(k): v for k, v in collections.Counter(g).items()}}
# drop one term at a time (width-held-out)
out["ablation"] = {}
for c in range(1, 9):
    cols = [k for k in ALL if k != c]; out["ablation"]["no " + NAMES[c]] = heldout(list(W), cols)[0]
out["ablation"]["no width terms (5,7,8)"] = heldout(list(W), [0, 1, 2, 3, 4, 6])[0]
out["ablation"]["no schedule terms (wd, final-lr, steps)"] = heldout(list(W), [0, 1, 5, 6, 7, 8])[0]
# noise floor from repeated settings in the public table
key = lambda r: tuple(r[k] for k in ["width", "layers", "heads", "hidden", "steps", "window", "fwlo", "fwhi", "peak", "wd", "min_lr_frac", "sname", "sched", "warmup", "fmarg", "gsplice"]) + (round(sum(r["lr"]), 6),)
d = collections.defaultdict(list)
for r in rows: d[key(r)].append(r["kept"][-1])
pairs = [(a, b) for v in d.values() if len(v) > 1 for i, a in enumerate(v) for b in v[i + 1:]]
out["repeats"] = {"settings": sum(len(v) > 1 for v in d.values()), "pairs": len(pairs),
                  "rmse_pair_diff": rmse([a - b for a, b in pairs]) if pairs else None,
                  "mean_abs_diff": float(np.mean([abs(a - b) for a, b in pairs])) if pairs else None}
# the same score with repeated rows removed (first row of each repeated setting kept)
seen, keep = set(), []
for i, r in enumerate(rows):
    k_ = key(r) + (round(r["kept"][-1], 3),)
    keep.append(k_ not in seen); seen.add(k_)
keep = np.array(keep); Xf, lf, yf, Wf = X, lnS, y, W
X, lnS, y, W = X[keep], lnS[keep], y[keep], W[keep]
out["dedup"] = {"rows": int(keep.sum()), "score1_end_z2": heldout(list(W), ALL)[0], "score1_lr_clock": heldout(list(W), [0])[0]}
X, lnS, y, W = Xf, lf, yf, Wf
# clock: steps vs summed lr, one shared 2-constant curve per model size, readings every 6 steps
def curve_r2(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    f = lambda p: 1 / (1 + np.exp(np.minimum(50, np.exp(p[1]) * (p[0] + np.log(xs))))) - ys
    best = min((least_squares(f, [a, 0.0]) for a in (-6, -3, 0, 3)), key=lambda r: r.cost)
    return float(1 - np.sum(best.fun ** 2) / np.sum((ys - ys.mean()) ** 2))
out["clock"] = []
by = collections.defaultdict(list)
for r in rows:
    if r["window"] in ("A", "B") and r["steps"] == 1024 and r["wd"] == 0 and r["sched"] != "crw": by[(r["width"], r["layers"])].append(r)
for (w, L), rs in sorted(by.items()):
    if len(rs) < 6: continue
    t, S, k = [], [], []
    for r in rs:
        cs = np.cumsum(r["lr"])
        for i in range(5, len(r["kept"]), 6): t.append(i + 1); S.append(cs[i]); k.append(r["kept"][i])
    out["clock"].append({"width": w, "layers": L, "runs": len(rs), "schedules": len({r["sname"] for r in rs}), "peaks": len({r["peak"] for r in rs}),
                         "R2_steps": curve_r2(t, k), "R2_S": curve_r2(S, k)})
# fair clock test 1: per model size, give both clocks their own peak exponent (3 constants each)
def curve3_r2(xs, lps, ys):
    xs, lps, ys = np.array(xs), np.array(lps), np.array(ys)
    f = lambda p: 1 / (1 + np.exp(np.minimum(50, np.exp(p[2]) * (p[0] + p[1] * lps + np.log(xs))))) - ys
    best = min((least_squares(f, [a, g, 0.0]) for a in (-6, -3, 0, 3) for g in (-1, 0, 1)), key=lambda r: r.cost)
    return float(1 - np.sum(best.fun ** 2) / np.sum((ys - ys.mean()) ** 2)), float(best.x[1])
out["clock_fair"] = []
for (w, L), rs in sorted(by.items()):
    if len(rs) < 6: continue
    t, S, k, lp = [], [], [], []
    for r in rs:
        cs = np.cumsum(r["lr"])
        for i in range(5, len(r["kept"]), 6): t.append(i + 1); S.append(cs[i]); k.append(r["kept"][i]); lp.append(math.log(r["peak"] / 8e-3))
    a, ga = curve3_r2(t, lp, k); b, gb = curve3_r2(S, lp, k)
    out["clock_fair"].append({"width": w, "layers": L, "runs": len(rs), "R2_steps": a, "R2_S": b, "gamma_steps": ga, "gamma_S": gb})
# fair clock test 2: the full formula with steps after the window in place of the summed learning rate
lnT = np.array([math.log(len(r["lr"])) for r in rows]); lnS_keep = lnS
lnS = lnT - math.log(1024) + math.log(8e-3 * 1024)   # steps times a fixed 8e-3, so the start point of the fit is on the same scale
sT, perT, oofT = heldout(list(W), ALL)
out["score1"]["end_z2_step_clock"] = sT; out["score1"]["end_z2_step_clock_per_width"] = perT
gs = [r["sname"] for r in rows]; out["other_holdouts"]["schedule"]["end_z2_step_clock"] = heldout(gs, ALL)[0]
lnS = lnS_keep
# whole fade inside each run: constants fitted on the end point only, clock read at every 6th step
p_all = fit(np.ones(len(y), bool), ALL); z_all = X @ p_all[:-1]; qq = math.exp(p_all[-1])
res, res_w = [], collections.defaultdict(list); traj = []
for r, z in zip(rows, z_all):
    cs = np.cumsum(r["lr"]); kk = np.array(r["kept"]); idx = np.arange(5, len(kk), 6)
    pr = 1 / (1 + np.exp(np.minimum(50, qq * (z + np.log(cs[idx])))))
    res += list(pr - kk[idx]); res_w[r["width"]] += list(pr - kk[idx])
    traj.append({"tag": r["tag"], "u": (z + np.log(cs[idx])).round(4).tolist(), "kept": kk[idx].round(4).tolist()})
out["within_run"] = {"readings": len(res), "rmse": rmse(res), "per_width": {str(k): rmse(v) for k, v in sorted(res_w.items())}}
out["traj"] = traj
out["z"] = z_all.round(5).tolist(); out["lnS"] = lnS.round(5).tolist()
# round 002: the three runs whose predictions were committed in git (19e61cc) before results (da0449e)
out["round002"] = []
actual = {json.loads(l)["tag"]: json.loads(l)["actual"] for l in open(os.path.join(REPO, "rounds/002-forgetting/results.jsonl"))}
for l in open(os.path.join(REPO, "rounds/002-forgetting/predictions.jsonl")):
    j = json.loads(l); st = j["settings"]; x = np.array([feats(st)])
    mine = float(pred(p_all, x, np.array([math.log(sum(st["lr"]))]), ALL)[0])
    out["round002"].append({"tag": j["tag"], "width": st["width"], "layers": st["layers"], "batch": st["batch"], "steps": st["steps"], "peak": st["peak"], "wd": st["wd"],
                            "S": float(sum(st["lr"])), "committed_pred": j["pred"], "refit_pred_end_z2": mine, "actual": actual.get(j["tag"])})
# same schedule at every width: cosine to zero, peak 8e-3, window A, 1,024 steps, no weight decay
p_in = pred(p_all, X, lnS, ALL)
out["width_ladder"] = [{"tag": r["tag"], "width": r["width"], "layers": r["layers"], "S": float(sum(r["lr"])), "kept": float(y[i]), "fit": float(p_in[i]), "oof": float(oof[i])}
                       for i, r in enumerate(rows) if r["sname"] == "cos0" and r["peak"] == 0.008 and r["window"] == "A" and r["steps"] == 1024 and r["wd"] == 0 and r["sched"] != "crw"]
# worked example: same model, same steps after the window, four schedules
ex = [r for r in rows if r["width"] == 512 and r["window"] == "A" and r["peak"] == 0.008 and r["wd"] == 0 and r["steps"] == 1024]
out["example_w512"] = [{"sname": r["sname"], "S": round(sum(r["lr"]), 3), "kept": round(r["kept"][-1], 3), "steps_after": len(r["lr"])} for r in ex]
# matched pairs: runs that differ in one setting only (the fact window, or the weight decay)
def pairs(field, a, b):
    keys = [k for k in ("width", "layers", "peak", "steps", "sname", "sched", "wd", "window") if k != field]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: g[tuple(r[k] for k in keys)][r[field]].append(r["kept"][-1])
    pr = [(float(np.mean(v[a])), float(np.mean(v[b]))) for v in g.values() if a in v and b in v]
    return {"pairs": len(pr), "mean_a": float(np.mean([p[0] for p in pr])) if pr else None,
            "mean_b": float(np.mean([p[1] for p in pr])) if pr else None, "b_higher": sum(q > p for p, q in pr)}
out["pairs_window"] = pairs("window", "A", "B")
out["pairs_wd"] = {str(w): pairs("wd", 0.0, w) for w in (0.1, 0.3, 1.0)}
json.dump(out, open(os.path.join(H, "numbers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("oof", "traj", "z", "lnS")}, indent=1))
