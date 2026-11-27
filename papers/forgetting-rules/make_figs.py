"""Figures of the paper, drawn from numbers.json and the public table.
    python3 paper_numbers.py && python3 make_figs.py      -> figures/fig1..fig6 as .png
Shares of facts are shown in percent and errors in percentage points (kept units x 100).
"""
import gzip, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TLF_REPO", os.path.abspath(os.path.join(H, "..", "..")))
N = json.load(open(os.path.join(H, "numbers.json")))
rows = [json.loads(l) for l in gzip.open(os.path.join(REPO, "problems/forgetting/data/llm.jsonl.gz"), "rt")]
F = os.path.join(H, "figures"); os.makedirs(F, exist_ok=True)

INK, MUTED, GRID = "#111111", "#55534e", "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8880"
RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]            # ordinal: narrow -> wide
BINS = [("64", [64]), ("128-256", [128, 256]), ("384-512", [384, 512]), ("768-1024", [768, 1024])]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white"})
P = 100.0                                                       # kept units -> percentage points

def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.6)

def pct(axis, full=1.0):
    axis.set_major_formatter(PercentFormatter(full, decimals=0))

def save(fig, name):
    fig.savefig(os.path.join(F, name)); plt.close(fig); print("wrote", name)

# ---------- Figure 1: the idea, on four runs that differ only in the schedule ----------
ex = [r for r in rows if r["width"] == 512 and r["window"] == "A" and r["peak"] == 0.008 and r["wd"] == 0 and r["steps"] == 1024 and r["sched"] != "crw"]
LAB = {"const": ("constant", ORANGE), "wsd50": ("hold, then decay", YELLOW), "cos01": ("cosine to 10%", AQUA), "cos0": ("cosine to zero", BLUE)}
ex = sorted(ex, key=lambda r: -sum(r["lr"]))
T = 1024; w0, w1 = int(0.20 * T), int(0.45 * T)
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.55), gridspec_kw={"wspace": 0.52})
a = ax[0]; a.axvspan(w0, w1, color="#efece4", lw=0)
a.text((w0 + w1) / 2, 9.05, "facts\nshown", ha="center", va="top", fontsize=8, color=MUTED)
for r in ex:
    name, c = LAB[r["sname"]]; n = len(r["lr"]); x = np.arange(T - n, T); lr = np.array(r["lr"]) * 1e3
    a.plot(x, lr, color=c, lw=1.8); a.fill_between(x, 0, lr, color=c, alpha=0.10, lw=0)
    t = np.arange(0, T - n + 1); cosw = 0.5 * (1 + np.cos(np.pi * np.clip(t - 20, 0, None) / (T - 20)))   # the part before the facts end, from the schedule's definition
    pre = {"const": np.ones_like(cosw), "wsd50": np.ones_like(cosw), "cos0": cosw, "cos01": 0.1 + 0.9 * cosw}[r["sname"]] * 8.0 * np.minimum(1, t / 20)
    a.plot(t, pre, color=c, lw=1.0, alpha=0.55)
a.annotate("shaded area\n= $S$", xy=(850, 5.3), fontsize=8.5, color=INK, ha="center", va="center")
a.set_xlim(0, T); a.set_ylim(0, 9.2); a.set_xlabel("training step"); a.set_ylabel("learning rate ($\\times10^{-3}$)")
a.set_title("(a) four schedules, same model"); grid(a, "y")
b = ax[1]
for r in ex:
    name, c = LAB[r["sname"]]; k = np.array(r["kept"]); b.plot(np.arange(1, len(k) + 1), k, color=c, lw=1.6)
    b.text(len(k) + 10, max(k[-1], 0.0) + {"cos01": -0.035, "cos0": 0.035}.get(r["sname"], 0), name, color=INK, fontsize=8, va="center")
b.set_xlim(0, 880); b.set_ylim(-0.03, 1.02); b.set_xlabel("steps after the facts"); b.set_ylabel("facts kept"); pct(b.yaxis)
b.set_title("(b) counted in steps: four curves"); grid(b, "y")
c_ = ax[2]
for r in ex:
    name, c = LAB[r["sname"]]; k = np.array(r["kept"]); c_.plot(np.cumsum(r["lr"]), k, color=c, lw=1.6)
    c_.plot([sum(r["lr"])], [k[-1]], "o", ms=5, color=c, mec="white", mew=1.2, zorder=5)
c_.set_xscale("log"); c_.set_xlim(0.03, 6); c_.set_ylim(-0.03, 1.02); c_.set_xlabel("summed learning rate $S$"); pct(c_.yaxis)
c_.set_title("(c) counted in $S$: close to one"); grid(c_, "y")
c_.text(0.045, 0.1, "dots: end of training", fontsize=8, color=MUTED)
save(fig, "fig1_idea.png")

# ---------- Figure 2: one curve for all runs, and held-out predictions ----------
s = N["score1"]; ab = N["ablation"]; FLOOR = 0.0126
z = np.array(N["z"]); lnS = np.array(N["lnS"]); q = N["q"]; W = np.array([r["width"] for r in rows]); y = np.array([r["kept"][-1] for r in rows])
oof = np.array([o["pred"] for o in N["oof"]])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.30})
a = ax[0]; xs = np.exp(z + lnS); g = np.logspace(-1.3, 2.2, 300)
a.plot(g, 1 / (1 + g ** q), color=INK, lw=1.2, zorder=4, label="formula")
for i, (lab, ws) in enumerate(BINS):
    m = np.isin(W, ws); a.scatter(xs[m], y[m], s=13, color=RAMP[i], edgecolor="white", lw=0.4, alpha=0.9, zorder=3 - i * 0.1, label=f"width {lab}")
a.set_xscale("log"); a.set_xlim(0.05, 160); a.set_ylim(-0.03, 1.0); a.set_xlabel("$r \\cdot S$  (rate from the settings $\\times$ summed learning rate)")
a.set_ylabel("facts kept at the end of training"); pct(a.yaxis); a.set_title("(a) all 375 runs on one curve"); grid(a); a.legend(loc="upper right", handletextpad=0.3)
b = ax[1]; b.plot([0, 1], [0, 1], color=INK, lw=0.9, zorder=1)
for i, (lab, ws) in enumerate(BINS):
    m = np.isin(W, ws); b.scatter(oof[m], y[m], s=13, color=RAMP[i], edgecolor="white", lw=0.4, alpha=0.9, zorder=3 - i * 0.1)
b.set_xlim(-0.02, 1.0); b.set_ylim(-0.03, 1.0); b.set_aspect("equal"); b.set_xlabel("predicted, width never seen in the fit"); b.set_ylabel("measured")
pct(b.xaxis); pct(b.yaxis); b.set_title("(b) held-out widths"); grid(b)
b.text(0.97, 0.05, f"error {s['end_z2'] * P:.1f} points\n$R^2$ = {N['oof_R2']:.2f}", ha="right", fontsize=8.5, color=INK, transform=b.transAxes)
save(fig, "fig3_one_curve.png")

# ---------- Figure 3: what each part of the formula buys ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.95})
a = ax[0]
items = [("average of the other widths", s["mean"], GREY), ("$S$ alone, 2 constants", s["lr_clock"], GREY),
         ("full formula with steps\nin place of $S$", s["end_z2_step_clock"], GREY), ("full formula", s["end_z2"], BLUE)]
for i, (lab, v, c) in enumerate(items[::-1]):
    a.barh(i, v * P, height=0.62, color=c); a.text(v * P + 0.4, i, f"{v * P:.1f}", va="center", fontsize=8.5, color=INK)
a.axvline(FLOOR * P, color=INK, lw=0.9, ls=(0, (3, 2))); a.text(FLOOR * P + 0.3, 3.62, f"floor {FLOOR * P:.1f}", fontsize=8, color=MUTED, va="center")
a.set_yticks(range(4)); a.set_yticklabels([i[0] for i in items[::-1]], color=INK); a.set_xlim(0, 25); a.set_ylim(-0.5, 3.9)
a.set_xlabel("error on held-out widths (percentage points)"); a.set_title("(a) against simpler predictors"); grid(a, "x"); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
b = ax[1]
names = {"no ln peak": "peak learning rate", "no ln steps": "run length", "no (ln width)^2": "width, squared", "no ln width": "width",
         "no weight decay": "weight decay", "no ln peak x ln width": "peak $\\times$ width", "no final-lr fraction": "final learning rate", "no ln layers": "depth"}
it = sorted(((names[k], v) for k, v in ab.items() if k in names), key=lambda t: t[1])
for i, (lab, v) in enumerate(it):
    b.barh(i, v * P, height=0.62, color=BLUE if v - s["end_z2"] >= 0.005 else "#b7d3f6"); b.text(v * P + 0.2, i, f"{v * P:.1f}", va="center", fontsize=8.5, color=INK)
b.axvline(s["end_z2"] * P, color=INK, lw=0.9); b.text(s["end_z2"] * P - 0.2, len(it) - 0.35, f"full formula {s['end_z2'] * P:.1f}", fontsize=8, color=MUTED, ha="right", va="center")
b.set_yticks(range(len(it))); b.set_yticklabels([t[0] for t in it], color=INK); b.set_xlim(0, 9.8); b.set_xticks([0, 2, 4, 6, 8]); b.set_ylim(-0.5, len(it) - 0.1)
b.set_xlabel("error with this term removed (points)"); b.set_title("(b) remove one term"); grid(b, "x"); b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
save(fig, "fig5_parts.png")

# ---------- Figure 4: the prediction committed before the run ----------
r2 = N["round002"][0]; pw = s["per_width"]; nw = s["n_per_width"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.62, "width_ratios": [1.15, 1]})
a = ax[0]; ws = sorted(int(k) for k in pw)
px = list(range(len(ws))); a.plot(px, [pw[str(w)] * P for w in ws], "-o", color=BLUE, lw=1.6, ms=5.5, mec="white", mew=1.2, label="error with this width held out")
a.plot([len(ws) + 0.2], [abs(r2["committed_pred"]["end_z2"] - r2["actual"]) * P], "D", color=ORANGE, ms=6.5, mec="white", mew=1.2, label="one run, predicted before it was trained")
a.axhline(FLOOR * P, color=INK, lw=0.9, ls=(0, (3, 2))); a.text(-0.2, FLOOR * P + 0.2, f"floor {FLOOR * P:.1f}", fontsize=8, color=MUTED)
a.axvspan(len(ws) - 0.45, len(ws) + 0.8, color="#f4f2ec", lw=0, zorder=0); a.text(len(ws) + 0.18, 5.2, "outside\nthe table", ha="center", fontsize=8, color=MUTED, va="top")
a.set_xticks(px + [len(ws) + 0.2]); a.set_xticklabels([str(w) for w in ws] + ["2048"]); a.set_xlim(-0.4, len(ws) + 0.8); a.set_ylim(-0.4, 9.8); a.set_yticks([0, 2, 4, 6])
a.set_xlabel("model width"); a.set_ylabel("error (percentage points)"); a.set_title("(a) error by width"); grid(a, "y"); a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, borderaxespad=0.2)
b = ax[1]; rows5 = [("formula of this paper", r2["committed_pred"]["end_z2"], BLUE), ("12 constants", r2["committed_pred"]["hA"], GREY), ("13 constants", r2["committed_pred"]["a_hA"], GREY)]
b.axvline(r2["actual"] * P, color=INK, lw=1.1); b.text(r2["actual"] * P + 0.2, 2.62, f"measured {r2['actual'] * P:.1f}%", fontsize=8.5, color=INK, va="center")
for i, (lab, v, c) in enumerate(rows5[::-1]):
    b.plot([v * P, r2["actual"] * P], [i, i], color="#cfcdc6", lw=1.6, zorder=1); b.plot([v * P], [i], "o", ms=7, color=c, mec="white", mew=1.2, zorder=3)
    b.text(v * P - 0.4, i, f"{v * P:.1f}%", ha="right", va="center", fontsize=8.5, color=INK)
b.set_yticks(range(3)); b.set_yticklabels([r[0] for r in rows5[::-1]], color=INK); b.set_xlim(15.2, 29.5); b.set_ylim(-0.5, 2.9); b.set_xticks([18, 20, 22, 24, 26, 28]); pct(b.xaxis, 100)
b.set_xlabel("facts kept, 503M-parameter run"); b.set_title("(b) three committed predictions"); grid(b, "x"); b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
save(fig, "fig6_prediction.png")

# ---------- Figure 5: steps against summed learning rate, per model size ----------
plt.rcParams.update({"font.size": 8, "legend.fontsize": 7.5})            # the last two figures are one column wide (3.3 in)
cf = N["clock_fair"]; fig, a = plt.subplots(figsize=(3.35, 2.6)); ys = np.arange(len(cf))[::-1]
for y_, c in zip(ys, cf):
    a.plot([c["R2_steps"], c["R2_S"]], [y_, y_], color="#cfcdc6", lw=1.6, zorder=1)
a.scatter([c["R2_steps"] for c in cf], ys, s=26, color=ORANGE, edgecolor="white", lw=0.8, zorder=3, label="steps after the facts")
a.scatter([c["R2_S"] for c in cf], ys, s=26, color=BLUE, edgecolor="white", lw=0.8, zorder=3, label="summed learning rate $S$")
a.set_yticks(ys); a.set_yticklabels([f"width {c['width']}, {c['layers']} layers ({c['runs']} runs)" for c in cf], color=INK)
a.set_xlim(0.4, 1.0); pct(a.xaxis); a.set_xlabel("variation in facts kept explained by one curve")
a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.2); grid(a, "x"); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
save(fig, "fig2_clock.png")

# ---------- Figure 6: hold out each setting in turn ----------
oh = N["other_holdouts"]
it = [("schedule", oh["schedule"]), ("peak learning rate", oh["peak"]), ("depth", oh["layers"]), ("width", {"end_z2": s["end_z2"], "lr_clock": s["lr_clock"]}),
      ("run length", oh["steps"]), ("fact window", oh["window"]), ("weight decay", oh["weight decay"])]
fig, a = plt.subplots(figsize=(3.35, 2.5)); yy = np.arange(len(it))[::-1]
for y_, (lab, d) in zip(yy, it):
    a.plot([d["end_z2"] * P, d["lr_clock"] * P], [y_, y_], color="#cfcdc6", lw=1.6, zorder=1)
    a.text(d["end_z2"] * P, y_ + 0.3, f"{d['end_z2'] * P:.1f}", ha="center", va="center", fontsize=7.5, color=INK)
a.scatter([d["lr_clock"] * P for _, d in it], yy, s=26, color=GREY, edgecolor="white", lw=0.8, zorder=3, label="$S$ alone, 2 constants")
a.scatter([d["end_z2"] * P for _, d in it], yy, s=26, color=BLUE, edgecolor="white", lw=0.8, zorder=3, label="full formula")
a.axvline(FLOOR * P, color=INK, lw=0.9, ls=(0, (3, 2))); a.text(FLOOR * P - 0.2, len(it) - 0.45, "floor", fontsize=7, color=MUTED, ha="right")
a.set_yticks(yy); a.set_yticklabels([f"hold out one {t[0]}" for t in it], color=INK); a.set_xlim(-1.2, 24); a.set_ylim(-0.6, len(it) - 0.2)
a.set_xlabel("error on the held-out group (points)"); a.legend(loc="lower right", handletextpad=0.2); grid(a, "x"); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
save(fig, "fig4_holdouts.png")
