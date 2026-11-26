"""Fit a candidate law to result rows and score it on held-out slices.

Two kinds of formula:

  linear in terms (fast, any number of terms):
    python3 fit/laws.py results.jsonl --y val_loss --log-y \
        --vars N=nonemb_params D=tokens --terms "log(N)" "log(D)" "log(N)*log(D)"

  any curve with named parameters (least squares):
    python3 fit/laws.py results.jsonl --y val_loss \
        --vars N=nonemb_params D=tokens \
        --curve "E + A*N**(-a) + B*D**(-b)" --p0 E=2 A=20 a=0.3 B=20 b=0.3 \
        --bounds E=0:10 a=0.05:2 b=0.05:2

Columns are the flattened row keys: nested dicts become dotted names, for
example config.d, config.lr, facts.acc. `--vars` gives them short names.

Score 1 (this tool): for each --holdout column, every value (or quantile bin of a
column with many values) is left out in turn, the formula is refit on the rest
and predicts the left-out rows. The R^2 of those predictions is that column's
held-out R^2. Score 1 is the worst of them. In-sample R^2 is printed but is not
the score.

--ceiling also fits gradient-boosted trees on the raw --vars with the same folds.
Their held-out R^2 is a rough ceiling for what any formula of those inputs can
reach; the gap to it says how much is left to find.

--save writes the fitted formula to JSON for fit/prereg.py.
"""
import argparse
import json
import math
import sys

import numpy as np

SAFE = {k: getattr(np, k) for k in ("log", "log2", "log10", "exp", "sqrt", "abs", "tanh",
                                     "minimum", "maximum", "where", "cos", "sin", "pi")}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load_rows(path, where=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = flatten(json.loads(line))
            if r.get("diverged"):
                continue
            if where and not all(str(r.get(k)) == v for k, v in where.items()):
                continue
            rows.append(r)
    return rows


def columns(rows, varmap):
    env = {}
    for short, col in varmap.items():
        env[short] = np.array([float(r[col]) for r in rows])
    return env


def r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 0 else float("nan")


class Law:
    """spec = dict(kind='linear'|'curve', y, log_y, vars, terms | expr+p0)."""

    def __init__(self, spec):
        self.spec = spec
        self.params = None

    def _y(self, rows):
        y = np.array([float(r[self.spec["y"]]) for r in rows])
        return np.log(y) if self.spec.get("log_y") else y

    def _env(self, rows):
        env = dict(SAFE)
        env.update(columns(rows, self.spec["vars"]))
        return env

    def _design(self, rows):
        env = self._env(rows)
        n = len(rows)
        cols = [np.ones(n)] + [np.broadcast_to(eval(t, {"__builtins__": {}}, env), (n,))
                               for t in self.spec["terms"]]
        return np.stack(cols, 1)

    def _curve(self, rows, p):
        env = self._env(rows)
        env.update(dict(zip(self.spec["p0"].keys(), p)))
        out = eval(self.spec["expr"], {"__builtins__": {}}, env)
        out = np.broadcast_to(out, (len(rows),)).astype(float)
        return np.log(np.maximum(out, 1e-12)) if self.spec.get("log_y") else out

    def fit(self, rows):
        y = self._y(rows)
        if self.spec["kind"] == "linear":
            X = self._design(rows)
            lam = self.spec.get("ridge", 1e-8)
            self.params = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y).tolist()
            return self
        from scipy.optimize import least_squares
        names = list(self.spec["p0"].keys())
        p0 = np.array([self.spec["p0"][n] for n in names], dtype=float)
        bnd = self.spec.get("bounds", {})
        lo = np.array([bnd.get(n, [-np.inf, np.inf])[0] for n in names], dtype=float)
        hi = np.array([bnd.get(n, [-np.inf, np.inf])[1] for n in names], dtype=float)
        rng = np.random.default_rng(0)
        best = None
        starts = [p0] + [p0 * np.exp(rng.normal(0, 0.5, p0.shape)) for _ in range(8)]
        for s in starts:
            with np.errstate(invalid="ignore"):
                s = np.clip(s, np.where(np.isfinite(lo), lo + 1e-6 * np.maximum(1, np.abs(lo)), lo),
                            np.where(np.isfinite(hi), hi - 1e-6 * np.maximum(1, np.abs(hi)), hi))
            try:
                with np.errstate(all="ignore"):
                    res = least_squares(lambda p: self._curve(rows, p) - y, s, bounds=(lo, hi),
                                        max_nfev=4000)
            except (ValueError, FloatingPointError):
                continue
            if np.isfinite(res.cost) and (best is None or res.cost < best.cost):
                best = res
        if best is None:
            raise RuntimeError("curve fit failed from every start")
        self.params = best.x.tolist()
        return self

    def predict(self, rows):
        """Prediction on the scale of the raw y column."""
        if self.spec["kind"] == "linear":
            z = self._design(rows) @ np.array(self.params)
        else:
            z = self._curve(rows, np.array(self.params))
        return np.exp(z) if self.spec.get("log_y") else z

    def describe(self):
        if self.spec["kind"] == "linear":
            names = ["const"] + self.spec["terms"]
        else:
            names = list(self.spec["p0"].keys())
        return {n: round(v, 6) for n, v in zip(names, self.params)}

    def to_json(self):
        return dict(spec=self.spec, params=self.params)

    @classmethod
    def from_json(cls, d):
        law = cls(d["spec"])
        law.params = d["params"]
        return law


def folds_for(rows, col, bins=4):
    """Group rows by column value; a column with many values is cut into quantile bins."""
    vals = np.array([r.get(col) for r in rows], dtype=object)
    uniq = sorted(set(vals.tolist()), key=lambda v: (str(type(v)), v))
    if len(uniq) <= 6 or not all(isinstance(v, (int, float)) for v in uniq):
        return [(str(u), np.where(vals == u)[0]) for u in uniq]
    x = vals.astype(float)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((x >= lo) & ((x <= hi) if i == bins - 1 else (x < hi)))[0]
        if len(idx):
            out.append((f"[{lo:.4g},{hi:.4g}]", idx))
    return out


def holdout(make_and_fit, predict, rows, y, cols, min_train=4):
    """Out-of-fold predictions per holdout column. Returns {col: (r2, [(slice, n, rmse)])}."""
    out = {}
    for col in cols:
        if len(folds_for(rows, col)) < 2:
            print(f"held out {col}: skipped, the rows have only one value of it (nothing to extrapolate to)")
            continue
        pred = np.full(len(rows), np.nan)
        detail = []
        for name, idx in folds_for(rows, col):
            train = np.setdiff1d(np.arange(len(rows)), idx)
            if len(train) < min_train:
                continue
            m = make_and_fit([rows[i] for i in train], train)
            pred[idx] = predict(m, [rows[i] for i in idx], idx)
            detail.append((name, len(idx), float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))))
        ok = np.isfinite(pred)
        out[col] = (r2(y[ok], pred[ok]) if ok.sum() > 2 else float("nan"), detail)
    return out


def parse_kv(items, cast=str):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = cast(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--y", required=True)
    ap.add_argument("--log-y", action="store_true", help="fit log(y)")
    ap.add_argument("--vars", nargs="+", required=True, help="short=column")
    ap.add_argument("--terms", nargs="+", help="linear terms, e.g. log(N) log(N)*log(D)")
    ap.add_argument("--curve", help="expression with named parameters")
    ap.add_argument("--p0", nargs="+", help="initial parameter values, name=value")
    ap.add_argument("--bounds", nargs="+", help="parameter bounds, name=lo:hi (e.g. E=0:10 a=0.05:2)")
    ap.add_argument("--holdout", nargs="+", default=[], help="columns to leave out in turn")
    ap.add_argument("--where", nargs="+", help="keep rows with column=value")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args(argv)

    varmap = parse_kv(a.vars)
    if a.terms:
        spec = dict(kind="linear", terms=a.terms)
    elif a.curve:
        spec = dict(kind="curve", expr=a.curve, p0=parse_kv(a.p0, float),
                    bounds={k: [float(x) for x in v.split(":")] for k, v in parse_kv(a.bounds).items()})
    else:
        sys.exit("give --terms or --curve")
    spec.update(y=a.y, log_y=a.log_y, vars=varmap)
    rows = load_rows(a.results, parse_kv(a.where))
    rows = [r for r in rows if all(r.get(c) is not None for c in list(varmap.values()) + [a.y])]
    y = np.array([float(r[a.y]) for r in rows])
    print(f"{len(rows)} rows from {a.results}")

    law = Law(spec).fit(rows)
    print("formula:", spec.get("expr") or " + ".join(["const"] + spec["terms"]),
          "(log y)" if a.log_y else "")
    print("params:", law.describe())
    print(f"in-sample R^2 {r2(y, law.predict(rows)):.4f} (not the score)")

    scores = holdout(lambda tr, idx: Law(spec).fit(tr), lambda m, te, idx: m.predict(te),
                     rows, y, a.holdout)
    ceil = {}
    if a.ceiling and a.holdout:
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.stack([columns(rows, varmap)[k] for k in varmap], 1)
        ceil = holdout(lambda tr, idx: HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=3).fit(X[idx], y[idx]),
                       lambda m, te, idx: m.predict(X[idx]), rows, y, a.holdout)
    for col, (s, detail) in scores.items():
        extra = f"   trees {ceil[col][0]:.4f}" if col in ceil else ""
        print(f"held out {col:<24} R^2 {s:.4f}{extra}")
        for name, n, rmse in detail:
            print(f"    slice {name:<24} n={n:<4} rmse {rmse:.4f}")
    if scores:
        worst = min(scores, key=lambda c: scores[c][0] if math.isfinite(scores[c][0]) else -1e9)
        print(f"SCORE 1 (worst held-out R^2) = {scores[worst][0]:.4f}  [{worst}]")
    if a.save:
        d = law.to_json()
        d["score1"] = {c: round(s, 4) for c, (s, _) in scores.items()}
        d["n_rows"] = len(rows)
        json.dump(d, open(a.save, "w"), indent=1)
        print("saved", a.save)
    return law, scores


if __name__ == "__main__":
    main()
