"""spec_dial.py PREFIX: best val (over hidden lr and, if tried, embedding lr E) for each spectral power p, per width, batch
and step count. Marks: '^'/'v' = best hidden lr at top/bottom of its grid, 'E^'/'Ev' = same for the embedding lr."""
import json, glob, sys, collections
pre = sys.argv[1] if len(sys.argv) > 1 else "spec1"
G = collections.defaultdict(list)
for f in glob.glob(f"out/{pre}*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("val") is None: continue
        t = r["tag"]; m = float(t.split("_lr")[-1]); E = float(t.split("_E")[-1].split("_")[0]) if "_E" in t else None
        G[(r["d"], r["batch"], r["steps"], r["pexp"])].append((r["val"], m, E))
for d, B, st in sorted({k[:3] for k in G}):
    ps = sorted(k[3] for k in G if k[:3] == (d, B, st))
    line, vals = [], {}
    for p in ps:
        rows = G[(d, B, st, p)]; v, m, E = min(rows); vals[p] = v
        ms = [x[1] for x in rows]; Es = [x[2] for x in rows if x[2] is not None]
        e = "^" if m == max(ms) else ("v" if m == min(ms) else " ")
        if E is not None: e += "E^" if E == max(Es) else ("Ev" if E == min(Es) else "  ")
        line.append(f"p{p:<4} {v:.4f} x{m:<4}" + (f"E{E:<6}" if E is not None else "") + e)
    best = min(ps, key=vals.get)
    print(f"d{d} B{B} steps {st:5d}: best p={best}  " + " | ".join(line))
