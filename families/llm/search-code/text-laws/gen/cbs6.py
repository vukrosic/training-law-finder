"""cbs6: widen the Muon batch grid: width 512 (256k, 1M), d256 at 2M, d32 at 2M, d64 at 4M, d128 at 4M.
4 lr members x0.35..x2.8 of C*sqrt(B/32) for new widths, 3 (x0.7..x2.8) elsewhere."""
import json, math
C = {32: 0.014, 64: 0.0075, 128: 0.0075, 256: 0.005, 512: 0.0035}
SHAPE = {32: 1, 64: 2, 128: 2, 256: 2, 512: 2}
DEV = {512: "mps", 256: "mps", 128: "mps", 64: "cpu", 32: "cpu"}
PRE = {262_144: "cbsq", 1_048_576: "cbs", 2_097_152: "cbsh", 4_194_304: "cbsx"}
plan = [(262_144, 512, B) for B in (4, 8, 16, 32, 64, 128)]
plan += [(2_097_152, 32, B) for B in (8, 16, 32, 64, 128)]
plan += [(1_048_576, 512, B) for B in (8, 16, 32, 64, 128)]
plan += [(4_194_304, 64, B) for B in (16, 32, 64, 128)]
plan += [(2_097_152, 256, B) for B in (16, 32, 64, 128)]
plan += [(4_194_304, 128, B) for B in (16, 32, 64, 128)]
jobs = {"mps": [], "cpu": []}
for D, d, B in plan:
    L = SHAPE[d]; name = f"{PRE[D]}_muon_d{d}L{L}_B{B}"
    c = C[d] * math.sqrt(B / 32)
    mult = (0.35, 0.7, 1.4, 2.8) if d == 512 else (0.7, 1.4, 2.8)
    jobs[DEV[d]].append(dict(pack=name, opt="muon", d=d, L=L, seq=64, batch=B, steps=D // (B * 64), warm=0.05,
                             members=[dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in mult]))
for k, v in jobs.items():
    open(f"jobs/cbs6_{k}.jsonl", "w").write("\n".join(json.dumps(j) for j in v) + "\n")
print({k: len(v) for k, v in jobs.items()})
