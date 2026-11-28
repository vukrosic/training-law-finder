"""cbs7: Adam batch grid on CPU (widths 32, 64): 64k tokens B2-128, d32 at 2M and 4M, d64 at 2M.
4 lr members x0.7..x5.6 of C*sqrt(B/32) (cbs3 showed the Adam optimum near x1.4-x2.8)."""
import json, math
C = {32: 0.003, 64: 0.002}
SHAPE = {32: 1, 64: 2}
PRE = {65_536: "cbst", 2_097_152: "cbsh", 4_194_304: "cbsx"}
plan = [(65_536, d, B) for d in (32, 64) for B in (2, 4, 8, 16, 32, 64, 128)]
plan += [(2_097_152, 32, B) for B in (8, 16, 32, 64, 128)]
plan += [(2_097_152, 64, B) for B in (8, 16, 32, 64, 128)]
plan += [(4_194_304, 32, B) for B in (8, 16, 32, 64, 128)]
jobs = []
for D, d, B in plan:
    L = SHAPE[d]; name = f"{PRE[D]}_adam_d{d}L{L}_B{B}"; c = C[d] * math.sqrt(B / 32)
    jobs.append(dict(pack=name, opt="adam", d=d, L=L, seq=64, batch=B, steps=D // (B * 64), warm=0.05,
                     members=[dict(tag=f"{name}_lr{m}", lr=round(c * m, 6)) for m in (0.7, 1.4, 2.8, 5.6)]))
open("jobs/cbs7_cpu.jsonl", "w").write("\n".join(json.dumps(j) for j in jobs) + "\n")
open("jobs/cbs7_mps.jsonl", "w").write("")
print(len(jobs))
