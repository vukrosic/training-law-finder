#!/bin/sh
# runpair.sh MPSNAME CPUNAME SECONDS : jobs/MPSNAME_mps.jsonl on MPS and jobs/CPUNAME_cpu.jsonl on CPU at once, each stops before a pack that would overrun
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.22 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12
( nice -n 10 python3 -I blaw.py --jobs jobs/$1_mps.jsonl --out out/$1.jsonl --device mps --budget $3 >> out/$1_mps.log 2>&1 &
  nice -n 10 python3 -I blaw.py --jobs jobs/$2_cpu.jsonl --out out/$2_c.jsonl --device cpu --threads 6 --budget $3 >> out/$2_cpu.log 2>&1 &
  wait )
tail -n 2 out/$1_mps.log out/$2_cpu.log
