#!/bin/sh
# run2.sh NAME : runs jobs/NAME_mps.jsonl on MPS and jobs/NAME_cpu.jsonl on CPU at the same time
cd "$(dirname "$0")"
( nice -n 10 python3 -I blaw.py --jobs jobs/$1_mps.jsonl --out out/$1.jsonl --device mps > out/$1_mps.log 2>&1 &
  nice -n 10 python3 -I blaw.py --jobs jobs/$1_cpu.jsonl --out out/$1_c.jsonl --device cpu --threads 4 > out/$1_cpu.log 2>&1 &
  wait )
tail -n 3 out/$1_mps.log out/$1_cpu.log
