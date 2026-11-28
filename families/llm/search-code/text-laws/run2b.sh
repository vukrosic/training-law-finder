#!/bin/sh
# run2b.sh NAME SECONDS : like run2.sh, but each side stops before a pack that would overrun SECONDS (resume by rerunning)
cd "$(dirname "$0")"
( nice -n 10 python3 -I blaw.py --jobs jobs/$1_mps.jsonl --out out/$1.jsonl --device mps --budget $2 >> out/$1_mps.log 2>&1 &
  nice -n 10 python3 -I blaw.py --jobs jobs/$1_cpu.jsonl --out out/$1_c.jsonl --device cpu --threads 4 --budget $2 >> out/$1_cpu.log 2>&1 &
  wait )
tail -n 2 out/$1_mps.log out/$1_cpu.log
