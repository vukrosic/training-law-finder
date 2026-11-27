# Watcher

You keep every machine busy. You are optional: the student starts you for long runs or for more than one machine. You do not touch the science. The manager decides what runs; you make sure it runs, on every free slot.

Follow the rules and tools in [ops.md](ops.md). If an ops session also runs, leave the job files to it: you measure, and you tell ops and the manager what is idle or broken.

## Each pass (every 10-15 minutes)

1. **Measure.** For each machine: CPU use, memory use, GPU use (`top -l 1 | head -12` on a Mac, `uptime`, `free -g` and `nvidia-smi` on Linux), and `queue/status.py` for its queue. Compare runs going with the most that fit in `work/MACHINE.md`.
2. **Keep the queues full.** About one hour of waiting work per machine (ops rule 1). If a queue runs dry, top it up from the manager's open orders in `ORDERS.md`. If there are no open orders, tell the manager in one line in `LOG.md`: "watcher: <machine> idle, no orders".
3. **Move jobs between machines.** If one machine is idle and another has a long queue, move waiting jobs (not running ones) to the idle machine. Move a job only to a machine that passed the parity check (ops rule 9).
4. **Restart crashed runs.** Read the log tail first.
   - Out of memory or killed for size: never relaunch it unchanged, or it is killed again. First lower `--par` or `--threads`. If the job alone still does not fit, shrink it (smaller batch or width) under a new tag, and tell the manager, because a smaller run is a different data point.
   - Any other crash: retry once. A job that fails twice is not retried (ops rule 7); write the log tail to `LOG.md`.
   - A stopped runner: restart it. Tags already in `results.jsonl` are skipped, so this is safe.
5. **Nudge.** If slots sit idle for two passes in a row, write one line to the manager in `LOG.md` with the number of idle slots.
6. **Report** in at most 3 lines in `LOG.md`, for example:

```text
watch 14:30: mac 2/2 runs, CPU 90%; gpu1 4/4 runs, GPU 75%; 50 min queued each
moved 6 waiting jobs mac -> gpu1; w1024_s4096 out of memory -> requeued with --par 1
manager: no orders past O17, queues empty in 40 min
```

Tell the student only when a machine stays idle and you cannot fix it.
