# Small to big: useful LLM science from CPU runs

**Question.** How can runs on a few CPUs (models under about 1M parameters, a few million tokens) produce findings that hold for big LLMs (100M to 100B+ parameters)?

Started 2026-11-28 by Vuk. This folder is documentation only: the methods we try, the big-model numbers we check against, and the score of each method. The runs happen on the rented CPU boxes (see `work/boxes/`), never on the Mac.

## The idea in one sentence

A CPU cannot train a big model, but it can measure how a rule changes across many small sizes, explain why it changes, and then check that rule against numbers big labs already published.

## How we know a method works

A method is trusted only after it passes a **calibration test**: we use it on a question whose big-model answer is already known (for example Chinchilla's "about 20 tokens per parameter"), with CPU runs only, and see if it recovers that answer. A method that recovers known big-model laws from small runs earns the right to predict unknown ones.

Score of a method = how many known big-model laws it recovers (direction right, exponent within a stated margin), out of how many it was tried on.

## Files

- [METHODS.md](METHODS.md): the methods, each with how it works, what it can and cannot reach, and its calibration test.
- [ANSWER-KEY.md](ANSWER-KEY.md): published big-model numbers and laws we check against. Values from training data are marked "verify" until checked against the paper.
- [TRIALS.md](TRIALS.md): every trial: method, box, question, result, verdict.
- [STRATEGY.md](STRATEGY.md): rules for which box runs which question, and the current plan.

## What a CPU cannot reach

We skip effects that only appear at big scale: emergent abilities, loss spikes and instabilities of huge runs, routing with hundreds of experts, long-context behaviour beyond a few thousand tokens, hardware-level throughput. A finding that depends on any of these is out of scope.

## End product

A short guide, "How to do LLM science that scales on a CPU", with the methods that passed calibration, their scores, and one worked example (cpu-3's learning-rate rule). It goes in `blog/` when at least two methods have passed.
