# Guide for students

An AI coding agent trains small models, searches for a formula and writes the paper. You pick the target, check in often and decide what goes in the paper.

## 1. Pick one target

A target is one number about training that a formula should predict. Pick one of these:

- An open problem from [problems/](problems/README.md): **forgetting** (how many facts a model still knows at the end of training) or **final-loss** (the final loss, from the training settings). Data and scoring are ready.
- One direction from [IDEAS.md](IDEAS.md), for example "the best learning rate for each model width".

One target per paper. Two targets split the work, and neither gets done.

Not sure? Skip this step. The AI proposes 8-10 targets and recommends one.

## 2. Start the AI

Paste the prompt from the [README](README.md#copy-this-prompt-to-your-ai-agent) into your coding agent, or just give it the link to this repo.

The AI first asks a few questions: how you want to work, how much it should talk, whether you are a beginner or an experienced researcher, and which decisions you want to make yourself. Then it lists targets. Name yours, even if it is not on the list.

## 3. Choose how to work

**Default: let the AI search on its own.** Use this when you want a formula and a paper:

- **Fully automatic:** the AI does everything and shows you the draft at the end.
- **3 stops:** the AI stops to pick the target, to check the first score, and to approve the draft. Answer "go" to take its recommendation.

The score is how well the formula predicts runs it never saw. "Score 1" uses old runs; "score 2" uses runs predicted before they existed.

**The 4-step loop.** Use this when you want to understand a result, or to try your own idea:

1. **Understand.** Ask the AI to explain the result or idea in plain words. Keep asking until you can say it back in two sentences.
2. **Find what it is good for.** Ask: "What could a lab do with this?" For example: "pick the learning rate without trying ten values".
3. **Check it.** Ask: "Has someone published this? What is the cheapest run that could prove it wrong?"
4. **Do experiments.** Small runs first, then the formula. Then start again at step 1.

For the loop, answer "talk at every stage" to the first question, so the AI stops after each step. You can mix the two: let the AI search alone, and use the loop when a result surprises you.

## 4. Your machine

You do not need to know your hardware. The AI checks it at the start: CPU cores, memory, any Mac or NVIDIA GPU, and how many runs fit at once. It writes this to `work/MACHINE.md` and sizes the runs to fit. A laptop is enough, because all runs are small scale. If you have more machines, tell the AI how to reach them.

## 5. Keep the AI busy

AI agents often slow down: they wait, run one experiment at a time, or leave a GPU idle. Every 15-30 minutes, paste one or more of these:

```text
Are all CPU cores and GPUs busy right now? Show me the numbers. If something is idle, give it work.
```

```text
How many experiments run in parallel right now? Is that the most this machine can take? If not, start more.
```

```text
What did the last batch of runs show? Give me score 1 before and after, in one line.
```

```text
What runs next, and how does it help the score?
```

If the score did not move for 2-3 check-ins, say: "The direction is stuck. Propose 3-5 new directions."

## 6. Optional: a watcher AI

For long runs or several machines, open a second AI session in the same folder. It only keeps the machines busy; the first AI still decides what runs. Paste this, with your topic name:

```text
You are the watcher for work/<topic>/ in training-law-finder. Read agents/watcher.md and agents/ops.md and follow them. Every 15 minutes, check that every machine is busy, restart crashed runs at a smaller size, and report in work/<topic>/LOG.md. Tell me only when a machine stays idle and you cannot fix it.
```

## 7. Before you share

The AI publishes nothing without your yes. Read the draft and ask about any number you do not understand. Then tell the AI where the paper should go ([options](CONTRIBUTING.md#sharing)).

Questions and help picking a target: [AI Research School](https://www.skool.com/become-ai-researcher-2669).
