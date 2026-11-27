# Guide for students

The AI trains small models, searches for a formula and writes the paper. You pick the target, check in, and decide what goes in the paper.

## 1. Pick one target

A target is one number per run that a formula predicts. Options:

- An open problem from [problems/](problems/README.md), with data and scoring ready: **forgetting** or **final-loss**.
- A direction from [IDEAS.md](IDEAS.md), for example "how large the weights grow during training".

A target must have one value per run. "Best learning rate per width" has one value per sweep, so it does not work directly. Predict each run's loss instead; the formula then tells you the best learning rate.

One target per paper. Unsure? Skip this; the AI proposes 8-10 and recommends one.

## 2. Start the AI

Paste the [prompt from the README](README.md#copy-this-prompt-to-your-ai-agent), or give the agent the repo link. It first asks how you want to work (questions in [agents/manager.md](agents/manager.md)), then lists targets. Name yours even if it is not listed.

## 3. Choose how to work

- **Fully automatic:** the AI does everything and shows you the draft.
- **3 stops:** the AI stops to pick the target, at the first score, and to approve the draft. Answer "go" to take its recommendation.
- **The 4-step loop**, to understand a result or try your own idea (answer "talk at every stage"):
  1. **Understand:** ask until you can say it back in two sentences.
  2. **Use:** "What could a lab do with this?" (e.g. pick the learning rate without trying ten values).
  3. **Check:** "Has someone published this? What is the cheapest run that could prove it wrong?"
  4. **Experiment:** small runs, then the formula, then step 1 again.

Modes mix: search automatically, loop when a result surprises you.

## 4. Keep the AI busy

A laptop is enough: the AI measures your machine, writes `work/MACHINE.md` and sizes runs to fit. Tell it how to reach any extra machines. Agents tend to idle or run one experiment at a time, so every 15-30 minutes paste:

```text
Are all CPU cores and GPUs busy, and is the number of parallel experiments the most this machine can take? Show the numbers and fill any idle capacity.
```

```text
What did the last batch show (score 1 before and after, one line)? What runs next, and how does it help the score?
```

If the score has not moved for 2-3 check-ins: "The direction is stuck. Propose 3-5 new directions."

## 5. Optional: a watcher AI

For long runs or several machines, open a second session inside the `training-law-finder` clone (not the empty starting folder). It only keeps machines busy; the first AI still decides what runs. Find your topic with `ls work/`, then paste:

```text
You are the watcher for work/<topic>/ in training-law-finder. Read agents/watcher.md and agents/ops.md and follow them. Every 15 minutes, check that every machine is busy, restart crashed runs at a smaller size, and report in work/<topic>/LOG.md. Tell me only when a machine stays idle and you cannot fix it.
```

A session acts only when it gets a message. To repeat the check, use your agent's loop feature (Claude Code: `/loop 15m <prompt>`) or paste it every 15 minutes.

## 6. Before you share

The AI publishes nothing without your yes. Read the draft, ask about any number you do not understand, then choose where it goes ([options](CONTRIBUTING.md#sharing)).

Help picking a target: [AI Research School](https://www.skool.com/become-ai-researcher-2669).
