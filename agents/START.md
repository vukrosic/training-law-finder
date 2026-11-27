# Start here (for AI agents)

This repo helps a student write a paper with one new formula about AI training.
The formula predicts one number (for example the final loss) and is tested on runs it never saw (score 1) and on runs predicted before they existed (score 2).
Runs are small scale, so a laptop is enough. You are the student's AI agent and the manager of the project.

You may have been given only the GitHub link, or the prompt in README.md. Either way, do these steps in order.

0. **Resume or role first.** If the student pasted a role prompt (for example the watcher prompt from GUIDE.md), follow that role's file (for example [watcher.md](watcher.md)) and skip the steps below. If `work/STUDENT.md` or `work/*/STUDENT.md` exists, a project is already running: read it, `work/MACHINE.md` and the last lines of `work/<topic>/LOG.md`, then pick up where it stopped. Do not ask the start questions again and never overwrite STUDENT.md. If you cannot run commands, tell the student to open Claude Code or Codex and paste the prompt from README.md.
1. **Get the repo.** Clone https://github.com/vukrosic/training-law-finder. If it is already cloned, run `git pull --rebase --autostash` in it. Pull again at every 15-minute check.
2. **Read** README.md, GUIDE.md, RESEARCH.md, IDEAS.md, problems/README.md, METHOD.md, agents/README.md, agents/manager.md and agents/WORKFLOW.md. GUIDE.md is what the student reads; it tells you what they expect from you.
3. **Ask the five start questions** in [manager.md](manager.md#first-ask-how-the-student-wants-to-work), all in one short message. Skip a question the student's request already answers: "choose everything for me" means fully automatic mode. Keep to the answers for the whole project and write them to `work/STUDENT.md`. Do not train anything yet.
4. **Find out the machine** as in [manager.md](manager.md#know-the-machine) and write `work/MACHINE.md`. Tell the student in one line what you found.
5. **Propose 8-10 options** by the rule in [manager.md](manager.md#starting-the-project-propose-ideas-then-talk-them-through). For each, one line each on: the number the formula would predict, why a formula might exist, the fastest test, and the closest published work and how this differs. Recommend one. Ask which ideas they like, what they would change or combine, and whether they have an idea of their own. If the student already named a target, check that one the same way instead of listing others.
6. **Pick together.** In fully automatic mode you pick. Then follow [WORKFLOW.md](WORKFLOW.md) stage by stage. Keep all project files in `work/<topic>/`, which git ignores.

Talk to the student short and simple the whole time: plain words, short sentences, no jargon. Keep each option to 4 short lines and the whole list under 500 words. Give details only when asked.

If the student has not seen GUIDE.md, say in one line that it explains how to work with you and how to check in.
