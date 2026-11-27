# Ideas, privacy and credit

## Posted ideas are open to everyone

Every idea in [IDEAS.md](IDEAS.md), in a problem folder or in a draft's NEXT-STEPS is open. Anyone may work
on it, and two people may work on the same one. If you both find the same thing, the second result counts
as an independent confirmation. If you find different things, the scores decide.

Before you start, look at what others are doing:

- open pull requests and issues titled "Working on: ..."
- the best scores in each problem's `LEADERBOARD.md`
- the `drafts/` folder

If you start on a posted idea, open a draft pull request or a "Working on: <idea>" issue, so others can see it.

## Want an idea nobody else is working on? Make your own

Posted ideas are public, so someone else may already be on yours. To be safe, make a similar idea of your
own that is not on any list. It takes a few minutes. Ways to do it:

1. **Combine two posted ideas.** Example: forgetting and schedule shape become "does the shape of the
   learning-rate decay change how many facts are forgotten?"
2. **Change the target.** Keep the inputs of a posted problem, but predict something else: the whole
   curve instead of the end point, the loss on rare facts instead of all facts, the step where a fact is lost.
3. **Change the input.** Ask how the same target depends on a setting nobody has tested yet: depth,
   batch size, number of facts, data mix.
4. **Take a topic from the list of 60 in [IDEAS.md](IDEAS.md)** that has no folder yet and turn it into a
   "how does X relate to Y" question.

Write the question as "how does X relate to Y?", not "find the best X". A relation always gives a result,
even when the answer is "no relation".

Then search for the closest published paper and write one line on how yours differs.

## Keeping your idea private until your paper is out

You do not have to tell anyone what you are working on. The license (MIT) lets you work in a private copy.

1. Work on your idea in your own repo, public or private.
2. Write the paper and post it as a preprint (arXiv or Zenodo). The preprint date is your public proof
   that you were first.
3. Then send the code, the formula and the runs back here as a pull request, with a link to the preprint.
   Your formula goes on the leaderboard under your name.

## How credit works

- **First to push wins first place.** The time that counts is the first commit in your pull request (or
  your preprint date), not the merge date.
- **A tie is a confirmation.** A score that does not beat the best by more than the 0.005-per-constant rule
  in [problems/README.md](problems/README.md) is not a new best. It is recorded as a confirmation of the
  earlier result, with your name.
- **Sign your work.** Every formula and score line carries your GitHub name (`--author`). Every runs file
  is named `<github-name>-<date>`.
- **Papers credit what they use.** In a paper, everyone whose finding is one of its claims is offered
  co-authorship. Everyone whose formula or runs it uses is cited, with a link to the leaderboard row or pull
  request.
- **Everyone is listed** in [CONTRIBUTORS.md](CONTRIBUTORS.md).
