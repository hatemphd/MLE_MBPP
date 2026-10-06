# How the Project Works: A Simple Explanation

## In one sentence

MBPP supplies problems and tests, AI models supply the code, running the tests supplies the labels, and
a trained model learns to predict trust **without** the hidden tests, so it can recommend **APPROVED**,
**REJECTED** or **NEEDS HUMAN REVIEW** for new code.

## Step by step

1. **We start with MBPP's ~970 well-defined Python problems.** Each problem comes with a description, a
   correct solution and a few tests. The **tests** are our reference for what "correct" means. The
   correct solution is only used to make sure each problem's tests are valid; it is never shown to the AI,
   and we never compare the AI's code with it, because there are many correct ways to write the same
   function.

2. **We ask several AI models, local and remote, to solve every problem.** Each model writes several
   attempts per problem, at different temperatures (more or less randomness). This gives us a realistic
   mix of good and bad code, and gives each attempt "siblings": other attempts at the same problem to
   compare with.

3. **We run every attempt against the tests.** Each problem's tests are split in two:
   - The **visible test** is the one example shown to the AI in the prompt. Whether an attempt passes it
     becomes a piece of evidence (a feature).
   - The **hidden tests** are never shown to the AI. They decide the label: an attempt is **Good** only
     if it passes all of them and has no serious security problem. Otherwise it is **Bad**.

4. **We now have a dataset.** Each row is one AI-written attempt with its evidence and its Good/Bad
   label.

5. **We turn each attempt into numbers the model can learn from:** code size and complexity, style and
   security warnings, whether it passed the visible test, how much it agrees with its siblings, and how
   confident the AI was while writing it. Hidden-test results are **never** included, because in real
   life nobody has a hidden test suite.

6. **We train and test several machine learning models.** The data is split by problem, so the model is
   always tested on problems it has never seen. We compare logistic regression, random forest and
   gradient boosting against a simple "did it pass the example test?" baseline, and measure accuracy,
   precision, recall, F1 and ROC AUC.

7. **The model gives a recommendation.** It outputs a probability that the code is trustworthy, which
   becomes:
   - **APPROVED** when it's very likely good,
   - **REJECTED** when it's very likely bad,
   - **NEEDS HUMAN REVIEW** when it's uncertain.

8. **People check that the labels make sense.** A team member reviews a sample of attempts to confirm that
   "passes the hidden tests" matches what a person would call trustworthy. The agreement is measured with
   Cohen's kappa.

## Why it's designed this way

- **Why use problems that are already solved?** Because we need to know the right answer for sure.
  MBPP's hidden tests grade every AI attempt automatically; brand-new problems would need hand-written
  tests.
- **Why have the AI write the code instead of using MBPP's solutions?** Because AI-written code is what
  we're studying. MBPP's solutions are all correct and written by people, so they can't teach us how AI
  code goes wrong.
- **Why keep the hidden tests out of the evidence?** If the model could see hidden-test results, it would
  just copy them and score nearly 100%, but it would be useless in real life, where no hidden tests
  exist.
- **One thing to watch:** MBPP is public and from 2021, so some AI models may have memorised it. That
  doesn't affect the labels, because they come from running the code, but it may make the AI look better
  than it would on new problems.

See `High_Level_Explanation.md` for more detail on the data, the features and the metrics.
