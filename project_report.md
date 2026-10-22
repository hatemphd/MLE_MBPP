# TrustGate: a trust gate for AI-generated code

Hatem El-Sebaaly, Shamik Bera, Connor Westcott

October 2026. Numbers are from the full MBPP run in `artifacts_pilot_openai/`
(968 problems, 38,720 solutions from `gpt-4o-mini`). The decision rule in section 6
was refit on 20 October 2026 without new model training and without new API calls.
Machine-readable tables are in `artifacts_pilot_openai/run_report.md`.

## 1. Problem

Agents now write code faster than a review team can read it. The usual gate is weak:
"does it pass the example test I showed the model?" On this dataset that gate is wrong
often enough to matter. Of the solutions that pass the one visible test, **10.6%** still
fail the hidden tests (3,102 solutions). A careful human reading the code is not a
replacement either: reviewing blind, the reviewer agreed with the tests on passing code
(97%) but approved **31%** of the wrong-output code.

TrustGate answers a narrower question than "is this good software?" It estimates whether
a generated solution is trustworthy, meaning it passes every hidden test and has no
medium or high Bandit finding, and it turns that estimate into one of three actions:
**APPROVED**, **REJECTED**, or **NEEDS HUMAN REVIEW**. The hidden tests grade the gate.
They are never used as inputs.

This is the sensor defined in `sensor_framing.md`: an objective, repeatable measurement,
checked against a human review, and limited on purpose to short single-function Python.

## 2. Data and labels

MBPP is about a thousand small Python problems, each with a description, a reference
solution, and a handful of assert tests. SWE-Bench was the original target, but it has
no ready-made set of labelled agent attempts, and each task needs a repository checkout.
MBPP is small enough to generate and grade tens of thousands of attempts. The generation
grid follows the SWE-Bench idea of varying the model configuration.

For each problem the first test is **visible**: it is shown to the model, the way a
developer shows an example. The remaining tests, plus the challenge tests, are
**hidden** and used only to grade.

A solution is labelled trustworthy only when both of these hold:

- it passes every hidden test;
- Bandit reports no medium or high finding.

Ruff findings are features, not part of the label. Nothing in the label comes from
another model judging the code. Execution is a subprocess with a 3 second limit and a
1 GB memory limit per solution.

The label is noisy in a known way. Most problems have only about two hidden tests, so
some wrong code passes by luck, and a few MBPP problems have tests that disagree with
the description. Section 7 measures that against a human.

## 3. Generation

One model, `gpt-4o-mini`, crossed with two temperatures (0.3 and 1.0) and four prompt
styles, five samples each:

| Prompt style | What the model is asked for | Trustworthy |
| --- | --- | --- |
| standard | Problem plus the visible test | 73.8% |
| constrained | Concise, standard library only, no comments | 72.5% |
| step_by_step | Brief reasoning, then the code | 74.5% |
| signature_only | Signature only, no example test | 54.9% |

That is 968 problems × 8 configurations × 5 samples = 38,720 solutions. Generation took
4 hours 43 minutes and about $2.40. Every attempt is cached in `generations.jsonl`, so
reruns do not call the API.

Two generation results matter:

- **Temperature barely matters.** Trustworthy rates at 0.3 and 1.0 differ by less than
  one point inside each prompt style. Prompt style is the useful knob.
- **Removing the example test** (`signature_only`) drops the trustworthy rate from about
  74% to 55%. Those failures are more often visible ones. With the example in the
  prompt, the failures that remain are more often silent.
- **step_by_step costs about five times as many output tokens** (about 239 against
  about 51) for the same trustworthy rate as `standard`.
- **Another sample does not fix a hard problem.** Pass@1 is 68.9%. Pass@5, the chance
  that at least one of the five samples is trustworthy, is only 73.7%. Of 968 problems,
  388 are always solved and 137 are never solved. Failures cluster by problem, which is
  why every train, calibration, and test split is grouped by problem.

A harness bug showed up in the pilot and was fixed before these numbers. `step_by_step`
answers often appended their own asserts. Those asserts were stripped before execution.
Runtime errors in that style fell from 776 to 179. The fix is in `strip_test_code` and
was applied by rebuilding candidates from the cache, with no new API calls.

## 4. The trust model

### Signals

Forty-two features, all available before the hidden tests run:

- the visible test, pass or fail;
- agreement with sibling attempts on the visible test and on probe inputs
  (other samples for the same problem, never the code's own hidden tests);
- Ruff issue counts and Bandit findings;
- structure: AST size and depth, cyclomatic complexity, maintainability;
- the generating model's token log-probabilities and token count.

Three signals carry almost all of the association with the label. Correlation with
trustworthy: visible-test pass 0.78, agreement with siblings on the visible test 0.72,
agreement on probes 0.59. Size, complexity, and lint are weak (|r| ≤ 0.12). Token
confidence is nearly useless for this model (r ≈ 0.03): `gpt-4o-mini` is often
confidently wrong.

### Training

Splits are by problem, 60% / 20% / 20%, seed 42:

| Split | Problems | Rows | Trustworthy |
| --- | --- | --- | --- |
| Train | 580 | 23,200 | 69.3% |
| Calibration | 194 | 7,760 | 72.7% |
| Test | 194 | 7,760 | 64.0% |

No problem appears in two splits. The calibration set is easier than the test set.
That gap is the reason the first decision rule missed its target (section 6).

Candidates were logistic regression, random forest, and histogram gradient boosting,
plus two baselines: predict the majority class, and "trust it if it passes the visible
test." Selection was grouped five-fold cross-validation on the train problems. The
random forest won and was calibrated with isotonic regression.

### Test metrics

Held-out test problems, probability cutoff 0.5. Brackets are 95% bootstrap intervals
from 1,000 resamples of the 194 problems.

| Model | AUROC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Visible test only | 0.856 | 0.872 | 0.964 | 0.916 |
| Logistic regression | 0.920 | 0.895 | 0.905 | 0.900 |
| Gradient boosting | 0.931 | 0.900 | 0.946 | 0.922 |
| **Random forest, calibrated** | **0.941** [0.910, 0.968] | **0.892** [0.849, 0.934] | **0.956** [0.929, 0.977] | **0.923** [0.895, 0.949] |

The useful gap is ranking, not the 0.5 cutoff. TrustGate's AUROC interval sits entirely
above the visible-test baseline. At one cutoff the F1 scores look similar, because both
methods already catch the obvious failures. Ranking is what makes a three-way gate
possible: high scores can be approved, low scores rejected, and the middle sent to a
person.

Permutation importance agrees with the correlations. The visible test dominates, then
the two agreement features. The rest are small.

## 5. What the visible test misses

Passing the example is a strong signal and an incomplete one.

On all 38,720 solutions, 75.6% pass the visible test and 68.9% are trustworthy.
The 3,102 solutions that pass the visible test and fail a hidden test are silent
failures. A visible-test gate approves every one of them.

On the 194 test problems there are 702 silent failures. The visible-test rule's
approval precision on that split is 87.2%: about one in eight approvals is wrong.

Problems are not uniformly hard. Tasks 483 and 537 pass the visible test on essentially
every attempt and fail the hidden tests on every attempt. Those are the problems where
"it passed the example" is most misleading. Untrustworthy solutions usually fail all of
the hidden tests, not most of them. When they crash, the crash is usually a TypeError
(966 solutions), from guessing the input format, especially under `signature_only`.

## 6. The decision rule

The calibrated probability is not yet an action. Two cutoffs turn it into approve,
review, or reject. The target for both automatic actions is **95% precision**: among
code the gate approves, at least 95% should actually be trustworthy, and among code it
rejects, at least 95% should actually be untrustworthy. Anything in between goes to a
human. A target that cannot be met disables that action rather than quietly lowering
the target.

### Why the first cutoffs missed

The first rule took the most lenient cutoff that hit 95% on the calibration rows
pooled together: approve at p ≥ 0.789, reject at p ≤ 0.473. On the calibration average
that was 95%. On the test problems, approval precision was **92.6%**.

Pooling caused part of this. Inside the calibration set, the worst fifth of problems
only reached about 92% precision at that same cutoff, which is what the test set then
did. The rest is a shift: calibration problems are 72.7% trustworthy and test problems
are 64.0%. A cutoff tuned on the easier group is too generous on the harder one.

### Grouped cross-validation

Cutoffs are now chosen on the calibration problems by grouped five-fold
cross-validation. A candidate cutoff is kept only when **every** held-out fold of
problems still meets 95%, and the fold has enough selected rows that a handful of
cases cannot decide it. The deployed cutoff is the most lenient one that survives
that check.

| | Pooled calibration | Grouped CV (deployed) | Conformal (α = 0.1) |
| --- | --- | --- | --- |
| Approve if | p ≥ 0.789 | **p ≥ 0.872** | conformal set |
| Reject if | p ≤ 0.473 | **p ≤ 0.090** | conformal set |
| Approved | 60.4% | **52.5%** (4,075) | 65.7% |
| Rejected | 30.3% | **20.7%** (1,605) | 27.9% |
| Sent to review | 9.3% | **26.8%** (2,080) | 6.4% |
| Approval precision | 92.6% | **95.1%** | 90.7% |
| Rejection precision | 92.3% | **95.8%** | 94.0% |
| Untrustworthy code approved | 345 | **201** | 473 |

On the held-out calibration folds, approval precision at p ≥ 0.872 ranged from 0.951
to 0.986. The test set, which was not used to choose the cutoff, landed at 95.1%.
That meets the target the pooled rule missed.

The cost is review. Automatic decisions fall from about 91% of solutions to about 73%.
Rejection in particular becomes conservative: only scores below 0.09 are rejected
automatically, because a higher bar did not keep 95% precision in every problem fold.
Of the 702 silent failures on the test problems, the pooled rule still approved 345
and the grouped rule approves 201. The other 501 are stopped, almost all by review
(499) rather than by an automatic reject (2).

Conformal prediction is reported as the alternative, not the deployed rule. At α = 0.1
it approves more and reviews less, and its approval precision on this test set is
90.7%, short of the target. The risk–coverage curve in
`artifacts_pilot_openai/figures/risk_coverage.png` is the same trade-off drawn as a
curve: approving the most confident half of the test solutions is about 95.7% precise,
and precision falls as more of the set is decided automatically.

## 7. Human validation

The labels are checked by a person. They are not trained on one.

Hatem reviewed 164 stratified solutions blind, in `human_edit/`, with the code and the
tests runnable from the page. The automated label was hidden until the row was
answered. Agreement with the execution label:

| | |
| --- | --- |
| Raw agreement | 88.4% |
| Cohen's kappa | **0.738**, 95% CI [0.62, 0.84], substantial |
| Human said yes, tests said no | 16 (13 wrong output, 2 timeouts, 1 runtime error) |
| Human said no, tests said yes | 3 (the reviewer's own edge-case checks failed) |
| Agreement on passing code | 97.1% (104 rows) |
| Agreement on wrong-output code | **69%** (42 rows) |

The disagreements are lopsided. Reading the code, the reviewer approved about one in
three solutions that return the wrong answer. The three cases the other way are
evidence of label noise: a few MBPP tests are stricter, or less sensible, than a
reviewer. Both results argue for execution-based labels, and for treating the benchmark
itself as imperfect. Kappa of 0.74 clears the 0.6 target set for the sensor. A second
independent reviewer was not done, so there is no inter-rater kappa.

## 8. Limitations

- **One model family.** Every solution is `gpt-4o-mini`. It rarely emits syntax errors
  or timeouts (about 0.1% each), so the gate is not tested on the crashes a weaker
  model produces. Token confidence, which might help for a weaker model, does not help
  here.
- **Function-level Python only.** MBPP does not exercise repository context, multi-file
  edits, or the security findings Bandit is there to catch. Almost no solution has a
  medium or high Bandit finding, so that half of the label is nearly unused.
- **Benchmark label noise.** Two hidden tests can pass wrong code, and a few problems
  have tests that a reviewer reasonably rejects. The human review measures this. It
  does not remove it. MBPP+ (EvalPlus) would be the direct fix.
- **The calibration shift remains.** Grouped cross-validation stops the cutoff from
  being set by the easiest calibration problems, and on this test split it was enough
  to hit 95%. It does not guarantee the next split of problems will be as hard as this
  one, or as easy. The review share is the slack that absorbs that shift.
- **The self-test artifact.** Models that were asked to reason appended their own
  tests. Those tests were stripped. The numbers in this report are after that fix.
  Any earlier pilot number from before 8 October 2026 is not comparable.
- **The reviewer is one person**, and the review sample is 164 rows, not the full
  38,720.

## 9. Related work

The pieces are not new. The combination is less common: a calibrated score, an explicit
"send to a human" band, and labels that come from running the code rather than from
another model's opinion. Closest published relatives, from `similar_concepts.md`:

- **AlphaCode** (DeepMind, 2022) filters on example tests, then clusters solutions by
  behaviour on generated inputs. That is the visible test plus sibling agreement.
- **MBR-exec** (Shi et al., 2022) and **CodeT** (Microsoft, 2022) also treat agreement
  among attempts as evidence of correctness.
- **LEVER** (2023) trains a verifier on the problem, the code, and execution results.
  That is the closest match to the classifier here.
- **Coder-Reviewer**, **CriticGPT**, and the multi-agent systems (AgentCoder, MetaGPT,
  Reflexion) use another model as the reviewer. TrustGate does not.
- In industry, SonarQube quality gates and CI branch protection are the same three-way
  shape, built from deterministic rules rather than a learned score.

## 10. What was built, and what is left

The pipeline is `trust-*`: prepare, generate, label, features, train, review, recommend,
and eda. The walkthrough notebook reads the saved run and does not call the API.
`clone_and_run.md` is how to run it after a clone, including Colab.

Done: the full run, the self-test fix, real human review, EDA, and a decision rule that
holds 95% precision on unseen problems.

Not done, in the order the data suggests:

1. A second reviewer, for inter-rater kappa.
2. MBPP+, so the hidden tests are harder and the faulty problems are fewer.
3. A second model family, so crashes and security findings actually occur.
4. A smaller model on the three strong signals, which should be easier to explain and
   is unlikely to lose much ranking quality.
5. Repository-level tasks, which is where SWE-Bench comes back.
