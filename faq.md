# FAQ

Short answers about TrustGate. The plain narrative is [goal_and_approach.md](goal_and_approach.md). The full write-up is [project_report.md](project_report.md).

## What is the project trying to do?

Decide, for one AI-written Python solution, whether to **approve** it, **reject** it, or send it to a **person**. The automatic actions have a target of 95% precision: at least 95% of approvals should really be trustworthy, and at least 95% of rejections should really be untrustworthy.

## What does "trustworthy" mean?

The solution passes every hidden test, and Bandit finds no medium or high security issue. A program in a sandbox runs the tests and Bandit. No language model judges the code.

## What did we train on?

968 MBPP problems and 38,720 solutions from `gpt-4o-mini`. Each solution has 42 features that are available before the hidden tests run: the one visible example test, agreement with other attempts at the same problem, Ruff and Bandit counts, code structure, and the model's token log-probabilities.

The hidden-test results are the answer key. They are blocked from the features. The model is trained to predict that answer from the earlier signals.

## How did we train, validate, and test?

Problems are split, not rows. All 40 solutions for one problem stay in the same split, so the test is on problems the model has never seen. Seed 42.

| Role | Split | What it is allowed to do |
| --- | --- | --- |
| Train | 60%: 580 problems, 23,200 rows | Fit the models. Choose which model wins, using five-fold cross-validation inside this split. Calibrate its probabilities, also inside this split. |
| Validate | Calibration, 20%: 194 problems, 7,760 rows | Choose the approve and reject cutoffs. Fit the conformal alternative. The model is not refit here. |
| Test | 20%: 194 problems, 7,760 rows | Score the chosen model and the chosen cutoffs once. Nothing is tuned on these problems. |

Cross-validation always groups by problem (`GroupKFold`), so a problem is never both in the fold that is fit and the fold that is scored.

The test set is harder than the calibration set (64.0% trustworthy versus 72.7%). That is why a cutoff that looks fine on the calibration rows pooled together can miss the target on new problems. The deployed rule checks every held-out group of calibration problems, not the average.

## Which machine learning methods did we try?

Five scorers, all classical models. The label is never an LLM's opinion.

| Method | What it is | Train CV, AUROC |
| --- | --- | --- |
| Majority class | Always predict the common class. A floor. | 0.500 |
| Visible test only | Logistic regression on one bit: did it pass the example? | 0.857 |
| Logistic regression | All 42 features, scaled, class-balanced. | 0.924 |
| Histogram gradient boosting | Tree boosting on the same features. | 0.928 |
| **Random forest** | 400 trees, class-balanced. **This one was selected.** | **0.940** |

Selection used the cross-validation AUROC on the training problems. The random forest won. Its probabilities were then calibrated with isotonic regression, again by grouped cross-validation on the training problems, so a score of 0.9 means about a 90% chance the label is trustworthy.

On the held-out test problems the calibrated forest has AUROC **0.941**. The visible-test baseline has 0.856. At a probability cutoff of 0.5 the forest's precision is 0.892 and its recall is 0.956. That 0.892 is not the project target. The target is the precision of the gate after the cutoffs, below.

## How does a score become approve, reject, or review?

We tried three ways to turn the probability into an action. Only one is deployed.

| Rule | How the cutoffs were chosen | Approval precision on test | Share sent to a person |
| --- | --- | --- | --- |
| Pooled calibration | Most lenient cutoff that hits 95% on all calibration rows together. | 92.6% (misses the target) | 9.3% |
| **Grouped cross-validation** | Most lenient cutoff that hits 95% in **every** held-out group of calibration problems. | **95.1%** | **26.8%** |
| Split conformal (α = 0.1) | A prediction set from the calibration scores. Reported as an alternative. | 90.7% | 6.4% |

The deployed gate approves when p ≥ 0.872, rejects when p ≤ 0.090, and sends everything else to a person. On the test problems that is 52.5% approved (precision 95.1%), 20.7% rejected (precision 95.8%), and 26.8% reviewed.

## What did the human do, and why?

One person, Hatem, reviewed **164** solutions. The review was blind: the code was on the page, the automated label stayed hidden until the row was answered. The review app can run tests, but the official label does not come from the reviewer. It comes from the sandbox.

The human was there for two reasons, and only those.

1. **Check the labels.** Hidden tests can be noisy, and code that returns the wrong answer can still look right. Cohen's kappa on the 164 rows is **0.738** (95% interval 0.62 to 0.84), against a project bar of 0.60. Raw agreement is 88.4%. The reviewer matched passing code 97.1% of the time and wrong-output code only 69% of the time: 16 solutions looked acceptable and failed the tests, and 3 looked wrong and passed. That is why we train on execution, and why a person reading code is not a substitute for running it.
2. **Take the uncertain cases.** When the score is between 0.090 and 0.872, the gate does not guess. A person is the action for that middle. On the test problems that is 2,080 solutions, 26.8%.

The human review is not a training target, and it was not used to pick the model or the cutoffs. A second independent reviewer was not done, so there is no score for agreement between two people.
