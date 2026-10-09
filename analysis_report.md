# Analysis Report: Agentic Trust for AI-Generated Code

*Sections 1–9 analyse the expanded OpenAI pilot (100 MBPP problems, 4,000 candidates). Section 0 adds
the full MBPP run and the real human review (8 Oct 2026).*

> Human-review numbers in sections 1–9 are **simulated** (pipeline demonstration only). Section 0 has the
> **real** human review. All other numbers come from real `gpt-4o-mini` code executed against MBPP's
> hidden tests.

## 0. Update: full MBPP run and real human review

Full details and commands are in `run_log.md`; the full report is in
`artifacts_pilot_openai/run_report.md`.

**Full dataset:** 968 problems, 38,720 candidates, 2 temperatures × 4 prompt styles × 5 samples, about
$2.40 of OpenAI usage. Results are after the self-written-test fix (section 6.1):

| Item | Value |
| --- | --- |
| Trustworthy rate | 68.9% |
| Silent failures (pass visible test, fail hidden) | **10.6% of visible passes** (5.9% in Pilot 2) |
| Test ROC AUC, calibrated random forest (194 unseen problems) | **0.941 [0.910, 0.968]** |
| Test ROC AUC, visible test only | 0.856 |
| Test F1 / precision / recall | 0.923 / 0.892 / 0.956 |
| Threshold rule: approve / reject / review | 60.4% / 30.3% / 9.3% |
| Threshold rule: approval precision | **92.6%** (target 95%) |
| Conformal: approval precision / review share | 90.7% / 6.4% |
| `step_by_step` runtime errors, before → after the fix | 776 → 179 |

**Trust model vs. baseline on the 194 held-out test problems:**

| Metric | Trust model, value [95% CI] | Visible-test-only baseline |
| --- | --- | --- |
| **AUROC** | **0.941** [0.910, 0.968] | 0.856 |
| **Precision** | 0.892 [0.849, 0.934] | 0.872 |
| **Recall** | 0.956 [0.929, 0.977] | 0.964 |
| **F1** | 0.923 [0.895, 0.949] | 0.916 |

- **Trust model:** the calibrated random forest, which combines the visible test, sibling agreement,
  model confidence, code structure and lint findings.
- **95% CI:** the range the score falls in across 1,000 bootstrap re-draws of the test problems.
  Narrow means reliable.
- **Baseline:** "trust it if it passes the one example test". This is the rule to beat.
- Precision, recall and F1 use a 0.5 probability cutoff. AUROC measures ranking across all cutoffs.

**Reading it:** the model's advantage is ranking. Its AUROC of 0.941 is well above the baseline's 0.856,
which lies outside the model's whole confidence interval. At a single 0.5 cutoff the two look similar:
precision is a bit higher and recall a bit lower. That ranking is what makes the three-way decision
work: confident cases are decided automatically, and uncertain ones go to human review. The decision
rule's own quality (92.6% approval precision) is reported separately above.

**What "trust model vs. visible-test-only" means.** Both answer the same question, "can we trust this
AI-generated code?":

| | Visible-test-only (baseline) | Trust model (ours) |
| --- | --- | --- |
| Signal used | Only whether the code passes the one example test, which the AI model also saw | The example test **plus** sibling agreement, model confidence, code structure and lint or security findings |
| Output | Yes or no | A trust score from 0 to 1, turned into APPROVED / NEEDS HUMAN REVIEW / REJECTED |
| Weakness | Approves every silent failure: 10.6% of code that passes the example still fails hidden tests | Still misses some failures (approval precision 92.6% against a 95% target) |

Neither method sees the hidden tests; those are used only to grade the methods. Example: five AI
attempts at one problem all pass the example test.

- The **baseline** approves all five.
- The **trust model** checks whether they behave the same on extra inputs. The three that agree get
  high scores and are approved. The two that disagree get low scores and go to review or are rejected,
  which catches silent failures the baseline would approve.

In one sentence: *passing the example test is a good first signal, but combining it with agreement
between independent attempts lets the trust model flag risky code that the example test alone would
approve.*

**Observations from the exploratory data analysis** (`uv run trust-eda --output-dir
artifacts_pilot_openai`; figures and tables in `artifacts_pilot_openai/eda_report.md`):

1. **Prompt style matters, temperature barely does.**
   - Trustworthy rates differ by less than 1 point between temperature 0.3 and 1.0.
   - Removing the example test (`signature_only`) drops trustworthiness from about 74% to about 55%.
     It also has the fewest silent failures (8% against 11–12%): without the example, mistakes are more
     often visible ones.
2. **Problems are mostly all-or-nothing.**
   - 388 of 968 problems are always solved and 137 are never solved; 124 are solved 1–50% of the time
     and 319 are solved 51–99%.
   - Failures cluster by problem, which is why splits must be by problem.
3. **Some problems are silent-failure traps.** Tasks 483 and 537, for example, pass the visible test
   100% of the time but never pass the hidden tests. These are where the visible-test rule is most
   misleading.
4. **Failing code is usually completely wrong, not almost right.** Most untrustworthy candidates pass 0%
   of the hidden tests; a second group passes about half (MBPP usually has two hidden tests).
5. **Crashes are mostly argument-type mistakes.** `TypeError` dominates hidden-test errors (966
   candidates), far ahead of `ValueError` (113) and `NameError` (52). This happens when the model
   guesses the input format, especially in `signature_only`.
6. **Three signals carry most of the information.** Correlation with the trustworthy label:
   - `visible_pass`: r = 0.78
   - `behavior_agreement_visible`: r = 0.72
   - `behavior_agreement`: r = 0.59
   
   Code size, complexity and lint counts are weak (|r| ≤ 0.12). A smaller model built on the strong
   signals should explain the results almost as well.
7. **Model confidence is a weak signal for `gpt-4o-mini`.** Token log-probabilities overlap heavily
   between trustworthy and untrustworthy code. Median mean log-prob is −0.031 for trustworthy code
   against −0.050 for untrustworthy code, and the correlation with the label is only r ≈ 0.03. The model
   is often confidently wrong.
8. **`step_by_step` costs about 5× more tokens** (about 238 against 42–57 output tokens per answer)
   for roughly the same trustworthiness as `standard`.

**Real human review:** 164 stratified rows, reviewed blind by Hatem.

| Item | Value |
| --- | --- |
| Raw agreement with the automated label | **88.4%** |
| Cohen's kappa | **0.738**, 95% CI [0.62, 0.84]: substantial |
| Human approved code that fails hidden tests | 16 rows (13 wrong output, 2 timeouts, 1 runtime error) |
| Human rejected code that passes hidden tests | 3 rows (reviewer's own edge-case checks failed) |
| Agreement on passing code / wrong-output code | 97.1% / **69.0%** |

**What changed compared with the pilot:**

- With 10× more problems, **silent failures nearly doubled** to 10.6%. The visible-only baseline drops to
  0.856 ROC AUC, while the trust model holds at 0.941. The gap between them is now clear and outside
  the confidence intervals.
- **Thresholds still miss the 95% target** (92.6%), but by less than in the pilot (89.9%). The larger
  calibration split (194 problems) helped. The remaining gap is the main open technical issue.
- **The automated labels are validated by humans** (kappa 0.74). Disagreements are lopsided: the
  reviewer was more lenient than the tests, approving about 1 in 3 wrong-output candidates. That is the
  core argument for execution-based trust signals instead of review alone. The few strict-human cases
  point to questionable benchmark tests (label noise).

## 1. Summary

- **The dataset is now usable.**
  - 71% of candidates are trustworthy, inside the 30–70% target band.
  - Every failure type appears except timeouts, including **178 silent failures**: code that passes the
    visible test but fails hidden tests (5.9% of visible-test passes).
- **The trust model ranks code better than the simple baseline.**
  - Cross-validated ROC AUC is 0.964, against 0.908 for "trust it if it passes the visible test".
  - Sibling agreement now contributes, alongside the visible test.
- **The three-way decision is not yet safe enough.**
  - The threshold rule targeted 95% precision on approvals but reached only **89.9%** on the test
    problems, worse than the visible-test-only rule (96.0%).
  - Conformal prediction did better: 96.8% approval precision, with 12.7% sent to human review.
- **Two findings explain most errors:**
  - One MBPP problem whose description contradicts its tests caused 38 of the 62 bad approvals.
  - A small calibration set (20 problems) produced thresholds that didn't transfer to the test problems.
- **One pipeline artifact needs fixing.** Self-written `assert` lines in generated code crash otherwise
  valid solutions, causing 79 false runtime errors in the `step_by_step` style.

## 2. What has been done so far

1. **Reviewed the original notebook.** It crashed at training, its labels leaked into the features, and
   the split wasn't by problem.
2. **Rebuilt the pipeline** as the `trust_pipeline` package plus `trust-*` commands and a walk-through
   notebook. It has:
   - labels from hidden tests and Bandit only, with hidden-test results never used as features,
   - a train / calibration / test split by problem,
   - calibrated models,
   - a three-way decision (two thresholds plus conformal prediction),
   - human-agreement checking with Cohen's kappa.
3. **Moved the project to uv**, and added clear progress logging, run reports (`run_report.md`,
   `runs_history.csv`), optional prompt styles, and `.env` support for API keys.
4. **Verified the pipeline** with a smoke test (reference solutions plus mutants) and a demo model with
   APPROVED / REVIEW / REJECTED examples.
5. **Ran two real pilots** with OpenAI `gpt-4o-mini`. Both used two temperature settings (0.3 and 1.0)
   and 5 attempts per setting.

| Run | Problems | Prompt styles | Rows | Trustworthy | Silent failures* | Test ROC AUC (model / visible-only) | Approve precision | Review share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke test | 60 | none | 447 | 19% | 13% | 1.00 / 1.00 | – (approval off) | 13% |
| Pilot 1 | 20 | standard | 200 | 76% | 0% | 1.00 / 1.00 | 100% | 20% |
| **Pilot 2** | **100** | **4 natural styles** | **4,000** | **71%** | **5.9%** | **0.967 / 0.947** | **89.9%** | **2%** |

\* Share of visible-test passes that fail at least one hidden test.

Pilot 2 took 32 minutes (27 of them generating) and used about 388,000 output tokens and 98,000 input
tokens.

## 3. The dataset (Pilot 2)

### 3.1 Prompt styles worked as intended

| Prompt style | Trustworthy | Runtime errors | Wrong output | Silent failures | Notes |
| --- | --- | --- | --- | --- | --- |
| standard | 76.0% | 4 | 230 | 50 | Baseline behaviour |
| signature_only | 57.1% | 86 | 333 | 27 | No example test: the model guesses argument types (genuine `TypeError`s) |
| step_by_step | 73.7% | 105 | 157 | 52 | Most runtime errors are a harness artifact (section 6.1) |
| constrained | 77.5% | 13 | 206 | 49 | Some syntax errors from over-compact code |
| **all** | **71.1%** | **208** | **926** | **178** | Plus 7 syntax errors and 16 security findings; **0 timeouts** |

- **`signature_only`** produced the most realistic variety: it lowered the pass rate by about 19 points
  and created genuine misunderstandings.
- **Temperature** barely matters for `gpt-4o-mini`. Pass@1 is almost identical at 0.3 and 1.0, but
  temperature 1.0 gives more diverse attempts: pass@5 is 83% against 78% for the standard prompt.

### 3.2 Visible vs. hidden tests

| | Hidden pass | Hidden fail |
| --- | --- | --- |
| Visible pass | 2,851 | **178** |
| Visible fail | 8 | 963 |

- The visible test is a strong signal but not sufficient: **5.9% of visible passes are silent
  failures**, steady across styles (4–7%).
- These are the cases the trust model exists to catch.
- With only about 2 hidden tests per MBPP problem, the true silent-failure rate is likely
  **under-counted**.

### 3.3 Failures cluster by problem

Most failures come from a handful of problems where nearly every attempt fails the same way. That points
to problems the model misreads, or problems whose description and tests disagree, rather than random
mistakes. Section 5.2 has the main example.

## 4. Model results

### 4.1 Model comparison

Grouped cross-validation on the 60 training problems:

| Model | ROC AUC | Average precision | F1 |
| --- | --- | --- | --- |
| Majority baseline | 0.500 | 0.659 | 0.787 |
| Visible-test-only baseline | 0.908 ± 0.060 | 0.915 | 0.954 |
| Logistic regression | 0.937 ± 0.042 | 0.958 | 0.924 |
| **Random forest (selected)** | **0.964 ± 0.025** | **0.978** | 0.924 |
| Gradient boosting | 0.947 ± 0.040 | 0.968 | 0.921 |

On the 20 held-out test problems, with 95% confidence intervals resampled by problem:

| Model | ROC AUC | PR AUC | Precision @0.5 | Recall @0.5 | F1 @0.5 |
| --- | --- | --- | --- | --- | --- |
| Visible-test-only | 0.947 [0.885, 0.992] | 0.959 | **0.960** | 0.998 | **0.979** |
| Logistic regression | **0.984** [0.965, 0.997] | **0.993** | **0.979** | 0.909 | 0.943 |
| Random forest (calibrated, selected) | 0.967 [0.921, 0.997] | 0.986 | 0.899 | 0.970 | 0.933 |

**Interpretation:**

- The model **ranks** code better than the baseline (higher ROC AUC and PR AUC). That's what a
  three-way decision needs.
- At a fixed 0.5 cutoff, the visible-test baseline has higher accuracy and F1, because it's a near-perfect
  rule for 94% of cases. The model's value is in separating the uncertain cases, not the easy ones.
- Logistic regression did best on the test problems, but its confidence intervals overlap the random
  forest's. With 20 test problems, the models can't be reliably separated yet.

### 4.2 What the model uses

Permutation importance:

| Feature | Importance |
| --- | --- |
| `visible_pass` | 0.148 |
| `behavior_agreement_visible` | 0.037 |
| `behavior_agreement` | 0.013 |
| `probe_error_frac` | 0.002 |
| AST depth, token log-probabilities, call counts | about 0.001–0.002 each |
| Code length, complexity, problem length | about 0 or slightly negative (noise) |

**Sibling agreement now helps,** unlike in the smoke test. Trustworthy code agrees more with its siblings,
and the model uses it. Code-size and complexity features add nothing on MBPP's short functions.

### 4.3 By prompt style (test set)

ROC AUC is 0.94–0.98 for every style. The model generalises across prompt styles; precision is lowest on
`signature_only` (0.82), the hardest style.

## 5. Three-way decision analysis

### 5.1 Results on the test problems

| Metric | Threshold rule | Conformal (α = 0.10) | Visible-test-only rule (for reference) |
| --- | --- | --- | --- |
| Approved | 77.0% | 66.4% | 74.3% |
| Rejected | 21.0% | 20.9% | 25.7% |
| Needs human review | 2.0% | 12.7% | 0% |
| Approval precision | **89.9%** (target 95%) | **96.8%** | 96.0% |
| Rejection precision | 96.4% | 97.0% | – |
| Untrustworthy code approved | 62 | **17** | about 24 |

- **The threshold rule missed its target.** Thresholds were chosen on the calibration problems
  (approve if p ≥ 0.514, reject if p ≤ 0.397), leaving almost no review band. The calibration split was
  86.4% trustworthy against 71.4% for test: with only 20 problems per split, the problem mix differs a
  lot, so the thresholds didn't transfer.
- **Conformal prediction is the better decision rule right now.** It slightly beat the visible-test rule
  on approval precision (96.8% vs 96.0%) and approved less bad code (17 vs about 24), at the cost of
  sending 12.7% of code to humans.

### 5.2 Why bad code was approved (62 cases, threshold rule)

1. **38 cases from one problem: MBPP task 704.** The description says "harmonic sum of **n−1**", but
   the tests expect the sum up to **n**. `gpt-4o-mini` follows the description, every attempt agrees
   with its siblings (agreement 0.90), and all fail the tests, including the visible test. The model
   approved them at p ≈ 0.58 because strong sibling agreement outweighed the visible-test failure.
   **This is label noise from the benchmark, not a model error in the usual sense.** A human reviewer
   would likely call this code correct for the stated task.
2. **24 silent failures from 4 problems.** All were approved by the threshold rule. The model did rank
   them lower (mean p = 0.66, against 0.86 for good visible-passing code), but the approve threshold of
   0.514 was too low. Conformal prediction sent half of them (12) to human review.

## 6. Issues found in the pipeline

### 6.1 Self-written test code crashes valid solutions

In the `step_by_step` style, `gpt-4o-mini` often appends its own example, for example
`assert find_missing([1, 2, 3, 5], 4) == 4`. That line runs when the code is loaded, so if the model's
own example is wrong, the whole candidate is labelled a runtime error, even when the function itself
might be right. **79 of the 105 `step_by_step` runtime errors are failing self-written `assert`s.**

**Fix (implemented, `strip_test_code` in `trust_pipeline/generation.py`):** top-level test code
(`assert`, `print` and other bare expressions, loops, `if __name__ == "__main__":` blocks, and
assignments that call the candidate's own functions) is removed. Imports, definitions and constants
are kept. This is standard in HumanEval and MBPP harnesses. It is applied when candidates are built,
including cached ones, so no new API calls are needed.

**Effect on Pilot 2:** 300 of the 4,000 candidates changed, all `step_by_step`. Re-running the 105
`step_by_step` runtime errors on the cleaned code gave:

| New outcome | Count | Meaning |
| --- | --- | --- |
| Wrong output | 76 | The function was wrong; the model's own example had caught it |
| Runtime error | 25 | Genuine crashes |
| Pass | 4 | Correct code that had been mislabelled |

So the fix mainly corrects the *type* of failure; only 4 labels flip. One more finding: when the model's
own example fails, the function is almost always wrong. "Run the agent's own tests" is a legitimate
trust signal and could become a feature.

### 6.2 Minor reporting gap

When an error happens only in the hidden tests, `exec_error` is empty. The error type is only in
`hidden_results`. This affects error analysis, not labels.

## 7. Recommendations (in priority order)

1. **Fix the harness artifact (section 6.1).** Done; re-label from the cache after the full run. No API
   cost.
2. **Use conformal prediction as the primary decision rule** for now, or make threshold selection
   robust: choose thresholds by cross-validation across problems instead of one 20-problem calibration
   split, and check the achieved precision on test.
3. **Scale to all of MBPP (~970 problems).** Confidence intervals are wide and calibration is unstable
   with 20 problems per split.
   - At Pilot 2's rate, the full run takes about 4–5 hours of generation.
   - At current `gpt-4o-mini` list prices it should cost a few dollars; check the usage dashboard.
   - Use the same output folder, so the 100 problems are reused.
4. **Handle benchmark label noise explicitly:**
   - Add **MBPP+** (EvalPlus), which has many more tests per problem and corrects faulty MBPP problems.
     It should reveal more silent failures and remove contradictions like task 704.
   - Use the **real human review** to flag ambiguous problems, and report how much label noise they
     cause.
5. **Add weaker models (Qwen on Colab)** to get syntax errors, crashes and timeouts, which
   `gpt-4o-mini` rarely produces.
6. **Simplify the feature set.** Code-size and complexity features add nothing; a smaller model is
   easier to explain, and logistic regression already matches the random forest.
7. **Optional:** run `--prompt-styles all` to add bug injection (training only), and test whether it
   helps catch silent failures on natural test data.

## 8. Key findings for the report

- On MBPP, passing the visible test is a strong but incomplete trust signal: **5.9% of visible passes
  are silent failures**.
- A trust model using sibling agreement improves ranking over the visible test (CV ROC AUC **0.964 vs
  0.908**).
- **Prompt variation matters more than temperature** for producing varied, realistic failures. Removing
  the example test (`signature_only`) cut the pass rate from 76% to 57%.
- **Calibration must be done across problems.** Thresholds tuned on 20 problems missed the 95% precision
  target on new problems (89.9%). Conformal prediction held up better (96.8%).
- **Benchmark label noise is real:** one MBPP problem with contradictory description and tests caused
  61% of the bad approvals. Human review is essential to separate model errors from benchmark errors.
- `gpt-4o-mini` can be **confidently and consistently wrong**: failing code had high token confidence
  and high agreement between attempts.

## 9. Next steps

```bash
# 1. Full MBPP run (same folder, reuses the 100 problems), with real human review
uv run trust-pipeline --no-local-hf --openai --all-problems --prompt-styles natural --samples-per-config 5 --human-feedback real --output-dir artifacts_pilot_openai

# 2. If that run started before the harness fix: rebuild candidates from the cache (no API calls),
#    then re-label and retrain
uv run trust-generate --output-dir artifacts_pilot_openai
uv run trust-label --output-dir artifacts_pilot_openai
uv run trust-features --output-dir artifacts_pilot_openai
uv run trust-train --output-dir artifacts_pilot_openai
```

Back up `artifacts_pilot_openai/generations.jsonl` before and after the full run.
