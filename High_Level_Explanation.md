# High-Level Explanation: Agentic Trust for AI-Generated Code

## 1. The project in one sentence

MBPP supplies problems and tests, AI models supply the code, running the tests supplies the labels, and a
trained model learns to predict trust **without** the hidden tests, so it can recommend **APPROVED**,
**REJECTED** or **NEEDS HUMAN REVIEW** for new code.

## 2. The pipeline step by step

1. **Problems (MBPP).** About 970 well-defined Python problems, each with a description, a correct
   reference solution and test checks. The **tests** are the real reference. We never compare AI code
   with MBPP's solution, because there are many correct ways to write a function. The reference solution
   is only used to check that each problem's tests are valid, and in smoke-test mode.
2. **Code generation.** Several AI models and settings (temperatures) each write several attempts per
   problem. This deliberately produces a mix of good and bad code, and gives each attempt **siblings** to
   compare against.
3. **Testing and labelling.**
   - The **hidden tests decide the label**: an attempt is *trustworthy* only if it passes all hidden
     tests and has no serious security finding. Otherwise it is *not trustworthy*.
   - The **visible test result is a feature, not part of the label**, because in real use you have an
     example test but no hidden test suite.
4. **Dataset.** One row per AI-written solution, with its features and its label.
5. **Features.** Each solution becomes numbers: code size and complexity, lint and security counts,
   whether it passed the visible test, agreement with siblings, and the AI's own confidence. Hidden-test
   results are never features.
6. **Training and evaluation.** Split by problem, so the model is tested on problems it has never seen.
   Train logistic regression, random forest and gradient boosting, compare them with a "passed the
   visible test" baseline, and measure accuracy, precision, recall, F1 and ROC AUC.
7. **Three-way recommendation.** The model's probability becomes **APPROVED** (very likely good),
   **REJECTED** (very likely bad) or **NEEDS HUMAN REVIEW** (uncertain).
8. **Human check.** People review a sample to confirm that the automatic labels match what a person would
   call trustworthy, measured with Cohen's kappa.

## 3. The MBPP data

### What's in it

974 problems in four parts: train (374), test (500), validation (90) and prompt (10). We combine all
four and make our own split by problem. Each problem has six fields; task 11 is shown as an example.

| Field | What it is | Task 11 |
| --- | --- | --- |
| `task_id` | Unique number | `11` |
| `text` | The problem in plain English | "Write a python function to remove first and last occurrence of a given character from the string." |
| `code` | Correct reference solution written by a person | `def remove_Occ(s,ch): ...` |
| `test_list` | Usually 3 `assert` checks | `assert remove_Occ("hello","l") == "heo"` and two more |
| `test_setup_code` | Code to run before the tests | Empty for all but 2 problems |
| `challenge_test_list` | Extra, harder tests | Only 11 problems have any; task 11 has 2 |

### Where the visible and hidden tests come from

Both come straight from MBPP. For task 11:

```python
# test_list
assert remove_Occ("hello","l") == "heo"             # 1st -> VISIBLE test
assert remove_Occ("abcda","a") == "bcd"             # HIDDEN
assert remove_Occ("PHP","P") == "H"                 # HIDDEN

# challenge_test_list
assert remove_Occ("hellolloll","l") == "helollol"   # HIDDEN
assert remove_Occ("","l") == ""                      # HIDDEN
```

- **Visible test:** the first entry in `test_list`. It is shown to the AI in the prompt, and the model may
  use whether it passes.
- **Hidden tests:** the rest of `test_list` plus all of `challenge_test_list`. They are never shown to the
  AI and never used as features; they only produce the label.

Most problems have no challenge tests, so they end up with 1 visible and 2 hidden tests.

### How we load it

1. Download with Hugging Face `datasets` (`load_dataset("google-research-datasets/mbpp")`). It is cached
   locally and falls back to the cache when offline.
2. Combine the four parts into one pool.
3. Take the first N problems (`--num-problems`) or all of them (`--all-problems`).
4. Split each problem's tests into visible and hidden.
5. Run each reference solution against its tests and drop any problem whose own answer fails, so a wrong
   test can't produce a wrong label.
6. Save to `problems.json` for the later stages.

### How each field is used

| Field | Used for |
| --- | --- |
| `text` | The prompt: "Write a single Python function that solves: ..." |
| Visible test | The prompt (as an example), the `visible_pass` feature, and the source of made-up inputs for sibling agreement |
| Hidden tests | The label only |
| `code` | **Never shown to the AI.** Validating problems, and the correct candidate in smoke-test mode |
| `test_setup_code` | Run before the tests for the 2 problems that need it |
| `task_id` | Keeps all attempts for a problem in the same split |

### Why already-solved problems, and why AI-written code

- **Already-solved problems** give a reliable answer key for free: MBPP's hidden tests label every AI
  attempt automatically, with no hand-written tests.
- **AI-written code** is what we're studying. MBPP's reference solutions are all correct and written by
  people, so on their own they would teach nothing about how AI code goes wrong.

## 4. Features vs. labels

| | Features (the evidence) | Label (the answer key) |
| --- | --- | --- |
| Comes from | The candidate code, the visible test, static analysis, sibling agreement, model confidence | **The hidden tests** plus Bandit security findings |
| Examples | `visible_pass`, `ruff_issues`, code length, complexity, `behavior_agreement` | `label_trustworthy` = 1 if every hidden test passes and there is no serious security finding |
| Used for | Model input | What the model learns to predict, and what it's graded against |

**Why hidden tests must never be features:** the model would just copy them, score nearly 100% and be
useless in practice, where no hidden test suite exists. The pipeline refuses to train if any hidden-test
or label column slips into the features.

## 5. Key terms

In this project, **"positive" means trustworthy**.

### Temperature (an AI-generation setting)

A dial that controls how random the AI's writing is.

- **Low (0 to 0.2):** predictable and repetitive; asking twice gives nearly the same code.
- **High (0.8 to 1.2):** more varied and creative, but more often wrong.

**Why we use it:** to get a realistic mix of good and bad code, and to make the several attempts per
problem differ from each other, which sibling agreement needs.

### Cohen's kappa (agreement between two judges)

A score for how much two judges agree, **after removing the agreement expected by pure luck**.

Plain percent agreement can mislead. If 90% of the code is bad, a lazy reviewer who always says "bad"
agrees with the automatic labels 90% of the time without looking. Kappa corrects for that.

| Kappa | Meaning |
| --- | --- |
| 1.0 | Perfect agreement |
| Above 0.8 | Excellent |
| 0.6–0.8 | Good |
| 0.4–0.6 | Moderate |
| 0 | No better than chance |
| Below 0 | Worse than chance |

**Why we use it:** to check that the automatic labels match what a human would call trustworthy. A kappa
of about 0.7 or higher on around 150 reviewed samples makes the labels credible. A low kappa means
"passes the tests" doesn't mean what we think, which is a finding in itself.

### The four outcomes behind every metric

| | Actually trustworthy | Actually not trustworthy |
| --- | --- | --- |
| **Model says trustworthy** | True positive (TP): correct approval | **False positive (FP): bad code approved** (dangerous) |
| **Model says not trustworthy** | False negative (FN): good code rejected (wasteful) | True negative (TN): correct rejection |

**Running example:** 100 test solutions, 40 truly good and 60 truly bad. The model approves 35, of which
30 are truly good, so TP = 30, FP = 5, FN = 10 and TN = 55.

### Accuracy

- **Definition:** share of all predictions that were right: (TP + TN) / total = (30 + 55) / 100 =
  **85%**.
- **Purpose:** a quick overall score.
- **Caution:** misleading when classes are unbalanced. If 90% of code is bad, always saying "bad" scores
  90% while being useless.

### Precision

- **Definition:** of the code the model approved, how much was actually good: TP / (TP + FP) = 30 / 35 =
  **86%**.
- **Plain English:** "When it says trust this, can I believe it?"
- **Purpose:** the **safety** metric. Low precision means bad code gets approved, which is why the
  APPROVED threshold targets 95% precision.

### Recall

- **Definition:** of all the truly good code, how much the model recognised: TP / (TP + FN) = 30 / 40 =
  **75%**.
- **Plain English:** "Does it find the good code, or throw a lot of it away?"
- **Purpose:** the **efficiency** metric. Low recall means good code is rejected or sent to humans
  unnecessarily.
- **Trade-off:** a stricter model raises precision and lowers recall; a looser one does the reverse.

### F1 score

- **Definition:** the balance of precision (P) and recall (R), their harmonic mean:
  2 × P × R / (P + R) = 2 × 0.86 × 0.75 / (0.86 + 0.75) ≈ **0.80**.
- **Plain English:** "How good is it at both approving the right things and not missing good ones?"
- **Purpose:** one number for comparing models. It is only high when precision and recall are both high,
  so a model can't score well by being extreme in one direction.

### ROC AUC

- **Definition:** the probability that, given one randomly chosen good and one randomly chosen bad
  solution, the model gives the good one a higher trust score. It ranges from 0.5 (coin flip) to 1.0
  (perfect ranking).
- **Plain English:** "How well does the model rank good code above bad code, whatever cutoff we choose?"
- **Purpose:** the other metrics depend on a chosen cutoff (for example "approve above 0.5"). AUC judges
  the scores themselves across every possible cutoff, so it is the best way to compare models; the
  pipeline uses it to choose the best model in cross-validation.
- **The ROC curve:** for every cutoff, the share of good code correctly approved (vertical axis) against
  the share of bad code wrongly approved (horizontal axis). The more it bulges toward the top-left, the
  better. AUC is the area under it.
- **Reading it:** 0.9 or above is excellent, 0.8–0.9 good, 0.7–0.8 fair, 0.5 useless.

### Summary

| Term | Question it answers in this project |
| --- | --- |
| Temperature | How do we get a realistic mix of good and bad AI code? |
| Cohen's kappa | Do our automatic labels match human judgment? |
| Accuracy | How often is the model right overall? |
| Precision | When it approves code, is that code really good? (safety) |
| Recall | Does it recognise most of the good code? (efficiency) |
| F1 | Balanced overall quality on the trustworthy class |
| ROC AUC | How well does it rank good above bad, independent of any cutoff? (model selection) |

**For the three-way decision:** high precision on APPROVED keeps bad code from slipping through, high
precision on REJECTED keeps good code from being thrown away, and everything uncertain goes to
**NEEDS HUMAN REVIEW**.

## 6. Limitations to mention in the report

- **Few tests per problem:** most problems have only 2 hidden tests, so some wrong code passes by luck and
  labels are slightly generous.
- **Contamination:** MBPP is from 2021, so models may have memorised it. That inflates pass rates but not
  the labels, because they come from running the code. LiveCodeBench is the remedy if it matters.
- **Messy problems:** some descriptions are unclear and function names are odd. A cleaner "sanitized"
  MBPP subset (427 problems) exists.
- **Subtle bugs:** a bug that passes the visible test can't be caught by any information available before
  the hidden tests run, so some bad code will still be approved.
- **Smoke and simulated numbers:** smoke-test and simulated-human-review results show the pipeline works.
  They are not findings.
