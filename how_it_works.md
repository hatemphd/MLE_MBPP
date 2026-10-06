# How It Works

## 1. What the smoke-test command does

```bash
uv run trust-pipeline --smoke-test --num-problems 60 --output-dir artifacts_smoke
```

In short: it builds a small practice dataset from MBPP, labels it by actually running the code against
tests, trains a model to predict which code is trustworthy, and saves everything to `artifacts_smoke/`.
No AI model writes code in this mode. It uses MBPP's correct solutions plus deliberately broken copies,
so it finishes in a few minutes and confirms that every stage works.

### The parts of the command

- **`uv run`**: runs the command in the project's own Python environment.
- **`trust-pipeline`**: the program that runs every stage in order.
- **`--smoke-test`**: practice mode. It uses reference solutions and broken copies instead of
  AI-generated code.
- **`--num-problems 60`**: only uses the first 60 MBPP problems, to keep it quick.
- **`--output-dir artifacts_smoke`**: keeps these results separate from real results in `artifacts/`.

### Step by step

**1. Load the problems.** Each MBPP problem has a description, a correct reference solution and a few
`assert` tests. One test becomes the **visible** test (the example an AI would be shown). The rest are
**hidden** tests that only the grader uses. Problems whose own reference solution fails are dropped.

**2. Create candidate solutions.** In a real run an AI model writes these. In smoke mode, each problem
gets its correct reference solution plus several **mutants**: copies with one small automatic bug, such
as `<` changed to `<=`, `+` to `-`, or `5` to `6`. That gives a mix of good and bad code.

**3. Run each candidate against the tests.** Each candidate runs in a separate, isolated process with a
time limit, so infinite loops or crashes can't freeze anything. The pipeline records whether it passes
the visible test, whether it passes all the hidden tests, and what it returns on a few made-up inputs
(created by tweaking the visible test's inputs, for step 6). Some mutants still pass every test because
the bug didn't matter for those inputs. That's fine: labels always come from running the code, never from
assuming a mutant is wrong.

**4. Check code quality and security.** Ruff checks style and common mistakes, and Bandit checks for
security risks. The pipeline counts what each tool finds.

**5. Assign the label (the answer key).** A candidate is **trustworthy** only if it passes all the hidden
tests and has no serious security findings. No AI judges anything.

**6. Turn each candidate into numbers (features).**

- Size and complexity of the code: lines, loops, conditions, nesting, cyclomatic complexity.
- Lint and security issue counts.
- Whether it passes the visible test.
- How often its outputs on the made-up inputs agree with other candidates for the same problem.

Hidden-test results are deliberately left out of the features. They are the answer, and you won't have
them when the model is used for real.

**7. Simulate the human review check.** A sample of candidates gets fake reviewer verdicts that mostly
agree with the labels, plus some noise. The pipeline then measures agreement (Cohen's kappa). This only
shows how the real human check will work; the fake verdicts are never used for training.

**8. Split the data by problem.** About 60% of problems go to training, 20% to calibration (tuning the
decision thresholds) and 20% to the final test. All candidates for one problem stay in the same group, so
the model is always tested on problems it has never seen.

**9. Train and compare models.** The candidates are a "trust it if it passes the visible test" baseline,
logistic regression, random forest and gradient boosting. The best one is chosen by cross-validation on
the training group, then calibrated so that a score of 0.8 means the code is trustworthy about 80% of the
time.

**10. Evaluate on the unseen test problems.** It reports accuracy, precision, recall, F1, ROC AUC and
calibration measures, and compares them with the visible-test-only baseline to show whether the model
adds anything beyond "did it pass the example?".

**11. Make the three-way decision.** Using the calibration group, it picks two cutoffs:

- **APPROVED** if the probability is high enough that approved code is correct about 95% of the time.
- **REJECTED** if it's low enough that rejected code is wrong about 95% of the time.
- **NEEDS HUMAN REVIEW** for everything in between.

It also runs an alternative method, conformal prediction, and draws a risk–coverage chart. That chart
shows the trade-off between how much code is decided automatically and how many mistakes those automatic
decisions make.

**12. Save everything to `artifacts_smoke/`.** Datasets (CSV), the trained model
(`trust_model.joblib`), the numbers (`metrics.json`) and the charts (`figures/`).

**What to take from it:** if it finishes without errors and creates those files, your setup works. The
scores will look very high because automatically broken code is easy to catch, so don't report them as
results. Real numbers come from the same command without `--smoke-test`, using AI-generated code.

## 2. Testing new code

You test new code with `trust-recommend`. You need a trained model first; the smoke test creates one in
`artifacts_smoke/`, so add `--output-dir artifacts_smoke` until you have a real model in `artifacts/`.

### Option 1: an existing MBPP problem

MBPP task 17 is "Write a function to find the perimeter of a square."

`good.py`

```python
def square_perimeter(a):
    return 4 * a
```

`bad.py`

```python
def square_perimeter(a):
    return 2 * a
```

```bash
uv run trust-recommend --output-dir artifacts_smoke --task-id 17 --code-file good.py
uv run trust-recommend --output-dir artifacts_smoke --task-id 17 --code-file bad.py
```

The tool finds the problem's description and visible test itself.

### Option 2: your own problem

`add.py`

```python
def add(a, b):
    return a + b
```

```bash
uv run trust-recommend --output-dir artifacts_smoke \
  --problem-text "Write a function to add two numbers." \
  --visible-test "assert add(1, 2) == 3" \
  --code-file add.py
```

The function name in the code has to match the one in the test (`add` here), because the test calls it.
`def (a, b):` with no name is a syntax error and comes back REJECTED.

### What happens when you run it

1. It runs the code against the visible test in an isolated process.
2. It runs Ruff and Bandit and measures size and complexity.
3. It builds the same features the model was trained on.
4. The model produces a probability that the code is trustworthy.
5. That probability is compared with the thresholds tuned during training: **APPROVED** if high enough,
   **REJECTED** if low enough, **NEEDS HUMAN REVIEW** in between.

It does **not** use hidden tests. In real use you won't have them, and that is the situation the model is
meant to handle.

### Reading the output

```json
{
  "p_trustworthy": 0.848,
  "decision": "NEEDS HUMAN REVIEW",
  "conformal_decision": "APPROVED",
  "visible_pass": 1,
  "behavior_agreement": null,
  "ruff_issues": 0,
  "bandit_med_high": 0
}
```

| Field | Meaning |
| --- | --- |
| `p_trustworthy` | The model's calibrated probability that the code is trustworthy. |
| `decision` | The main two-cutoff decision (95% precision target). |
| `conformal_decision` | The conformal prediction decision (about 90% overall coverage guarantee). |
| `visible_pass` | 1 if the code passed the visible test, 0 if not. |
| `behavior_agreement` | Share of sibling files that gave the same outputs on the made-up inputs. `null` if none were given. |
| `ruff_issues` | Number of Ruff (lint) issues. A syntax error counts as one. |
| `bandit_med_high` | Number of medium or high severity Bandit (security) findings. |

**Examples from testing `add.py`:**

- **Broken version** (`def (a,b):`): `p_trustworthy` 0.061, REJECTED by both methods, `visible_pass` 0,
  `ruff_issues` 1 (the syntax error).
- **Fixed version:** `p_trustworthy` 0.848, `visible_pass` 1, no lint or security issues. The main
  decision is NEEDS HUMAN REVIEW and the conformal decision is APPROVED.

**Why the two decisions can disagree:** the main method only approves when approved code is correct at
least 95% of the time. On the small smoke-test data no cutoff met that bar, so approval is effectively off
and anything not clearly bad goes to a human. Conformal prediction targets a 90% overall guarantee, which
is less strict. In practice, use the stricter main decision. A disagreement means the code is in the
uncertain zone, which is exactly where human review adds the most.

**Why 0.848 and not closer to 1.0:**

- The smoke model learned from broken copies of reference solutions, and many of those also pass the
  visible test, so passing one example isn't enough for high confidence.
- No sibling files were given, so there was no agreement signal.
- A one-line `add` function looks different from the MBPP solutions it trained on.

A model trained on real AI-generated data has a larger calibration set, so a 95%-precision approve cutoff
can exist.

## 3. Sibling files

Sibling files are **other attempts at the same problem**, saved as separate `.py` files. The tool
compares your main solution with them. When several attempts give the same outputs they are usually
right, because there are many ways to be wrong but only one right answer. A solution that disagrees with
most of its siblings is suspicious.

**How the comparison works:** the tool takes the visible test (`add(1, 2)`), creates a few similar inputs
by tweaking the numbers (`add(2, 2)`, `add(0, 2)`, `add(1, 3)`), runs your solution and every sibling on
them, and reports the share of siblings whose outputs match yours as `behavior_agreement`. It never uses
the hidden tests.

### Example

`add_v2.py`

```python
def add(a, b):
    return sum([a, b])
```

`add_v3.py`

```python
def add(a, b):
    result = a
    result += b
    return result
```

`add_wrong.py`

```python
def add(a, b):
    return a * b
```

```bash
uv run trust-recommend --output-dir artifacts_smoke \
  --problem-text "Write a function to add two numbers." \
  --visible-test "assert add(1, 2) == 3" \
  --code-file add.py \
  --sibling add_v2.py --sibling add_v3.py --sibling add_wrong.py
```

`add.py` agrees with `add_v2` and `add_v3` on every input but not with `add_wrong`, so
`behavior_agreement` is about 0.67 (2 out of 3). Scoring `add_wrong.py` as the main file gives a low
agreement score as well as a failed visible test.

### Things to know

- In the pipeline, siblings come for free: `--samples-per-config 5` generates 5 attempts per model
  setting, so every candidate has siblings during training. `--sibling` lets you supply them when scoring
  by hand.
- Siblings must solve the same problem with the same function name.
- More siblings make the signal more reliable: 2–3 at minimum, 5 or more is better.
- If every sibling makes the same mistake, they agree on the wrong answer. That's why agreement is only
  one feature among several.

## 4. Worked examples

[`examples/README.md`](examples/README.md) has ready-to-run files and exact commands, with real outputs,
that produce each decision: **APPROVED**, **NEEDS HUMAN REVIEW** and **REJECTED**. They use a demo model
trained with:

```bash
uv run trust-pipeline --smoke-test --num-problems 400 --output-dir artifacts_demo
```

These are smoke-test (reference + mutant) numbers that demonstrate the tool, not findings. One
smoke-data artifact matters for siblings: in that data, agreement with siblings is a *negative* signal,
because siblings are mostly mutants. That's the opposite of what this section describes for real LLM
samples. The examples README explains it.
