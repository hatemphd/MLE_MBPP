# Examples: APPROVED, NEEDS HUMAN REVIEW, REJECTED

Ready-to-run inputs for `trust-recommend`. The outputs below were produced by a **smoke-test demo
model** (MBPP reference solutions plus mutants, no LLM). They show how the tool behaves; they are not
findings.

## 1. Train the demo model (about 4 minutes on CPU)

```bash
uv run trust-pipeline --smoke-test --num-problems 400 --output-dir artifacts_demo
```

Resulting thresholds: **REJECT if p ≤ 0.469, APPROVE if p ≥ 0.924** (95% precision target).
On its test split, 96.4% of APPROVED candidates and 97.0% of REJECTED candidates were labelled correctly.
The 30-problem `artifacts_smoke` model never approves: its calibration split is too small for any
cutoff to reach 95% precision with at least 10 approved rows.

## 2. MBPP task 176: "find the perimeter of a triangle"

Task 176 is in the **test split** of `artifacts_demo`, so the model never trained on it. Its visible test
is `assert perimeter_triangle(10,20,30)==60`.

| File | What it is |
| --- | --- |
| `perimeter_triangle.py` | correct (`a + b + c`) |
| `perimeter_triangle_v2.py`, `perimeter_triangle_v3.py` | correct, written differently |
| `perimeter_triangle_subtle_bug.py` | `2 * c`: passes the visible test, fails the hidden tests |
| `perimeter_triangle_wrong.py` | `a + b`: fails the visible test |

**APPROVED**

```bash
uv run trust-recommend --output-dir artifacts_demo --task-id 176 \
  --code-file examples/perimeter_triangle.py
```
```json
{"p_trustworthy": 0.992, "decision": "APPROVED", "conformal_decision": "APPROVED",
 "visible_pass": 1, "behavior_agreement": null, "ruff_issues": 0, "bandit_med_high": 0}
```

**NEEDS HUMAN REVIEW**: the subtle bug disagrees with every correct sibling

```bash
uv run trust-recommend --output-dir artifacts_demo --task-id 176 \
  --code-file examples/perimeter_triangle_subtle_bug.py \
  --sibling examples/perimeter_triangle.py --sibling examples/perimeter_triangle_v2.py \
  --sibling examples/perimeter_triangle_v3.py
```
```json
{"p_trustworthy": 0.489, "decision": "NEEDS HUMAN REVIEW", "conformal_decision": "NEEDS HUMAN REVIEW",
 "visible_pass": 1, "behavior_agreement": 0.0, "ruff_issues": 0, "bandit_med_high": 0}
```

**REJECTED**

```bash
uv run trust-recommend --output-dir artifacts_demo --task-id 176 \
  --code-file examples/perimeter_triangle_wrong.py \
  --sibling examples/perimeter_triangle.py --sibling examples/perimeter_triangle_v2.py
```
```json
{"p_trustworthy": 0.015, "decision": "REJECTED", "conformal_decision": "REJECTED",
 "visible_pass": 0, "behavior_agreement": 0.0, "ruff_issues": 0, "bandit_med_high": 0}
```

## 3. Custom problem: add two numbers

Files: `add.py` (correct), `add_v2.py`, `add_v3.py` (correct variants), `add_wrong.py` (`a - b`),
`add_broken.py` (uses an undefined name).

```bash
P="Write a function to add two numbers."; V="assert add(1, 2) == 3"
uv run trust-recommend --output-dir artifacts_demo --problem-text "$P" --visible-test "$V" --code-file examples/add.py
#   p 0.824 -> NEEDS HUMAN REVIEW (conformal: APPROVED)
uv run trust-recommend --output-dir artifacts_demo --problem-text "$P" --visible-test "$V" \
  --code-file examples/add_wrong.py --sibling examples/add.py --sibling examples/add_v2.py
#   p 0.036 -> REJECTED
uv run trust-recommend --output-dir artifacts_demo --problem-text "$P" --visible-test "$V" \
  --code-file examples/add_broken.py --sibling examples/add.py --sibling examples/add_v2.py
#   p 0.052 -> REJECTED
```

## 4. Results that look wrong, and why

These come from the smoke-test training data, not from a code bug:

- **The subtle bug on its own is APPROVED** (`perimeter_triangle_subtle_bug.py` with no siblings:
  p 0.981). It passes the visible test and looks clean, so nothing available at decision time exposes
  it. Hidden tests or disagreeing siblings are what catch this kind of bug.
- **Correct code with agreeing siblings scores lower.** `perimeter_triangle.py` with v2, v3 and the
  subtle bug as siblings gets p 0.523 (REVIEW). `add.py` with v2, v3 and `add_wrong.py` gets p 0.433
  (REJECTED). In smoke data each problem has one correct reference and seven mutants, so a group that
  agrees is usually several mutants making the *same* mistake. Among candidates that pass the visible
  test, those with agreement above 0.3 are only 49% trustworthy, against 86% for those that agree with
  nobody. The model learns that agreement is a bad sign. With real LLM samples, where correct answers
  are usually the majority, this should reverse.
- **No siblings at all** (`behavior_agreement: null`) never occurred in training, where every problem
  had 8 candidates. The value is imputed, so treat single-file scores with care.
