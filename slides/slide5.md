# Slide 5 — Why these features were selected

This slide answers **why these inputs go into the trust model**, and why we did not just pile on every possible metric.

The rule at the top matters. Every feature must be knowable **before** the hidden tests run. Hidden tests are the answer key for training and grading; they are not allowed as inputs when the gate scores new code.

## 1. Visible test pass/fail

Did the solution pass the one example shown in the prompt? That is today’s usual check. It is the strongest single clue (r = 0.78). We include it so the model has to beat “just trust the example,” not ignore it.

## 2. Sibling agreement

For the same problem we have other AI attempts. Do they behave the same on the example and on probe inputs? If this solution is a lucky one-off, agreement drops. That is the second-strongest signal (r ≈ 0.59–0.72).

## 3. Ruff + Bandit

Lint and security scans that do not need the hidden tests. Bandit medium/high is also part of the **label**. On MBPP almost nothing trips Bandit, but we keep it for real “factory” code where security findings matter.

## 4. AST / complexity (radon)

Cheap structure stats: size, depth, cyclomatic complexity, maintainability. They barely correlate with trustworthiness (|r| ≤ 0.12). Kept because they are cheap, readable, and a weak proxy for structural mess — not because they win alone.

## 5. Token log-probabilities

How confident the generator was in its own tokens. We hoped self-confidence would flag bad code. For `gpt-4o-mini` it almost does not (r ≈ 0.03): the model is often confidently wrong. Kept for completeness and for weaker models later.

## Bottom line

Features were chosen to cover different failure modes you can see **before deployment**, not to maximize feature count. Behavior (visible test + siblings) carries almost all the weight; lint, complexity, and token confidence are secondary.
