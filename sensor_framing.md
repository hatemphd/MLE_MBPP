# Defining the Sensor

To study trust in AI-generated code, we define the "sensor": **what** we measure, **how**, and **why**.
In ML terms, the sensor is our data-generation and labelling process, its coverage is the kind of code we
sample, and calibrating it means checking our labels against human judgment.

## 1. Sensor specification

| | Definition |
| --- | --- |
| **What we measure** | Is the AI-generated code trustworthy? Yes if it passes all hidden tests and has no serious security issue. |
| **How we measure** | Run the code against MBPP's hidden tests, plus Ruff (lint) and Bandit (security). No AI judges the code. |
| **Calibration** | People review about 150–200 samples; agreement with our labels is measured with Cohen's kappa (target 0.6 or higher). |
| **Output** | A trust probability, turned into **APPROVED**, **REJECTED** or **NEEDS HUMAN REVIEW**. |
| **Known noise** | Only about 2 hidden tests per problem, so some wrong code passes by luck. |
| **Range** | Short, single-function Python problems only. |

**Why this measurement:** it's objective, repeatable, and doesn't depend on an AI's opinion.

## 2. Models

A range of strengths, from more than one family, so we see good and bad code:

| Tier | Model | Role |
| --- | --- | --- |
| Weak | Qwen2.5-Coder-0.5B-Instruct (free) | Many realistic mistakes |
| Medium | Qwen2.5-Coder-1.5B-Instruct (free) | A mix of good and bad |
| Strong | GPT-4o-mini and/or Claude Haiku (API) | Mostly correct; the hard cases that look right but fail |

On CW's personal system, only models that don't need FlashAttention can be used.

## 3. Core design: good and bad examples

We use the standard prompt (problem description plus one example test) and never ask for mistakes.
Variety comes from:

- **Model:** weak, medium and strong.
- **Temperature:** 0.2, 0.8 and 1.2 (more randomness, more mistakes).

All code is labelled by running it. We always evaluate on this naturally generated code, because it's
what AI agents really produce.

## 4. Coverage and bias

The dataset should contain each type of failure:

- syntax errors, crashes and timeouts,
- wrong answers (logic errors, missed edge cases),
- security issues,
- **code that passes the example test but fails the hidden tests** (the most dangerous kind).

Biases to report: Python only, short functions, few hidden tests, limited model families, and MBPP may
already be in the models' training data. Every sample records its model and temperature, so we can check
for bias.

## 5. Extra: prompt variations

Optional, off by default, switched on with `--prompt-styles`.

| Style | What the AI is given | Why |
| --- | --- | --- |
| `standard` (default) | Description plus example test | The core design |
| `signature_only` | Description plus function signature, no example | More misunderstandings |
| `step_by_step` | Asked to reason first, then write the function | Different kinds of mistakes |
| `constrained` | Asked for a concise solution (no imports, no comments) | Different coding style |
| `inject_bug` | Asked to include one subtle, realistic bug | More of the rare failure types |

Rules for injected bugs:

- Used for training only, never for testing.
- Comments are removed so the bug isn't given away.
- Not used for sibling agreement.

```bash
uv run trust-pipeline --num-problems 20 --samples-per-config 5 --prompt-styles standard,signature_only,step_by_step,constrained,inject_bug
```

Each extra style multiplies the dataset size: candidates = problems × model settings × prompt styles ×
attempts.

## 6. Next steps

1. Run a 20-problem pilot with the core design.
2. Check in `run_report.md` that 30–70% of the code is trustworthy and every failure type appears.
3. If some failure types are rare, add the extra prompt styles and rerun.
4. Fix the setup before the full MBPP run.
