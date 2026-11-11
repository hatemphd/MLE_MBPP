# Run report `20261111-075050-d4f62a`

> **SIMULATED HUMAN FEEDBACK.** Human-agreement numbers come from a simulated reviewer and are not a finding.

## Run

| Item | Value |
|---|---|
| Run id | 20261111-075050-d4f62a |
| Mode | real |
| Human feedback | simulated |
| Started | 2026-11-11T07:50:50-05:00 |
| Finished | running / incomplete |
| Wall-clock duration | – |
| Problems requested | 200 |
| Model configs | phi4_mini_it, phi4_mini_it, gemma4_12b_it, gemma4_12b_it |
| Prompt styles | standard |
| Samples per config | 5 |
| Seed | 42 |
| Output dir | `/run/media/khan/5TB_HDD/FIT_PhD/CSE_5695_MLE/TermProject/MLE_MBPP/artifacts` |
| Commands | trust-prepare --num-problems 200; trust-generate --samples-per-config 5; trust-label; trust-features; trust-train |
| Python / platform | 3.12.15 / Linux-6.12.0-211.74.1.el10_2.x86_64-x86_64-with-glibc2.39 |
| GPU | not checked |
| Package versions | scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, torch 2.14.1, transformers 5.19.0, datasets 5.1.0, openai 3.24.0, anthropic 1.11.0 |

## Stage timings

| Stage | Time | Seconds |
|---|---|---|
| prepare | 4s | 4.0 |
| generate | 25m 48s | 1547.6 |
| execute | 34s | 33.5 |
| static | 2s | 2.1 |
| features | 2s | 1.7 |
| human | 0s | 0.1 |
| **total** | 26m 29s | 1589.0 |

## Generation (per model config and prompt style)

New / reused / failed refer to the latest generate session; tokens and latency cover every sample in the dataset that recorded them. Local models generate the samples for a problem in one batch, so seconds per sample is the batch time divided by the batch size.

| Config | Model | Temp | Prompt style | Provenance | Problems | Samples | New | Reused | Failed | Out tokens (total) | Out tokens (mean) | In tokens (total) | s / sample | tokens / s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi4_mini_it | ../models/phi_4_mini_it | 1.0 | standard | natural | 200 | 2,000 | 0 | 1,000 | 0 | 107052 | 53.5 | 47260 | 0.28 | 188.9 |
| gemma4_12b_it | ../models/gemma_4_12B_it | 1.0 | standard | natural | 200 | 2,000 | 0 | 1,000 | 0 | 235848 | 117.9 | 57030 | 1.26 | 93.3 |

## Execution outcomes and labels

Outcome on the hidden tests. Trustworthy = passes every hidden test and has no medium/high-severity bandit finding. Config = model config, plus `/style` for non-standard prompt styles.

Provenance: natural 4,000. `natural` = LLM output from a normal prompt; `injected` = LLM asked for a subtle bug (training only); `mutant` / `reference` = mutated / original reference solutions.

| Config | Temp | Prompt style | Provenance | Rows | pass | wrong_output | runtime_error | timeout | syntax_error | generation_error | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| phi4_mini_it | – | standard | natural | 2,000 | 1,194 | 556 | 248 | 2 | 0 | 0 | 59.7% |
| gemma4_12b_it | – | standard | natural | 2,000 | 1,588 | 294 | 36 | 8 | 74 | 0 | 79.4% |
| **all** | – | – | – | 4,000 | 2,782 | 850 | 284 | 10 | 74 | 0 | 69.5% |

### Failure-mode coverage (rows showing each failure mode)

Target: every failure mode is present. The last column is the dangerous silent failure.

| Prompt style | Provenance | Rows | Syntax error | Runtime error | Timeout | Wrong output | Security finding | Visible pass, hidden fail | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|
| standard | natural | 4,000 | 74 | 284 | 10 | 850 | 0 | 284 | 2,782 |
| all | all | 4,000 | 74 | 284 | 10 | 850 | 0 | 284 | 2,782 |

### pass@k on hidden tests (unbiased estimator)

| Config | pass@1 | pass@3 | pass@5 |
|---|---|---|---|
| phi4_mini_it | 59.7% | 63.4% | 64.7% |
| gemma4_12b_it | 79.4% | 80.8% | 81.2% |

### Visible test vs hidden tests

**9.4% of candidates that pass the visible test still fail at least one hidden test.**

|  | Hidden pass | Hidden fail |
|---|---|---|
| Visible pass | 2,722 | 284 |
| Visible fail | 60 | 934 |

| Config | Visible pass | Hidden pass | Hidden fail given visible pass |
|---|---|---|---|
| phi4_mini_it | 65.5% | 59.7% | 11.3% |
| gemma4_12b_it | 84.8% | 79.4% | 8.0% |
| **all** | 75.1% | 69.5% | 9.4% |

### Static analysis

| Config | With ruff issues | With undefined names | With bandit medium/high | Mean ruff issues |
|---|---|---|---|---|
| phi4_mini_it | 20.6% | 0.2% | 0.0% | 0.37 |
| gemma4_12b_it | 46.5% | 0.0% | 0.0% | 1.68 |
| **all** | 33.6% | 0.1% | 0.0% | 1.03 |

## Dataset and splits

Splits are by problem: no problem appears in two splits.

| Split | Rows | Problems | Trustworthy | Rate | natural |
|---|---|---|---|---|---|
| train | 2,400 | 120 | 1,774 | 73.9% | 2,400 |
| calibration | 800 | 40 | 538 | 67.2% | 800 |
| test | 800 | 40 | 470 | 58.8% | 800 |
| **total** | 4,000 | 200 | 2,782 | 69.5% | 4,000 |

## Human review agreement

| Item | Value |
|---|---|
| Reviewer | SIMULATED (demo only) |
| Reviewed rows | 157 |
| Raw agreement | 94.3% |
| Cohen's kappa | 0.870 |

## Files

`run_report.json` has every number above (machine-readable). Other outputs in the same folder: `metrics.json`, `cv_results.csv`, `model_comparison.csv`, `test_predictions.csv`, `feature_importance.csv`, `figures/`.
