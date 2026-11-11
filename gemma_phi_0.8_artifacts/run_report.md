# Run report `20261103-082636-9cc8c8`

> **SIMULATED HUMAN FEEDBACK.** Human-agreement numbers come from a simulated reviewer and are not a finding.

## Run

| Item | Value |
|---|---|
| Run id | 20261103-082636-9cc8c8 |
| Mode | real |
| Human feedback | simulated |
| Started | 2026-11-03T08:26:36-05:00 |
| Finished | 2026-11-03T08:49:40-05:00 |
| Wall-clock duration | 23m 04s |
| Problems requested | 200 |
| Model configs | phi4_mini_it, gemma4_12b_it |
| Prompt styles | standard |
| Samples per config | 5 |
| Seed | 42 |
| Output dir | `/run/media/khan/5TB_HDD/FIT_PhD/CSE_5695_MLE/TermProject/MLE_MBPP/artifacts` |
| Commands | trust-prepare --num-problems 200; trust-generate --samples-per-config 5; trust-generate --samples-per-config 5; trust-label; trust-features; trust-train |
| Python / platform | 3.12.15 / Linux-6.12.0-211.69.1.el10_2.x86_64-x86_64-with-glibc2.39 |
| GPU | not checked |
| Package versions | scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, torch 2.14.1, transformers 5.19.0, datasets 5.1.0, openai 3.24.0, anthropic 1.11.0 |

## Stage timings

| Stage | Time | Seconds |
|---|---|---|
| prepare | 2s | 1.6 |
| generate | 15m 03s | 903.3 |
| execute | 27s | 27.4 |
| static | 1s | 0.7 |
| features | 1s | 0.7 |
| human | 0s | 0.0 |
| train | 5s | 5.0 |
| decide | 4s | 4.4 |
| **total** | 15m 43s | 943.2 |

## Generation (per model config and prompt style)

New / reused / failed refer to the latest generate session; tokens and latency cover every sample in the dataset that recorded them. Local models generate the samples for a problem in one batch, so seconds per sample is the batch time divided by the batch size.

| Config | Model | Temp | Prompt style | Provenance | Problems | Samples | New | Reused | Failed | Out tokens (total) | Out tokens (mean) | In tokens (total) | s / sample | tokens / s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi4_mini_it | ../models/phi_4_mini_it | 0.8 | standard | natural | 200 | 1,000 | 1,000 | 0 | 0 | 57816 | 57.8 | 23630 | 0.20 | 289.4 |
| gemma4_12b_it | ../models/gemma_4_12B_it | 0.8 | standard | natural | 200 | 1,000 | 670 | 330 | 0 | 121019 | 121.0 | 28515 | 1.08 | 112.5 |

## Execution outcomes and labels

Outcome on the hidden tests. Trustworthy = passes every hidden test and has no medium/high-severity bandit finding. Config = model config, plus `/style` for non-standard prompt styles.

Provenance: natural 2,000. `natural` = LLM output from a normal prompt; `injected` = LLM asked for a subtle bug (training only); `mutant` / `reference` = mutated / original reference solutions.

| Config | Temp | Prompt style | Provenance | Rows | pass | wrong_output | runtime_error | timeout | syntax_error | generation_error | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| phi4_mini_it | 0.8 | standard | natural | 1,000 | 603 | 270 | 125 | 2 | 0 | 0 | 60.3% |
| gemma4_12b_it | 0.8 | standard | natural | 1,000 | 785 | 153 | 17 | 0 | 45 | 0 | 78.5% |
| **all** | – | – | – | 2,000 | 1,388 | 423 | 142 | 2 | 45 | 0 | 69.4% |

### Failure-mode coverage (rows showing each failure mode)

Target: every failure mode is present. The last column is the dangerous silent failure.

| Prompt style | Provenance | Rows | Syntax error | Runtime error | Timeout | Wrong output | Security finding | Visible pass, hidden fail | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|
| standard | natural | 2,000 | 45 | 142 | 2 | 423 | 0 | 131 | 1,388 |
| all | all | 2,000 | 45 | 142 | 2 | 423 | 0 | 131 | 1,388 |

### pass@k on hidden tests (unbiased estimator)

| Config | pass@1 | pass@3 | pass@5 |
|---|---|---|---|
| phi4_mini_it | 60.3% | 68.3% | 72.0% |
| gemma4_12b_it | 78.5% | 81.0% | 82.0% |

### Visible test vs hidden tests

**8.8% of candidates that pass the visible test still fail at least one hidden test.**

|  | Hidden pass | Hidden fail |
|---|---|---|
| Visible pass | 1,359 | 131 |
| Visible fail | 29 | 481 |

| Config | Visible pass | Hidden pass | Hidden fail given visible pass |
|---|---|---|---|
| phi4_mini_it | 65.0% | 60.3% | 9.4% |
| gemma4_12b_it | 84.0% | 78.5% | 8.3% |
| **all** | 74.5% | 69.4% | 8.8% |

### Static analysis

| Config | With ruff issues | With undefined names | With bandit medium/high | Mean ruff issues |
|---|---|---|---|---|
| phi4_mini_it | 24.3% | 0.6% | 0.0% | 0.42 |
| gemma4_12b_it | 47.4% | 0.0% | 0.0% | 1.79 |
| **all** | 35.9% | 0.3% | 0.0% | 1.10 |

## Dataset and splits

Splits are by problem: no problem appears in two splits.

| Split | Rows | Problems | Trustworthy | Rate | natural |
|---|---|---|---|---|---|
| train | 1,200 | 120 | 882 | 73.5% | 1,200 |
| calibration | 400 | 40 | 266 | 66.5% | 400 |
| test | 400 | 40 | 240 | 60.0% | 400 |
| **total** | 2,000 | 200 | 1,388 | 69.4% | 2,000 |

## Model selection (grouped cross-validation on train, mean ± std)

| Model | roc_auc | average_precision | f1 | accuracy |
|---|---|---|---|---|
| majority | 0.500 ± 0.000 | 0.735 ± 0.063 | 0.846 ± 0.043 | 0.735 ± 0.063 |
| visible_test_only | 0.869 ± 0.067 | 0.921 ± 0.034 | 0.952 ± 0.023 | 0.926 ± 0.037 |
| logistic_regression | 0.913 ± 0.057 | 0.950 ± 0.045 | 0.935 ± 0.017 | 0.906 ± 0.027 |
| random_forest | 0.923 ± 0.051 | 0.965 ± 0.024 | 0.947 ± 0.024 | 0.920 ± 0.038 |
| **hist_gradient_boosting** | 0.928 ± 0.040 | 0.965 ± 0.026 | 0.948 ± 0.021 | 0.921 ± 0.035 |

Selected: **hist_gradient_boosting**, calibrated with isotonic calibration. Baselines: majority, visible_test_only.

## Test-set metrics (held-out problems, 95% CI from 1000 bootstrap resamples by problem)

Threshold 0.5 for accuracy / precision / recall / F1; lower Brier is better.

| Model | accuracy | precision | recall | f1 | roc_auc | pr_auc | brier |
|---|---|---|---|---|---|---|---|
| majority | 0.600 [0.472, 0.720] | 0.600 [0.472, 0.720] | 1.000 [1.000, 1.000] | 0.750 [0.642, 0.837] | 0.500 [0.500, 0.500] | 0.600 [0.472, 0.720] | 0.258 [0.202, 0.318] |
| visible_test_only | 0.858 [0.765, 0.940] | 0.817 [0.701, 0.922] | 0.983 [0.954, 1.000] | 0.892 [0.816, 0.955] | 0.826 [0.724, 0.927] | 0.813 [0.695, 0.917] | 0.126 [0.057, 0.203] |
| logistic_regression | 0.812 [0.725, 0.895] | 0.831 [0.714, 0.940] | 0.863 [0.777, 0.934] | 0.847 [0.762, 0.914] | 0.834 [0.715, 0.942] | 0.808 [0.677, 0.951] | 0.151 [0.082, 0.225] |
| random_forest | 0.845 [0.760, 0.927] | 0.835 [0.717, 0.940] | 0.925 [0.873, 0.969] | 0.877 [0.801, 0.941] | 0.860 [0.751, 0.957] | 0.875 [0.738, 0.973] | 0.127 [0.066, 0.195] |
| hist_gradient_boosting | 0.840 [0.745, 0.925] | 0.824 [0.699, 0.939] | 0.933 [0.876, 0.978] | 0.875 [0.791, 0.942] | 0.871 [0.767, 0.962] | 0.888 [0.768, 0.975] | 0.142 [0.062, 0.231] |
| **hist_gradient_boosting (calibrated)** | 0.838 [0.743, 0.925] | 0.821 [0.696, 0.935] | 0.933 [0.876, 0.978] | 0.873 [0.792, 0.939] | 0.857 [0.743, 0.957] | 0.846 [0.706, 0.971] | 0.135 [0.066, 0.215] |

## Three-way decisions on the test split

Threshold rule (chosen by grouped cross-validation across calibration problems, target precision 0.95 / 0.95): APPROVE if p ≥ 0.933; REJECT if p ≤ 0.061; otherwise NEEDS HUMAN REVIEW. At the approve cutoff, every held-out calibration fold cleared the target (precision 1.000 to 1.000, mean 1.000). Conformal (alpha 0.10): qhat 0.378, test coverage 0.825.

| Metric | threshold_rule | pooled_calibration | conformal |
|---|---|---|---|
| approved share | 45.5% | 67.8% | 66.8% |
| rejected share | 10.5% | 27.8% | 31.2% |
| review share | 44.0% | 4.5% | 2.0% |
| approved precision | 83.5% | 82.3% | 82.0% |
| rejected precision | 100.0% | 97.3% | 88.8% |
| automated error rate | 13.4% | 13.4% | 15.8% |
| untrustworthy code approved | 30 | 48 | 48 |

## Human review agreement

| Item | Value |
|---|---|
| Reviewer | SIMULATED (demo only) |
| Reviewed rows | 152 |
| Raw agreement | 95.4% |
| Cohen's kappa | 0.895 |

## Files

`run_report.json` has every number above (machine-readable). Other outputs in the same folder: `metrics.json`, `cv_results.csv`, `model_comparison.csv`, `test_predictions.csv`, `feature_importance.csv`, `figures/`. Cross-run history: `/run/media/khan/5TB_HDD/FIT_PhD/CSE_5695_MLE/TermProject/MLE_MBPP/runs_history.csv`.
