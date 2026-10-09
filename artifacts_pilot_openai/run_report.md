# Run report `20261008-001942-774ca5`

## Run

| Item | Value |
|---|---|
| Run id | 20261008-001942-774ca5 |
| Mode | real |
| Human feedback | real |
| Started | 2026-10-08T00:19:42-04:00 |
| Finished | 2026-10-08T08:39:04-04:00 |
| Wall-clock duration | 8h 19m |
| Problems requested | all |
| Model configs | gpt-4o-mini-t0.3, gpt-4o-mini-t1.0 |
| Prompt styles | standard, signature_only, step_by_step, constrained |
| Samples per config | 5 |
| Seed | 42 |
| Output dir | `/Users/hatem/MLE_NEW/artifacts_pilot_openai` |
| Commands | trust-pipeline --no-local-hf --openai --all-problems --prompt-styles natural --samples-per-config 5 --human-feedback real --output-dir artifacts_pilot_openai; trust-generate --output-dir artifacts_pilot_openai; trust-label --output-dir artifacts_pilot_openai; trust-label --output-dir artifacts_pilot_openai; trust-features --output-dir artifacts_pilot_openai; trust-train --output-dir artifacts_pilot_openai; trust-train --output-dir artifacts_pilot_openai; trust-review --output-dir artifacts_pilot_openai; trust-review --output-dir artifacts_pilot_openai; trust-review --output-dir artifacts_pilot_openai |
| Python / platform | 3.12.12 / macOS-15.7.4-x86_64-i386-64bit |
| GPU | not checked |
| Package versions | mle-trust-pipeline 0.1.0, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, datasets 5.1.0, openai 3.24.0, anthropic 1.11.0 |

## Stage timings

| Stage | Time | Seconds |
|---|---|---|
| prepare | 18s | 17.5 |
| generate | 4s | 4.1 |
| execute | 20m 36s | 1236.4 |
| static | 1m 41s | 101.1 |
| features | 57s | 56.5 |
| human | 0s | 0.1 |
| train | 2m 42s | 162.4 |
| decide | 3m 43s | 222.8 |
| **total** | 30m 01s | 1801.1 |

## Generation (per model config and prompt style)

New / reused / failed refer to the latest generate session; tokens and latency cover every sample in the dataset that recorded them. Local models generate the samples for a problem in one batch, so seconds per sample is the batch time divided by the batch size.

| Config | Model | Temp | Prompt style | Provenance | Problems | Samples | New | Reused | Failed | Out tokens (total) | Out tokens (mean) | In tokens (total) | s / sample | tokens / s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini-t0.3 | gpt-4o-mini | 0.3 | standard | natural | 968 | 4,840 | 0 | 4,840 | 0 | 244866 | 50.6 | 115334 | 0.31 | 165.3 |
| gpt-4o-mini-t0.3 | gpt-4o-mini | 0.3 | signature_only | natural | 968 | 4,840 | 0 | 4,840 | 0 | 271458 | 56.1 | 86970 | 0.31 | 182.3 |
| gpt-4o-mini-t0.3 | gpt-4o-mini | 0.3 | step_by_step | natural | 968 | 4,840 | 0 | 4,840 | 0 | 1151437 | 237.9 | 128886 | 0.89 | 266.3 |
| gpt-4o-mini-t0.3 | gpt-4o-mini | 0.3 | constrained | natural | 968 | 4,840 | 0 | 4,840 | 0 | 204313 | 42.2 | 146310 | 0.28 | 150.9 |
| gpt-4o-mini-t1.0 | gpt-4o-mini | 1.0 | standard | natural | 968 | 4,840 | 0 | 4,840 | 0 | 247723 | 51.2 | 115334 | 0.31 | 166.2 |
| gpt-4o-mini-t1.0 | gpt-4o-mini | 1.0 | signature_only | natural | 968 | 4,840 | 0 | 4,840 | 0 | 274703 | 56.8 | 86970 | 0.32 | 177.9 |
| gpt-4o-mini-t1.0 | gpt-4o-mini | 1.0 | step_by_step | natural | 968 | 4,840 | 0 | 4,840 | 0 | 1161077 | 239.9 | 128886 | 0.88 | 271.6 |
| gpt-4o-mini-t1.0 | gpt-4o-mini | 1.0 | constrained | natural | 968 | 4,840 | 0 | 4,840 | 0 | 203842 | 42.1 | 146310 | 0.27 | 154.5 |

## Execution outcomes and labels

Outcome on the hidden tests. Trustworthy = passes every hidden test and has no medium/high-severity bandit finding. Config = model config, plus `/style` for non-standard prompt styles.

Provenance: natural 38,720. `natural` = LLM output from a normal prompt; `injected` = LLM asked for a subtle bug (training only); `mutant` / `reference` = mutated / original reference solutions.

| Config | Temp | Prompt style | Provenance | Rows | pass | wrong_output | runtime_error | timeout | syntax_error | generation_error | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini-t0.3 | 0.3 | standard | natural | 4,840 | 3,590 | 1,203 | 41 | 1 | 5 | 0 | 74.0% |
| gpt-4o-mini-t0.3/signature_only | 0.3 | signature_only | natural | 4,840 | 2,672 | 1,731 | 437 | 0 | 0 | 0 | 55.1% |
| gpt-4o-mini-t0.3/step_by_step | 0.3 | step_by_step | natural | 4,840 | 3,626 | 1,124 | 81 | 8 | 1 | 0 | 74.9% |
| gpt-4o-mini-t0.3/constrained | 0.3 | constrained | natural | 4,840 | 3,525 | 1,240 | 65 | 5 | 5 | 0 | 72.8% |
| gpt-4o-mini-t1.0 | 1.0 | standard | natural | 4,840 | 3,572 | 1,194 | 59 | 6 | 9 | 0 | 73.7% |
| gpt-4o-mini-t1.0/signature_only | 1.0 | signature_only | natural | 4,840 | 2,654 | 1,737 | 439 | 7 | 3 | 0 | 54.7% |
| gpt-4o-mini-t1.0/step_by_step | 1.0 | step_by_step | natural | 4,840 | 3,586 | 1,146 | 98 | 4 | 6 | 0 | 74.1% |
| gpt-4o-mini-t1.0/constrained | 1.0 | constrained | natural | 4,840 | 3,495 | 1,249 | 81 | 8 | 7 | 0 | 72.2% |
| **all** | – | – | – | 38,720 | 26,720 | 10,624 | 1,301 | 39 | 36 | 0 | 68.9% |

### Failure-mode coverage (rows showing each failure mode)

Target: every failure mode is present. The last column is the dangerous silent failure.

| Prompt style | Provenance | Rows | Syntax error | Runtime error | Timeout | Wrong output | Security finding | Visible pass, hidden fail | Trustworthy |
|---|---|---|---|---|---|---|---|---|---|
| standard | natural | 9,680 | 14 | 100 | 7 | 2,397 | 14 | 892 | 7,148 |
| signature_only | natural | 9,680 | 3 | 876 | 7 | 3,468 | 10 | 445 | 5,316 |
| step_by_step | natural | 9,680 | 7 | 179 | 12 | 2,270 | 0 | 855 | 7,212 |
| constrained | natural | 9,680 | 12 | 146 | 13 | 2,489 | 11 | 910 | 7,019 |
| all | all | 38,720 | 36 | 1,301 | 39 | 10,624 | 35 | 3,102 | 26,695 |

### pass@k on hidden tests (unbiased estimator)

| Config | pass@1 | pass@3 | pass@5 |
|---|---|---|---|
| gpt-4o-mini-t0.3 | 74.2% | 75.8% | 76.4% |
| gpt-4o-mini-t0.3/signature_only | 55.2% | 57.1% | 57.7% |
| gpt-4o-mini-t0.3/step_by_step | 74.9% | 77.7% | 78.9% |
| gpt-4o-mini-t0.3/constrained | 72.8% | 74.8% | 75.5% |
| gpt-4o-mini-t1.0 | 73.8% | 77.7% | 79.2% |
| gpt-4o-mini-t1.0/signature_only | 54.8% | 59.4% | 61.1% |
| gpt-4o-mini-t1.0/step_by_step | 74.1% | 79.5% | 81.1% |
| gpt-4o-mini-t1.0/constrained | 72.2% | 77.6% | 79.4% |

### Visible test vs hidden tests

**10.6% of candidates that pass the visible test still fail at least one hidden test.**

|  | Hidden pass | Hidden fail |
|---|---|---|
| Visible pass | 26,184 | 3,102 |
| Visible fail | 536 | 8,898 |

| Config | Visible pass | Hidden pass | Hidden fail given visible pass |
|---|---|---|---|
| gpt-4o-mini-t0.3 | 82.0% | 74.2% | 11.3% |
| gpt-4o-mini-t0.3/signature_only | 58.9% | 55.2% | 7.8% |
| gpt-4o-mini-t0.3/step_by_step | 82.1% | 74.9% | 10.5% |
| gpt-4o-mini-t0.3/constrained | 80.6% | 72.8% | 11.8% |
| gpt-4o-mini-t1.0 | 81.5% | 73.8% | 11.3% |
| gpt-4o-mini-t1.0/signature_only | 58.4% | 54.8% | 7.9% |
| gpt-4o-mini-t1.0/step_by_step | 81.5% | 74.1% | 11.1% |
| gpt-4o-mini-t1.0/constrained | 80.0% | 72.2% | 11.6% |
| **all** | 75.6% | 69.0% | 10.6% |

### Static analysis

| Config | With ruff issues | With undefined names | With bandit medium/high | Mean ruff issues |
|---|---|---|---|---|
| gpt-4o-mini-t0.3 | 19.3% | 0.0% | 0.2% | 0.43 |
| gpt-4o-mini-t0.3/signature_only | 24.5% | 0.0% | 0.1% | 0.55 |
| gpt-4o-mini-t0.3/step_by_step | 38.2% | 0.1% | 0.0% | 1.04 |
| gpt-4o-mini-t0.3/constrained | 14.1% | 0.2% | 0.1% | 0.18 |
| gpt-4o-mini-t1.0 | 20.4% | 0.0% | 0.1% | 0.43 |
| gpt-4o-mini-t1.0/signature_only | 25.2% | 0.1% | 0.1% | 0.53 |
| gpt-4o-mini-t1.0/step_by_step | 40.0% | 0.0% | 0.0% | 0.98 |
| gpt-4o-mini-t1.0/constrained | 14.9% | 0.2% | 0.1% | 0.20 |
| **all** | 24.6% | 0.1% | 0.1% | 0.54 |

## Dataset and splits

Splits are by problem: no problem appears in two splits.

| Split | Rows | Problems | Trustworthy | Rate | natural |
|---|---|---|---|---|---|
| train | 23,200 | 580 | 16,086 | 69.3% | 23,200 |
| calibration | 7,760 | 194 | 5,639 | 72.7% | 7,760 |
| test | 7,760 | 194 | 4,970 | 64.0% | 7,760 |
| **total** | 38,720 | 968 | 26,695 | 68.9% | 38,720 |

## Model selection (grouped cross-validation on train, mean ± std)

| Model | roc_auc | average_precision | f1 | accuracy |
|---|---|---|---|---|
| majority | 0.500 ± 0.000 | 0.693 ± 0.035 | 0.818 ± 0.025 | 0.693 ± 0.035 |
| visible_test_only | 0.857 ± 0.026 | 0.888 ± 0.028 | 0.933 ± 0.016 | 0.903 ± 0.021 |
| logistic_regression | 0.924 ± 0.019 | 0.945 ± 0.016 | 0.920 ± 0.012 | 0.889 ± 0.015 |
| **random_forest** | 0.940 ± 0.009 | 0.963 ± 0.012 | 0.933 ± 0.015 | 0.906 ± 0.019 |
| hist_gradient_boosting | 0.928 ± 0.012 | 0.956 ± 0.015 | 0.929 ± 0.016 | 0.900 ± 0.021 |

Selected: **random_forest**, calibrated with isotonic calibration. Baselines: majority, visible_test_only.

## Test-set metrics (held-out problems, 95% CI from 1000 bootstrap resamples by problem)

Threshold 0.5 for accuracy / precision / recall / F1; lower Brier is better.

| Model | accuracy | precision | recall | f1 | roc_auc | pr_auc | brier |
|---|---|---|---|---|---|---|---|
| majority | 0.640 [0.583, 0.703] | 0.640 [0.583, 0.703] | 1.000 [1.000, 1.000] | 0.781 [0.737, 0.825] | 0.500 [0.500, 0.500] | 0.640 [0.583, 0.703] | 0.233 [0.209, 0.255] |
| visible_test_only | 0.886 [0.848, 0.922] | 0.872 [0.824, 0.919] | 0.964 [0.932, 0.988] | 0.916 [0.885, 0.944] | 0.856 [0.816, 0.898] | 0.864 [0.815, 0.912] | 0.101 [0.071, 0.131] |
| logistic_regression | 0.871 [0.835, 0.906] | 0.895 [0.847, 0.940] | 0.905 [0.868, 0.937] | 0.900 [0.868, 0.929] | 0.920 [0.884, 0.953] | 0.942 [0.909, 0.970] | 0.099 [0.075, 0.124] |
| random_forest | 0.900 [0.868, 0.932] | 0.900 [0.857, 0.940] | 0.948 [0.919, 0.973] | 0.924 [0.896, 0.949] | 0.941 [0.909, 0.968] | 0.956 [0.926, 0.983] | 0.080 [0.059, 0.102] |
| hist_gradient_boosting | 0.898 [0.864, 0.929] | 0.900 [0.857, 0.940] | 0.946 [0.915, 0.970] | 0.922 [0.893, 0.948] | 0.931 [0.896, 0.961] | 0.948 [0.913, 0.977] | 0.085 [0.058, 0.115] |
| **random_forest (calibrated)** | 0.898 [0.864, 0.930] | 0.892 [0.849, 0.934] | 0.956 [0.929, 0.977] | 0.923 [0.895, 0.949] | 0.941 [0.910, 0.968] | 0.957 [0.927, 0.982] | 0.081 [0.058, 0.104] |

### Test metrics of random_forest (calibrated) by prompt style (natural samples only)

| Prompt style | Rows | Problems | Trustworthy rate | accuracy | precision | recall | f1 | roc_auc | pr_auc | brier |
|---|---|---|---|---|---|---|---|---|---|---|
| standard | 1,940 | 194 | 67.8% | 0.888 | 0.882 | 0.964 | 0.921 | 0.937 | 0.965 | 0.091 |
| signature_only | 1,940 | 194 | 51.5% | 0.943 | 0.938 | 0.953 | 0.945 | 0.963 | 0.960 | 0.052 |
| step_by_step | 1,940 | 194 | 69.7% | 0.876 | 0.888 | 0.942 | 0.914 | 0.916 | 0.953 | 0.093 |
| constrained | 1,940 | 194 | 67.1% | 0.884 | 0.874 | 0.967 | 0.918 | 0.932 | 0.950 | 0.087 |

## Three-way decisions on the test split

Threshold rule (chosen on the calibration split, target precision 0.95 / 0.95): APPROVE if p ≥ 0.789; REJECT if p ≤ 0.473; otherwise NEEDS HUMAN REVIEW. Conformal (alpha 0.10): qhat 0.333, test coverage 0.859.

| Metric | threshold_rule | conformal |
|---|---|---|
| approved share | 60.4% | 65.7% |
| rejected share | 30.3% | 27.9% |
| review share | 9.3% | 6.4% |
| approved precision | 92.6% | 90.7% |
| rejected precision | 92.3% | 94.0% |
| automated error rate | 7.5% | 8.3% |
| untrustworthy code approved | 345 | 473 |

## Human review agreement

| Item | Value |
|---|---|
| Reviewer | real reviewers |
| Reviewed rows | 164 |
| Raw agreement | 88.4% |
| Cohen's kappa | 0.738 |

## Files

`run_report.json` has every number above (machine-readable). Other outputs in the same folder: `metrics.json`, `cv_results.csv`, `model_comparison.csv`, `test_predictions.csv`, `feature_importance.csv`, `figures/`. Cross-run history: `/Users/hatem/MLE_NEW/runs_history.csv`.
