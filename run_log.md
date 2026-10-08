# Run log

What we ran, in order, with the results and the state after each step. All OpenAI runs use the same
folder, `artifacts_pilot_openai`, so generated code is cached and reused. Every run also appends a row
to `runs_history.csv` (project root), and `artifacts_pilot_openai/run_report.md` is rewritten by the
latest run, so earlier numbers are kept here.

## Summary

| # | Date (2026) | What | Problems | Rows | Duration | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Oct 5 | Smoke test (reference solutions + mutants, no API) | 60 | 447 | 1.6 min | Done |
| 1 | Oct 7, 23:34 | OpenAI pilot 1, standard prompt | 20 | 200 | 1.7 min | Done |
| 2 | Oct 7, 23:39 | OpenAI pilot 2, 4 natural prompt styles | 100 | 4,000 | 32 min | Done, analysed in `analysis_report.md` |
| 3 | Oct 8, 00:19 | **Full MBPP**, 4 natural prompt styles | 968 | 38,720 | 4h 43m | Done (before the self-written-test fix) |
| 4 | Oct 8, 07:49 | Relabel and retrain from the cache with the fix | 968 | 38,720 | about 25 min | **In progress** |

## 0. Smoke test

```bash
uv run trust-pipeline --smoke-test --num-problems 60 --output-dir artifacts_smoke
```

Checks that the pipeline runs end to end. Numbers are not findings (mutant data, simulated review).

## 1. Pilot 1

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 20 --samples-per-config 5 --output-dir artifacts_pilot_openai
```

76% trustworthy and no silent failures (no code passed the visible test but failed hidden tests). Only
the visible test mattered, so we needed more problems and more varied prompts.

## 2. Pilot 2

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 100 --prompt-styles natural --samples-per-config 5 --output-dir artifacts_pilot_openai
```

- 71.1% trustworthy; 5.9% of visible passes were silent failures.
- Test ROC AUC 0.967 for the random forest, against 0.947 for the visible test alone.
- Found a harness problem: in `step_by_step`, the model's own `assert` lines crashed the code at load
  time. Fixed afterwards (see run 4).

Full analysis: `analysis_report.md`.

## 3. Full MBPP run (before the fix)

```bash
uv run trust-pipeline --no-local-hf --openai --all-problems --prompt-styles natural --samples-per-config 5 --human-feedback real --output-dir artifacts_pilot_openai
```

Generated 870 new problems (the first 100 came from the cache). Cost was estimated at $2–3; check the
OpenAI usage dashboard for the real figure. `generations.jsonl` now holds all 38,720 attempts.

**Data**

| Item | Value |
| --- | --- |
| Problems / rows | 968 / 38,720 (2 temperatures × 4 prompt styles × 5 samples) |
| Trustworthy | 68.9% |
| Outcomes | 26,685 pass, 10,072 wrong output, 1,899 runtime error, 36 syntax error, 28 timeout |
| Silent failures | 3,102 (**10.6% of visible passes**; 5.9% in pilot 2) |
| Splits (by problem) | train 580, calibration 194, test 194 problems |

Trustworthy rate by prompt style: standard 73.9%, step_by_step 74.2%, constrained 72.5%, signature_only
54.9%. Temperature 0.3 against 1.0 makes almost no difference at pass@1. Temperature 1.0 gains at pass@5
(for example standard: 76.4% at 0.3, 79.2% at 1.0).

**Model** (random forest, isotonic calibration, chosen by grouped cross-validation)

| Metric (test, 95% CI) | Trust model | Visible test only |
| --- | --- | --- |
| ROC AUC | **0.941** [0.910, 0.968] | 0.856 [0.816, 0.898] |
| F1 | 0.924 | 0.916 |
| Precision | 0.893 | 0.872 |
| Recall | 0.958 | 0.964 |
| Brier (lower is better) | 0.080 | 0.101 |

Cross-validation ROC AUC: random forest 0.940, against 0.858 for the visible test alone. The gap is wider
than in pilot 2, because the full dataset has more silent failures.

**Decisions on the test split**

| Metric | Threshold rule (approve p ≥ 0.782, reject p ≤ 0.487) | Conformal (α = 0.10) |
| --- | --- | --- |
| Approved | 60.9% | 66.0% |
| Rejected | 30.3% | 28.7% |
| Needs human review | 8.8% | 5.2% |
| Approval precision | 92.4% | 90.8% |
| Untrustworthy code approved | 357 | 472 |

Both rules miss the 95% approval-precision target. The calibration split was 72.7% trustworthy and the
test split 64.0%, so thresholds tuned on one set of problems carry over imperfectly to another.

**Human review: the reported number is not valid.** The report shows kappa 0.868 from 156 "real"
reviews. Those rows were simulated answers left over from pilot 2 in `human_review_sample.csv`; nobody
has reviewed yet. Fixed in the code (`real_reviews` in `trust_pipeline/human_review.py`): simulated
answers are now ignored in real mode, and a fresh blank sample is written for reviewers.

## 4. Relabel and retrain with the fix (done)

Applies the self-written-test fix (`strip_test_code` in `trust_pipeline/generation.py`) to all cached
code, then redoes everything after generation. No API calls.

```bash
uv run trust-generate --output-dir artifacts_pilot_openai   # done 07:49 (cache only)
rm -f artifacts_pilot_openai/human_agreement.json           # removes the invalid simulated kappa
uv run trust-label --output-dir artifacts_pilot_openai      # first attempt stopped ~07:56; rerun, ~18 min
uv run trust-features --output-dir artifacts_pilot_openai   # also writes a new blank human_review_sample.csv
uv run trust-train --output-dir artifacts_pilot_openai
```

The stopped `trust-label` saved nothing; `labeled.csv`, `dataset.csv` and the model were still from run 3.

**Result (finished 08:39).** `runs_history.csv` reuses run 3's id, so its 499-minute duration spans both
runs. Ignore it; the relabel itself took about 25 minutes.

| Item | Run 3 (before fix) | Run 4 (after fix) |
| --- | --- | --- |
| `step_by_step` runtime errors (t0.3 / t1.0) | 376 / 400 | **81 / 98** |
| All runtime errors | 1,899 | 1,301 |
| All wrong output | 10,072 | 10,624 |
| Trustworthy rate | 68.9% | 68.9% |
| Silent failures (share of visible passes) | 10.6% | 10.6% |
| Test ROC AUC, calibrated random forest | 0.941 | 0.941 [0.910, 0.968] |
| Test ROC AUC, visible test only | 0.856 | 0.856 |
| Threshold rule: approve / review / approval precision | 60.9% / 8.8% / 92.4% | 60.4% / 9.3% / 92.6% |
| Conformal: approve / review / approval precision | 66.0% / 5.2% / 90.8% | 65.7% / 6.4% / 90.7% |
| Untrustworthy code approved (threshold rule) | 357 | 345 |
| Human kappa | 0.868 (invalid, simulated) | Pending real review |

The fix corrected the failure *types* (crashes were really wrong answers) without changing the label
balance or the model's ranking quality. Top features are unchanged: `visible_pass`, then sibling agreement
(`behavior_agreement_visible`, `behavior_agreement`).

What to expect: `step_by_step` runtime errors (776 in run 3) should drop sharply, mostly turning into wrong
output. A few labels flip to trustworthy (4 of 105 in the pilot 2 check).

## Next

1. After run 4: compare its `run_report.md` with run 3 above, then update `analysis_report.md` for the
   full dataset.
2. Human review: fill in `human_trustworthy` (yes/no) in the new `human_review_sample.csv`, then
   `uv run trust-review --output-dir artifacts_pilot_openai`.
3. Back up the folder: `cp -r artifacts_pilot_openai ~/Backups/artifacts_pilot_openai_$(date +%Y%m%d)`.
4. Open: approval precision is below 95% with both rules. Options: choose thresholds by cross-validation
   across problems, or raise the conformal confidence (smaller α).
