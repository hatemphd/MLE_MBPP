# Project plan and run log

Living document: what has been run, where we are, and what is left to finish the project.
Update the run log after every run.

## 1. Where we are (8 Oct 2026)

- The pipeline is complete and tested: data, generation, labelling, features, training, three-way
  decisions, run report.
- The full MBPP dataset is generated with `gpt-4o-mini`: 968 problems, 38,720 attempts, about $2.40 of
  OpenAI cost. It is cached in `artifacts_pilot_openai/generations.jsonl`, so **no more API spending is
  needed**.
- **Phase 1 is done:** relabelled and retrained with the self-written-test fix.
- **Phase 2 is done:** real human review of 164 rows gives kappa **0.738 [0.62, 0.84]** and 88.4%
  agreement. The old 0.868 was simulated and must not be reported.
- **Next:** Phase 3 (decision rule) and Phase 5 (write-up). Results are summarised in
  `analysis_report.md` section 0.

## 2. Run log

Short version; full commands and results per run are in `run_log.md`.

| # | When | Command (short) | Problems / rows | Time | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 Oct | `trust-pipeline --smoke-test --num-problems 60` | 60 / 447 | 1.6 min | Pipeline works end to end (reference solutions and mutants, no LLM) |
| 2 | 7 Oct 23:34 | Pilot 1: `--openai --num-problems 20` | 20 / 200 | 1.7 min | 76% trustworthy, 0 silent failures; too easy to learn from |
| 3 | 7 Oct 23:38 | Pilot 2: `--openai --num-problems 100 --prompt-styles natural` | 100 / 4,000 | 32 min | 5.9% silent failures; RF ROC AUC 0.967 vs 0.947 visible-only; found the self-test artifact |
| 4 | 8 Oct 00:19 | Full: `--openai --all-problems --prompt-styles natural --human-feedback real` | 968 / 38,720 | 4h 43m | See section 3 (before the fixes) |
| 5 | 8 Oct 07:49 | `trust-generate` (cache only) | 968 / 38,720 | 4 s | Candidates rebuilt with self-test code removed; 0 API calls |
| 6 | 8 Oct ~07:50 | `trust-label` | – | – | Stopped by user before saving; nothing lost |
| 7 | 8 Oct 08:39 | `trust-label`, `trust-features`, `trust-train` (Phase 1) | 968 / 38,720 | ~25 min | Done: `step_by_step` runtime errors 776 → 179; test ROC AUC 0.941; approval precision 92.6% |
| 8 | 8 Oct 20:52 | Human review in `human_edit/app.py`, then `trust-review` | 164 rows | ~1.5 h | Kappa 0.738 [0.62, 0.84], agreement 88.4%; human approved 16 failing candidates, rejected 3 passing ones |

All OpenAI runs use the same folder, `artifacts_pilot_openai`, so earlier attempts are reused.

## 3. Full-run results (before the fixes; final numbers are in `run_log.md` sections 4–5)

| Item | Value |
| --- | --- |
| Trustworthy rate | 68.9% |
| Silent failures (pass visible test, fail hidden tests) | 3,102 = **10.6% of visible passes** |
| Selected model | Random forest, isotonic calibration |
| Cross-validation ROC AUC | 0.940 vs 0.858 visible-test-only |
| Test ROC AUC (194 unseen problems) | **0.941 [0.910, 0.968]** vs 0.856 visible-test-only |
| Test F1 | 0.924 |
| Threshold rule | Approve 60.9%, reject 30.3%, review 8.8%; approval precision **92.4%** (target 95%) |
| Conformal (alpha 0.10) | Approve 66.0%, review 5.2%; approval precision 90.8% |
| Human kappa | Invalid (simulated leftovers); real review pending |

## 4. Plan to finish

### Phase 1: relabel and retrain with the fixes (done 8 Oct 08:39)

Run one line at a time from `~/MLE_NEW`:

```bash
rm artifacts_pilot_openai/human_agreement.json
uv run trust-label --output-dir artifacts_pilot_openai
uv run trust-features --output-dir artifacts_pilot_openai
uv run trust-train --output-dir artifacts_pilot_openai
cp -r artifacts_pilot_openai ~/Backups/artifacts_pilot_openai_final_$(date +%Y%m%d)
```

Done when the new `run_report.md` shows far fewer `step_by_step` runtime errors and "no reviews filled
in" for human review.

### Phase 2: real human review (done 8 Oct 20:52; kappa 0.738)

Full reviewer guide: `human_review_helper.md`.

1. Open the review editor (`human_edit/app.py`):
   `uv run --with streamlit streamlit run human_edit/app.py`. It edits
   `artifacts_pilot_openai/human_review_sample.csv` (164 rows) and saves each answer immediately.
2. For each row, read the problem and the code and write `yes` or `no` in `human_trustworthy`. Do
   **not** look at `label_trustworthy` first. Hiding that column in Excel keeps the review blind.
3. Best practice: two team members review independently. Their agreement with each other
   (inter-rater kappa) shows how hard the judgement is.
4. Write a short reason in `human_notes` for disagreements, especially "problem is ambiguous". This
   measures benchmark label noise, as with task 704.
5. Run `uv run trust-review --output-dir artifacts_pilot_openai` to get the real kappa.

Done when `human_agreement.json` says `"simulated": false` with about 150 reviewed rows.

### Phase 3: one improvement to the decision rule (half a day, $0)

The 95% approval-precision target was missed (92.4%). This is the main open technical problem. Choose
**one** option:

- **(Recommended)** Choose thresholds by grouped cross-validation instead of a single calibration split,
  then report the achieved precision on test. Or make the target stricter on calibration (for example
  97%) and show the trade-off with review share.
- Report a risk–coverage curve: approval precision versus share approved. It shows the honest trade-off
  without needing a single "right" threshold.

### Phase 4: optional extras (choose at most one or two)

| Extra | Effort | Cost | Value for the report |
| --- | --- | --- | --- |
| "Agent's own tests" feature (did the model's self-written asserts pass?) | Small | $0 | New trust signal found in this project |
| Feature simplification (logistic regression on 5–8 features) | Small | $0 | Easier to explain |
| MBPP+ (EvalPlus) stronger hidden tests | Medium | $0 | More silent failures found; less label noise |
| Weaker model (Qwen on Colab) | Medium | $0 (GPU time) | More syntax errors, crashes and timeouts |
| Bug-injection prompts (`--prompt-styles all`) | Small | ~$0.60 | Tests whether synthetic bugs help |

### Phase 5: write the report (1–2 days)

EDA figures and tables for the report: `uv run trust-eda --output-dir artifacts_pilot_openai`. This writes
`artifacts_pilot_openai/eda_report.md` and 8 figures in `artifacts_pilot_openai/figures/eda/`, in about
5 seconds with no API calls. Rerun it after any retrain.

Suggested structure, with sources:

1. **Problem and sensor framing:** `sensor_framing.md`, `High_Level_Explanation.md`.
2. **Data and labelling:** MBPP, visible vs hidden tests, execution plus static analysis
   (`how_it_works.md`).
3. **Generation:** model, two temperatures, four prompt styles (positioned as an extra); pass@k table.
4. **Key finding:** passing the visible test is not enough; 10.6% silent failures.
5. **Trust model:** features, grouped splits, baselines, metrics with confidence intervals, feature
   importance (`run_report.md`, `figures/`).
6. **Decisions:** APPROVED / REJECTED / NEEDS HUMAN REVIEW, threshold rule vs conformal, precision vs
   review share.
7. **Human validation:** real kappa and examples of disagreement.
8. **Limitations:**
   - one model family;
   - MBPP label noise (task 704);
   - calibration shift between problems;
   - the self-test artifact and its fix;
   - Python only.
9. **Related work:** `similar_concepts.md`.

## 5. Risks and rules

- **Never delete `generations.jsonl`.** It is the only paid artifact. Back it up after every change.
- **Always use `--output-dir artifacts_pilot_openai`.** A new folder means regenerating and paying again.
- **Do not report simulated kappa as a finding.**
- Run commands one line at a time. Copied multi-line commands with `\` broke before.
- Freeze the dataset after Phase 1. Later changes should only touch training and decisions, so results
  stay comparable.
