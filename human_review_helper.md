# Human review helper

How to review the 164 sampled candidates, so the project can report a **real** human-vs-automated
agreement score (Cohen's kappa).

## 1. Why we do this

The automated label says code is trustworthy if it passes every hidden test and has no medium or high
security finding. Human review checks whether that label matches expert judgement. It is a **check on
label quality, not training data**: your answers are never used to train the model.

The sample is stratified by model setting and execution outcome, so it includes passing code, wrong
output, crashes and other failures.

## 2. Start the editor

From `~/MLE_NEW`:

```bash
uv run --with streamlit streamlit run human_edit/app.py
```

A browser tab opens at `http://localhost:8501`. Answers are saved to
`artifacts_pilot_openai/human_review_sample.csv` as soon as you click **Save** or **Save and next**. You
can stop (Ctrl+C in the terminal) and continue later; the editor reopens at the first unreviewed row.

## 3. How to review one row

1. Read the **problem** text.
2. Look at the **example test**. It shows the expected function name, argument types and output format.
3. Read the **generated code** and decide:

   | Answer | Choose when |
   | --- | --- |
   | **Yes** | You would approve this code for the task as described: it is correct for all reasonable inputs and has no obvious security problem |
   | **No** | It is wrong for some reasonable input, crashes, never finishes, misreads the task, or is unsafe |

4. If unsure, use the **Try the code** panel below the code:

   | Button | What it tells you |
   | --- | --- |
   | **Run code** | Whether the code loads at all (syntax error, crash at import) |
   | **Run example test** | Whether it passes the one test the model was shown |
   | **Run my checks** | Results of your own lines: `assert ...` lines show pass or fail; any other line, such as `func([])`, shows the returned value |
   | **Run hidden tests** | Only after turning on "Reveal": the tests behind the automated label |

   The checks box starts with a few inputs derived from the example test. Edit them into edge cases
   (empty, zero, negative, one element, duplicates). Code runs in a separate process with a 3-second
   limit per check, so a hanging loop shows as a timeout instead of freezing the app.
5. Write a short note, especially for **No** or for any row where you hesitated.
6. Click **Save and next**.

Aim for 30–60 seconds per row: about 1.5–2.5 hours for all 164. Take breaks; accuracy drops when you
rush.

## 4. Rules that keep the review valid

- **Stay blind.** Keep the "Reveal automated label and hidden tests" switch **off** until you have saved
  your answer. Peeking makes the agreement score meaningless.
- **Judge the code, not the style.** Ugly but correct code is **Yes**. Clean but wrong code is **No**.
- **Judge the task as written.** If the problem text and the example test disagree, follow the example
  test, since that is what the model was given, and write `ambiguous problem` in the notes.
- **Use the "Try the code" panel for your own edge cases**, not the hidden tests. Running the hidden
  tests before answering just copies the automated label. That button stays disabled until you turn on
  "Reveal".
- **Don't change other columns.** Only `human_trustworthy` and `human_notes` are edited; the editor
  handles this.

## 5. Quick checklist for each piece of code

- Does it define the function name the example test calls, with the right arguments?
- Does it return the value (not just `print` it) in the expected type and format?
- Edge cases: empty, zero, negative, single element, duplicates, ties?
- Loop bounds: off-by-one at the start or end?
- Does it mutate its input when it shouldn't?
- Could it hang (unbounded loop, huge recursion)?
- Anything unsafe: `eval`, `exec`, `os.system`, `subprocess`, `pickle`, writing files?

## 6. Note conventions

Short, consistent notes make the disagreement analysis easy. Start with one of these tags:

| Tag | Meaning |
| --- | --- |
| `edge case` | Fails on an edge case (say which) |
| `wrong logic` | Misunderstands the algorithm or task |
| `ambiguous problem` | Task text is unclear or contradicts the example test |
| `crash` | Would raise an error for normal inputs |
| `unsafe` | Security concern |
| `unsure` | You could not decide confidently |

Example: `edge case: empty list returns None instead of 0`.

## 7. Optional: two independent reviewers (recommended)

Two people reviewing the same rows separately shows how hard the judgement is (inter-rater agreement).

1. Before anyone starts, make a copy for reviewer B:

   ```bash
   cp artifacts_pilot_openai/human_review_sample.csv artifacts_pilot_openai/human_review_sample_B.csv
   ```

2. Reviewer A uses the normal command. Reviewer B runs:

   ```bash
   uv run --with streamlit streamlit run human_edit/app.py -- --file artifacts_pilot_openai/human_review_sample_B.csv
   ```

3. Don't discuss rows until both are done.
4. Agreement between the two reviewers:

   ```bash
   uv run python -c "import pandas as pd; from sklearn.metrics import cohen_kappa_score as k; a=pd.read_csv('artifacts_pilot_openai/human_review_sample.csv'); b=pd.read_csv('artifacts_pilot_openai/human_review_sample_B.csv'); m=a.human_trustworthy.isin(['yes','no']) & b.human_trustworthy.isin(['yes','no']); print(m.sum(), 'rows, agreement', round((a.human_trustworthy[m]==b.human_trustworthy[m]).mean(),3), 'kappa', round(k(a.human_trustworthy[m], b.human_trustworthy[m]),3))"
   ```

5. The official human-vs-automated score uses reviewer A's file (`human_review_sample.csv`).

## 8. When you are done

```bash
uv run trust-review --output-dir artifacts_pilot_openai
```

This prints the reviewed-row count, raw agreement, Cohen's kappa and the confusion table. It writes them
to `human_agreement.json` and the human-review section of `run_report.md`. Check that it says
`"simulated": false`.

How to read kappa:

| Kappa | Agreement |
| --- | --- |
| below 0.40 | Weak: the automated label or the review guidelines need work |
| 0.40–0.60 | Moderate |
| 0.60–0.80 | Substantial |
| above 0.80 | Almost perfect |

## 9. After the review: what to look at

- **Automated yes, human no:** problems the hidden tests miss (weak tests) or human false alarms. Check
  the notes.
- **Automated no, human yes:** often `ambiguous problem` (benchmark label noise, like task 704) or a
  subtle bug the reviewer missed.
- Count the `ambiguous problem` notes. This estimates how much MBPP label noise affects the results.
- Pick two or three clear disagreement examples for the report.

## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| `File not found` in the editor | Run `uv run trust-features --output-dir artifacts_pilot_openai` to create the sample |
| Port 8501 already in use | Add `--server.port 8502` after `app.py` |
| Made a mistake on a row | Go back with **Previous** or **Go to row**, change the answer, save again |
| Want to undo a whole session | Restore the `human_review_sample.csv.<time>.bak` copy in the same folder |
| `trust-review` says nothing is filled in | Answers must be exactly `yes` or `no`; save at least one row in the editor |
| Accidentally reran `trust-features` mid-review | No harm: files with real answers are kept, not replaced |
