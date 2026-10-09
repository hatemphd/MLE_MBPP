# Running the Project with OpenAI

This runs the full pipeline on your Mac using OpenAI's `gpt-4o-mini` to write the code. No GPU or Colab
is needed. The `--openai` option uses two settings: `gpt-4o-mini` at temperature 0.3 and at 1.0.

**Use one output folder for every OpenAI run** (`artifacts_pilot_openai` below). Generated code is cached
there, so each bigger run only pays for what's new. A different `--output-dir` starts over (see
section 9).

## 1. One-time setup

```bash
cd /Users/hatem/MLE_NEW
uv sync --extra api --extra notebook
cp .env.example .env
```

Open `.env` and add your key (leave the Anthropic line empty if you don't have one):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=
```

`.env` is git-ignored and loaded automatically by every `trust-*` command. Never commit it or put it in
the Colab zip.

## 2. First pilot (20 problems, 200 API calls, about 2 minutes)

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 20 --samples-per-config 5 --output-dir artifacts_pilot_openai
```

- The first lines include `Loaded OPENAI_API_KEY from .env`, confirming the key was found.
- Each stage prints a banner such as `==== [Stage 2/8] Generating candidates ====`.
- It's done when it prints `Pipeline finished ...` and the location of
  `artifacts_pilot_openai/run_report.md`.

## 3. Check the pilot

Open `artifacts_pilot_openai/run_report.md` and check:

1. **Label balance:** the share labelled trustworthy should be roughly 30–70%.
2. **Failure coverage:** the failure-mode table should show wrong answers, crashes and, most importantly,
   code that **passes the visible test but fails the hidden tests**.
3. **Siblings help:** trustworthy code should agree with its siblings more, and `behavior_agreement`
   should matter in `feature_importance.csv`.

**Add variety if either of these is true:**

- more than about 85% is trustworthy, or
- the failure coverage is thin, especially if "visible pass, hidden fail" is 0.

### First pilot results (2026-10-07)

| Check | Result | Verdict |
| --- | --- | --- |
| Label balance | 76% trustworthy (75% at temperature 0.3, 77% at 1.0) | Slightly above target |
| Failure coverage | 48 wrong answers; no crashes, syntax errors, timeouts or security findings | Too narrow |
| Visible pass, hidden fail | **0 cases**: passing the example test always meant passing the hidden tests | Key case missing |
| Siblings | Trustworthy code agrees more with siblings (0.91 vs 0.72), but only `visible_pass` mattered to the model | Right direction, not used yet |
| Model | Test metrics 1.0, but the visible-test-only baseline also scored 1.0 | No value beyond the baseline yet |

The 48 failures came from just 6 of the 20 problems, and in 4 of them almost every attempt failed:
`gpt-4o-mini` consistently misreads those problems, so it fails the example test too. Failing code also
had higher token confidence than correct code: the model was confidently wrong. **Decision: add
variety (section 4).**

## 4. Expanded pilot: 100 problems and four prompt styles

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 100 --prompt-styles natural --samples-per-config 5 --output-dir artifacts_pilot_openai
```

| Part | Meaning |
| --- | --- |
| `--num-problems 100` | The first 100 MBPP problems (they include the first pilot's 20) |
| `--prompt-styles natural` | Four styles that never ask for bugs: `standard`, `signature_only`, `step_by_step`, `constrained` |
| `--samples-per-config 5` | 5 attempts per problem, model setting and prompt style |
| Same `--output-dir` | The first pilot's 200 attempts are reused |

**Size:** 100 problems × 2 settings × 4 styles × 5 attempts = 4,000 attempts, of which about 3,800 are
new API calls. Expect roughly 15–30 minutes, and likely well under a dollar with `gpt-4o-mini` (check your
[usage dashboard](https://platform.openai.com/usage)).

| Style | What `gpt-4o-mini` is given | Expected effect |
| --- | --- | --- |
| `standard` | Problem plus the example test | Same as the first pilot |
| `signature_only` | Problem plus function name and arguments, **no example test** | More misunderstandings, more varied wrong code |
| `step_by_step` | Asked to reason briefly, then write the function | Different kinds of mistakes |
| `constrained` | Concise, standard library only, no comments | Different style and shortcuts |

Every attempt is still run against the visible test (a feature) and the hidden tests (the label).

**Then repeat the three checks.** If "visible pass, hidden fail" is still 0, add bug injection, which
deliberately asks for bugs that pass the example test (training-only samples):

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 100 --prompt-styles all --samples-per-config 5 --output-dir artifacts_pilot_openai
```

Crashes and syntax errors mainly come from weaker models; add the Colab Qwen results
(`run_in_colab.md`) for those.

The chosen prompt styles are saved in the folder's `config.json`, so later runs in this folder keep them.
Pass `--prompt-styles standard` to go back.

## 5. Medium run (300 problems)

Same folder, so the 100-problem attempts are reused. Use the prompt styles you settled on in section 4:

```bash
uv run trust-pipeline --no-local-hf --openai --num-problems 300 --prompt-styles natural --samples-per-config 5 --human-feedback real --output-dir artifacts_pilot_openai
```

Afterwards, check your usage dashboard to estimate the cost of the full run. The report also records the
tokens used.

## 6. Full run (all ~970 problems)

```bash
uv run trust-pipeline --no-local-hf --openai --all-problems --prompt-styles natural --samples-per-config 5 --human-feedback real --output-dir artifacts_pilot_openai
```

If it stops (network drop, rate limit, laptop sleep), rerun the same command. It resumes where it
stopped.

## 7. Human review

`--human-feedback real` exports `artifacts_pilot_openai/human_review_sample.csv`. Fill in the
`human_trustworthy` column (yes or no), then run:

```bash
uv run trust-review --output-dir artifacts_pilot_openai
```

This adds Cohen's kappa to `run_report.md`.

## 8. Results for the report

| What | Where |
| --- | --- |
| Every table, ready to paste | `artifacts_pilot_openai/run_report.md` |
| The same numbers, machine-readable | `artifacts_pilot_openai/run_report.json` |
| Comparison of the pilot, medium and full runs | `runs_history.csv` (project folder) |
| Charts | `artifacts_pilot_openai/figures/` |

To explore and present the results:

```bash
uv run jupyter lab
```

## 9. Where the data is saved, and how to avoid reruns

### Files in the output folder

| File | What it contains | Cost to recreate |
| --- | --- | --- |
| `problems.json` | The MBPP problems used, with visible and hidden tests | Seconds |
| **`generations.jsonl`** | **Every attempt the AI wrote: code, model, temperature, prompt style, tokens, timing** | **Expensive: API calls, money and time** |
| `candidates.csv` | Those attempts as a table | Seconds |
| `labeled.csv` | Plus test results, lint and security findings, and the label | Minutes |
| **`dataset.csv`** | **The final ML dataset: one row per attempt with all features and the label** | Minutes |
| `human_review_sample.csv` | The sample for human review | Seconds |
| `trust_model.joblib`, `metrics.json`, `run_report.md`, `figures/` | The trained model and results | Minutes |
| `config.json` | The settings used for this folder | — |

`dataset.csv` is the file for your own analysis; `generations.jsonl` is the one to protect.

### Avoiding reruns

1. **Generation is cached automatically.** Each finished problem, model setting and prompt style is
   written to `generations.jsonl`. Reruns print a line like `100/100 problems cached, 0 to generate`
   and only generate what's missing. Failed calls aren't cached, so they're retried.
2. **Always use the same `--output-dir`.** A new folder name means an empty cache.
3. **Rerun only the stages you need.** `trust-pipeline` reuses generation but always redoes execution,
   features and training (minutes). For less:

   | You want to... | Run |
   | --- | --- |
   | Retrain and re-evaluate (fastest) | `uv run trust-train --output-dir artifacts_pilot_openai` |
   | Rebuild features, then retrain | `uv run trust-features --output-dir artifacts_pilot_openai`, then `trust-train` |
   | Re-execute tests, rebuild, retrain | `uv run trust-label --output-dir artifacts_pilot_openai`, then `trust-features`, then `trust-train` |
   | Pick up a code-cleaning change (for example the self-written-test fix) | `uv run trust-generate --output-dir artifacts_pilot_openai` (cache only, no API calls), then `trust-label`, `trust-features`, `trust-train` |
   | Update kappa after human review | `uv run trust-review --output-dir artifacts_pilot_openai` |

   These read the folder's `config.json`, so you don't need to repeat `--openai` or `--prompt-styles`.
4. **Back up after expensive runs.** `artifacts*` folders are git-ignored (except the files of
   `artifacts_pilot_openai` listed in the next section), so nothing else protects them:

   ```bash
   cp -r artifacts_pilot_openai ~/Backups/artifacts_pilot_openai_$(date +%Y%m%d)
   ```

   Losing `dataset.csv` or the model costs minutes; losing `generations.jsonl` means paying for
   generation again.
5. **Renaming the folder is fine** (for example to `artifacts_openai`) as long as no run is in progress
   and you use the new name in every later command. The cache travels with the folder.

### The full run is saved in GitHub

The repository tracks the expensive and the reported files of `artifacts_pilot_openai`:
`generations.jsonl` (all OpenAI outputs), `problems.json`, `config.json`, `dataset.csv`, the human
review, metrics, reports and figures. Not tracked: `candidates.csv` and `labeled.csv` (rebuilt in
about 20 minutes) and `trust_model.joblib` (160 MB, over GitHub's 100 MB file limit).

After cloning, no OpenAI key or API calls are needed:

```bash
git clone git@github.com:hatemphd/MLE_MBPP.git && cd MLE_MBPP
uv sync --extra api --extra notebook
uv run trust-train --output-dir artifacts_pilot_openai     # about 4 minutes; recreates trust_model.joblib
```

The notebook and reports work straight away; only the scoring demo needs the model. To also rebuild
`candidates.csv` and `labeled.csv`, run `trust-generate` (reads the cache, prints `0 to generate`),
`trust-label`, `trust-features` and `trust-train` with `--output-dir artifacts_pilot_openai`.

### What causes regeneration

- A different `--output-dir`.
- Deleting or editing `generations.jsonl`.
- Raising `--samples-per-config` (only the extra attempts are generated).
- Changing model names or temperatures in the code.

## 10. Test new code with the trained model

```bash
uv run trust-recommend --output-dir artifacts_pilot_openai --task-id 17 --code-file solution.py --sibling other.py
```

## Tips

- **`No module named 'openai'`:** the API packages aren't installed. Run
  `uv sync --extra api --extra notebook`, then rerun the same command; failed calls aren't cached, so
  nothing needs cleaning up. Always list every extra you use: `uv sync` keeps only the extras you name,
  so a plain `uv sync` (or `uv sync --extra notebook` alone) removes the OpenAI package again.
- Write each command on one line, or make sure each `\` is the very last character on its line.
- A key exported in the shell (`export OPENAI_API_KEY=...`) takes priority over `.env`.
- To add Anthropic models later, put `ANTHROPIC_API_KEY` in `.env` and add `--anthropic`.
