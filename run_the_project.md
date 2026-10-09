# Running the Real Project

**Use the Python pipeline for the real run, then the notebook to analyse the results.**

## Why the pipeline, not the notebook

- **Resumes:** generated code is saved as it goes. If the run stops (network error, laptop sleeps, Colab
  disconnects), rerunning the same command picks up where it left off. A notebook cell would start over.
- **Runs for hours:** it can be left running in a terminal.
- **Same code:** the notebook calls the same `trust_pipeline` package, so nothing is lost.

## Where to run it

The free local Qwen models can't run on an Intel Mac, so choose one of these:

| Option | When to choose it | Command |
| --- | --- | --- |
| **A. Your Mac with API models** | You have an OpenAI and/or Anthropic API key (small cost per call) | `uv run trust-pipeline --no-local-hf --openai --anthropic ...` |
| **B. Google Colab with local Qwen** | You want it free and can use a Colab GPU | Run the pipeline command from a Colab cell (see Step 1) |
| **C. Both** | Best data: a mix of strong and weak models gives a better spread of good and bad code | Run A and B on separate machines into separate output folders |

## Step 1: a small pilot first (about 20 problems)

**Option A, on your Mac:**

Put your API keys in a `.env` file in the project folder. It is git-ignored, and the `trust-*` commands
load it automatically (a key already exported in the shell takes priority):

```bash
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...   (leave empty if you don't have one)
```

Run with the backends you have keys for (drop `--anthropic` if you only have an OpenAI key):

```bash
uv run trust-pipeline --no-local-hf --openai --anthropic \
  --num-problems 20 --samples-per-config 5 --output-dir artifacts_pilot
```

The first lines of output include `Loaded OPENAI_API_KEY from .env`, confirming the key was found.

**Option B, in Colab:** the notebook does not contain the pipeline code; it imports the `trust_pipeline`
package, so Colab needs the **whole project folder**, not just the `.ipynb`.

1. On your Mac, zip the project without the large local folders and without your `.env` keys file:

   ```bash
   cd /Users/hatem
   zip -r MLE_NEW.zip MLE_NEW -x "MLE_NEW/.venv/*" "MLE_NEW/artifacts*" "MLE_NEW/.hf_cache_test/*" "MLE_NEW/.cache/*" "MLE_NEW/.env"
   ```

2. Open Colab and switch the runtime to a GPU (Runtime → Change runtime type).
3. Upload `MLE_NEW.zip` (Files panel on the left), then run:

   ```python
   !unzip -q MLE_NEW.zip -d /content
   %cd /content/MLE_NEW
   !pip install -q uv
   !uv pip install --system -q -e ".[llm,api]"
   ```

   This installs the project into Colab's own Python. Alternatively, open the notebook from the
   unzipped folder and run its install cell, which does the same thing.
4. Run the pilot. On Colab, call `trust-pipeline` directly, **not** `uv run trust-pipeline`; `uv run`
   would build a separate environment without the local-model packages.

   ```python
   !trust-pipeline --num-problems 20 --samples-per-config 5 --output-dir artifacts_pilot
   ```

5. Colab deletes files when the session ends. Download the results before closing:

   ```python
   !zip -rq artifacts_pilot.zip artifacts_pilot
   from google.colab import files
   files.download("artifacts_pilot.zip")
   ```

   Or mount Google Drive (`from google.colab import drive; drive.mount("/content/drive")`) and use
   `--output-dir /content/drive/MyDrive/artifacts_pilot`, which also lets an interrupted run resume in a
   new session.

**How to know the Colab run is done.** The most reliable check is that the final output files exist.

1. **The cell stops running.** While it runs, the circle next to the cell spins. When it finishes, a
   green check mark and the elapsed time appear. A red mark or a `Traceback` in the output means it
   failed; copy the last ~20 lines of the error to investigate.
2. **The output reaches the end.** Each of the 8 stages prints a banner like
   `==== [Stage 2/8] Generating candidates ====`, progress bars and a `Stage '...' finished in ...`
   line. Generation (stage 2) is by far the longest; its progress bar names the model config and
   counts problems done out of the total. The first time, the Qwen models are downloaded (about 1 GB
   for 0.5B, 3 GB for 1.5B). The run is finished when it prints `Pipeline finished in ...`, followed by
   a list of output files that starts with `RUN REPORT ... run_report.md`.
3. **The final files exist.** Run this in a new cell, using your own output folder:

   ```python
   !ls -lh /content/drive/MyDrive/artifacts_pilot
   ```

   It's done when you see `run_report.md`, `metrics.json`, `trust_model.joblib`,
   `test_predictions.csv` and a `figures/` folder. While it's still generating, you'll mainly see
   `problems.json` and a growing `generations.jsonl`.

**Watch progress during generation.** Each line of `generations.jsonl` is one attempt:

```python
!wc -l /content/drive/MyDrive/artifacts_pilot/generations.jsonl
```

The target is problems × model settings × attempts per setting (for example 20 × 3 × 5 = 300). The
settings are listed in `config.json` in the output folder.

**Look at the results when it's done:**

```python
from IPython.display import Markdown
Markdown(open("/content/drive/MyDrive/artifacts_pilot/run_report.md").read())
```

**Then check three things:**

1. **It finished without errors.**
2. **The label balance is sensible.** The share of trustworthy candidates should be somewhere between 30%
   and 70%. If almost everything passes, add weaker settings (higher temperature or a smaller model). If
   almost everything fails, use stronger models.
3. **Siblings help.** Look at the agreement feature in `artifacts_pilot/feature_importance.csv` and the
   figures. With real AI code, agreeing with siblings should raise the trust score, the opposite of the
   smoke test. If it still lowers it, investigate before scaling up.

## Step 2: the real run

```bash
uv run trust-pipeline --no-local-hf --openai --anthropic \
  --num-problems 300 --samples-per-config 5 --human-feedback real
```

- Scale to `--all-problems` once you're confident.
- In Colab, use `!trust-pipeline ...` (not `uv run`), drop `--no-local-hf --openai --anthropic` to use
  the local Qwen models, and write to Google Drive so results survive the session.
- `--human-feedback real` exports `artifacts/human_review_sample.csv` for your team to fill in.

## Step 3: human review

Fill in the `human_trustworthy` column (yes or no) in `artifacts/human_review_sample.csv`, then run:

```bash
uv run trust-review
```

This gives you the Cohen's kappa for the report. It also refreshes `run_report.md` and this run's row
in `runs_history.csv`.

## Step 4: analyse in the notebook

```bash
uv run jupyter lab
```

Open `AI_Code_Trust_Pipeline.ipynb` and set `ARTIFACTS` to your results folder (default
`artifacts_pilot_openai`). It loads the saved results and walks through every stage in under a minute,
with no API calls and no regeneration. Section 9 displays the run report. Retraining is optional
(`RETRAIN = True`, about 4 minutes).

The notebook doesn't generate data. To generate (for example with Qwen on a Colab GPU), use the
`trust-pipeline` commands in `run_in_colab.md`.

## Where to find the numbers for your report

| You need | Where |
| --- | --- |
| Every table, ready to paste | `<output-dir>/run_report.md` |
| The same numbers, machine-readable | `<output-dir>/run_report.json` |
| Comparison across runs (pilot / medium / full, different model mixes) | `runs_history.csv` in the parent of the output folder (or `--history-file`) |
| Run settings, versions, GPU, commands used | "Run" section of `run_report.md` |
| Time per stage; tokens, seconds per sample and cache reuse per model | "Stage timings" and "Generation" sections |
| Outcomes per model and prompt style, pass@k, visible-pass-but-hidden-fail rate, lint/security shares | "Execution outcomes and labels" section |
| Which failure types the dataset covers, by prompt style and provenance | "Failure-mode coverage" table |
| Test metrics per prompt style (natural samples only) | "Test metrics ... by prompt style" table (shown when several styles are used) |
| Split sizes and class balance | "Dataset and splits" section |
| CV mean ± std; test metrics with 95% CIs (bootstrap by problem) vs baselines | "Model selection" and "Test-set metrics" sections |
| Approve / reject / review shares and precision | "Three-way decisions" section |
| Cohen's kappa | "Human review agreement" section (filled in after `trust-review` when reviews are real) |
| Figures | `<output-dir>/figures/` |

Each report starts with a clear warning when the run used the smoke test or simulated human feedback.
Don't report those numbers as findings.

## Building a rich dataset: use the whole MBPP dataset

Use all of MBPP (about 970 problems) for the final dataset, but grow to it in stages. Because the
train/test split is by problem, the **number of different problems** matters more than the number of
attempts per problem.

### Why all the problems

- **More reliable evaluation.** With 20% held out for testing, all of MBPP gives about 190 unseen test
  problems; 100 problems would give only 20.
- **More variety.** Each extra problem adds new kinds of tasks and mistakes. A sixth or tenth attempt at
  the same problem adds much less.
- **It fits the plan.** `ideas.md` capped the dataset at 50,000 samples, and the whole of MBPP stays well
  under that.

### What "rich" depends on, in order of importance

1. **Many different problems:** use `--all-problems`.
2. **Several different models and settings:** mix strong and weak models, plus low and high
   temperatures, so the dataset has a real mix of good and bad code. One strong model alone produces
   mostly passing code, and that teaches the trust model little.
3. **A balanced label:** aim for roughly 30–70% trustworthy.
4. **Enough attempts per problem for sibling agreement:** 5 is fine, and 10 is better if you have time
   and budget.

### How big it gets

Candidates = problems × model settings × prompt styles × attempts per setting. With the default
`standard` prompt only, the prompt-styles factor is 1.

| Setup | Candidates |
| --- | --- |
| 970 × 3 settings × 5 attempts | about 14,500 |
| 970 × 4 settings × 5 attempts | about 19,400 |
| 970 × 4 settings × 10 attempts | about 38,800 |
| 20 × 3 settings × 5 prompt styles × 5 attempts (pilot with every style) | 1,500 |
| 300 × 3 settings × 5 prompt styles × 5 attempts | 22,500 |

Generation time grows by the same factor, so try extra styles on a pilot first.

### Optional extra: prompt variations

The core design is models × temperatures with the standard prompt. If the pilot's
**failure-mode coverage** table in `run_report.md` is thin (for example, very few "visible pass, hidden
fail" cases), add prompt styles:

```bash
uv run trust-pipeline --num-problems 20 --samples-per-config 5 --output-dir artifacts_pilot \
  --prompt-styles standard,signature_only,step_by_step,constrained,inject_bug
```

In Colab, write `!trust-pipeline ...` with the same flags. `--prompt-styles natural` is every style
except `inject_bug`, and `--prompt-styles all` is every style.

How the styles are handled:

- **Natural styles** (`signature_only`, `step_by_step`, `constrained`) are realistic agent prompts. Their
  samples are treated like standard ones: they can land in any split and count as siblings.
- **`inject_bug`** samples (`provenance = injected`) are training-only. They are never in the
  calibration or test splits, and they never count as siblings for agreement. Their comments and
  docstrings are stripped, and the comment features are switched off while they are present.
- The report breaks generation, outcomes, pass@k, failure modes and test metrics down by prompt style.
- You can reuse the same output folder: standard-prompt samples already in the cache are reused, and only
  the new styles are generated.

### Suggested plan

1. **Pilot:** `--num-problems 20`. Check that it runs and that the label balance and sibling effect look
   right.
2. **Medium:** `--num-problems 300`. Look at the metrics, adjust the model mix if the balance is off, and
   make sure it fits your time and API budget.
3. **Full:** `--all-problems --samples-per-config 5`, with the same output folder.

Use the same output folder from the medium run onward. Generated code is cached, so attempts already made
are reused rather than paid for again. Later, if you have time left, raise `--samples-per-config` to 10.

```bash
uv run trust-pipeline --no-local-hf --openai --anthropic \
  --all-problems --samples-per-config 5 --human-feedback real
```

### Practical tips

- **On Colab,** write results to Google Drive so a disconnect doesn't lose hours of work. Just rerun the
  same command to continue. A full local-model run can take several hours on a free GPU. See
  `run_in_colab.md`.
- **With API models,** run `--num-problems 300` first and check your usage dashboard to estimate the cost
  of the full run.
- **Human review stays small,** around 150–200 samples, however big the dataset gets.
- **Mention the scope in the report:** even with all of MBPP, the dataset only contains short,
  single-function Python problems.

## Summary

| Step | Tool | Output |
| --- | --- | --- |
| 1. Pilot (20 problems) | Pipeline | `artifacts_pilot/`: a check that everything works and labels are balanced |
| 2. Real run | Pipeline | `artifacts/`: dataset, trained model, `run_report.md` (report tables), `metrics.json`, figures; a row in `runs_history.csv` |
| 3. Human review | `trust-review` | Cohen's kappa |
| 4. Analysis | Notebook | Plots, tables and write-up |

See also `get_started.md` (setup), `how_it_works.md` (what each step does) and `High_Level_Explanation.md`
(concepts and metrics).
