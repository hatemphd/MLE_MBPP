# Agentic Trust for AI-Generated Code (MBPP)

Predict whether AI-generated code can be trusted and recommend **APPROVED**, **REJECTED** or
**NEEDS HUMAN REVIEW**.

- **Labels** come from running each candidate against MBPP's *hidden* tests plus Bandit security
  findings. No AI model judges the code.
- **Features** only use information available before the hidden tests run: code structure, lint and
  security findings, the one *visible* test shown in the prompt, agreement with sibling samples on
  probe inputs, and the generating model's token log-probabilities.
- **Human review** (real, or simulated for demos) checks the labels via Cohen's kappa. It is never a
  training target.
- **Splits** are grouped by problem. The calibrated probability is turned into the three-way decision
  with precision-targeted thresholds, plus split-conformal prediction as an alternative.

## Layout

| Path | What it is |
| --- | --- |
| `trust_pipeline/` | All pipeline logic (single source of truth), installed as a package by uv |
| `trust_pipeline/cli.py` | The `trust-*` commands |
| `scripts/` | The same commands as plain scripts (`uv run python scripts/<name>.py`) |
| `AI_Code_Trust_Pipeline.ipynb` | Results walkthrough on the prepared data (`artifacts_pilot_openai`): every stage in under a minute, no API calls || `pyproject.toml`, `uv.lock`, `.python-version` | uv project definition, lockfile, Python 3.12 pin |
| `requirements.txt` | Exported from `uv.lock` for environments without uv (core + API backends) |

## Setup (uv)

```bash
uv sync                                  # core pipeline (enough for --smoke-test)
uv sync --extra llm                      # + local Hugging Face generation (torch, transformers)
uv sync --extra api                      # + OpenAI / Anthropic backends
uv sync --extra notebook                 # + Jupyter
uv sync --all-extras                     # everything
```

Extras can be combined, e.g. `uv sync --extra llm --extra notebook`. The locked torch has no wheels
for Intel Macs; use Apple Silicon, Linux or Colab for the `llm` extra.

## Run

Check the pipeline end to end without any LLM (MBPP reference solutions plus single-bug mutants,
a few minutes on CPU):

```bash
uv run trust-pipeline --smoke-test --num-problems 60 --output-dir artifacts_smoke
```

Real run with the local Qwen2.5-Coder models (needs `uv sync --extra llm`; a GPU is recommended):

```bash
uv run trust-pipeline --num-problems 100 --samples-per-config 5
uv run trust-pipeline --all-problems --samples-per-config 10   # full MBPP
```

Add `--openai` / `--anthropic` to include API models (needs `uv sync --extra api` and
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`), and `--no-local-hf` to skip the local models. Put the keys
in a git-ignored `.env` file (`cp .env.example .env`); the `trust-*` commands load it automatically, and
variables already exported in the shell take priority. Generations
are cached in `artifacts/generations.jsonl`, so interrupted runs resume where they stopped.

**Optional extra: prompt variations.** By default, every model config uses the standard prompt
(problem plus the visible test). `--prompt-styles` adds more prompt styles, each crossed with every
model config:

- `standard`
- `signature_only`: function signature only, no visible test in the prompt
- `step_by_step`: brief reasoning, then the final code block
- `constrained`: concise, standard library only, no comments
- `inject_bug`: asks for one subtle, realistic bug

Presets: `natural` (all but `inject_bug`) and `all`.

```bash
uv run trust-pipeline --num-problems 20 --samples-per-config 5 \
    --prompt-styles standard,signature_only,step_by_step,constrained,inject_bug
```

Candidates = problems × model configs × prompt styles × samples.

Every row records `model`, `temperature`, `prompt_style` and `provenance`. Provenance is `natural`
for normal prompts and `injected` for `inject_bug`.

Rules for injected samples:

- They are training-only: never in the calibration or test splits.
- Their comments and docstrings are stripped.
- They are scored against natural siblings but never counted as siblings.
- While they are present, the comment features are switched off.

`prompt_style` and `provenance` are never features. Old caches stay valid: entries without a
`prompt_style` are treated as `standard`.

Stages can also be run one at a time. They share `artifacts/config.json`, and any flags you pass
override it.

```bash
uv run trust-prepare --num-problems 100   # problems.json (visible/hidden split, validated)
uv run trust-generate                     # candidates.csv
uv run trust-label                        # labeled.csv (execution, static analysis, labels)
uv run trust-features                     # dataset.csv, feature_columns.json, review sample
uv run trust-train                        # model, metrics, figures
```

Every command has `--help`, and each one is also available as a script, e.g.
`uv run python scripts/run_pipeline.py --smoke-test`. Each stage prints a banner, progress bars and
timestamped status lines. Add `--quiet` to see only warnings, or `--verbose` for extra detail.

Real human review: run with `--human-feedback real`, fill in `human_trustworthy` (yes/no) in
`artifacts/human_review_sample.csv`, then run `uv run trust-review`.

Score a new candidate with the trained model:

```bash
uv run trust-recommend --task-id 17 --code-file solution.py --sibling other_sample.py
uv run trust-recommend --problem-text "Add two numbers." \
    --visible-test "assert add(1, 2) == 3" --code-file add.py
```

If the Hugging Face Hub can't be reached, MBPP loads from the local Hugging Face cache. Set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to force offline mode.

## Notebook

```bash
uv sync --extra notebook
uv run jupyter lab
# or, for VS Code / Cursor, register a kernel and select "mle-trust":
uv run python -m ipykernel install --user --name mle-trust
```

On **Colab**, upload or clone this folder, `%cd` into it, and run the notebook's install cell. It
installs uv and runs `uv pip install --system -e ".[llm,api]"`, reusing Colab's preinstalled torch.

## Maintaining dependencies

```bash
uv add <package>                   # core dependency
uv add --optional llm <package>    # into an extra
uv lock --upgrade                  # refresh the lockfile
uv export --no-hashes --no-emit-project --extra api --format requirements.txt -o requirements.txt
```

## Artifacts (`--output-dir`, default `artifacts/`)

`problems.json`, `generations.jsonl`, `candidates.csv`, `labeled.csv`, `dataset.csv`,
`feature_columns.json`, `human_review_sample.csv`, `human_agreement.json`, `trust_model.joblib`,
`metrics.json`, `cv_results.csv`, `model_comparison.csv`, `test_predictions.csv`,
`feature_importance.csv`, and `figures/` (EDA, confusion matrix / ROC / calibration, risk–coverage,
feature importance).

**Run report.** Every stage, whether it runs inside `trust-pipeline` or as a separate command, records
its numbers in `run_report.json`. The training stage then writes:

- `run_report.md`: tables ready to paste into a report, with a warning banner for smoke-test runs or
  simulated human feedback;
- one summary row per run in `runs_history.csv`, for comparing pilot, medium and full runs.

The report covers:

- the run manifest: run id, timestamps, commands, config, seed, versions, platform, GPU;
- seconds per stage;
- generation per model config: new vs cached samples, failures, tokens, seconds per sample, tokens/s;
- outcomes per config, trustworthy rate, pass@k, and how often a visible-test pass still fails hidden
  tests;
- static-analysis shares, and rows / problems / class balance per split;
- cross-validation mean ± std per model and test metrics with 95% bootstrap confidence intervals
  (resampled by problem), compared with the baselines;
- the three-way decisions (threshold rule and conformal) and human agreement.

`runs_history.csv` sits in the parent of `--output-dir` (the project root for `artifacts/`); change the
location with `--history-file`. `trust-pipeline` and `trust-prepare` start a new run id; the other
stage commands add to the current report.

API token counts are recorded, but no cost estimate is computed.

## Caveats

- Smoke-test and simulated-human-review numbers show that the pipeline works. They are not findings.
- MBPP problems are small, isolated functions with few hidden tests, and may be in models' training
  data (consider a LiveCodeBench slice).
- Candidate code runs in a subprocess with time and memory limits. That is fine for MBPP, but use a
  container for untrusted sources.
