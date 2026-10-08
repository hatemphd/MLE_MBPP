# Getting Started

Everything runs from the project folder with `uv run`. On an Intel Mac you can run the full pipeline in
smoke-test mode, and generate real data with the OpenAI / Anthropic models. The local Qwen models need
Colab, Apple Silicon, Linux or a GPU machine.

## Recommended order

1. **Smoke test from the terminal.** This confirms your setup works, and errors are easier to read in
   the terminal than inside notebook cells.
2. **Open the notebook in smoke mode** to understand each stage: labels, features, the grouped split,
   the metrics and the three-way decision. The explanations and plots are useful for the report.
3. **Generate real data from the terminal** (API models, or Qwen in Colab or on a GPU). It's a long job,
   and the command-line run resumes cleanly if it's interrupted.
4. **Back in the notebook**, point it at the real `artifacts/` folder for the analysis, plots and
   write-up.

The terminal is for producing results reliably; the notebook is for understanding and presenting them.

## 1. One-time setup

```bash
cd /Users/hatem/MLE_NEW
uv sync --extra notebook --extra api
```

This installs the core pipeline, Jupyter, and the OpenAI and Anthropic clients. Leave out `--extra llm`
on an Intel Mac, because the locked torch version has no Intel-Mac build.

## 2. Smoke test (no AI model, a few minutes)

```bash
uv run trust-pipeline --smoke-test --num-problems 60 --output-dir artifacts_smoke
```

This uses MBPP's reference solutions plus deliberately broken copies, so the numbers only show that the
pipeline works. They are not findings.

To see the results:

- `artifacts_smoke/metrics.json` has accuracy, precision, recall, F1, ROC AUC and the three-way decision
  rates.
- `artifacts_smoke/figures/` has the confusion matrix, ROC curve, calibration plot and risk–coverage
  plot.

## 3. Generate real data

**Option A: on your Mac, with API models**

```bash
cp .env.example .env    # then put your OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env
uv run trust-pipeline --no-local-hf --openai --anthropic --num-problems 100 --samples-per-config 5
```

`.env` is git-ignored and loaded automatically by the `trust-*` commands. Drop `--anthropic` if you only
have an OpenAI key.

**Option B: in Colab or on a GPU machine, with the free local Qwen models**

```bash
uv sync --extra llm
uv run trust-pipeline --num-problems 100 --samples-per-config 5
uv run trust-pipeline --all-problems --samples-per-config 10    # full MBPP
```

Start with about 20–100 problems. Check that between 30% and 70% of candidates are labelled
trustworthy, then scale up. Generated code is cached in `artifacts/generations.jsonl`, so an
interrupted run picks up where it stopped.

## 4. Real human review

```bash
uv run trust-pipeline ... --human-feedback real
```

Fill in the `human_trustworthy` column (yes or no) in `artifacts/human_review_sample.csv`, then run:

```bash
uv run trust-review
```

This reports how well the automated labels agree with your reviewers (Cohen's kappa), which is the
number for your write-up. Simulated review (`--human-feedback simulated`) only demonstrates the
pipeline.

## 5. Score a new solution

```bash
uv run trust-recommend --task-id 17 --code-file solution.py --sibling other_sample.py
```

It prints the probability that the solution is trustworthy and the decision: **APPROVED**,
**REJECTED** or **NEEDS HUMAN REVIEW**. `--sibling` adds other samples for the same problem, which
improves the agreement signal.

## 6. Notebook

```bash
uv run jupyter lab
```

To use it inside Cursor or VS Code instead, register a kernel and select "mle-trust" as the notebook
kernel:

```bash
uv run python -m ipykernel install --user --name mle-trust
```

On Colab, upload the whole folder, `%cd` into it, and run the notebook's first install cell.

## Tips

- **Running stages separately:** `trust-prepare`, `trust-generate`, `trust-label`, `trust-features`
  and `trust-train` run one stage at a time. Add `--help` to any command to see its options.
- **Offline:** add `HF_HOME=$PWD/.hf_cache_test HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` in front of a
  command to load MBPP from the local copy.
- **Full details:** see `README.md`.
- **Plain-English explanations:** see `how_it_works.md` for what the pipeline does step by step, how to
  test new code, how to read the output, and what sibling files are.
