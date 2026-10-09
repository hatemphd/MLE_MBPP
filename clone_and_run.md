# Clone and Run

How to run the project after `git clone`: from the command line, in a local notebook, and in Google
Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hatemphd/MLE_MBPP/blob/main/AI_Code_Trust_Pipeline.ipynb)

(Read [section 4](#4-run-the-notebook-in-google-colab) first: the link opens only the notebook, and the
project files must be loaded into `/content` before it runs).

## Contents

1. [What the clone contains](#1-what-the-clone-contains)
2. [Run the Python pipeline locally](#2-run-the-python-pipeline-locally)
3. [Run the notebook locally](#3-run-the-notebook-locally)
4. [Run the notebook in Google Colab](#4-run-the-notebook-in-google-colab)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. What the clone contains

The repository includes the results of the full OpenAI run in `artifacts_pilot_openai/`, so
**nothing has to be generated again, no OpenAI key is needed, and nothing costs money.**

| File | What it is | In git? |
| --- | --- | --- |
| `generations.jsonl` | Every attempt `gpt-4o-mini` wrote: 38,720 attempts on 968 problems, with tokens and timing (the expensive part) | Yes |
| `problems.json` | The MBPP problems with visible and hidden tests | Yes |
| `config.json` | The settings of the run | Yes |
| `dataset.csv` | One row per attempt: features and the label | Yes |
| `human_review_sample.csv`, `human_agreement.json` | The 164 real human reviews and Cohen's kappa | Yes |
| `metrics.json`, `model_comparison.csv`, `cv_results.csv`, `test_predictions.csv`, `feature_importance.csv` | Results | Yes |
| `run_report.md`, `run_report.json`, `eda_report.md`, `figures/` | Reports and charts | Yes |
| `trust_model.joblib` | The trained model (160 MB, over GitHub's 100 MB limit) | **No**, rebuild in a few minutes |
| `candidates.csv`, `labeled.csv` | Intermediate tables | **No**, rebuild in about 20 minutes (optional) |

The notebook and the reports work straight after cloning. Only scoring new code (notebook section 8
and `trust-recommend`) needs `trust_model.joblib`, which `trust-train` recreates from `dataset.csv`.

---

## 2. Run the Python pipeline locally

### 2.1 Install uv and clone

[uv](https://docs.astral.sh/uv/) manages Python and the packages. It downloads Python 3.12 itself
(pinned in `.python-version`), so no separate Python install is needed.

```bash
# macOS / Linux (or: brew install uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone git@github.com:hatemphd/MLE_MBPP.git        # or https://github.com/hatemphd/MLE_MBPP.git
cd MLE_MBPP
```

### 2.2 Install the packages

```bash
uv sync                                   # core pipeline: enough for everything below
uv sync --extra notebook                  # add Jupyter, for section 3
```

Add `--extra api` only if you want to generate *new* attempts with OpenAI (needs a key in `.env`, see
[run_openai.md](run_openai.md)). It is not needed to use the saved run.

Without uv: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`, then drop the
`uv run` prefix from the commands below.

### 2.3 Rebuild the model (about 4 minutes, no API calls)

```bash
uv run trust-train --output-dir artifacts_pilot_openai
```

This recreates `trust_model.joblib` and rewrites the metrics, reports and figures from `dataset.csv`.
The data and the random seed are the same, so the numbers match the committed ones.

### 2.4 Other commands on the saved data

Always pass `--output-dir artifacts_pilot_openai`; each command reads `config.json` from that folder,
so `--openai` and `--prompt-styles` don't need repeating.

| You want to... | Command | Time |
| --- | --- | --- |
| Rebuild the EDA figures and `eda_report.md` | `uv run trust-eda --output-dir artifacts_pilot_openai` | Seconds |
| Recompute human agreement (kappa) | `uv run trust-review --output-dir artifacts_pilot_openai` | Seconds |
| Retrain and re-evaluate | `uv run trust-train --output-dir artifacts_pilot_openai` | ~4 min |
| Score a new piece of code (needs the model) | `uv run trust-recommend --output-dir artifacts_pilot_openai --task-id 17 --code-file solution.py` | Seconds |
| Edit the human review in a browser | `uv run --with streamlit streamlit run human_edit/app.py` | — |

### 2.5 Rebuild everything from the cached generations (optional, about 25 minutes)

To recreate `candidates.csv` and `labeled.csv` too, run the stages in order:

```bash
uv run trust-generate --output-dir artifacts_pilot_openai   # reads the cache: "... cached, 0 to generate"
uv run trust-label    --output-dir artifacts_pilot_openai   # runs every attempt against the tests (~20 min)
uv run trust-features --output-dir artifacts_pilot_openai
uv run trust-train    --output-dir artifacts_pilot_openai
```

`trust-generate` prints `N/N problems cached, 0 to generate` for every model setting and makes no API
calls. If it ever says something other than `0 to generate`, stop it (Ctrl+C): it means the cache
or the settings changed. `trust-pipeline` (all stages at once) also works, but always re-executes
everything.

To check that the code works end to end without any saved data or LLM:

```bash
uv run trust-pipeline --smoke-test --num-problems 60 --output-dir artifacts_smoke
```

---

## 3. Run the notebook locally

`AI_Code_Trust_Pipeline.ipynb` is a walkthrough of the saved results: data, labels, features, models,
decisions, human review and EDA, in under a minute. It does not call OpenAI and does not retrain
unless you ask it to.

### 3.1 Jupyter Lab in the browser

```bash
uv sync --extra notebook
uv run jupyter lab
```

Open `AI_Code_Trust_Pipeline.ipynb` and choose **Run > Run All Cells**.

### 3.2 Cursor or VS Code

1. `uv sync --extra notebook`
2. Open `AI_Code_Trust_Pipeline.ipynb`.
3. Click **Select Kernel** (top right) > **Python Environments** and choose `.venv/bin/python` inside
   the cloned folder. If it isn't listed, register it once and pick **mle-trust**:

   ```bash
   uv run python -m ipykernel install --user --name mle-trust
   ```

4. **Run All**.

### 3.3 Settings in the second cell

| Setting | Default | Meaning |
| --- | --- | --- |
| `ARTIFACTS` | `"artifacts_pilot_openai"` | Folder with the saved run |
| `RETRAIN` | `False` | `True` retrains the model (~4 min) and overwrites the model and metrics |
| `RUN_EDA` | `True` | Rebuild the EDA figures (~5 s) |

On a fresh clone the first cells print a note that `trust_model.joblib` is missing. Section 8 (scoring
new code) then prints "Skipped". Run `uv run trust-train --output-dir artifacts_pilot_openai` once, or
set `RETRAIN = True`, and rerun.

---

## 4. Run the notebook in Google Colab

Colab runs the notebook on a Google machine. It needs no install on your computer, but it starts
empty: **the Colab link opens only the `.ipynb` file, not the rest of the repository.** The notebook
imports `trust_pipeline/` and reads `artifacts_pilot_openai/`, so the whole project must first be
loaded into Colab's `/content` folder. The steps below do that.

### 4.1 Open the notebook

Choose one:

- **The link:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hatemphd/MLE_MBPP/blob/main/AI_Code_Trust_Pipeline.ipynb)
  This works directly if the repository is **public**.
- **Private repository:** in Colab choose **File > Open notebook > GitHub**, tick **Include private
  repos**, authorize Colab with GitHub, search for `hatemphd/MLE_MBPP` and pick
  `AI_Code_Trust_Pipeline.ipynb`. (You can also download the notebook from GitHub and use **File >
  Upload notebook**.)

Colab opens a copy. Use **File > Save a copy in Drive** if you want to keep your changes.

The default CPU runtime is enough; no GPU is needed (**Runtime > Change runtime type > CPU**).

### 4.2 Load the project into `/content`

Insert a new code cell at the very top (hover above the first cell and click **+ Code**, or
**Insert > Code cell** and move it up), paste one of the options below, and run it with Shift+Enter.

The project must end up in **`/content/MLE_NEW`**: the notebook's setup cell looks for it there (and
in `/content/drive/MyDrive/MLE_NEW`).

**Option A: public repository**

```python
!git clone https://github.com/hatemphd/MLE_MBPP.git /content/MLE_NEW
%cd /content/MLE_NEW
!pip install -q -e .
```

**Option B: private repository, with a token stored in Colab Secrets**

1. On GitHub: **Settings > Developer settings > Personal access tokens > Fine-grained tokens >
   Generate new token**, give it read-only **Contents** access to `MLE_MBPP`, and copy it.
2. In Colab: click the **key icon** (Secrets) in the left sidebar, **Add new secret**, name it
   `GITHUB_TOKEN`, paste the token, and switch on **Notebook access**.
3. Run:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/hatemphd/MLE_MBPP.git /content/MLE_NEW
%cd /content/MLE_NEW
!git remote set-url origin https://github.com/hatemphd/MLE_MBPP.git
!pip install -q -e .
```

The last `git remote` line removes the token from the saved git settings. Never paste the token
directly into a cell: notebooks are easy to share or commit by accident.

**Option C: keep everything in Google Drive (survives session resets)**

```python
from google.colab import drive
drive.mount("/content/drive")                       # approve the Google sign-in prompt

import os
if not os.path.exists("/content/drive/MyDrive/MLE_NEW"):
    !git clone https://github.com/hatemphd/MLE_MBPP.git /content/drive/MyDrive/MLE_NEW
%cd /content/drive/MyDrive/MLE_NEW
!pip install -q -e .
```

For a private repository, use the token URL from option B in the `git clone` line. Instead of
cloning, you can also upload the project folder to **My Drive** with the Drive website, named
`MLE_NEW`. Anything the notebook writes (the rebuilt model, figures) then stays in Drive.

**What the cell does**

- `git clone ... /content/MLE_NEW` downloads the code and the saved OpenAI run (about 60 MB) into the
  folder name the notebook expects.
- `%cd` makes it the working folder. Use `%cd`, not `!cd`: `!cd` only changes folder for that one line.
- `pip install -q -e .` installs the project and its core packages (pandas, scikit-learn, matplotlib,
  joblib, bandit, ruff, radon). Jupyter is already part of Colab, and the OpenAI package isn't needed.
  In Colab, use plain `pip` and call the commands directly (`!trust-train`), **not** `uv run`.

Check that it worked:

```python
!ls /content/MLE_NEW/artifacts_pilot_openai
```

You should see `dataset.csv`, `generations.jsonl`, `metrics.json`, `figures` and the other files from
[section 1](#1-what-the-clone-contains).

### 4.3 Run the notebook

Choose **Runtime > Run all** (or run the cells one by one). The setup cell prints
`Project: /content/MLE_NEW` when it finds the project.

To include the section 8 demo (scoring new code), build the model once per session, in a new cell
before section 8:

```python
!trust-train --output-dir artifacts_pilot_openai
```

or set `RETRAIN = True` in the second cell. It takes about 4 minutes on a laptop and can be slower on
Colab's free 2-core CPU. Neither option calls OpenAI.

### 4.4 Save results before the session ends

Colab deletes `/content` when the runtime disconnects (after idle time or about 12 hours), unless you
used option C. To download the outputs:

```python
!zip -qr /content/results.zip artifacts_pilot_openai/figures artifacts_pilot_openai/*.md artifacts_pilot_openai/*.json
from google.colab import files
files.download("/content/results.zip")
```

After a reset, run the cell from 4.2 again (option C only needs the `drive.mount` and `%cd` lines).

### 4.5 Get the latest code in a running session

```python
%cd /content/MLE_NEW
!git pull
```

Then **Runtime > Restart session** and run all cells again, so the updated `trust_pipeline` code is
imported.

---

## 5. Troubleshooting

| Problem | Fix |
| --- | --- |
| `RuntimeError: Can't find the project folder` | The project isn't in `/content/MLE_NEW` (Colab) or the kernel isn't in the cloned folder (local). In Colab, run the cell from 4.2 and check `!ls /content`. Locally, pick the `.venv` kernel. Or set `os.environ["MLE_PROJECT_ROOT"] = "/path/to/folder"` before the setup cell. |
| `ModuleNotFoundError: No module named 'trust_pipeline'` | Same cause: rerun the 4.2 cell (Colab) or select the `.venv` kernel (local). |
| `fatal: could not read Username` when cloning in Colab | The repository is private: use option B. |
| `Missing in artifacts_pilot_openai: [...]` | The folder is incomplete: clone again rather than copying single files. |
| Section 8 prints "Skipped" | `trust_model.joblib` isn't built yet: run `trust-train` (section 2.3 or 4.3). |
| `trust-generate` doesn't say `0 to generate` | Stop it: the `--output-dir` or settings differ from the saved run, and it would call the API. |
| `uv: command not found` in Colab | Expected: in Colab use `pip` and `!trust-...` commands without `uv run`. |
