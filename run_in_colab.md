# Running the Project in Google Colab

Colab gives you a free GPU, so it can run the local Qwen models that don't work on an Intel Mac.

**Important:** the notebook does **not** contain the pipeline code. It imports the `trust_pipeline`
package, so Colab needs the **whole project folder**, not just the `.ipynb` file. Uploading only the
notebook fails at the `import trust_pipeline` cell.

## 1. Zip the project on your Mac

Leave out the large local folders (virtual environment, results, caches) and your `.env` keys file:

```bash
cd /Users/hatem
zip -r MLE_NEW.zip MLE_NEW -x "MLE_NEW/.venv/*" "MLE_NEW/artifacts*" "MLE_NEW/.hf_cache_test/*" "MLE_NEW/.cache/*" "MLE_NEW/.env"
```

Never put `.env` in the zip: anything uploaded to Colab can end up in shared notebooks or Drive.

## 2. Set up Colab

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook (or upload
   `AI_Code_Trust_Pipeline.ipynb`).
2. Switch the runtime to a GPU: **Runtime → Change runtime type → T4 GPU** (or better).
3. Upload `MLE_NEW.zip` using the **Files** panel on the left.

## 3. Install the project

Run in a Colab cell:

```python
!unzip -q MLE_NEW.zip -d /content
%cd /content/MLE_NEW
!pip install -q uv
!uv pip install --system -q -e ".[llm,api]"
```

This installs the project, the local-model packages (`llm`) and the API clients (`api`) into Colab's own
Python, reusing Colab's preinstalled torch. Alternatively, open `AI_Code_Trust_Pipeline.ipynb` from the
unzipped folder and run its install cell, which does the same thing.

## 4. (Recommended) Save results to Google Drive

Colab deletes all files when the session ends. Writing results to Drive keeps them, and lets an
interrupted run resume in a new session.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 5. Run the pilot (about 20 problems)

On Colab, call `trust-pipeline` directly, **not** `uv run trust-pipeline`. `uv run` would build a
separate environment without the local-model packages.

```python
!trust-pipeline --num-problems 20 --samples-per-config 5 \
  --output-dir /content/drive/MyDrive/artifacts_pilot
```

Then check:

1. **It finished without errors.**
2. **The label balance is sensible:** roughly 30–70% of candidates labelled trustworthy.
3. **Siblings help:** agreeing with siblings should raise the trust score (see `feature_importance.csv`
   and the figures).

## 6. Run the real project

```python
!trust-pipeline --num-problems 300 --samples-per-config 5 --human-feedback real \
  --output-dir /content/drive/MyDrive/artifacts
```

- Scale to `--all-problems` once you're confident.
- To add API models too, add `--openai --anthropic`. Don't upload your `.env`; store the keys in
  Colab's **Secrets** panel (the key icon on the left, enable notebook access), then load them:

  ```python
  import os
  from google.colab import userdata
  os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
  os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")   # skip if you don't have one
  ```

- Optional extra: add `--prompt-styles standard,signature_only,step_by_step,constrained,inject_bug` (or
  `natural` / `all`) for more varied code. Each style multiplies generation time, so try it on the pilot
  first. Without the flag, the run uses only the standard prompt, as before. The existing cache stays
  valid, and adding styles later to the same output folder only generates the new styles.
- If the session disconnects, rerun the same command. Generated code is cached in the output folder, so
  it resumes where it stopped.

## 7. What the run prints, and the report files

Each of the 8 stages starts with a banner such as `==== [Stage 2/8] Generating candidates ====`. The
banner gives a one-line description and the key settings. After that you see timestamped progress
messages, a progress bar with a clear label (for example `generating qwen0.5b-t0.3`), and a closing
`Stage '...' finished in ...` line. While loading, the model prints a line such as
`Loading Qwen/Qwen2.5-Coder-1.5B-Instruct on GPU (Tesla T4)`; the first time, its weights are
downloaded (about 1 GB for 0.5B models, 3 GB for 1.5B). Add `--quiet` to see only warnings, or
`--verbose` for extra detail, including messages from the Hugging Face libraries.

When the run finishes, the output folder contains:

- `run_report.md`: tables ready to paste into your report. They cover run settings and versions, time
  per stage, generation stats per model config (tokens, seconds per sample, cache reuse, failures),
  execution outcomes, pass@k, how often a visible-test pass still fails hidden tests, static analysis,
  splits, cross-validation (mean ± std), test metrics with 95% bootstrap confidence intervals
  (resampled by problem), the three-way decisions and human agreement.
- `run_report.json`: the same numbers in machine-readable form.
- `runs_history.csv`, in the folder *above* the output folder (for example `MyDrive/` when the output
  folder is `MyDrive/artifacts_pilot`): one row per run, so pilot, medium and full runs can be compared.
  Use `--history-file` to put it somewhere else.

Preview the report in Colab:

```python
from IPython.display import Markdown
Markdown(open("/content/drive/MyDrive/artifacts_pilot/run_report.md").read())
```

## 8. Updating the code while a run is in progress

To pick up a new version of the project (for example, these logging and report changes):

1. **Stop the running cell**: press the stop button next to it, or use **Runtime → Interrupt execution**.
2. **On your Mac, make a new zip** with the same command as in step 1.
3. **Upload the new `MLE_NEW.zip`** in Colab's Files panel (replace the old one), then run:

   ```python
   !unzip -oq MLE_NEW.zip -d /content
   %cd /content/MLE_NEW
   !uv pip install --system -q -e ".[llm,api]"
   ```

   `-o` overwrites the old files. The project is installed in editable mode, so the new code is used
   straight away; rerunning the install is a quick safety step that also picks up any new dependencies.
4. **Rerun exactly the same `!trust-pipeline ...` command with the same `--output-dir`.**

What is kept:

- Generated code is cached in `<output-dir>/generations.jsonl`, one entry per problem and model config.
  An entry is written as soon as all the samples for that problem and config are done. So stopping
  loses at most the problem in progress, and the rerun prints something like
  `[1/3] qwen0.5b-t0.3: ... 57/100 problems cached, 43 to generate`, then carries on from there.
- The cache is only found if you reuse **the same output folder**. A different `--output-dir`
  starts over.
- The cache lasts as long as the folder does. On Google Drive it survives the session ending. Under
  `/content` it survives only while the same runtime is alive: interrupting and rerunning is fine,
  but **Runtime → Disconnect and delete runtime**, or a session timeout, deletes it.
- Execution, labelling, features and training are always recomputed, which takes minutes rather than
  hours. The rerun starts a new run in `run_report.md`; its generation table shows how many samples
  were reused from the cache and how many are new.

## 9. Get the results back

If you used Google Drive, the results are already in `MyDrive/artifacts`. Otherwise, download them before
closing the session:

```python
!zip -rq artifacts.zip artifacts
from google.colab import files
files.download("artifacts.zip")
```

On your Mac, unzip them into `/Users/hatem/MLE_NEW/artifacts` and continue with the human review
(`uv run trust-review`) and the notebook analysis (`uv run jupyter lab`), as described in
`run_the_project.md`.

## Quick reference

| Do | Don't |
| --- | --- |
| Upload the whole project folder (zipped) | Upload only the notebook |
| Use a GPU runtime | Run the local models on CPU (very slow) |
| Call `!trust-pipeline ...` | Call `!uv run trust-pipeline ...` |
| Write results to Google Drive | Leave results only in `/content` (deleted when the session ends) |
| After updating the code, rerun with the same `--output-dir` | Change the output folder (generation starts over) |
| Paste numbers from `run_report.md` | Copy numbers by hand from the console output |
