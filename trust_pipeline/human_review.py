"""Human validation of the automated labels. Human labels are never a training target.

With `human_added_feedback=True` a simulated reviewer fills the sample so the agreement analysis
can run end-to-end (demo only). Otherwise the sample is exported blank for real reviewers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REVIEW_COLUMNS = ["candidate_id", "task_id", "model_name", "problem_text", "generated_code",
                  "exec_status", "label_trustworthy"]
SIMULATED_NOTE = "(simulated -- replace with real reviewer notes)"


def stratified_sample(frame, total, keys=("model_name", "exec_status"), min_per_group=3, seed=42):
    parts = []
    for _, g in frame.groupby(list(keys)):
        k = min(len(g), max(min_per_group, round(total * len(g) / len(frame))))
        parts.append(g.sample(k, random_state=seed))
    return pd.concat(parts).sort_values("candidate_id")


def simulate_human_label(row, rng):
    """Mostly agrees with the automated label; sometimes rejects messy passing code, sometimes
    misses a subtle bug in partially correct code, plus a little random noise."""
    trustworthy = bool(row["label_trustworthy"])
    if trustworthy and row["ruff_issues"] > 5:
        trustworthy = rng.random() > 0.3
    elif not trustworthy and row["exec_status"] == "wrong_output" and row["hidden_pass_frac"] > 0:
        trustworthy = rng.random() < 0.25
    if rng.random() < 0.05:
        trustworthy = not trustworthy
    return "yes" if trustworthy else "no"


def build_review_sample(frame, sample_size, simulated, seed=42):
    sample = stratified_sample(frame, sample_size, seed=seed)
    out = sample[REVIEW_COLUMNS].copy()
    if simulated:
        rng = np.random.default_rng(seed)
        out["human_trustworthy"] = [simulate_human_label(r, rng) for _, r in sample.iterrows()]
        out["human_notes"] = SIMULATED_NOTE
    else:
        out["human_trustworthy"] = ""
        out["human_notes"] = ""
    return out


def real_reviews(frame):
    """Rows a real reviewer answered: yes/no and not left over from a simulated run."""
    simulated_rows = frame["human_notes"].fillna("").astype(str).str.startswith(SIMULATED_NOTE)
    return frame[frame["human_trustworthy"].isin(["yes", "no"]) & ~simulated_rows]


def human_agreement(review_csv, simulated):
    """Cohen's kappa between human and automated labels, or None if nothing is filled in yet."""
    from sklearn.metrics import cohen_kappa_score

    reviewed = pd.read_csv(review_csv)
    reviewed = (reviewed[reviewed["human_trustworthy"].isin(["yes", "no"])] if simulated
                else real_reviews(reviewed))
    if reviewed.empty:
        return None
    human = (reviewed["human_trustworthy"] == "yes").astype(int)
    auto = reviewed["label_trustworthy"].astype(int)
    table = pd.crosstab(auto.rename("automated label"), human.rename("human label"))
    return {
        "reviewed_rows": int(len(reviewed)),
        "raw_agreement": float((human == auto).mean()),
        "cohen_kappa": float(cohen_kappa_score(human, auto)) if human.nunique() > 1 or auto.nunique() > 1 else float("nan"),
        "simulated": bool(simulated),
        "confusion": {f"auto={a}": {f"human={h}": int(table.loc[a, h]) for h in table.columns}
                      for a in table.index},
    }
