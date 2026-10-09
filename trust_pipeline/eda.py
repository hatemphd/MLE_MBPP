"""Exploratory data analysis of a finished run: report-ready figures plus `eda_report.md`.

Reads only files the pipeline already wrote (dataset.csv, test_predictions.csv, human_review_sample.csv),
so it is fast and never calls an API.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import log

STATUS_ORDER = ["pass", "wrong_output", "runtime_error", "timeout", "syntax_error"]
STATUS_COLORS = {"pass": "#4c9f70", "wrong_output": "#e0a03a", "runtime_error": "#d0573f",
                 "timeout": "#8a6bbf", "syntax_error": "#555555"}
KEY_FEATURES = ["behavior_agreement", "behavior_agreement_visible", "probe_error_frac", "gen_mean_logprob",
                "gen_num_tokens", "code_lines", "cc_total", "ruff_issues"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, folder, name, saved):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    _plt().close(fig)
    saved.append(name)


def _config(df):
    return df["prompt_style"].fillna("standard") + " · t" + df["temperature"].astype(str)


def _md_table(frame, floatfmt="{:.3f}"):
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _error_types(series):
    found = series.fillna("").astype(str).str.findall(r"error:(\w+)")
    return pd.Series([e for errors in found for e in set(errors)]).value_counts()


def plot_outcomes(df, folder, saved):
    plt = _plt()
    shares = pd.crosstab(_config(df), df["exec_status"], normalize="index").reindex(
        columns=[s for s in STATUS_ORDER if s in df["exec_status"].unique()])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shares.plot(kind="barh", stacked=True, ax=ax, color=[STATUS_COLORS[c] for c in shares.columns])
    ax.set(title="Hidden-test outcome by prompt style and temperature", xlabel="share of candidates", ylabel="")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, folder, "01_outcomes_by_config.png", saved)


def plot_trust_and_silent(df, folder, saved):
    plt = _plt()
    by = df.groupby(["prompt_style", "temperature"])
    trust = by["label_trustworthy"].mean().unstack()
    vis = df[df["visible_pass"] == 1]
    silent = (1 - vis.groupby(["prompt_style", "temperature"])["label_trustworthy"].mean()).unstack()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    trust.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set(title="Trustworthy rate", ylabel="share", xlabel="prompt style", ylim=(0, 1))
    silent.plot(kind="bar", ax=axes[1], rot=0, color=["#d0573f", "#f0a08a"])
    axes[1].set(title="Silent failures: fail hidden tests although the visible test passes",
                ylabel="share of visible-test passes", xlabel="prompt style")
    for ax in axes:
        ax.legend(title="temperature")
    _save(fig, folder, "02_trust_and_silent_failures.png", saved)


def plot_problem_difficulty(df, folder, saved):
    plt = _plt()
    per_problem = df.groupby("task_id")["label_trustworthy"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(per_problem, bins=20, color="#4c78a8", edgecolor="white")
    axes[0].set(title="Per-problem trustworthy rate (40 candidates each)", xlabel="trustworthy rate",
                ylabel="number of problems")
    frac = df.loc[df["label_trustworthy"] == 0, "hidden_pass_frac"].dropna()
    axes[1].hist(frac, bins=np.linspace(0, 1, 11), color="#e0a03a", edgecolor="white")
    axes[1].set(title="Untrustworthy candidates: share of hidden tests passed", xlabel="hidden tests passed",
                ylabel="candidates")
    _save(fig, folder, "03_problem_difficulty.png", saved)
    return per_problem


def plot_feature_distributions(df, folder, saved):
    plt = _plt()
    features = [f for f in KEY_FEATURES if f in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, feature in zip(axes.flat, features):
        groups = [df.loc[df["label_trustworthy"] == k, feature].dropna() for k in (0, 1)]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks([1, 2], ["not trustworthy", "trustworthy"])
        ax.set_title(feature)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    fig.suptitle("Key features by label (outliers hidden)")
    fig.tight_layout()
    _save(fig, folder, "04_feature_distributions.png", saved)


def feature_label_correlation(df, feature_columns):
    cols = [c for c in feature_columns if c in df.columns and df[c].nunique() > 1]
    corr = df[cols].corrwith(df["label_trustworthy"]).dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def plot_correlations(df, corr, folder, saved):
    plt = _plt()
    top = corr.head(15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.3]})
    top[::-1].plot(kind="barh", ax=axes[0], color=["#4c9f70" if v > 0 else "#d0573f" for v in top[::-1]])
    axes[0].set(title="Correlation with the trustworthy label (top 15)", xlabel="Pearson r")
    matrix = df[list(top.index[:10])].corr()
    im = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(matrix)), matrix.columns, rotation=60, ha="right")
    axes[1].set_yticks(range(len(matrix)), matrix.columns)
    axes[1].set_title("Correlation between the top 10 features")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    _save(fig, folder, "05_feature_correlations.png", saved)


def plot_generation(df, folder, saved):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    styles = sorted(df["prompt_style"].dropna().unique())
    axes[0].boxplot([df.loc[df["prompt_style"] == s, "gen_num_tokens"].dropna() for s in styles],
                    showfliers=False)
    axes[0].set_xticks(range(1, len(styles) + 1), styles)
    axes[0].set(title="Output tokens per answer", ylabel="tokens")
    for k, color in ((1, "#4c9f70"), (0, "#d0573f")):
        axes[1].hist(df.loc[df["label_trustworthy"] == k, "gen_mean_logprob"].dropna(), bins=40, alpha=0.6,
                     color=color, label="trustworthy" if k else "not trustworthy", density=True)
    axes[1].set(title="Model confidence (mean token log-probability)", xlabel="mean log-prob", ylabel="density")
    axes[1].legend()
    _save(fig, folder, "06_generation.png", saved)


def plot_predictions(pred, folder, saved):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for k, color in ((1, "#4c9f70"), (0, "#d0573f")):
        axes[0].hist(pred.loc[pred["label_trustworthy"] == k, "p_trustworthy"], bins=25, alpha=0.6, color=color,
                     label="trustworthy" if k else "not trustworthy")
    axes[0].set(title="Predicted probability on test problems", xlabel="p(trustworthy)", ylabel="candidates")
    axes[0].legend()
    table = pd.crosstab(pred["decision"], pred["label_trustworthy"].map({1: "trustworthy", 0: "not trustworthy"}))
    table.plot(kind="bar", ax=axes[1], rot=0, color=["#d0573f", "#4c9f70"])
    axes[1].set(title="Three-way decision vs. true label (threshold rule)", xlabel="", ylabel="candidates")
    _save(fig, folder, "07_predictions.png", saved)


def human_summary(review):
    answered = review[review["human_trustworthy"].isin(["yes", "no"])
                      & ~review["human_notes"].fillna("").astype(str).str.startswith("(")].copy()
    if answered.empty:
        return answered, None
    answered["human"] = (answered["human_trustworthy"] == "yes").astype(int)
    answered["agree"] = answered["human"] == answered["label_trustworthy"].astype(int)
    by_status = answered.groupby("exec_status")["agree"].agg(["mean", "size"]).rename(
        columns={"mean": "agreement", "size": "rows"})
    return answered, by_status


def plot_human(answered, by_status, folder, saved):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    matrix = pd.crosstab(answered["label_trustworthy"].astype(int), answered["human"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0)
    axes[0].imshow(matrix, cmap="Blues")
    for (r, c), v in np.ndenumerate(matrix.values):
        axes[0].text(c, r, str(v), ha="center", va="center", fontsize=14)
    axes[0].set_xticks([0, 1], ["human: no", "human: yes"])
    axes[0].set_yticks([0, 1], ["automated: no", "automated: yes"])
    axes[0].set_title("Human vs. automated label")
    by_status["agreement"].plot(kind="bar", ax=axes[1], rot=30, color="#4c78a8", ylim=(0, 1.05))
    axes[1].set(title="Agreement by execution outcome", ylabel="agreement", xlabel="")
    for i, (value, n) in enumerate(zip(by_status["agreement"], by_status["rows"])):
        axes[1].text(i, value + 0.02, f"n={n}", ha="center", fontsize=9)
    _save(fig, folder, "08_human_review.png", saved)


def run_eda(output_dir):
    dataset_path = os.path.join(output_dir, "dataset.csv")
    if not os.path.exists(dataset_path):
        raise SystemExit(f"{dataset_path} not found; run trust-features first.")
    folder = os.path.join(output_dir, "figures", "eda")
    os.makedirs(folder, exist_ok=True)
    df = pd.read_csv(dataset_path, low_memory=False)
    df = df[df["provenance"].fillna("natural") == "natural"] if "provenance" in df else df
    columns_path = os.path.join(output_dir, "feature_columns.json")
    feature_columns = KEY_FEATURES
    if os.path.exists(columns_path):
        with open(columns_path) as f:
            feature_columns = json.load(f)["feature_columns"]
    saved = []
    log.info(f"EDA on {len(df):,} candidates from {df['task_id'].nunique()} problems -> {folder}")

    plot_outcomes(df, folder, saved)
    plot_trust_and_silent(df, folder, saved)
    per_problem = plot_problem_difficulty(df, folder, saved)
    plot_feature_distributions(df, folder, saved)
    corr = feature_label_correlation(df, feature_columns)
    plot_correlations(df, corr, folder, saved)
    plot_generation(df, folder, saved)

    pred_path = os.path.join(output_dir, "test_predictions.csv")
    pred = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
    if pred is not None:
        plot_predictions(pred, folder, saved)

    review_path = os.path.join(output_dir, "human_review_sample.csv")
    answered, by_status = (human_summary(pd.read_csv(review_path, keep_default_na=False))
                           if os.path.exists(review_path) else (pd.DataFrame(), None))
    if by_status is not None:
        plot_human(answered, by_status, folder, saved)

    report = _report(df, per_problem, corr, pred, answered, by_status, saved)
    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Wrote {len(saved)} figures and {report_path}")
    return report_path


def _report(df, per_problem, corr, pred, answered, by_status, saved):
    vis = df["visible_pass"] == 1
    silent = int((vis & (df["label_trustworthy"] == 0)).sum())
    config = df.assign(config=_config(df)).groupby("config").agg(
        rows=("label_trustworthy", "size"), trustworthy=("label_trustworthy", "mean"),
        visible_pass=("visible_pass", "mean"), mean_tokens=("gen_num_tokens", "mean")).reset_index()
    status = df["exec_status"].value_counts().rename_axis("outcome").reset_index(name="rows")
    status["share"] = status["rows"] / len(df)
    hardest = (df.groupby("task_id").agg(trustworthy=("label_trustworthy", "mean"),
                                         visible_pass=("visible_pass", "mean"),
                                         problem=("problem_text", "first"))
               .query("visible_pass > 0.5").sort_values("trustworthy").head(10).reset_index())
    hardest["problem"] = hardest["problem"].str.slice(0, 80)
    errors = _error_types(df.get("hidden_results", pd.Series(dtype=str))).head(10).rename_axis(
        "error type").reset_index(name="candidates")
    bands = pd.cut(per_problem, [-0.01, 0, 0.5, 0.99, 1], labels=["0% (always fails)", "1–50%", "51–99%",
                                                                   "100% (always correct)"])
    band_table = bands.value_counts().reindex(bands.cat.categories).rename_axis("problem trust rate") \
        .reset_index(name="problems")

    lines = [
        "# Exploratory data analysis",
        "",
        (f"{len(df):,} natural candidates from {df['task_id'].nunique()} MBPP problems "
        f"({df['model_name'].nunique()} model settings × {df['prompt_style'].nunique()} prompt styles). "
        "Figures are in `figures/eda/`."),
        "",
        "## 1. Overview",
        "",
        f"- Trustworthy: **{df['label_trustworthy'].mean():.1%}**",
        f"- Pass the visible test: {vis.mean():.1%}",
        (f"- Pass the visible test but untrustworthy (hidden-test failure or security finding): **{silent:,}** = "
        f"{silent / max(vis.sum(), 1):.1%} of visible passes"),
        "",
        _md_table(status),
        "",
        "![](figures/eda/01_outcomes_by_config.png)",
        "",
        "## 2. By prompt style and temperature",
        "",
        _md_table(config),
        "",
        "![](figures/eda/02_trust_and_silent_failures.png)",
        "",
        "## 3. Problems",
        "",
        "Failures cluster by problem: most problems are either always or never solved.",
        "",
        _md_table(band_table),
        "",
        "Hardest problems that usually pass the visible test (silent-failure hot spots):",
        "",
        _md_table(hardest),
        "",
        "![](figures/eda/03_problem_difficulty.png)",
        "",
        "## 4. Error types in hidden tests",
        "",
        _md_table(errors) if len(errors) else "No error types recorded.",
        "",
        "## 5. Features",
        "",
        ("Correlation of each feature with the trustworthy label (top 15 by absolute value). Correlation is not "
        "the same as model importance; see `feature_importance.csv`."),
        "",
        _md_table(corr.head(15).rename("r").rename_axis("feature").reset_index()),
        "",
        "![](figures/eda/04_feature_distributions.png)",
        "",
        "![](figures/eda/05_feature_correlations.png)",
        "",
        "![](figures/eda/06_generation.png)",
    ]
    if pred is not None:
        decisions = pd.crosstab(pred["decision"], pred["label_trustworthy"], margins=True).reset_index()
        decisions.columns = ["decision", "not trustworthy", "trustworthy", "total"]
        lines += ["", "## 6. Model predictions on the test problems", "", _md_table(decisions), "",
                  "![](figures/eda/07_predictions.png)"]
    if by_status is not None:
        lines += ["", "## 7. Human review", "",
                  (f"{len(answered)} rows reviewed; agreement with the automated label "
                  f"{answered['agree'].mean():.1%}. Kappa is in `human_agreement.json`."), "",
                  _md_table(by_status.reset_index()), "", "![](figures/eda/08_human_review.png)"]
    return "\n".join(lines) + "\n"
