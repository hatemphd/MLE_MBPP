"""Figures. Each function returns the matplotlib figure and saves it when `path` is given."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recommendation import decision_report, risk_coverage


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_eda(df, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df.groupby("model_name")["label_trustworthy"].mean().sort_values().plot(
        kind="barh", ax=axes[0], title="Trustworthy rate by model/setting")
    df["exec_status"].value_counts().plot(kind="bar", ax=axes[1], title="Hidden-test outcome")
    groups = [df.loc[df.label_trustworthy == k, "behavior_agreement"].dropna() for k in (0, 1)]
    axes[2].boxplot(groups)
    axes[2].set_xticks([1, 2], ["not trustworthy", "trustworthy"])
    axes[2].set_title("Behaviour agreement vs. label")
    return _finish(fig, path)


def plot_evaluation(y_test, test_proba, calibrated_name, path=None):
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    p = test_proba[calibrated_name]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (p >= 0.5).astype(int), display_labels=["not_trustworthy", "trustworthy"],
        cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion matrix ({calibrated_name})")
    if pd.Series(y_test).nunique() > 1:
        for name, proba in test_proba.items():
            if name != "majority":
                RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set_title("ROC curves (test set)")
        CalibrationDisplay.from_predictions(y_test, p, n_bins=10, strategy="quantile",
                                            name=calibrated_name, ax=axes[2])
        axes[2].set_title("Reliability diagram")
    return _finish(fig, path)


def plot_risk_coverage(y_test, p_test, operating_points, path=None):
    """operating_points: {label: decisions array}."""
    import matplotlib.pyplot as plt
    coverage, risk = risk_coverage(y_test, p_test)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coverage, risk, label="calibrated model")
    for (label, decisions), marker in zip(operating_points.items(), ["o", "s", "^"]):
        rep = decision_report(y_test, decisions)
        if not np.isnan(rep["automated_error_rate"]):
            ax.scatter(1 - rep["review_share"], rep["automated_error_rate"], marker=marker, s=80,
                       zorder=3, label=label)
    ax.set_xlabel("Coverage (share of candidates decided automatically)")
    ax.set_ylabel("Error rate among automated decisions")
    ax.set_title("Risk–coverage curve (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_feature_importance(importances, top=15, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.head(top).iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top} features (permutation importance, test ROC AUC)")
    ax.set_xlabel("Mean drop in ROC AUC")
    return _finish(fig, path)
