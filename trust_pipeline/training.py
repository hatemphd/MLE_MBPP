"""Grouped splitting, baselines vs. classifiers, calibration and evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TRAINING_ONLY_PROVENANCE

BASELINES = {"majority", "visible_test_only"}
LABEL = "label_trustworthy"


def grouped_split(df, seed=42, smoke_test=False):
    """60/20/20 train/calibration/test split by task_id. Training-only rows (mutants, injected-bug
    samples) are removed from calibration and test, except in the smoke test (which has only mutants)."""
    groups = df["task_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_rel, cal_rel = next(inner.split(df.iloc[trainval_idx], groups=groups[trainval_idx]))
    train = df.iloc[trainval_idx[tr_rel]]
    calib = df.iloc[trainval_idx[cal_rel]]
    test = df.iloc[test_idx]
    if not smoke_test:
        calib = calib[~calib["provenance"].isin(TRAINING_ONLY_PROVENANCE)]
        test = test[~test["provenance"].isin(TRAINING_ONLY_PROVENANCE)]
    assert not set(train.task_id) & set(test.task_id)
    assert not set(train.task_id) & set(calib.task_id)
    return {"train": train, "calibration": calib, "test": test}


def split_summary(splits):
    return pd.DataFrame({
        name: {"rows": len(part), "problems": part.task_id.nunique(),
               "trustworthy_rate": round(part[LABEL].mean(), 3)}
        for name, part in splits.items()
    }).T


def build_models(seed=42):
    return {
        "majority": DummyClassifier(strategy="prior"),
        "visible_test_only": make_pipeline(
            ColumnTransformer([("visible", "passthrough", ["visible_pass"])]),
            LogisticRegression()),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced")),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                                   n_jobs=-1, random_state=seed)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed),
    }


def grouped_cv_splits(X, y, groups, n_splits=5):
    n_splits = min(n_splits, pd.Series(groups).nunique())
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))


def cross_validate_models(models, X, y, cv_splits):
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv_splits,
                                scoring=["roc_auc", "average_precision", "f1", "accuracy"])
        row = {"model": name}
        for k, v in scores.items():
            if k.startswith("test_"):
                row[k.replace("test_", "cv_")] = float(np.nanmean(v))
                row[k.replace("test_", "cv_") + "_std"] = float(np.nanstd(v))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_best(cv_table):
    return cv_table.drop(index=[m for m in BASELINES if m in cv_table.index])["cv_roc_auc"].idxmax()


def fit_calibrated(model, X, y, cv_splits):
    method = "isotonic" if len(X) >= 1000 else "sigmoid"
    calibrated = CalibratedClassifierCV(clone(model), method=method, cv=cv_splits)
    return calibrated.fit(X, y), method


def binary_metrics(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, proba)),
    }
