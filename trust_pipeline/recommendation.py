"""Three-way trust decision: APPROVED / REJECTED / NEEDS HUMAN REVIEW."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

APPROVED, REJECTED, REVIEW = "APPROVED", "REJECTED", "NEEDS HUMAN REVIEW"
NAN = float("nan")


def choose_thresholds(y, p, target_approve, target_reject, min_support=10):
    """(t_low, t_high) chosen so the approved / rejected sets reach the target precision.

    Approval is only possible at p >= 0.5 and rejection only at p < 0.5. A target that can't be
    met disables that decision (infinite threshold) instead of relaxing the target.
    """
    y, p = np.asarray(y), np.asarray(p)
    candidates = np.unique(p)
    t_high = next((t for t in candidates[candidates >= 0.5]
                   if (p >= t).sum() >= min_support and y[p >= t].mean() >= target_approve), np.inf)
    t_low = next((t for t in candidates[candidates < 0.5][::-1]
                  if (p <= t).sum() >= min_support and 1 - y[p <= t].mean() >= target_reject), -np.inf)
    return float(t_low), float(t_high)


def trust_decision(p, t_low, t_high):
    if p >= t_high:
        return APPROVED
    if p <= t_low:
        return REJECTED
    return REVIEW


def conformal_quantile(y, p, alpha):
    """Split-conformal quantile of the nonconformity score 1 - p(true class)."""
    y, p = np.asarray(y), np.asarray(p)
    scores = np.where(y == 1, 1 - p, p)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_decision(p, qhat):
    can_be_good, can_be_bad = (1 - p) <= qhat, p <= qhat
    if can_be_good and not can_be_bad:
        return APPROVED
    if can_be_bad and not can_be_good:
        return REJECTED
    return REVIEW


def conformal_coverage(y, p, qhat):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.where(y == 1, (1 - p) <= qhat, p <= qhat).mean())


def decision_report(y, decisions):
    y, decisions = np.asarray(y), np.asarray(decisions)
    approved, rejected = decisions == APPROVED, decisions == REJECTED
    auto = approved | rejected
    errors = (approved & (y == 0)) | (rejected & (y == 1))
    return {
        "approved_share": float(approved.mean()),
        "rejected_share": float(rejected.mean()),
        "review_share": float(1 - auto.mean()),
        "approved_precision": float(y[approved].mean()) if approved.any() else NAN,
        "rejected_precision": float(1 - y[rejected].mean()) if rejected.any() else NAN,
        "automated_error_rate": float(errors[auto].mean()) if auto.any() else NAN,
        "untrustworthy_code_approved": int((approved & (y == 0)).sum()),
    }


def risk_coverage(y, p):
    """Error rate among the most confident fraction of decisions, for every coverage level."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-np.maximum(p, 1 - p))
    wrong = ((p >= 0.5).astype(int) != y)[order]
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(wrong) / n


@dataclass
class TrustRecommender:
    """Calibrated model + decision thresholds; saved as one joblib artifact."""
    model: object
    feature_columns: list
    model_names: list
    t_low: float
    t_high: float
    qhat: float
    model_label: str = ""
    config: dict = field(default_factory=dict)

    def predict_proba(self, frame):
        return self.model.predict_proba(frame[self.feature_columns])[:, 1]

    def decide(self, p):
        return trust_decision(p, self.t_low, self.t_high)

    def decide_conformal(self, p):
        return conformal_decision(p, self.qhat)

    def recommend(self, problem, code, siblings=(), executor=None):
        """Score one new candidate for `problem` (a dict from data.make_problem).

        Runs the visible test, probes, static analysis and code features. Pass other samples for the
        same problem as `siblings` to get self-consistency features. Log-probabilities are imputed.
        """
        from .execution import Executor
        from .features import add_code_features, add_consistency_features, add_execution_features
        from .static_analysis import add_static_features

        executor = executor or Executor(progress=False)
        codes = [code, *siblings]
        frame = pd.DataFrame({
            "candidate_id": [f"new::{i}" for i in range(len(codes))],
            "task_id": problem["task_id"], "model_name": "new", "provenance": "new",
            "generated_code": codes,
            "gen_mean_logprob": NAN, "gen_min_logprob": NAN, "gen_num_tokens": NAN,
        })
        by_id = {problem["task_id"]: problem}
        frame = add_execution_features(frame, by_id, executor)
        frame = add_consistency_features(frame, all_in_pool=True)
        frame = add_static_features(frame)
        frame = add_code_features(frame, by_id, self.model_names)
        p = float(self.predict_proba(frame.iloc[[0]])[0])
        row = frame.iloc[0]
        return {
            "p_trustworthy": round(p, 3),
            "decision": self.decide(p),
            "conformal_decision": self.decide_conformal(p),
            "visible_pass": int(row["visible_pass"]),
            "behavior_agreement": None if pd.isna(row["behavior_agreement"]) else round(float(row["behavior_agreement"]), 3),
            "ruff_issues": int(row["ruff_issues"]),
            "bandit_med_high": int(row["bandit_med_high"]),
        }

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
