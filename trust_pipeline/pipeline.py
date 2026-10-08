"""End-to-end stages shared by the CLI scripts and the notebook.

Each stage reads its inputs from the artifacts directory when they aren't passed in, and writes its
outputs there, so stages can run one at a time (scripts/) or all together (run_pipeline.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import data, features, generation, human_review, log, plots, report, training
from .config import TRAINING_ONLY_PROVENANCE, Config
from .execution import Executor
from .recommendation import (TrustRecommender, choose_thresholds, conformal_coverage,
                             conformal_decision, conformal_quantile, decision_report, trust_decision)
from .static_analysis import RUFF_RULES, add_static_features

LABEL = "label_trustworthy"
EXEC_STATUS_ORDER = ["pass", "wrong_output", "runtime_error", "timeout", "syntax_error"]

STAGES = {
    "prepare": "Preparing MBPP problems",
    "generate": "Generating candidates",
    "execute": "Executing candidates (features + labels)",
    "static": "Static analysis (ruff + bandit) and labels",
    "features": "Building features",
    "human": "Human review sample",
    "train": "Training and calibrating the classifier",
    "decide": "Decision rules, evaluation and artifacts",
}


def _banner(key, description="", **settings):
    keys = list(STAGES)
    log.banner(STAGES[key], description, keys.index(key) + 1, len(keys), **settings)


@contextmanager
def _timed(paths, key):
    """Log and record (in the run report) how long a stage took."""
    start = time.time()
    yield
    seconds = time.time() - start
    log.info(f"Stage '{key}' finished in {log.format_seconds(seconds)}")
    report.record_stage(paths, key, seconds)


class ArtifactPaths:
    def __init__(self, root):
        self.root = root
        self.figures = os.path.join(root, "figures")
        os.makedirs(self.figures, exist_ok=True)

    def __getattr__(self, name):
        files = {
            "config": "config.json",
            "problems": "problems.json",
            "generations": "generations.jsonl",
            "candidates": "candidates.csv",
            "labeled": "labeled.csv",
            "dataset": "dataset.csv",
            "feature_columns": "feature_columns.json",
            "human_review": "human_review_sample.csv",
            "human_agreement": "human_agreement.json",
            "model": "trust_model.joblib",
            "metrics": "metrics.json",
            "cv_results": "cv_results.csv",
            "model_comparison": "model_comparison.csv",
            "predictions": "test_predictions.csv",
            "importance": "feature_importance.csv",
            "run_report": "run_report.json",
            "run_report_md": "run_report.md",
        }
        if name in files:
            return os.path.join(self.root, files[name])
        raise AttributeError(name)

    def figure(self, name):
        return os.path.join(self.figures, name)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    raise TypeError(type(o))


def _write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _load_problems(paths, problems):
    return problems if problems is not None else data.read_problems(paths.problems)


def _describe_models(cfg):
    if cfg.smoke_test:
        return "SMOKE_TEST (reference solutions + AST mutants, no LLM)"
    return ", ".join(f"{m['name']} ({m.get('model', m['provider'])})" for m in cfg.resolved_model_configs())


# ------------------------------------------------------------------------------------- stages ----

def stage_prepare(cfg, paths, executor=None):
    """Load MBPP, split visible/hidden tests, validate reference solutions."""
    _banner("prepare", "Load MBPP, keep test 1 as the visible test and hide the rest, and drop problems "
                       "whose reference solution fails its own tests.",
            problems="all" if cfg.num_problems is None else cfg.num_problems,
            seed=cfg.seed, output_dir=cfg.output_dir)
    report.ensure(paths, cfg, "prepare", fresh_if_new=True)
    with _timed(paths, "prepare"):
        executor = executor or Executor.from_config(cfg)
        problems, dropped = data.load_problems(cfg, executor)
        data.save_problems(problems, paths.problems)
        cfg.save(paths.config)
        log.info(f"Kept {len(problems)} problems ({dropped} dropped: reference failed its own tests) "
                 f"-> {paths.problems}")
    return problems


def stage_generate(cfg, paths, problems=None):
    problems = _load_problems(paths, problems)
    _banner("generate", "Ask each model config for candidate solutions (visible test shown in the prompt).",
            problems=len(problems), model_configs=_describe_models(cfg),
            prompt_styles="not used in SMOKE_TEST" if cfg.smoke_test else ", ".join(cfg.resolved_prompt_styles()),
            samples_per_config=1 if cfg.smoke_test else cfg.samples_per_config,
            expected_candidates="" if cfg.smoke_test else
            f"{len(problems)} problems x {len(cfg.resolved_model_configs())} model configs x "
            f"{len(cfg.resolved_prompt_styles())} prompt styles x {cfg.samples_per_config} samples = "
            f"{len(problems) * len(cfg.resolved_model_configs()) * len(cfg.resolved_prompt_styles()) * cfg.samples_per_config}",
            mutant_augmentation=cfg.use_mutant_augmentation,
            cache=paths.generations if not cfg.smoke_test else "not used in SMOKE_TEST")
    report.ensure(paths, cfg, "generate")
    with _timed(paths, "generate"):
        candidates, _ = generation.generate_candidates(problems, cfg, paths.generations)
        candidates.to_csv(paths.candidates, index=False)
        log.info(f"{len(candidates)} candidates for {candidates.task_id.nunique()} problems -> {paths.candidates}")
        log.info("By provenance: " + log.counts_line(candidates["provenance"]))
        report.update(paths, "generation", report.generation_stats(
            candidates, candidates.attrs.get("generation_session"), cfg))
    return candidates


def stage_label(cfg, paths, problems=None, candidates=None, executor=None):
    """Execute (visible + probes for features, hidden tests for labels) and run static analysis."""
    problems = _load_problems(paths, problems)
    candidates = candidates if candidates is not None else features.read_frame(paths.candidates)
    _banner("execute", "Run every candidate in a sandboxed subprocess twice: once on the visible test and "
                       "probe inputs (features), once on the hidden tests (labels only).",
            candidates=len(candidates), workers=cfg.max_workers or os.cpu_count(),
            timeout_per_test=f"{cfg.item_timeout}s", memory_limit=f"{cfg.memory_limit_mb} MB")
    report.ensure(paths, cfg, "label")
    with _timed(paths, "execute"):
        executor = executor or Executor.from_config(cfg)
        by_id = data.problems_by_id(problems)
        labeled = features.add_execution_features(candidates, by_id, executor)
        log.info(f"Visible test passed by {labeled['visible_pass'].sum()}/{len(labeled)} candidates "
                 f"({labeled['visible_pass'].mean():.0%}).")
        labeled = features.add_hidden_test_labels(labeled, by_id, executor)
        log.info("Hidden-test outcomes: " + log.counts_line(labeled["exec_status"], EXEC_STATUS_ORDER))

    _banner("static", "Lint every candidate with ruff and scan it with bandit, then assign the label: "
                      "trustworthy = passes all hidden tests AND no medium/high bandit finding.",
            ruff_rules=RUFF_RULES)
    with _timed(paths, "static"):
        labeled = add_static_features(labeled)
        log.info(f"ruff: {int((labeled['ruff_issues'] > 0).sum())} candidates with lint issues; "
                 f"bandit: {int((labeled['bandit_med_high'] > 0).sum())} with medium/high findings.")
        labeled = features.add_label(labeled)
        labeled.to_csv(paths.labeled, index=False)
        rate = labeled[LABEL].mean()
        log.info(f"Labels: {int(labeled[LABEL].sum())}/{len(labeled)} trustworthy ({rate:.0%}) -> {paths.labeled}")
        if not 0.2 <= rate <= 0.8:
            log.warning("labels are very imbalanced; mix in weaker/stronger configs or temperatures.")
        stats = report.execution_stats(labeled, cfg, report.section(paths, "generation"))
        report.update(paths, "execution", stats)
        if len(stats["per_config"]) > 1:
            for label, block in stats["per_config"].items():
                log.info(f"  {label:<34} {block['rows']:>5} rows, trustworthy {block['trustworthy_rate']:.0%}, "
                         f"visible pass {block['visible_pass_rate']:.0%}, hidden pass {block['hidden_pass_rate']:.0%}")
        coverage = stats["failure_modes"]["all"]
        log.info("Failure-mode coverage: " + " | ".join(
            f"{m.replace('_', ' ')} {coverage[m]}" for m in report.FAILURE_MODES))
        rate_vh = stats["overall"]["visible_vs_hidden"]["hidden_fail_given_visible_pass"]
        if rate_vh is not None:
            log.info(f"Key number: {rate_vh:.0%} of candidates that pass the visible test fail a hidden test.")
    return labeled


def stage_features(cfg, paths, problems=None, labeled=None):
    problems = _load_problems(paths, problems)
    labeled = labeled if labeled is not None else features.read_frame(paths.labeled)
    _banner("features", "Decision-time features only: code structure, visible test, self-consistency "
                        "across sibling samples, static analysis, generation confidence. "
                        "Hidden-test results are never features.",
            rows=len(labeled))
    report.ensure(paths, cfg, "features")
    with _timed(paths, "features"):
        by_id = data.problems_by_id(problems)
        log.info("Computing self-consistency (agreement of outputs on probe inputs across siblings)...")
        if not cfg.smoke_test and labeled["provenance"].isin(TRAINING_ONLY_PROVENANCE).any():
            log.info("  Sibling pool = natural samples only; mutants / injected samples are scored "
                     "against it but never added to it.")
        dataset = features.add_consistency_features(labeled, all_in_pool=cfg.smoke_test)
        log.info("Computing code-structure features (AST, complexity, prompt words)...")
        dataset = features.add_code_features(dataset, by_id, cfg.model_names)
        feature_columns, dropped = features.select_feature_columns(dataset, cfg.model_names)
        dataset.to_csv(paths.dataset, index=False)
        _write_json({"feature_columns": feature_columns, "dropped_uninformative": dropped}, paths.feature_columns)
        log.info(f"{len(feature_columns)} features -> {paths.dataset}")
        if dropped:
            log.info(f"Not used (constant in this run, or comment features when injected samples are "
                     f"present): {', '.join(dropped)}")
        log.debug("Features: " + ", ".join(feature_columns))
    return dataset, feature_columns


def stage_human(cfg, paths, dataset=None):
    """Export (or simulate) the human review sample and report agreement with the automated label.

    In real mode an existing, partially filled review file is never overwritten.
    """
    dataset = dataset if dataset is not None else features.read_frame(paths.dataset)
    simulated = cfg.human_added_feedback
    _banner("human", "Stratified sample for human reviewers; agreement with the automated label is a "
                     "check on label quality, not a training signal.",
            mode="simulated reviewer (demo only)" if simulated else "real reviewers",
            sample_size=cfg.human_sample_size)
    report.ensure(paths, cfg, "human")
    with _timed(paths, "human"):
        keep_existing = False
        if not simulated and os.path.exists(paths.human_review):
            existing = pd.read_csv(paths.human_review)
            keep_existing = existing["human_trustworthy"].isin(["yes", "no"]).any()
        if keep_existing:
            log.info(f"Keeping existing reviews in {paths.human_review}")
        else:
            sample = human_review.build_review_sample(dataset, cfg.human_sample_size, simulated, cfg.seed)
            sample.to_csv(paths.human_review, index=False)
            mode = "simulated reviewer, demo only" if simulated else "blank: fill in 'human_trustworthy' with yes/no"
            log.info(f"{len(sample)} rows -> {paths.human_review} ({mode})")
        agreement = human_review.human_agreement(paths.human_review, simulated)
        if agreement:
            _write_json(agreement, paths.human_agreement)
            log.info(f"Agreement {agreement['raw_agreement']:.1%}, Cohen's kappa {agreement['cohen_kappa']:.3f}"
                     + ("  (SIMULATED labels -- not a finding)" if simulated else ""))
        elif not simulated:
            log.info("No reviews filled in yet; agreement will be computed once reviewers answer yes/no.")
        report.update(paths, "human_agreement", agreement)
    return agreement


def stage_train(cfg, paths, dataset=None, feature_columns=None, make_plots=True):
    """Grouped split, model selection, calibration, test metrics, thresholds, conformal, artifacts."""
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    dataset = dataset if dataset is not None else features.read_frame(paths.dataset)
    if feature_columns is None:
        with open(paths.feature_columns) as f:
            feature_columns = json.load(f)["feature_columns"]
    features.check_no_leakage(feature_columns)

    _banner("train", "Split by problem (no problem in two splits), compare models with grouped "
                     "cross-validation, calibrate the best one, evaluate on the held-out test split.",
            rows=len(dataset), features=len(feature_columns), split="train 60% / calibration 20% / test 20%")
    report.ensure(paths, cfg, "train")
    stage_start = time.time()
    splits = training.grouped_split(dataset, cfg.seed, cfg.smoke_test)
    train, calib, test = splits["train"], splits["calibration"], splits["test"]
    X_train, y_train, g_train = train[feature_columns], train[LABEL], train["task_id"]
    X_calib, y_calib = calib[feature_columns], calib[LABEL]
    X_test, y_test = test[feature_columns], test[LABEL]
    if y_train.nunique() < 2:
        raise RuntimeError("Training data has a single class; increase --num-problems or vary model configs.")
    split_table = training.split_summary(splits)
    log.block("Split:", split_table.to_string())
    if not cfg.smoke_test:
        held_out = dataset[dataset["task_id"].isin(set(calib["task_id"]) | set(test["task_id"]))]
        n_removed = int(held_out["provenance"].isin(TRAINING_ONLY_PROVENANCE).sum())
        if n_removed:
            log.info(f"Removed {n_removed} training-only rows (mutants / injected bugs) from calibration "
                     "and test; those splits contain natural samples only.")
    report.update(paths, "dataset", report.dataset_stats(dataset, splits, LABEL))

    models = training.build_models(cfg.seed)
    cv_splits = training.grouped_cv_splits(X_train, y_train, g_train)
    log.info(f"Grouped {len(cv_splits)}-fold cross-validation of {len(models)} models on the train split...")
    cv_table = training.cross_validate_models(models, X_train, y_train, cv_splits)
    for name, row in cv_table.iterrows():
        log.info(f"  CV {name:<22} " + "  ".join(
            f"{k.replace('cv_', '')} {row[k]:.3f}±{row[k + '_std']:.3f}"
            for k in cv_table.columns if not k.endswith("_std")))
    best_name = training.select_best(cv_table)
    log.info(f"Selected model: {best_name}")

    trust_model, calibration_method = training.fit_calibrated(models[best_name], X_train, y_train, cv_splits)
    log.info(f"Calibrated {best_name} with {calibration_method} calibration.")
    log.info("Refitting every model on the full train split for the test-set comparison...")
    fitted = {name: clone(m).fit(X_train, y_train) for name, m in models.items()}
    calibrated_name = f"{best_name} (calibrated)"
    test_proba = {name: m.predict_proba(X_test)[:, 1] for name, m in fitted.items()}
    test_proba[calibrated_name] = trust_model.predict_proba(X_test)[:, 1]
    comparison = pd.DataFrame({n: training.binary_metrics(y_test, p) for n, p in test_proba.items()}).T
    log.block("Test-set metrics (held-out problems):", comparison.round(3).to_string())
    log.info(f"Bootstrapping 95% confidence intervals ({cfg.bootstrap_samples} resamples of the "
             f"{test['task_id'].nunique()} test problems)...")
    ci = report.bootstrap_ci(y_test, test_proba, test["task_id"], cfg.bootstrap_samples, cfg.seed)
    by_style = report.metrics_by_prompt_style(test, test_proba[calibrated_name])
    if len(by_style) > 1:
        for style, row in by_style.items():
            log.info(f"  test by prompt style {style:<15} {row['rows']:>4} rows  "
                     + (f"ROC AUC {row['roc_auc']:.3f}  F1 {row['f1']:.3f}" if "roc_auc" in row else row["note"]))
    lo, hi = ci[calibrated_name]["roc_auc"]
    if lo is not None:
        log.info(f"{calibrated_name}: test ROC AUC {comparison.loc[calibrated_name, 'roc_auc']:.3f} "
                 f"(95% CI {lo:.3f}-{hi:.3f})")
    seconds = time.time() - stage_start
    log.info(f"Stage 'train' finished in {log.format_seconds(seconds)}")
    report.record_stage(paths, "train", seconds)

    _banner("decide", "Pick the APPROVE/REJECT thresholds on the calibration split, check them on the "
                      "test split, and save the model, predictions, metrics and figures.",
            target_approve_precision=cfg.target_approve_precision,
            target_reject_precision=cfg.target_reject_precision,
            min_decision_support=cfg.min_decision_support, conformal_alpha=cfg.conformal_alpha)
    stage_start = time.time()
    p_calib = trust_model.predict_proba(X_calib)[:, 1]
    p_test = test_proba[calibrated_name]
    t_low, t_high = choose_thresholds(y_calib, p_calib, cfg.target_approve_precision,
                                      cfg.target_reject_precision, cfg.min_decision_support)
    qhat = conformal_quantile(y_calib, p_calib, cfg.conformal_alpha)
    threshold_decisions = np.array([trust_decision(p, t_low, t_high) for p in p_test])
    conformal_decisions = np.array([conformal_decision(p, qhat) for p in p_test])
    decisions = pd.DataFrame({
        "threshold_rule": decision_report(y_test, threshold_decisions),
        "conformal": decision_report(y_test, conformal_decisions),
    })
    coverage = conformal_coverage(y_test, p_test, qhat)
    log.info("Thresholds: " + (f"REJECT if p <= {t_low:.3f}" if np.isfinite(t_low) else "REJECT disabled") + ", "
             + (f"APPROVE if p >= {t_high:.3f}" if np.isfinite(t_high) else "APPROVE disabled")
             + ", otherwise NEEDS HUMAN REVIEW.")
    if not np.isfinite(t_high):
        log.info("  APPROVE is disabled: no threshold reached the target precision with enough "
                 "calibration support (more problems usually fixes this).")
    if not np.isfinite(t_low):
        log.info("  REJECT is disabled: no threshold reached the target precision with enough "
                 "calibration support.")
    log.info(f"Conformal: qhat {qhat:.3f}, test coverage {coverage:.3f} (target {1 - cfg.conformal_alpha:.2f})")
    log.block("Three-way decisions on the test split:", decisions.round(3).to_string())

    importances = pd.Series(dtype=float)
    if y_test.nunique() > 1:
        log.info("Computing permutation importance on the test split (can take a minute)...")
        perm = permutation_importance(trust_model, X_test, y_test, scoring="roc_auc",
                                      n_repeats=5, random_state=cfg.seed)
        importances = pd.Series(perm.importances_mean, index=feature_columns).sort_values(ascending=False)
        importances.rename("permutation_importance").to_csv(paths.importance)
        log.info("Top features: " + ", ".join(f"{k} {v:.3f}" for k, v in importances.head(5).items()))

    recommender = TrustRecommender(
        model=trust_model, feature_columns=feature_columns, model_names=cfg.model_names,
        t_low=t_low, t_high=t_high, qhat=qhat, model_label=calibrated_name, config=cfg.to_dict(),
    )
    recommender.save(paths.model)

    predictions = test[["candidate_id", "task_id", "model_name"]
                       + [c for c in ("temperature", "prompt_style") if c in test]
                       + ["provenance", "exec_status",
                        "visible_pass", "behavior_agreement", LABEL, "generated_code"]].copy()
    predictions["p_trustworthy"] = p_test
    predictions["decision"] = threshold_decisions
    predictions["conformal_decision"] = conformal_decisions
    predictions.to_csv(paths.predictions, index=False)
    cv_table.to_csv(paths.cv_results)
    comparison.to_csv(paths.model_comparison)

    metrics = {
        "smoke_test": cfg.smoke_test,
        "human_feedback_simulated": cfg.human_added_feedback,
        "split": split_table.to_dict(orient="index"),
        "features": feature_columns,
        "cv": cv_table.to_dict(orient="index"),
        "selected_model": best_name,
        "calibration_method": calibration_method,
        "test_metrics": comparison.to_dict(orient="index"),
        "thresholds": {"t_low": t_low, "t_high": t_high,
                       "target_approve_precision": cfg.target_approve_precision,
                       "target_reject_precision": cfg.target_reject_precision},
        "conformal": {"alpha": cfg.conformal_alpha, "qhat": qhat, "test_coverage": coverage},
        "decisions_test": decisions.to_dict(),
    }
    _write_json(metrics, paths.metrics)

    if make_plots:
        import matplotlib.pyplot as plt
        log.info("Drawing figures...")
        for fig in [
            plots.plot_eda(dataset, paths.figure("eda.png")),
            plots.plot_evaluation(y_test, test_proba, calibrated_name, paths.figure("evaluation.png")),
            plots.plot_risk_coverage(y_test, p_test, {"threshold rule": threshold_decisions,
                                                      "conformal": conformal_decisions},
                                     paths.figure("risk_coverage.png")),
        ] + ([plots.plot_feature_importance(importances, path=paths.figure("feature_importance.png"))]
             if len(importances) else []):
            plt.close(fig)
    log.info(f"Model -> {paths.model}; metrics -> {paths.metrics}; figures -> {paths.figures}")
    report.update(paths, "model_results", report.model_results(
        cv_table, best_name, calibrated_name, calibration_method, comparison, ci, cfg.bootstrap_samples,
        thresholds={**metrics["thresholds"], "approve_enabled": bool(np.isfinite(t_high)),
                    "reject_enabled": bool(np.isfinite(t_low)), "min_decision_support": cfg.min_decision_support},
        decisions=decisions, conformal=metrics["conformal"], baselines=training.BASELINES,
        by_prompt_style=by_style))
    seconds = time.time() - stage_start
    log.info(f"Stage 'decide' finished in {log.format_seconds(seconds)}")
    report.record_stage(paths, "decide", seconds)
    written = report.finalize(paths, cfg)
    if written:
        log.info(f"Run report -> {written[0]} (all numbers: {paths.run_report}); history row -> {written[1]}")

    return {
        "splits": splits, "split_table": split_table, "cv_table": cv_table, "best_name": best_name,
        "calibrated_name": calibrated_name, "comparison": comparison, "test_proba": test_proba,
        "y_test": y_test, "p_test": p_test, "threshold_decisions": threshold_decisions,
        "conformal_decisions": conformal_decisions, "decisions": decisions,
        "conformal_coverage": coverage, "importances": importances, "recommender": recommender,
        "predictions": predictions, "metrics": metrics,
    }


def log_summary(cfg, paths, results, elapsed, title="Pipeline finished"):
    calibrated = results["calibrated_name"]
    test = results["comparison"].loc[calibrated]
    t_low, t_high = results["metrics"]["thresholds"]["t_low"], results["metrics"]["thresholds"]["t_high"]
    decisions = results["decisions"]["threshold_rule"]
    vh = ((report.section(paths, "execution") or {}).get("overall", {})
          .get("visible_vs_hidden", {}).get("hidden_fail_given_visible_pass"))
    lines = [
        f"RUN REPORT (tables for your write-up): {paths.run_report_md}",
        f"all numbers (JSON):   {paths.run_report}",
        f"trained model:        {paths.model}",
        f"test predictions:     {paths.predictions}",
        f"labelled dataset:     {paths.dataset}",
        f"human review sample:  {paths.human_review}",
        f"figures:              {paths.figures}/",
        "",
        f"model: {calibrated}; test ROC AUC {test.get('roc_auc', float('nan')):.3f}, "
        f"F1 {test.get('f1', float('nan')):.3f}",
        "decisions: " + (f"REJECT if p <= {t_low:.3f}" if np.isfinite(t_low) else "REJECT disabled") + ", "
        + (f"APPROVE if p >= {t_high:.3f}" if np.isfinite(t_high) else "APPROVE disabled")
        + f"; on test: {decisions['approved_share']:.0%} approved, {decisions['rejected_share']:.0%} rejected, "
          f"{decisions['review_share']:.0%} to human review",
    ]
    if vh is not None:
        lines.append(f"visible test passed but hidden tests failed: {vh:.0%} of visible-passing candidates")
    lines += ["", f"Try it:  trust-recommend --output-dir {paths.root} --task-id <MBPP id> --code-file <file.py>"]
    log.block(f"{title} in {log.format_seconds(elapsed)}. Artifacts in {paths.root}/:", "\n".join(lines))
    if cfg.smoke_test:
        log.info("SMOKE_TEST run: this checks the pipeline end to end; the numbers are not a research result.")


def run_all(cfg, make_plots=True):
    start = time.time()
    paths = ArtifactPaths(cfg.output_dir)
    log.info(f"Running the full pipeline ({len(STAGES)} stages); outputs go to {os.path.abspath(cfg.output_dir)}")
    if os.path.abspath(paths.root) not in report._active:
        report.begin(paths, cfg, "python: run_all", fresh=True)
    executor = Executor.from_config(cfg)
    problems = stage_prepare(cfg, paths, executor)
    candidates = stage_generate(cfg, paths, problems)
    labeled = stage_label(cfg, paths, problems, candidates, executor)
    dataset, feature_columns = stage_features(cfg, paths, problems, labeled)
    stage_human(cfg, paths, dataset)
    results = stage_train(cfg, paths, dataset, feature_columns, make_plots=make_plots)
    log_summary(cfg, paths, results, time.time() - start)
    return results


# ------------------------------------------------------------------------------------- CLI -------

def add_verbosity_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quiet", action="store_true", help="only warnings and errors, no progress bars")
    group.add_argument("--verbose", action="store_true",
                       help="extra detail, including third-party library messages")
    return parser


def verbosity_from_args(args, default="info"):
    if getattr(args, "quiet", False):
        return "quiet"
    if getattr(args, "verbose", False):
        return "verbose"
    return default


def add_common_args(parser):
    parser.add_argument("--output-dir", default="artifacts", help="artifacts directory (default: artifacts)")
    parser.add_argument("--num-problems", type=int, help="number of MBPP problems (default 100)")
    parser.add_argument("--all-problems", action="store_true", help="use all 974 MBPP problems")
    parser.add_argument("--samples-per-config", type=int, help="samples per (problem, model config)")
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None,
                        help="reference solutions + AST mutants instead of an LLM (pipeline check only)")
    parser.add_argument("--human-feedback", choices=["simulated", "real"],
                        help="simulated reviewer (demo) or a blank CSV for real reviewers")
    parser.add_argument("--mutant-augmentation", action=argparse.BooleanOptionalAction, default=None,
                        help="add mutated reference solutions as training-only rows")
    parser.add_argument("--local-hf", action=argparse.BooleanOptionalAction, default=None,
                        help="use the local Hugging Face models (default on)")
    parser.add_argument("--openai", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--anthropic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-precision", type=float,
                        help="target precision for both APPROVED and REJECTED (default 0.95)")
    parser.add_argument("--max-workers", type=int, help="parallel execution workers")
    parser.add_argument("--prompt-styles",
                        help="optional extra: comma list of prompt styles crossed with every model config "
                             f"({', '.join(generation.PROMPT_STYLES)}), or a preset: 'natural' (all but "
                             "inject_bug) or 'all'. Default: standard")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int,
                        help="bootstrap resamples (by problem) for test-metric 95%% CIs (default 1000)")
    parser.add_argument("--history-file",
                        help="cross-run summary CSV (default: runs_history.csv in the parent of --output-dir)")
    add_verbosity_args(parser)
    return parser


def config_from_args(args):
    """Start from <output-dir>/config.json if present (so stages agree), then apply CLI overrides."""
    log.setup_logging(verbosity_from_args(args))
    path = os.path.join(args.output_dir, "config.json")
    existing = os.path.exists(path)
    cfg = Config.load(path) if existing else Config()
    cfg.output_dir = args.output_dir
    overrides = {
        "num_problems": args.num_problems,
        "samples_per_config": args.samples_per_config,
        "smoke_test": args.smoke_test,
        "use_mutant_augmentation": args.mutant_augmentation,
        "use_local_hf": args.local_hf,
        "use_openai": args.openai,
        "use_anthropic": args.anthropic,
        "max_workers": args.max_workers,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "history_file": args.history_file,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.all_problems:
        cfg.num_problems = None
    if args.prompt_styles:
        try:
            cfg.prompt_styles = generation.resolve_prompt_styles(args.prompt_styles)
        except ValueError as e:
            raise SystemExit(f"error: --prompt-styles: {e}") from None
    if args.human_feedback:
        cfg.human_added_feedback = args.human_feedback == "simulated"
    if args.target_precision is not None:
        cfg.target_approve_precision = cfg.target_reject_precision = args.target_precision
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(path)
    if existing:
        log.info(f"Using settings from {path} (plus any command-line overrides).")
    return cfg
