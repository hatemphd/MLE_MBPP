"""Report-ready run metrics: `run_report.json` (everything), `run_report.md` (tables to paste into a
report) and one row per run in a cross-run `runs_history.csv`.

Every stage merges its own section into `<output-dir>/run_report.json`, so a full `trust-pipeline`
run and stage-by-stage runs (`trust-generate`, `trust-label`, `trust-train`, ...) produce the same
report. A new run (new run_id, fresh report) starts with `trust-pipeline`, `trust-prepare`, or the
first stage called in a new Python process when no report exists yet.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import platform
import sys
import uuid
from importlib import metadata

import numpy as np
import pandas as pd

from . import log

EXEC_STATUS_ORDER = ["pass", "wrong_output", "runtime_error", "timeout", "syntax_error"]
METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
COUNT_KEYS = {"untrustworthy_code_approved"}
VERSION_PACKAGES = ["mle-trust-pipeline", "scikit-learn", "numpy", "pandas", "torch", "transformers",
                    "datasets", "openai", "anthropic"]
_active = {}   # output_dir (abs path) -> run_id started in this process


def report_paths(root):
    return os.path.join(root, "run_report.json"), os.path.join(root, "run_report.md")


def default_history_file(output_dir):
    """runs_history.csv next to the output directory, so sibling runs share one history."""
    return os.path.join(os.path.dirname(os.path.abspath(output_dir)), "runs_history.csv")


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _clean(o):
    """JSON-safe: numpy scalars to Python, NaN/inf to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _load(root):
    path = report_paths(root)[0]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save(root, report):
    json_path, md_path = report_paths(root)
    with open(json_path, "w") as f:
        json.dump(_clean(report), f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_markdown(report))


def _safe(fn):
    """Reporting must never kill a long run: log a warning instead of raising."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning(f"could not update the run report ({fn.__name__}: {type(e).__name__}: {e})")
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ------------------------------------------------------------------------------- manifest -------

def _versions():
    out = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _gpu_info():
    if "torch" not in sys.modules:
        return {"checked": False, "note": "torch not loaded in this run (smoke test or API-only)"}
    import torch
    if not torch.cuda.is_available():
        return {"checked": True, "count": 0, "names": []}
    return {"checked": True, "count": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def _manifest(cfg, command, args):
    return {
        "run_id": dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "started_at": _now(),
        "finished_at": None,
        "duration_seconds": None,
        "mode": "smoke_test" if cfg.smoke_test else "real",
        "human_feedback": "simulated" if cfg.human_added_feedback else "real",
        "output_dir": os.path.abspath(cfg.output_dir),
        "commands": [],
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "model_configs": cfg.resolved_model_configs(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": _versions(),
        "gpu": _gpu_info(),
    }


@_safe
def begin(paths, cfg, command, args=None, fresh=False):
    """Start (fresh=True or no report on disk) or continue the run report for this output dir."""
    root = paths.root
    report = None if fresh else _load(root)
    if report is None:
        report = {"manifest": _manifest(cfg, command, args)}
    m = report["manifest"]
    m["config"], m["model_configs"] = cfg.to_dict(), cfg.resolved_model_configs()
    m["mode"] = "smoke_test" if cfg.smoke_test else "real"
    m["human_feedback"] = "simulated" if cfg.human_added_feedback else "real"
    m["commands"].append({"command": command, "argv": sys.argv[1:] if args is not None else None,
                          "args": args, "at": _now()})
    _active[os.path.abspath(root)] = m["run_id"]
    _save(root, report)
    return m["run_id"]


def ensure(paths, cfg, stage, fresh_if_new=False):
    """Called by each stage: no-op if this process already began a run for this output dir."""
    if os.path.abspath(paths.root) not in _active:
        begin(paths, cfg, f"python: stage_{stage}", fresh=fresh_if_new)


def section(paths, name):
    return (_load(paths.root) or {}).get(name)


@_safe
def update(paths, section, data):
    report = _load(paths.root) or {}
    report[section] = data
    _save(paths.root, report)


@_safe
def record_stage(paths, stage, seconds):
    report = _load(paths.root) or {}
    report.setdefault("stage_seconds", {})[stage] = round(float(seconds), 2)
    _save(paths.root, report)


# ------------------------------------------------------------------------------ generation ------

def generation_stats(candidates, session, cfg):
    """Per model config. Token/latency totals cover every sample in the dataset (cached ones too, if
    they were generated by a version that recorded them); new/reused/failed refer to this session."""
    out = {}
    for model_cfg in cfg.resolved_model_configs():
        name = model_cfg["name"]
        rows = candidates[candidates["model_name"] == name]
        s = (session or {}).get(name, {})
        tokens = pd.to_numeric(rows.get("gen_num_tokens"), errors="coerce") if len(rows) else pd.Series(dtype=float)
        inputs = pd.to_numeric(rows.get("gen_input_tokens"), errors="coerce") if "gen_input_tokens" in rows else pd.Series(dtype=float)
        secs = pd.to_numeric(rows.get("gen_seconds"), errors="coerce") if "gen_seconds" in rows else pd.Series(dtype=float)
        timed = secs.notna() & tokens.notna() if len(secs) else pd.Series(dtype=bool)
        llm = model_cfg["provider"] not in ("smoke",)
        out[name] = {
            "provider": model_cfg["provider"],
            "model": model_cfg.get("model"),
            "temperature": model_cfg.get("temperature"),
            "problems": int(rows["task_id"].nunique()) if len(rows) else 0,
            "samples_in_dataset": int(len(rows)),
            "attempts_requested": s.get("attempts_requested"),
            "new_samples": s.get("new_samples"),
            "reused_from_cache": s.get("reused_samples"),
            "failed_attempts": s.get("failed_attempts"),
            "session_seconds": s.get("session_seconds"),
            "output_tokens_total": float(tokens.sum()) if llm and tokens.notna().any() else None,
            "output_tokens_mean": float(tokens.mean()) if llm and tokens.notna().any() else None,
            "input_tokens_total": float(inputs.sum()) if llm and inputs.notna().any() else None,
            "mean_seconds_per_sample": float(secs.mean()) if secs.notna().any() else None,
            "output_tokens_per_second": (float(tokens[timed].sum() / secs[timed].sum())
                                         if llm and timed.any() and secs[timed].sum() > 0 else None),
            "timed_samples": int(timed.sum()) if len(secs) else 0,
        }
    return out


# ------------------------------------------------------------------------------- execution ------

def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _pass_at_k_table(frame, ks):
    per_problem = frame.groupby("task_id").agg(n=("hidden_pass", "size"), c=("hidden_pass", "sum"))
    out = {}
    for k in ks:
        eligible = per_problem[per_problem["n"] >= k]
        out[f"pass@{k}"] = (float(np.mean([pass_at_k(n, c, k) for n, c in zip(eligible["n"], eligible["c"])]))
                            if len(eligible) else None)
    return out


def _execution_block(frame, gen_errors=0):
    counts = frame["exec_status"].value_counts()
    statuses = EXEC_STATUS_ORDER + [s for s in counts.index if s not in EXEC_STATUS_ORDER]
    vis, hid = frame["visible_pass"].astype(int), frame["hidden_pass"].astype(int)
    n_vis = int(vis.sum())
    return {
        "rows": int(len(frame)),
        "problems": int(frame["task_id"].nunique()),
        "outcomes": {**{s: int(counts.get(s, 0)) for s in statuses}, "generation_error": int(gen_errors or 0)},
        "trustworthy_rate": float(frame["label_trustworthy"].mean()),
        "visible_pass_rate": float(vis.mean()),
        "hidden_pass_rate": float(hid.mean()),
        "visible_vs_hidden": {
            "visible_pass_hidden_pass": int(((vis == 1) & (hid == 1)).sum()),
            "visible_pass_hidden_fail": int(((vis == 1) & (hid == 0)).sum()),
            "visible_fail_hidden_pass": int(((vis == 0) & (hid == 1)).sum()),
            "visible_fail_hidden_fail": int(((vis == 0) & (hid == 0)).sum()),
            "hidden_fail_given_visible_pass": float(((vis == 1) & (hid == 0)).sum() / n_vis) if n_vis else None,
        },
        "static": {
            "share_with_ruff_issues": float((frame["ruff_issues"] > 0).mean()),
            "share_with_undefined_names": float((frame["ruff_undefined_names"] > 0).mean()),
            "share_with_bandit_med_high": float((frame["bandit_med_high"] > 0).mean()),
            "mean_ruff_issues": float(frame["ruff_issues"].mean()),
        },
    }


def execution_stats(labeled, cfg, generation=None):
    frame = labeled.copy()
    frame["hidden_pass"] = (frame["exec_status"] == "pass").astype(int)
    gen_errors = {k: (v or {}).get("failed_attempts") or 0 for k, v in (generation or {}).items()}
    n = 1 if cfg.smoke_test else cfg.samples_per_config
    ks = sorted({k for k in (1, 3, 5, 10) if k <= max(n, 1)} | {max(n, 1)})
    per_config = {}
    for name, group in frame.groupby("model_name", sort=False):
        block = _execution_block(group, gen_errors.get(name, 0))
        block["pass_at_k"] = _pass_at_k_table(group, ks)
        per_config[name] = block
    overall = _execution_block(frame, sum(gen_errors.values()))
    overall["pass_at_k_note"] = ("pass@k uses hidden tests and the unbiased estimator over each problem's "
                                 "samples for one model config; in SMOKE_TEST the 'samples' are the "
                                 "reference plus its mutants.")
    return {"overall": overall, "per_config": per_config}


# ---------------------------------------------------------------------------------- dataset -----

def dataset_stats(dataset, splits, label="label_trustworthy"):
    def block(df):
        return {"rows": int(len(df)), "problems": int(df["task_id"].nunique()),
                "trustworthy": int(df[label].sum()), "trustworthy_rate": float(df[label].mean()) if len(df) else None,
                "provenance": {k: int(v) for k, v in df["provenance"].value_counts().items()}}
    return {"total": block(dataset), "splits": {name: block(df) for name, df in splits.items()}}


# ---------------------------------------------------------------------------------- metrics -----

def _fast_metrics(y, p):
    """Same definitions as training.binary_metrics, vectorised for bootstrapping."""
    pred = p >= 0.5
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"accuracy": float(np.mean(pred == (y == 1))), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "brier": float(np.mean((p - y) ** 2)), "roc_auc": np.nan, "pr_auc": np.nan}
    n_pos = int(y.sum())
    if 0 < n_pos < len(y):
        from scipy.stats import rankdata
        ranks = rankdata(p)
        out["roc_auc"] = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))
        order = np.argsort(-p, kind="mergesort")
        ps, ys = p[order], y[order]
        last = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
        tps = np.cumsum(ys)[last]
        precision = tps / (last + 1)
        recall = tps / n_pos
        out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return out


def bootstrap_ci(y_true, proba_by_model, groups, n_boot=1000, seed=42, level=0.95):
    """Percentile CIs resampling whole problems (task_id) with replacement."""
    y = np.asarray(y_true).astype(int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([index[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
             for _ in range(n_boot)]
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for name, proba in proba_by_model.items():
        p = np.asarray(proba, dtype=float)
        samples = pd.DataFrame([_fast_metrics(y[idx], p[idx]) for idx in draws])
        out[name] = {m: [float(np.nanpercentile(samples[m], lo)), float(np.nanpercentile(samples[m], hi))]
                     if samples[m].notna().any() else [None, None] for m in METRIC_ORDER}
    return out


def model_results(cv_table, best_name, calibrated_name, calibration_method, comparison, ci, n_boot,
                  thresholds, decisions, conformal, baselines):
    test = {}
    for name, row in comparison.iterrows():
        test[name] = {m: {"value": row[m], "ci95": ci.get(name, {}).get(m)} for m in METRIC_ORDER if m in row}
    return {
        "cv": cv_table.to_dict(orient="index"),
        "selected_model": best_name,
        "calibrated_model": calibrated_name,
        "calibration_method": calibration_method,
        "baselines": list(baselines),
        "test": test,
        "bootstrap": {"resamples": n_boot, "unit": "problem (task_id)", "level": 0.95},
        "thresholds": thresholds,
        "decisions": decisions.to_dict(),
        "conformal": conformal,
    }


# ---------------------------------------------------------------------------------- finalize ----

HISTORY_COLUMNS = [
    "run_id", "started_at", "finished_at", "duration_min", "mode", "human_feedback", "output_dir",
    "problems", "model_configs", "samples_per_config", "seed", "rows", "trustworthy_rate",
    "hidden_fail_given_visible_pass", "selected_model", "test_roc_auc", "test_roc_auc_ci_low",
    "test_roc_auc_ci_high", "test_f1", "test_precision", "test_recall", "test_brier",
    "visible_only_roc_auc", "approve_threshold", "reject_threshold", "approved_share",
    "approved_precision", "rejected_share", "rejected_precision", "review_share",
    "automated_error_rate", "human_kappa",
]


def _history_row(report):
    m = report["manifest"]
    ex = report.get("execution", {}).get("overall", {})
    ds = report.get("dataset", {}).get("total", {})
    mr = report.get("model_results", {})
    cal = mr.get("test", {}).get(mr.get("calibrated_model"), {})
    vis = mr.get("test", {}).get("visible_test_only", {})
    dec = mr.get("decisions", {}).get("threshold_rule", {})
    th = mr.get("thresholds", {})
    hum = report.get("human_agreement") or {}

    def val(d, k):
        return (d.get(k) or {}).get("value")
    ci = (cal.get("roc_auc") or {}).get("ci95") or [None, None]
    return {
        "run_id": m["run_id"], "started_at": m["started_at"], "finished_at": m.get("finished_at"),
        "duration_min": round(m["duration_seconds"] / 60, 1) if m.get("duration_seconds") else None,
        "mode": m["mode"], "human_feedback": m["human_feedback"], "output_dir": m["output_dir"],
        "problems": ds.get("problems"), "model_configs": ";".join(c["name"] for c in m["model_configs"]),
        "samples_per_config": 1 if m["mode"] == "smoke_test" else m["config"].get("samples_per_config"),
        "seed": m["seed"],
        "rows": ds.get("rows"), "trustworthy_rate": ds.get("trustworthy_rate"),
        "hidden_fail_given_visible_pass": ex.get("visible_vs_hidden", {}).get("hidden_fail_given_visible_pass"),
        "selected_model": mr.get("selected_model"),
        "test_roc_auc": val(cal, "roc_auc"), "test_roc_auc_ci_low": ci[0], "test_roc_auc_ci_high": ci[1],
        "test_f1": val(cal, "f1"), "test_precision": val(cal, "precision"), "test_recall": val(cal, "recall"),
        "test_brier": val(cal, "brier"), "visible_only_roc_auc": val(vis, "roc_auc"),
        "approve_threshold": th.get("t_high"), "reject_threshold": th.get("t_low"),
        "approved_share": dec.get("approved_share"), "approved_precision": dec.get("approved_precision"),
        "rejected_share": dec.get("rejected_share"), "rejected_precision": dec.get("rejected_precision"),
        "review_share": dec.get("review_share"), "automated_error_rate": dec.get("automated_error_rate"),
        "human_kappa": hum.get("cohen_kappa"),
    }


def _upsert_history(path, row):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_id") != row["run_id"]]
    def cell(v):
        if v is None:
            return ""
        return round(v, 4) if isinstance(v, float) else v
    rows.append({k: cell(row.get(k)) for k in HISTORY_COLUMNS})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@_safe
def finalize(paths, cfg, stamp_end=True):
    """Stamp the end time, refresh environment info, write the markdown and the history row.
    `stamp_end=False` (e.g. after reviewers fill in labels later) only refreshes the outputs."""
    report = _load(paths.root)
    if report is None:
        return None
    m = report["manifest"]
    if stamp_end or not m.get("finished_at"):
        m["finished_at"] = _now()
        m["duration_seconds"] = (dt.datetime.fromisoformat(m["finished_at"])
                                 - dt.datetime.fromisoformat(m["started_at"])).total_seconds()
        m["versions"], m["gpu"] = _versions(), _gpu_info()
    m["stage_seconds_total"] = round(sum(report.get("stage_seconds", {}).values()), 2)
    history = cfg.history_file or default_history_file(cfg.output_dir)
    m["history_file"] = os.path.abspath(history)
    _save(paths.root, report)
    _upsert_history(history, _history_row(_clean(report)))
    return report_paths(paths.root)[1], history


# ---------------------------------------------------------------------------------- markdown ----

def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _seconds(v):
    return "–" if v is None else log.format_seconds(v)


def render_markdown(report):
    report = _clean(report)
    m = report.get("manifest", {})
    out = [f"# Run report `{m.get('run_id', '?')}`", ""]
    if m.get("mode") == "smoke_test":
        out += ["> **SMOKE TEST.** Candidates are reference solutions and AST mutants, not LLM output. "
                "These numbers check that the pipeline works; they are not research results.", ""]
    if m.get("human_feedback") == "simulated":
        out += ["> **SIMULATED HUMAN FEEDBACK.** Human-agreement numbers come from a simulated reviewer "
                "and are not a finding.", ""]

    cfg = m.get("config", {})
    gpu = m.get("gpu", {})
    gpu_text = (", ".join(gpu.get("names", [])) or "none") if gpu.get("checked") else "not checked"
    versions = ", ".join(f"{k} {v}" for k, v in (m.get("versions") or {}).items() if v)
    out += ["## Run", "", _table(["Item", "Value"], [
        ["Run id", m.get("run_id")], ["Mode", m.get("mode")], ["Human feedback", m.get("human_feedback")],
        ["Started", m.get("started_at")], ["Finished", m.get("finished_at") or "running / incomplete"],
        ["Wall-clock duration", _seconds(m.get("duration_seconds"))],
        ["Problems requested", "all" if cfg.get("num_problems") is None else cfg.get("num_problems")],
        ["Model configs", ", ".join(c["name"] for c in m.get("model_configs", []))],
        ["Samples per config", 1 if cfg.get("smoke_test") else cfg.get("samples_per_config")],
        ["Seed", m.get("seed")], ["Output dir", f"`{m.get('output_dir')}`"],
        ["Commands", "; ".join(" ".join([c["command"]] + (c.get("argv") or [])) for c in m.get("commands", []))],
        ["Python / platform", f"{m.get('python')} / {m.get('platform')}"],
        ["GPU", gpu_text], ["Package versions", versions],
    ]), ""]

    stages = report.get("stage_seconds", {})
    if stages:
        out += ["## Stage timings", "", _table(["Stage", "Time", "Seconds"],
                [[k, _seconds(v), _fmt(v, 1)] for k, v in stages.items()]
                + [["**total**", _seconds(sum(stages.values())), _fmt(sum(stages.values()), 1)]]), ""]

    gen = report.get("generation")
    if gen:
        out += ["## Generation (per model config)", "",
                "New / reused / failed refer to the latest generate session; tokens and latency cover "
                "every sample in the dataset that recorded them. Local models generate the samples for a "
                "problem in one batch, so seconds per sample is the batch time divided by the batch size.", "",
                _table(["Config", "Model", "Temp", "Problems", "Samples", "New", "Reused", "Failed",
                        "Out tokens (total)", "Out tokens (mean)", "In tokens (total)", "s / sample", "tokens / s"],
                       [[k, v.get("model") or v.get("provider"), _fmt(v.get("temperature"), 1), _fmt(v.get("problems")),
                         _fmt(v.get("samples_in_dataset")), _fmt(v.get("new_samples")), _fmt(v.get("reused_from_cache")),
                         _fmt(v.get("failed_attempts")), _fmt(v.get("output_tokens_total"), 0),
                         _fmt(v.get("output_tokens_mean"), 1), _fmt(v.get("input_tokens_total"), 0),
                         _fmt(v.get("mean_seconds_per_sample"), 2), _fmt(v.get("output_tokens_per_second"), 1)]
                        for k, v in gen.items()]), ""]

    ex = report.get("execution")
    if ex:
        blocks = {**ex["per_config"], "**all**": ex["overall"]}
        statuses = list(ex["overall"]["outcomes"].keys())
        out += ["## Execution outcomes and labels", "",
                "Outcome on the hidden tests. Trustworthy = passes every hidden test and has no "
                "medium/high-severity bandit finding.", "",
                _table(["Config", "Rows"] + statuses + ["Trustworthy"],
                       [[k, _fmt(v["rows"])] + [_fmt(v["outcomes"].get(s, 0)) for s in statuses]
                        + [_pct(v["trustworthy_rate"])] for k, v in blocks.items()]), ""]
        ks = sorted({k for v in ex["per_config"].values() for k in v.get("pass_at_k", {})},
                    key=lambda s: int(s.split("@")[1]))
        if ks:
            out += ["### pass@k on hidden tests (unbiased estimator)", "",
                    _table(["Config"] + ks, [[k] + [_pct(v["pass_at_k"].get(c)) for c in ks]
                                             for k, v in ex["per_config"].items()]), ""]
        vh = ex["overall"]["visible_vs_hidden"]
        out += ["### Visible test vs hidden tests", "",
                f"**{_pct(vh['hidden_fail_given_visible_pass'])} of candidates that pass the visible test "
                "still fail at least one hidden test.**", "",
                _table(["", "Hidden pass", "Hidden fail"], [
                    ["Visible pass", _fmt(vh["visible_pass_hidden_pass"]), _fmt(vh["visible_pass_hidden_fail"])],
                    ["Visible fail", _fmt(vh["visible_fail_hidden_pass"]), _fmt(vh["visible_fail_hidden_fail"])]]), "",
                _table(["Config", "Visible pass", "Hidden pass", "Hidden fail given visible pass"],
                       [[k, _pct(v["visible_pass_rate"]), _pct(v["hidden_pass_rate"]),
                         _pct(v["visible_vs_hidden"]["hidden_fail_given_visible_pass"])] for k, v in blocks.items()]), "",
                "### Static analysis", "",
                _table(["Config", "With ruff issues", "With undefined names", "With bandit medium/high", "Mean ruff issues"],
                       [[k, _pct(v["static"]["share_with_ruff_issues"]), _pct(v["static"]["share_with_undefined_names"]),
                         _pct(v["static"]["share_with_bandit_med_high"]), _fmt(v["static"]["mean_ruff_issues"], 2)]
                        for k, v in blocks.items()]), ""]

    ds = report.get("dataset")
    if ds:
        provs = sorted({p for b in [ds["total"], *ds["splits"].values()] for p in b["provenance"]})
        out += ["## Dataset and splits", "", "Splits are by problem: no problem appears in two splits.", "",
                _table(["Split", "Rows", "Problems", "Trustworthy", "Rate"] + provs,
                       [[k, _fmt(b["rows"]), _fmt(b["problems"]), _fmt(b["trustworthy"]), _pct(b["trustworthy_rate"])]
                        + [_fmt(b["provenance"].get(p, 0)) for p in provs]
                        for k, b in {**ds["splits"], "**total**": ds["total"]}.items()]), ""]

    mr = report.get("model_results")
    if mr:
        cv_metrics = [c for c in next(iter(mr["cv"].values())) if not c.endswith("_std")]
        out += ["## Model selection (grouped cross-validation on train, mean ± std)", "",
                _table(["Model"] + [c.replace("cv_", "") for c in cv_metrics],
                       [[("**" + k + "**") if k == mr["selected_model"] else k]
                        + [f"{_fmt(row.get(c))} ± {_fmt(row.get(c + '_std'))}" for c in cv_metrics]
                        for k, row in mr["cv"].items()]), "",
                f"Selected: **{mr['selected_model']}**, calibrated with {mr['calibration_method']} calibration. "
                f"Baselines: {', '.join(mr['baselines'])}.", ""]
        b = mr["bootstrap"]
        out += [f"## Test-set metrics (held-out problems, 95% CI from {b['resamples']} bootstrap resamples by problem)", "",
                "Threshold 0.5 for accuracy / precision / recall / F1; lower Brier is better.", "",
                _table(["Model"] + METRIC_ORDER,
                       [[("**" + k + "**") if k == mr["calibrated_model"] else k]
                        + [(f"{_fmt(v[m_]['value'])} [{_fmt(v[m_]['ci95'][0])}, {_fmt(v[m_]['ci95'][1])}]"
                            if v.get(m_) and v[m_].get("ci95") else _fmt((v.get(m_) or {}).get("value")))
                           for m_ in METRIC_ORDER]
                        for k, v in mr["test"].items()]), ""]
        th, conf = mr["thresholds"], mr["conformal"]
        approve = (f"APPROVE if p ≥ {_fmt(th.get('t_high'))}" if th.get("t_high") is not None
                   else "APPROVE disabled (target precision not reachable with enough calibration support)")
        reject = (f"REJECT if p ≤ {_fmt(th.get('t_low'))}" if th.get("t_low") is not None
                  else "REJECT disabled (target precision not reachable with enough calibration support)")
        dec = mr["decisions"]
        names = list(dec.keys())
        out += ["## Three-way decisions on the test split", "",
                f"Threshold rule (chosen on the calibration split, target precision "
                f"{_fmt(th.get('target_approve_precision'), 2)} / {_fmt(th.get('target_reject_precision'), 2)}): "
                f"{approve}; {reject}; otherwise NEEDS HUMAN REVIEW. "
                f"Conformal (alpha {_fmt(conf.get('alpha'), 2)}): qhat {_fmt(conf.get('qhat'))}, test coverage "
                f"{_fmt(conf.get('test_coverage'))}.", "",
                _table(["Metric"] + names,
                       [[k.replace("_", " ")] + [(_fmt(int(dec[n][k])) if k in COUNT_KEYS and dec[n].get(k) is not None
                                                  else _pct(dec[n].get(k))) for n in names]
                        for k in dec[names[0]]]), ""]

    hum = report.get("human_agreement")
    if hum:
        out += ["## Human review agreement", "", _table(["Item", "Value"], [
            ["Reviewer", "SIMULATED (demo only)" if hum.get("simulated") else "real reviewers"],
            ["Reviewed rows", _fmt(hum.get("reviewed_rows"))], ["Raw agreement", _pct(hum.get("raw_agreement"))],
            ["Cohen's kappa", _fmt(hum.get("cohen_kappa"))]]), ""]
    elif "human_agreement" in report:
        out += ["## Human review agreement", "", "No reviews filled in yet.", ""]

    out += ["## Files", "", "`run_report.json` has every number above (machine-readable). Other outputs in the "
            "same folder: `metrics.json`, `cv_results.csv`, `model_comparison.csv`, `test_predictions.csv`, "
            "`feature_importance.csv`, `figures/`." + (f" Cross-run history: `{m['history_file']}`." if m.get("history_file") else ""), ""]
    return "\n".join(out)
