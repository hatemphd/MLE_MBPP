"""Decision-time features and hidden-test labels.

Features only use information available before the hidden tests run: the code, the visible test,
probe inputs derived from the visible test, sibling samples, static analysis, generation
confidence and the model config. Hidden-test results are only used for the label.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from .astutils import parse_quietly
from .config import TRAINING_ONLY_PROVENANCE
from .execution import hidden_outcome
from .static_analysis import STATIC_COLUMNS

NAN = float("nan")

AST_KEYS = ["parses", "ast_nodes", "ast_depth", "n_functions", "n_loops", "n_ifs", "n_returns",
            "n_try", "n_comprehensions", "n_lambdas", "n_imports", "n_calls", "uses_print",
            "uses_input", "defines_expected_function", "n_args_expected_function",
            "is_recursive", "has_docstring", "n_magic_numbers", "cc_total", "cc_max",
            "maintainability_index"]
SIZE_KEYS = ["code_chars", "code_lines", "avg_line_len", "max_line_len", "comment_lines"]

CODE_FEATURES = SIZE_KEYS + AST_KEYS
EXECUTION_FEATURES = ["def_ok", "visible_pass"]
CONSISTENCY_FEATURES = ["behavior_agreement", "behavior_agreement_visible", "probe_error_frac"]
STATIC_FEATURES = list(STATIC_COLUMNS)
GENERATION_FEATURES = ["gen_mean_logprob", "gen_min_logprob", "gen_num_tokens"]

LEAKY_COLUMNS = {"exec_status", "hidden_pass_frac", "hidden_results", "exec_error", "label_trustworthy",
                 # How a row was produced: `injected` / `mutant` almost always means "buggy".
                 "provenance", "mutation_type", "prompt_style"}
# Injected samples have their comments and docstrings stripped, so with them in the training data
# these features would mostly detect "was injected" rather than "is buggy".
INJECTION_SENSITIVE_FEATURES = ["comment_lines", "has_docstring"]
TEXT_COLUMNS = ["generated_code", "gen_error", "mutation_type", "problem_text", "probe_outputs",
                "hidden_results", "exec_error", "def_status", "exec_status", "model", "prompt_style",
                "provenance"]


# ---------------------------------------------------------------- execution ----------------------

def add_execution_features(frame, problems_by_id, executor):
    """Feature run: module definitions, visible test and probes (decision-time information)."""
    jobs = []
    for _, row in frame.iterrows():
        p = problems_by_id[row["task_id"]]
        jobs.append(dict(code=row["generated_code"], setup=p["setup_code"],
                         tests=p["visible_tests"], probes=p["probes"]))
    runs = executor.run_many(jobs, "feature run: visible test + probes")
    frame = frame.copy()
    frame["def_status"] = [r["status"] for r in runs]
    frame["def_ok"] = (frame["def_status"] == "ok").astype(int)
    frame["visible_pass"] = [int(r["status"] == "ok" and all(t == "pass" for t in r["tests"]))
                             for r in runs]
    frame["probe_outputs"] = [json.dumps(r["probes"]) for r in runs]
    return frame


def add_hidden_test_labels(frame, problems_by_id, executor):
    """Label run: hidden tests only. Produces ground truth; never used as a feature."""
    jobs = []
    for _, row in frame.iterrows():
        p = problems_by_id[row["task_id"]]
        jobs.append(dict(code=row["generated_code"], setup=p["setup_code"], tests=p["hidden_tests"]))
    runs = executor.run_many(jobs, "label run: hidden tests")
    frame = frame.copy()
    outcomes = [hidden_outcome(r) for r in runs]
    frame["exec_status"] = [o[0] for o in outcomes]
    frame["hidden_pass_frac"] = [o[1] for o in outcomes]
    frame["hidden_results"] = [json.dumps(r["tests"]) for r in runs]
    frame["exec_error"] = [r.get("error", "") for r in runs]
    return frame


def add_label(frame):
    """Trustworthy = passes every hidden test and has no medium/high-severity Bandit finding."""
    frame = frame.copy()
    frame["label_trustworthy"] = ((frame["exec_status"] == "pass")
                                  & (frame["bandit_med_high"] == 0)).astype(int)
    return frame


# ---------------------------------------------------------------- self-consistency ---------------

def _is_probe_error(x):
    return x.startswith("<error:") or x == "<timeout>"


def add_consistency_features(frame, all_in_pool=False):
    """Behaviour agreement of each candidate with its siblings on the probe inputs.

    The pool is the NATURAL samples of the problem (all models and prompt styles). Training-only rows
    (mutants, injected-bug samples) are scored against the pool but never added to it (unless
    `all_in_pool`, used by the smoke test), so they cannot change the features of evaluation rows.
    Signatures where every probe errored agree with nobody.
    """
    frame = frame.copy()
    in_pool = (pd.Series(True, index=frame.index) if all_in_pool
               else ~frame["provenance"].isin(TRAINING_ONLY_PROVENANCE))
    agree, agree_vis, err_frac = {}, {}, {}
    for _, g in frame.groupby("task_id"):
        pool = g[in_pool.loc[g.index]]
        pool_items = list(zip(pool.index, pool["probe_outputs"], pool["visible_pass"]))
        for idx, row in g.iterrows():
            outputs = json.loads(row["probe_outputs"])
            errors = [_is_probe_error(x) for x in outputs]
            err_frac[idx] = float(np.mean(errors)) if outputs else 1.0
            others = [(sig, vis) for i, sig, vis in pool_items if i != idx]
            if not outputs or all(errors):
                agree[idx] = 0.0
                agree_vis[idx] = 0.0
                continue
            same = [sig == row["probe_outputs"] for sig, _ in others]
            agree[idx] = float(np.mean(same)) if others else NAN
            vis_same = [sig == row["probe_outputs"] for sig, vis in others if vis]
            agree_vis[idx] = float(np.mean(vis_same)) if vis_same else NAN
    frame["behavior_agreement"] = pd.Series(agree)
    frame["behavior_agreement_visible"] = pd.Series(agree_vis)
    frame["probe_error_frac"] = pd.Series(err_frac)
    return frame


# ---------------------------------------------------------------- code structure -----------------

def _ast_depth(node, depth=0):
    return max((_ast_depth(c, depth + 1) for c in ast.iter_child_nodes(node)), default=depth)


def code_features(code_str, function_name):
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    code_str = code_str if isinstance(code_str, str) else ""
    lines = [l for l in code_str.splitlines() if l.strip()]
    f = {
        "code_chars": len(code_str),
        "code_lines": len(lines),
        "avg_line_len": float(np.mean([len(l) for l in lines])) if lines else 0.0,
        "max_line_len": max((len(l) for l in lines), default=0),
        "comment_lines": sum(l.strip().startswith("#") for l in lines),
    }
    f.update(dict.fromkeys(AST_KEYS, 0))
    try:
        tree = parse_quietly(code_str)
    except (SyntaxError, ValueError):
        return f
    nodes = list(ast.walk(tree))
    called = {n.func.id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    funcs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    target = next((fn for fn in funcs if fn.name == function_name), None)
    f.update({
        "parses": 1,
        "ast_nodes": len(nodes),
        "ast_depth": _ast_depth(tree),
        "n_functions": len(funcs),
        "n_loops": sum(isinstance(n, (ast.For, ast.While)) for n in nodes),
        "n_ifs": sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes),
        "n_returns": sum(isinstance(n, ast.Return) for n in nodes),
        "n_try": sum(isinstance(n, ast.Try) for n in nodes),
        "n_comprehensions": sum(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                                               ast.GeneratorExp)) for n in nodes),
        "n_lambdas": sum(isinstance(n, ast.Lambda) for n in nodes),
        "n_imports": sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes),
        "n_calls": sum(isinstance(n, ast.Call) for n in nodes),
        "uses_print": int("print" in called),
        "uses_input": int("input" in called),
        "n_magic_numbers": sum(isinstance(n, ast.Constant) and type(n.value) in (int, float)
                               and n.value not in (0, 1, 2, -1) for n in nodes),
    })
    if target is not None:
        f["defines_expected_function"] = 1
        f["n_args_expected_function"] = len(target.args.args)
        f["has_docstring"] = int(ast.get_docstring(target) is not None)
        f["is_recursive"] = int(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                    and n.func.id == function_name for n in ast.walk(target)))
    try:
        blocks = cc_visit(code_str)
        f["cc_total"] = sum(b.complexity for b in blocks)
        f["cc_max"] = max((b.complexity for b in blocks), default=0)
    except Exception:
        pass
    try:
        f["maintainability_index"] = float(mi_visit(code_str, multi=True))
    except Exception:
        pass
    return f


def add_code_features(frame, problems_by_id, model_names):
    frame = frame.copy()
    feats = pd.DataFrame(
        [code_features(c, problems_by_id[t]["function_name"])
         for c, t in zip(frame["generated_code"], frame["task_id"])],
        index=frame.index,
    )
    for col in feats.columns:
        frame[col] = feats[col]
    frame["problem_words"] = frame["task_id"].map(lambda t: len(problems_by_id[t]["text"].split()))
    for name in model_names:
        frame[f"model_is_{name}"] = (frame["model_name"] == name).astype(int)
    return frame


# ---------------------------------------------------------------- feature selection --------------

def all_feature_columns(model_names):
    return (CODE_FEATURES + EXECUTION_FEATURES + CONSISTENCY_FEATURES + STATIC_FEATURES
            + GENERATION_FEATURES + ["problem_words"] + [f"model_is_{n}" for n in model_names])


def select_feature_columns(frame, model_names):
    """Explicit feature list minus columns with no information in this run; guards against leakage."""
    candidates = all_feature_columns(model_names)
    if "provenance" in frame and (frame["provenance"] == "injected").any():
        candidates = [c for c in candidates if c not in INJECTION_SENSITIVE_FEATURES]
    selected = [c for c in candidates
                if c in frame and frame[c].notna().any() and frame[c].nunique(dropna=True) > 1]
    check_no_leakage(selected)
    return selected, sorted(set(all_feature_columns(model_names)) - set(selected))


def check_no_leakage(feature_columns):
    leaked = LEAKY_COLUMNS & set(feature_columns)
    if leaked or any(c.startswith(("hidden", "prompt_style", "provenance")) for c in feature_columns):
        raise ValueError(f"hidden-test / label / provenance information leaked into the features: "
                         f"{sorted(leaked)}")


def read_frame(path):
    """pd.read_csv that keeps empty code/text cells as '' instead of NaN."""
    df = pd.read_csv(path)
    for col in TEXT_COLUMNS:
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    return df
