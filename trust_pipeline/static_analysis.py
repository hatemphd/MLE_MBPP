"""Batched rule-based static analysis: ruff (lint) and bandit (security)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

RUFF_RULES = "E,W,F,B,SIM,C4"
STATIC_COLUMNS = ["ruff_issues", "ruff_undefined_names", "ruff_unused", "bandit_issues", "bandit_med_high"]


def batch_static_analysis(codes, chunk=2000):
    """One ruff and one bandit call per batch of files. Returns a list of count dicts."""
    results = [dict.fromkeys(STATIC_COLUMNS, 0) for _ in codes]
    for start in range(0, len(codes), chunk):
        with tempfile.TemporaryDirectory() as d:
            for k, code_str in enumerate(codes[start:start + chunk], start=start):
                with open(os.path.join(d, f"c{k}.py"), "w") as f:
                    f.write((code_str or "") + "\n")
            ruff = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero",
                 "--output-format=json", "--select", RUFF_RULES, d],
                capture_output=True, text=True,
            )
            for item in json.loads(ruff.stdout or "[]"):
                k = int(os.path.basename(item["filename"])[1:-3])
                rule = item.get("code") or ""
                results[k]["ruff_issues"] += 1
                results[k]["ruff_undefined_names"] += rule == "F821"
                results[k]["ruff_unused"] += rule in ("F401", "F841")
            bandit = subprocess.run(
                [sys.executable, "-m", "bandit", "-r", d, "-f", "json", "-q"],
                capture_output=True, text=True,
            )
            try:
                bandit_results = json.loads(bandit.stdout or "{}").get("results", [])
            except json.JSONDecodeError:
                bandit_results = []
            for item in bandit_results:
                k = int(os.path.basename(item["filename"])[1:-3])
                results[k]["bandit_issues"] += 1
                results[k]["bandit_med_high"] += item.get("issue_severity") in ("MEDIUM", "HIGH")
    return results


def add_static_features(frame):
    frame = frame.copy()
    static = pd.DataFrame(batch_static_analysis(frame["generated_code"].tolist()), index=frame.index)
    for col in STATIC_COLUMNS:
        frame[col] = static[col]
    return frame
