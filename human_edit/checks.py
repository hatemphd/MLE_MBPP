"""Helpers for the review editor (`app.py`): CSV loading/saving and sandboxed checks."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from trust_pipeline.execution import Executor
from trust_pipeline.human_review import AUTOMATED_NOTE, SIMULATED_NOTE

DEFAULT_FILE = PROJECT_ROOT / "artifacts_pilot_openai" / "human_review_sample.csv"
MACHINE_NOTES = (SIMULATED_NOTE, AUTOMATED_NOTE)
__all__ = ["AUTOMATED_NOTE", "DEFAULT_FILE", "MACHINE_NOTES", "SIMULATED_NOTE", "load_problems",
           "load_reviews", "make_executor", "run_checks", "save_reviews", "split_checks"]
TEXT_COLUMNS = ("human_trustworthy", "human_notes", "human_checks", "test_status", "test_failure",
                "tested_at")


def load_reviews(path):
    frame = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=False)
    for col in ("human_trustworthy", "human_notes", "human_checks"):
        if col not in frame.columns:
            frame[col] = ""
    frame["human_trustworthy"] = frame["human_trustworthy"].str.strip().str.lower()
    return frame


def save_reviews(frame, path):
    """Write to a temporary file first so a crash never leaves a half-written CSV."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".csv.tmp")
    os.close(fd)
    frame.to_csv(tmp, index=False)
    if os.path.exists(path):
        os.chmod(tmp, os.stat(path).st_mode & 0o777 | 0o644)
    os.replace(tmp, path)


def load_problems(folder):
    path = Path(folder) / "problems.json"
    if not path.exists():
        return {}
    return {str(p["task_id"]): p for p in json.loads(path.read_text())}


def make_executor(max_workers=1):
    """Same sandbox as the pipeline: separate process, 3 s per check, 1 GB memory cap."""
    return Executor(item_timeout=3.0, memory_limit_mb=1024, max_workers=max_workers, progress=False)


def split_checks(lines):
    """Lines starting with `assert` are tests; any other non-empty, non-comment line is an expression."""
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return ([line for line in lines if line.startswith("assert")],
            [line for line in lines if not line.startswith("assert")])


def summarize(result, tests, probes):
    return {"status": result["status"], "error": result.get("error", ""),
            "tests": list(zip(tests, result["tests"])), "probes": list(zip(probes, result["probes"]))}


def run_checks(executor, code, setup, lines):
    tests, probes = split_checks(lines)
    return summarize(executor.execute(code, setup, tests, probes), tests, probes)
