"""Sandboxed subprocess execution of candidate code against tests and probe inputs.

MBPP code is benign, so a subprocess with per-item time limits and a memory cap is enough here.
Use a container (Docker / nsjail) before running candidates from untrusted sources.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from . import log

RUNNER_SOURCE = r"""
import json
import signal
import sys

payload = json.loads(sys.stdin.read())
item_timeout = payload["item_timeout"]

try:
    import resource
    limit = payload["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except Exception:
    pass


class ItemTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise ItemTimeout()


HAS_ALARM = hasattr(signal, "setitimer")
if HAS_ALARM:
    signal.signal(signal.SIGALRM, _on_alarm)


def guarded(fn):
    if HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, item_timeout)
    try:
        return fn()
    finally:
        if HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)


out = {"status": "ok", "error": "", "tests": [], "probes": []}


def emit():
    sys.stdout.write("\n__RESULT__" + json.dumps(out) + "\n")
    sys.stdout.flush()


ns = {"__name__": "__candidate__"}
try:
    compiled = compile(payload["setup"] + "\n" + payload["code"], "<candidate>", "exec")
except (SyntaxError, ValueError) as e:
    out["status"], out["error"] = "syntax_error", repr(e)[:300]
    emit()
    sys.exit(0)

try:
    guarded(lambda: exec(compiled, ns))
except ItemTimeout:
    out["status"], out["error"] = "timeout", "module-level code exceeded the time limit"
    emit()
    sys.exit(0)
except BaseException as e:
    out["status"], out["error"] = "runtime_error", (type(e).__name__ + ": " + str(e))[:300]
    emit()
    sys.exit(0)


def run_test(src):
    try:
        guarded(lambda: exec(src, ns))
        return "pass"
    except ItemTimeout:
        return "timeout"
    except AssertionError:
        return "fail"
    except BaseException as e:
        return "error:" + type(e).__name__


def run_probe(expr):
    try:
        return guarded(lambda: repr(eval(expr, ns)))[:200]
    except ItemTimeout:
        return "<timeout>"
    except BaseException as e:
        return "<error:" + type(e).__name__ + ">"


out["tests"] = [run_test(t) for t in payload["tests"]]
out["probes"] = [run_probe(p) for p in payload["probes"]]
emit()
"""


class Executor:
    """Runs candidates in separate Python processes; one assertion/probe gets `item_timeout` seconds."""

    def __init__(self, item_timeout=3.0, memory_limit_mb=1024, max_workers=4, progress=True):
        self.item_timeout = item_timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_workers = max_workers
        self.progress = progress
        self.sandbox_dir = tempfile.mkdtemp(prefix="mbpp_sandbox_")
        self.runner_path = os.path.join(self.sandbox_dir, "_runner.py")
        with open(self.runner_path, "w") as f:
            f.write(RUNNER_SOURCE)
        # Fixed hash seed so repr() of sets is comparable across candidates.
        self._env = {**os.environ, "PYTHONHASHSEED": "0"}
        atexit.register(shutil.rmtree, self.sandbox_dir, ignore_errors=True)

    @classmethod
    def from_config(cls, cfg, progress=True):
        return cls(cfg.item_timeout, cfg.memory_limit_mb, cfg.max_workers, progress)

    def execute(self, code, setup="", tests=(), probes=()):
        """Returns {status, error, tests: [pass|fail|error:X|timeout], probes: [repr | <error:X> | <timeout>]}."""
        tests, probes = list(tests), list(probes)
        payload = json.dumps({"code": code or "", "setup": setup or "", "tests": tests,
                              "probes": probes, "item_timeout": self.item_timeout,
                              "memory_mb": self.memory_limit_mb})
        budget = self.item_timeout * (len(tests) + len(probes) + 1) + 5
        crash = {"status": "runtime_error", "tests": ["error:Crash"] * len(tests),
                 "probes": ["<error:Crash>"] * len(probes)}
        try:
            proc = subprocess.run([sys.executable, self.runner_path], input=payload,
                                  capture_output=True, text=True, timeout=budget,
                                  cwd=self.sandbox_dir, env=self._env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "process exceeded its overall time budget",
                    "tests": ["timeout"] * len(tests), "probes": ["<timeout>"] * len(probes)}
        lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
        if not lines:
            return {**crash, "error": (proc.stderr or "process exited without a result")[-300:]}
        try:
            return json.loads(lines[-1][len("__RESULT__"):])
        except json.JSONDecodeError:
            return {**crash, "error": "unreadable result"}

    def run_many(self, jobs, desc="executing"):
        start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = pool.map(lambda job: self.execute(**job), jobs)
            results = list(tqdm(results, total=len(jobs), desc=f"  {desc}", unit="run", file=sys.stdout,
                                mininterval=1.0, disable=not (self.progress and log.progress_enabled())))
        log.debug(f"{desc}: {len(jobs)} subprocess runs in {log.format_seconds(time.time() - start)} "
                  f"({self.max_workers} workers)")
        return results


def hidden_outcome(result):
    """Collapse a label run into (exec_status, fraction of hidden tests passed)."""
    if result["status"] != "ok":
        return result["status"], 0.0
    tests = result["tests"]
    frac = sum(t == "pass" for t in tests) / max(len(tests), 1)
    if all(t == "pass" for t in tests):
        return "pass", 1.0
    if any(t == "timeout" for t in tests):
        return "timeout", frac
    if all(t in ("pass", "fail") for t in tests):
        return "wrong_output", frac
    return "runtime_error", frac
