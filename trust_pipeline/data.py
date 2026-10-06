"""Load MBPP, split each problem's tests into visible (prompt + features) and hidden (label only)."""
from __future__ import annotations

import json
import random

from . import log
from .astutils import expected_call, make_probes

MBPP_DATASET = "google-research-datasets/mbpp"


def make_problem(task_id, text, visible_tests, hidden_tests, reference_code="", setup_code="",
                 probes_per_problem=6, seed=42):
    """Problem dict used throughout the pipeline; also usable for problems outside MBPP."""
    visible_tests = list(visible_tests)
    function_name, _ = expected_call(visible_tests[0]) if visible_tests else (None, None)
    return {
        "task_id": task_id,
        "text": text,
        "reference_code": (reference_code or "").replace("\r\n", "\n"),
        "setup_code": (setup_code or "").replace("\r\n", "\n"),
        "visible_tests": visible_tests,
        "hidden_tests": list(hidden_tests),
        "function_name": function_name,
        "probes": make_probes(visible_tests[0], probes_per_problem, seed) if visible_tests else [],
    }


def mbpp_record_to_problem(r, probes_per_problem=6, seed=42):
    return make_problem(
        task_id=r["task_id"],
        text=r["text"],
        visible_tests=r["test_list"][:1],
        hidden_tests=list(r["test_list"][1:]) + list(r["challenge_test_list"]),
        reference_code=r["code"],
        setup_code=r["test_setup_code"],
        probes_per_problem=probes_per_problem,
        seed=seed,
    )


def load_mbpp_records():
    """All 974 problems from every split of the `full` config, sorted by task_id.

    Falls back to the local Hugging Face cache when the Hub can't be reached.
    """
    from datasets import concatenate_datasets, load_dataset
    log.quiet_third_party()
    log.info(f"Loading MBPP ({MBPP_DATASET}) from Hugging Face...")
    try:
        raw = load_dataset(MBPP_DATASET)
    except Exception as e:
        log.warning(f"could not reach the Hugging Face Hub ({type(e).__name__}); using the local cache.")
        import datasets.config
        import huggingface_hub.constants
        datasets.config.HF_HUB_OFFLINE = True
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        raw = load_dataset(MBPP_DATASET)
    records = concatenate_datasets([raw[s] for s in raw.keys()]).sort("task_id")
    log.info(f"Loaded {len(records)} MBPP problems (all splits).")
    return records


def load_problems(cfg, executor):
    """Random subset of MBPP whose reference solutions pass all of their own tests."""
    records = load_mbpp_records()
    problems = [mbpp_record_to_problem(r, cfg.probes_per_problem, cfg.seed) for r in records]
    problems = [p for p in problems if p["function_name"] and p["hidden_tests"]]
    random.Random(cfg.seed).shuffle(problems)
    selected = problems if cfg.num_problems is None else problems[:cfg.num_problems]
    log.info(f"Selected {len(selected)} problems (seed {cfg.seed}); checking that each reference "
             "solution passes its own tests...")
    runs = executor.run_many(
        [dict(code=p["reference_code"], setup=p["setup_code"],
              tests=p["visible_tests"] + p["hidden_tests"]) for p in selected],
        "validating reference solutions",
    )
    valid = [p for p, r in zip(selected, runs)
             if r["status"] == "ok" and all(t == "pass" for t in r["tests"])]
    return sorted(valid, key=lambda p: p["task_id"]), len(selected) - len(valid)


def save_problems(problems, path):
    with open(path, "w") as f:
        json.dump(problems, f, indent=1)


def read_problems(path):
    with open(path) as f:
        return json.load(f)


def problems_by_id(problems):
    return {p["task_id"]: p for p in problems}
