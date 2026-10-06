"""Command-line entry points (installed as `trust-*` commands by `uv sync`)."""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import time  # noqa: E402

from . import log, report  # noqa: E402
from .pipeline import (ArtifactPaths, add_common_args, add_verbosity_args, config_from_args,  # noqa: E402
                       log_summary, run_all, stage_features, stage_generate, stage_human,
                       stage_label, stage_prepare, stage_train, verbosity_from_args)


def _parse(command, description, extra=None, fresh_report=False):
    """`fresh_report`: this command starts a new run (new run_id) in the run report."""
    parser = add_common_args(argparse.ArgumentParser(
        prog=command, description=description, formatter_class=argparse.RawDescriptionHelpFormatter))
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = ArtifactPaths(cfg.output_dir)
    report.begin(paths, cfg, command, args=vars(args), fresh=fresh_report)
    return args, cfg, paths


def _no_plots(parser):
    parser.add_argument("--no-plots", action="store_true")


def run_pipeline():
    """Run every stage end to end: prepare -> generate -> execute -> static analysis -> features ->
human review -> train -> decision rules. Each stage prints a banner, progress and a timing line.

Quick pipeline check without any LLM:
  uv run trust-pipeline --smoke-test --num-problems 40 --output-dir artifacts_smoke
"""
    args, cfg, _ = _parse("trust-pipeline", run_pipeline.__doc__, _no_plots, fresh_report=True)
    if cfg.smoke_test:
        log.info("SMOKE TEST: candidates are reference solutions + AST mutants; numbers are not findings.")
    run_all(cfg, make_plots=not args.no_plots)


def prepare():
    """Stage 1: load MBPP, split visible/hidden tests, validate reference solutions -> problems.json.
Starts a new run in run_report.json."""
    _, cfg, paths = _parse("trust-prepare", prepare.__doc__, fresh_report=True)
    stage_prepare(cfg, paths)


def generate():
    """Stage 2: generate candidate solutions (LLMs, or reference + mutants with --smoke-test) -> candidates.csv."""
    _, cfg, paths = _parse("trust-generate", generate.__doc__)
    stage_generate(cfg, paths)


def label():
    """Stages 3-4: execute candidates (visible test + probes, hidden tests), static analysis, labels -> labeled.csv."""
    _, cfg, paths = _parse("trust-label", label.__doc__)
    stage_label(cfg, paths)


def build_features():
    """Stages 5-6: self-consistency + code-structure features, then the human review sample -> dataset.csv."""
    _, cfg, paths = _parse("trust-features", build_features.__doc__)
    dataset, _ = stage_features(cfg, paths)
    stage_human(cfg, paths, dataset)


def train():
    """Stages 7-8: grouped split, baselines vs. classifiers, calibration, thresholds, conformal, metrics,
figures, run_report.md and a runs_history.csv row."""
    start = time.time()
    args, cfg, paths = _parse("trust-train", train.__doc__, _no_plots)
    results = stage_train(cfg, paths, make_plots=not args.no_plots)
    log_summary(cfg, paths, results, time.time() - start, title="Training finished")


def review_agreement():
    """Recompute human-vs-automated agreement (Cohen's kappa) after reviewers fill in the review CSV.
Also refreshes run_report.md and this run's row in runs_history.csv."""
    _, cfg, paths = _parse("trust-review", review_agreement.__doc__)
    agreement = stage_human(cfg, paths)
    report.finalize(paths, cfg, stamp_end=False)
    print(json.dumps(agreement, indent=2) if agreement else "No human labels filled in yet.")


def recommend():
    """Score a new candidate with the trained model: APPROVED / REJECTED / NEEDS HUMAN REVIEW.

Examples:
  uv run trust-recommend --task-id 17 --code-file my_solution.py
  uv run trust-recommend --task-id 17 --code-file a.py --sibling b.py --sibling c.py
  uv run trust-recommend --problem-text "Write a function to add two numbers." \\
      --visible-test "assert add(1, 2) == 3" --code-file add.py
"""
    from . import data
    from .recommendation import TrustRecommender

    parser = argparse.ArgumentParser(description=recommend.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--task-id", type=int, help="MBPP task id")
    parser.add_argument("--problem-text", help="custom problem statement (instead of --task-id)")
    parser.add_argument("--visible-test", help="one assert statement for a custom problem")
    parser.add_argument("--code-file", required=True, help="candidate solution to score")
    parser.add_argument("--sibling", action="append", default=[],
                        help="other candidate solutions for the same problem (self-consistency)")
    add_verbosity_args(parser)
    args = parser.parse_args()
    # The JSON result is the output here, so progress messages stay off unless --verbose.
    log.setup_logging(verbosity_from_args(args, default="quiet"))
    if not args.problem_text and args.task_id is None:
        parser.error("give --task-id or --problem-text/--visible-test")
    if args.problem_text and not args.visible_test:
        parser.error("--problem-text needs --visible-test")

    model_path = os.path.join(args.output_dir, "trust_model.joblib")
    if not os.path.exists(model_path):
        parser.error(f"{model_path} not found; train first with `uv run trust-pipeline --output-dir {args.output_dir}`")
    recommender = TrustRecommender.load(model_path)

    if args.problem_text:
        problem = data.make_problem("custom", args.problem_text, [args.visible_test], [])
    else:
        problem = None
        problems_path = os.path.join(args.output_dir, "problems.json")
        if os.path.exists(problems_path):
            problem = next((p for p in data.read_problems(problems_path) if p["task_id"] == args.task_id), None)
        if problem is None:
            record = next((r for r in data.load_mbpp_records() if r["task_id"] == args.task_id), None)
            if record is None:
                parser.error(f"MBPP task {args.task_id} not found")
            problem = data.mbpp_record_to_problem(record)

    with open(args.code_file) as f:
        code = f.read()
    siblings = []
    for path in args.sibling:
        with open(path) as f:
            siblings.append(f.read())
    print(json.dumps(recommender.recommend(problem, code, siblings), indent=2))
