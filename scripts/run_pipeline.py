"""Same as `uv run trust-pipeline` -- see `uv run python scripts/run_pipeline.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import run_pipeline

if __name__ == "__main__":
    run_pipeline()
