"""Same as `uv run trust-prepare` -- see `uv run python scripts/prepare_data.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import prepare

if __name__ == "__main__":
    prepare()
