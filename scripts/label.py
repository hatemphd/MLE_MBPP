"""Same as `uv run trust-label` -- see `uv run python scripts/label.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import label

if __name__ == "__main__":
    label()
