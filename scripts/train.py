"""Same as `uv run trust-train` -- see `uv run python scripts/train.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import train

if __name__ == "__main__":
    train()
