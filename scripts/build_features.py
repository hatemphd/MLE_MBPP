"""Same as `uv run trust-features` -- see `uv run python scripts/build_features.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import build_features

if __name__ == "__main__":
    build_features()
