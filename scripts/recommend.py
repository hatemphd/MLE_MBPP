"""Same as `uv run trust-recommend` -- see `uv run python scripts/recommend.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import recommend

if __name__ == "__main__":
    recommend()
