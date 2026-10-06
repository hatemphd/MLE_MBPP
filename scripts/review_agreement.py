"""Same as `uv run trust-review` -- see `uv run python scripts/review_agreement.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import review_agreement

if __name__ == "__main__":
    review_agreement()
