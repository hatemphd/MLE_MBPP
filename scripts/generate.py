"""Same as `uv run trust-generate` -- see `uv run python scripts/generate.py --help`."""
import _bootstrap  # noqa: F401
from trust_pipeline.cli import generate

if __name__ == "__main__":
    generate()
