"""Fallback so scripts/ also works when trust_pipeline isn't installed (`uv sync` installs it)."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import trust_pipeline  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
