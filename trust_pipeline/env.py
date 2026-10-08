"""Load API keys from a `.env` file (KEY=value lines) into the environment."""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | os.PathLike | None = None) -> list[str]:
    """Load `.env` from `path`, else the current directory, else the project root.

    Variables already set in the environment win, so `export KEY=...` still overrides the file.
    Returns the names of the variables that were set (never their values).
    """
    candidates = [Path(path)] if path else [Path.cwd() / ".env", _PROJECT_ROOT / ".env"]
    env_file = next((p for p in candidates if p.is_file()), None)
    if env_file is None:
        return []

    loaded = []
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
