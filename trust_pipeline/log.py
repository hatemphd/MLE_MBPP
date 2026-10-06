"""Readable run-time output: timestamped messages, stage banners, timers, quieter third-party libraries.

Messages go to stdout through `tqdm.write`, so they don't break progress bars, and are flushed right
away (Colab buffers output otherwise).
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

from tqdm.auto import tqdm

logger = logging.getLogger("trust_pipeline")
_state = {"configured": False, "progress": True, "verbosity": "info"}


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def setup_logging(verbosity="info"):
    """verbosity: 'quiet' (warnings only, no progress bars), 'info' (default) or 'verbose' (debug)."""
    level = {"quiet": logging.WARNING, "info": logging.INFO, "verbose": logging.DEBUG}[verbosity]
    logger.handlers.clear()
    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _state.update(configured=True, progress=verbosity != "quiet", verbosity=verbosity)
    quiet_third_party()


def _ensure():
    if not _state["configured"]:
        setup_logging()


def progress_enabled():
    _ensure()
    return _state["progress"]


def info(msg):
    _ensure()
    logger.info(msg)


def debug(msg):
    _ensure()
    logger.debug(msg)


def warning(msg):
    _ensure()
    logger.warning(f"WARNING: {msg}")


def block(title, text):
    """Log a multi-line block (e.g. a table) under a title, indented."""
    info(title + "\n" + "\n".join("    " + line for line in str(text).splitlines()))


def banner(title, description="", step=None, total=None, **settings):
    """Stage banner like `==== [Stage 2/6] Generating candidates ====` plus description and settings."""
    _ensure()
    if not logger.isEnabledFor(logging.INFO):
        return
    prefix = f"[Stage {step}/{total}] " if step and total else ""
    tqdm.write("", file=sys.stdout)
    lines = [f"==== {prefix}{title} " + "=" * max(4, 70 - len(prefix) - len(title))]
    if description:
        lines.append(description)
    lines += [f"  {k.replace('_', ' ')}: {v}" for k, v in settings.items()]
    logger.info("\n".join(lines))


@contextmanager
def timed(label):
    start = time.time()
    yield
    info(f"{label} finished in {format_seconds(time.time() - start)}")


def format_seconds(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def counts_line(series, order=None):
    """'pass 58 (19%) | wrong_output 74 (25%) | ...' for a pandas Series of categories."""
    counts = series.value_counts()
    total = max(len(series), 1)
    keys = list(order or []) + [k for k in counts.index if k not in (order or [])]
    return " | ".join(f"{k} {int(counts.get(k, 0))} ({counts.get(k, 0) / total:.0%})" for k in keys)


def quiet_third_party(verbose=None):
    """Hide Hugging Face / datasets chatter. Model download bars stay visible; weight-loading and
    dataset-preparation bars and deprecation warnings are hidden unless verbose.

    Only touches libraries that are already imported; call again right after importing one.
    """
    if verbose is None:
        verbose = _state["verbosity"] == "verbose"
    if verbose:
        return
    for name in ("transformers", "huggingface_hub", "datasets", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if "datasets" in sys.modules:
        try:
            import datasets
            datasets.utils.logging.set_verbosity_error()
            datasets.disable_progress_bars()
        except Exception:
            pass
    if "transformers" in sys.modules:
        try:
            import transformers
            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
        except Exception:
            pass
    if "huggingface_hub" in sys.modules:
        try:
            import huggingface_hub
            huggingface_hub.utils.logging.set_verbosity_error()
            if _state["progress"]:
                huggingface_hub.utils.enable_progress_bars()
            else:
                huggingface_hub.utils.disable_progress_bars()
        except Exception:
            pass
