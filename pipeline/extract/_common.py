"""Shared helpers for the extract scripts.

Each script runs as a file (uv run python pipeline/extract/<script>.py), which
puts this directory on sys.path, so `from _common import ...` works without any
packaging. Keep this module dependency-light: only stdlib + yaml.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FatalApiError(Exception):
    """Unrecoverable API failure (e.g. invalid key). Deliberately NOT a
    RuntimeError and never listed in a retry loop's except tuple, so it aborts
    the run instead of being retried and poisoning every pending item."""


def load_dotenv(env_path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments, quotes stripped,
    real environment always wins (setdefault)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def active_db_path() -> Path:
    """DB path of the active destination from config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg["destinations"][cfg["destination"]]["db_path"])
