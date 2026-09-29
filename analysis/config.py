"""Loader for analysis/config.yaml — Stage-A defaults (single source of truth).

Scope: estimator/tuning defaults (warehouse path, seed, trip rule, gate
thresholds, DML partitions, Optuna trials). Pre-registered refutation floors
stay next to their tests as dataclass defaults in refute.py / sensitivity.py.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CFG: dict = yaml.safe_load(
    (Path(__file__).resolve().parent / "config.yaml").read_text())
