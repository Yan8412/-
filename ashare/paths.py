"""Project-root paths so a launch from another folder does not write to C:."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Directory that contains ``pyproject.toml`` (the repository root)."""
    return Path(__file__).resolve().parents[1]


def resolve_dir(value: Path | None, env_name: str, relative: str) -> Path:
    """Explicit flag, then env var, then ``<project root>/<relative>``."""
    if value is not None:
        return Path(value).expanduser().resolve()
    env = os.environ.get(env_name)
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / relative


def default_config_path() -> Path:
    env = os.environ.get("ASHARE_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "config.json"


def report_dir(value: Path | None = None) -> Path:
    return resolve_dir(value, "ASHARE_REPORT_DIR", "reports")


def cache_dir(value: Path | None = None) -> Path:
    return resolve_dir(value, "ASHARE_CACHE_DIR", "data/cache")


def ledger_path(value: Path | None = None) -> Path:
    return resolve_dir(value, "ASHARE_LEDGER", "data/paper/ledger.json")


def model_path(value: Path | None = None) -> Path:
    return resolve_dir(value, "ASHARE_MODEL", "data/models/ranker.pkl")
