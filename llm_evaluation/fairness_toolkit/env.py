"""Environment variable loading for LLM evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
_ENV_LOADED = False


def load_env() -> None:
    """Load variables from the project ``.env`` file if present."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(ENV_FILE)
    _ENV_LOADED = True


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear configuration error."""
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Falta {name} en .env")
    return value
