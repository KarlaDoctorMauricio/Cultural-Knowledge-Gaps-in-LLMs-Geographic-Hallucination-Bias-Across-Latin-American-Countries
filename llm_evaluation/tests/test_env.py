import importlib.util
from pathlib import Path

import pytest

ENV_MODULE_PATH = Path(__file__).resolve().parents[1] / "fairness_toolkit" / "env.py"


def _load_env_module():
    spec = importlib.util.spec_from_file_location("env_module", ENV_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_env_raises_clear_message(monkeypatch):
    env = _load_env_module()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Falta ANTHROPIC_API_KEY en \\.env"):
        env.require_env("ANTHROPIC_API_KEY")
