import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clients import (  # noqa: E402
    clean_latamgpt_response,
    is_latamgpt_response_broken,
    is_latamgpt_unavailable_error,
    query_claude,
    query_gpt,
    query_latamgpt,
    query_latamgpt_with_meta,
    reset_latamgpt_status,
)


def test_clean_latamgpt_response_strips_assistant_leakage():
    raw = "Antonio Gonzassistant\n\nAntassistant\n\nAntonio Gonzaga nassistant"
    cleaned = clean_latamgpt_response(raw)
    assert "assistant" not in cleaned.lower()
    assert "Gonzaga" in cleaned


def test_is_latamgpt_response_broken_detects_loops():
    raw = "Elassistant\n\nEl mateassistant\n\nEl mate esassistant"
    cleaned = clean_latamgpt_response(raw)
    assert is_latamgpt_response_broken(raw, cleaned)


def test_is_latamgpt_response_broken_accepts_normal_text():
    assert not is_latamgpt_response_broken("Paris.", "Paris.")


def test_query_latamgpt_falls_back_to_completion(monkeypatch):
    reset_latamgpt_status()
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    mock_client = MagicMock()
    bad = MagicMock(message=MagicMock(content="Antassistant\n\nAnt"))
    good = MagicMock(text="Antonio Gonzaga nacio en Corrientes, Argentina.")
    mock_client.chat.completions.create.return_value = MagicMock(choices=[bad])
    mock_client.completions.create.return_value = MagicMock(choices=[good])

    with patch("openai.OpenAI", return_value=mock_client):
        result, method, _ = query_latamgpt_with_meta("¿Dónde nació Antonio Gonzaga?")

    assert method == "completion"
    assert "Corrientes" in (result or "")
    assert mock_client.completions.create.call_count == 1


def test_query_gpt_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert query_gpt("¿Qué es el mate?") is None


def test_query_claude_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert query_claude("¿Qué es el mate?") is None


def test_query_latamgpt_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reset_latamgpt_status()
    assert query_latamgpt("¿Qué es el mate?") is None


def test_query_gpt_retries_once_then_returns_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Respuesta GPT"))]
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("timeout"),
        mock_response,
    ]

    with patch("openai.OpenAI", return_value=mock_client):
        result = query_gpt("¿Qué es el mate?")

    assert result == "Respuesta GPT"
    assert mock_client.chat.completions.create.call_count == 2


def test_query_gpt_returns_none_after_two_failures(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("quota exceeded")

    with patch("openai.OpenAI", return_value=mock_client):
        with patch("clients.time.sleep", return_value=None):
            result = query_gpt("¿Qué es el mate?")

    assert result is None
    assert mock_client.chat.completions.create.call_count == 2


def test_is_latamgpt_unavailable_error_detects_not_deployed():
    exc = RuntimeError("Model latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0 is not deployed")
    assert is_latamgpt_unavailable_error(exc)


def test_query_latamgpt_marks_unavailable_on_not_deployed(monkeypatch):
    reset_latamgpt_status()
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError(
        "Model is not deployed on Inference API"
    )

    with patch("openai.OpenAI", return_value=mock_client):
        with patch("clients.time.sleep", return_value=None):
            result = query_latamgpt("¿Qué es el mate?")

    assert result is None
    assert query_latamgpt("otra") is None
