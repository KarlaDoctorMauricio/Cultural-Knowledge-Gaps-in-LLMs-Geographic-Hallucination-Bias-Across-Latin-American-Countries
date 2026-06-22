"""Thin API clients for CHOCLO LLM evaluation."""

from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

ENV_MODULE_PATH = Path(__file__).resolve().parent / "fairness_toolkit" / "env.py"

LATAMGPT_UNAVAILABLE_MARKERS = (
    "not deployed",
    "model is not deployed",
    "model not deployed",
    "not found",
    "no inference provider",
    "inference provider",
    "not available",
    "currently unavailable",
    "503",
    "404",
    "failed to load model",
)

_latamgpt_unavailable_note: Optional[str] = None


def _load_env_module():
    spec = importlib.util.spec_from_file_location("llm_env", ENV_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_env = _load_env_module()
load_env = _env.load_env

load_env()

SYSTEM_PROMPT = (
    "Responde la siguiente pregunta de forma breve y directa, en español."
)
LATAMGPT_SYSTEM_PROMPT = (
    "Responde de forma clara y sin repetir palabras."
)
LATAMGPT_CHAT_TEMPERATURE = 0.3
LATAMGPT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 60.0
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_LATAMGPT_MODEL = "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0:featherless-ai"
DEFAULT_GPT_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

_last_query_errors: dict[str, str] = {}
LATAMGPT_UNAVAILABLE_DEFAULT_NOTE = (
    "LatamGPT no disponible: modelo sin Inference Provider activo en Hugging Face"
)

T = TypeVar("T")


def reset_latamgpt_status() -> None:
    """Reset cached LatamGPT availability state between pipeline runs."""
    global _latamgpt_unavailable_note
    _latamgpt_unavailable_note = None


def get_latamgpt_unavailability_note() -> Optional[str]:
    """Return a human-readable note when LatamGPT is unavailable."""
    return _latamgpt_unavailable_note


def is_latamgpt_unavailable_error(exc: Exception) -> bool:
    """Detect Hugging Face errors indicating the model is not served."""
    parts = [str(exc).lower()]
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("text", "content", "body"):
            value = getattr(response, attr, None)
            if value is not None:
                parts.append(str(value).lower())
    combined = " ".join(parts)
    return any(marker in combined for marker in LATAMGPT_UNAVAILABLE_MARKERS)


def _set_latamgpt_unavailable(note: str) -> None:
    global _latamgpt_unavailable_note
    _latamgpt_unavailable_note = note


def _get_env(name: str) -> Optional[str]:
    load_env()
    value = os.environ.get(name, "").strip()
    return value or None


def _chat_messages(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _latamgpt_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Option A: chat.completions with a consistency-focused system prompt."""
    return [
        {"role": "system", "content": LATAMGPT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _latamgpt_raw_prompt(prompt: str) -> str:
    """Option B: manual Llama-style prompt to bypass chat router quirks."""
    return (
        "<|system|>\n"
        "Eres un asistente util. Responde en espanol de forma breve y clara.\n"
        "<|user|>\n"
        f"{prompt}\n"
        "<|assistant|>\n"
    )


def clean_latamgpt_response(content: str) -> str:
    """
    Strip ``assistant`` token leakage common in LatamGPT via HF router.

    The featherless-ai endpoint sometimes emits partial Llama chat-template
    tokens (e.g. ``Antassistant``, ``Gonzaga nassistant``) instead of clean text.
    """
    if not content or not content.strip():
        return content.strip()

    candidates: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"(?i)assistant", "", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
        if len(cleaned) >= 8:
            candidates.append(cleaned)

    if candidates:
        return max(candidates, key=len)

    fallback = re.sub(r"(?i)assistant", "", content)
    return re.sub(r"\s+", " ", fallback).strip()


def is_latamgpt_response_broken(raw: str, cleaned: str) -> bool:
    """
    Detect unusable LatamGPT output (Option C signal).

    Featherless router may return assistant-token loops or truncated fragments.
    """
    if not cleaned or len(cleaned.strip()) < 5:
        return True

    raw_lower = raw.lower()
    assistant_hits = raw_lower.count("assistant")

    if assistant_hits >= 2:
        return True
    if assistant_hits >= 1 and len(cleaned.strip()) < 30:
        return True
    if re.search(r"(?i)(assistant\s*){2,}", raw):
        return True
    if re.search(r"¿qu[eé] quieres decir", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"(?i)no entiendo tu pregunta", cleaned):
        return True
    if len(cleaned.strip()) < 15 and cleaned.strip()[-1] not in ".!?":
        return True

    words = cleaned.split()
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.45:
            return True

    return False


def _latamgpt_chat_call(client, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=_latamgpt_chat_messages(prompt),
        temperature=LATAMGPT_CHAT_TEMPERATURE,
        max_tokens=LATAMGPT_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LatamGPT chat returned an empty response.")
    return content


def _latamgpt_completion_call(client, model: str, prompt: str) -> str:
    response = client.completions.create(
        model=model,
        prompt=_latamgpt_raw_prompt(prompt),
        temperature=LATAMGPT_CHAT_TEMPERATURE,
        max_tokens=LATAMGPT_MAX_TOKENS,
    )
    content = response.choices[0].text
    if not content:
        raise RuntimeError("LatamGPT completion returned an empty response.")
    return content


def query_latamgpt_with_meta(prompt: str) -> tuple[Optional[str], str, str]:
    """
    Query LatamGPT trying Option A (chat) then Option B (raw completion).

    Returns ``(cleaned_text_or_none, method_used, raw_text)``.
    """
    api_key = _get_env("HF_TOKEN")
    if not api_key:
        return None, "unavailable", ""

    from openai import OpenAI

    model = os.environ.get("LATAMGPT_MODEL", DEFAULT_LATAMGPT_MODEL)
    client = OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT,
    )

    raw_chat = _latamgpt_chat_call(client, model, prompt)
    cleaned_chat = clean_latamgpt_response(raw_chat.strip())
    if not is_latamgpt_response_broken(raw_chat, cleaned_chat):
        return cleaned_chat, "chat", raw_chat

    try:
        raw_completion = _latamgpt_completion_call(client, model, prompt)
        cleaned_completion = clean_latamgpt_response(raw_completion.strip())
        if not is_latamgpt_response_broken(raw_completion, cleaned_completion):
            return cleaned_completion, "completion", raw_completion
    except Exception:
        pass

    # Option C: provider still broken — return best-effort chat cleanup for logging/judge.
    return cleaned_chat or None, "broken", raw_chat


def get_query_error(model_name: str) -> Optional[str]:
    """Return the last error message recorded for a model query."""
    return _last_query_errors.get(model_name)


def _record_query_error(model_name: str, exc: Exception) -> None:
    _last_query_errors[model_name] = f"{type(exc).__name__}: {exc}"


def _retry_call(fn: Callable[[], T], model_name: str | None = None) -> Optional[T]:
    """Run ``fn`` up to two times, returning None if both attempts fail."""
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = fn()
            if model_name:
                _last_query_errors.pop(model_name, None)
            return result
        except Exception as exc:
            last_exc = exc
            if model_name:
                _record_query_error(model_name, exc)
            if attempt == 0:
                time.sleep(1.0)
                continue
            return None
    return None


def query_latamgpt(prompt: str) -> Optional[str]:
    """
    Query LatamGPT via the Hugging Face Inference Router (OpenAI-compatible API).

    Uses chat completions first (Option A), then raw completions (Option B) if
    the response looks broken. Requires ``HF_TOKEN`` in the environment.
    """
    if _latamgpt_unavailable_note:
        return None

    if not _get_env("HF_TOKEN"):
        _set_latamgpt_unavailable("LatamGPT no disponible: falta HF_TOKEN en .env")
        return None

    model = os.environ.get("LATAMGPT_MODEL", DEFAULT_LATAMGPT_MODEL)

    for attempt in range(2):
        try:
            result, method, _raw = query_latamgpt_with_meta(prompt)
            if result:
                _last_query_errors.pop("LatamGPT", None)
                if method == "broken":
                    _record_query_error(
                        "LatamGPT",
                        RuntimeError(
                            "LatamGPT respondio con texto degradado en HF router "
                            "(featherless-ai puede no estar bien servido)."
                        ),
                    )
                return result
            raise RuntimeError("LatamGPT returned an empty response.")
        except Exception as exc:
            _record_query_error("LatamGPT", exc)
            if is_latamgpt_unavailable_error(exc):
                _set_latamgpt_unavailable(LATAMGPT_UNAVAILABLE_DEFAULT_NOTE)
                return None
            if attempt == 0:
                time.sleep(1.0)
                continue
            return None

    return None


def query_gpt(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Query OpenAI GPT-4o mini.

    Requires ``OPENAI_API_KEY`` in the environment.
    """
    api_key = _get_env("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", DEFAULT_GPT_MODEL)
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    def _call() -> str:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=DEFAULT_TIMEOUT)
        response = client.chat.completions.create(
            model=model,
            messages=_chat_messages(prompt, system_prompt=system),
            max_tokens=256,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("GPT returned an empty response.")
        return content.strip()

    return _retry_call(_call, model_name="GPT")


def query_claude(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Query Anthropic Claude Haiku (or the model configured in ``ANTHROPIC_MODEL``).

    Requires ``ANTHROPIC_API_KEY`` in the environment.
    """
    api_key = _get_env("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    def _call() -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT)
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        if not content:
            raise RuntimeError("Claude returned an empty response.")
        return content.strip()

    return _retry_call(_call, model_name="Claude")


QUERY_FUNCTIONS = {
    "LatamGPT": query_latamgpt,
    "GPT": query_gpt,
    "Claude": query_claude,
}
