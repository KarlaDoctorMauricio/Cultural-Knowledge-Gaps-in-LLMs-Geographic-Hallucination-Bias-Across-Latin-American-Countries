"""LLM response generators for CHOCLO evaluation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .choclo import normalize_country
from .env import load_env, require_env

load_env()


class LLMClient(ABC):
    """Minimal interface for generating answers to CHOCLO questions."""

    name: str
    api_key_env: str

    @abstractmethod
    def generate(
        self,
        question: str,
        *,
        country: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    Deterministic mock client for offline runs and tests.

    Simulates different model behavior without external APIs.
    """

    api_key_env = ""

    LATAM_COUNTRIES = {
        "argentina",
        "bolivia",
        "chile",
        "colombia",
        "costa rica",
        "cuba",
        "ecuador",
        "el salvador",
        "guatemala",
        "honduras",
        "mexico",
        "nicaragua",
        "panama",
        "paraguay",
        "peru",
        "republica dominicana",
        "uruguay",
        "venezuela",
    }

    def __init__(self, name: str):
        self.name = name

    def generate(
        self,
        question: str,
        *,
        country: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        answer = reference_answer or "No tengo informacion suficiente."
        country = normalize_country(country or "")

        if self.name == "LatamGPT":
            if country in self.LATAM_COUNTRIES:
                return answer
            return answer[: max(len(answer) // 2, 20)]

        if self.name == "GPT":
            if difficulty and difficulty.strip().upper().startswith("DIF"):
                return answer[: max(len(answer) // 3, 15)]
            return answer[: max(int(len(answer) * 0.75), 20)]

        if self.name == "Claude":
            if country in self.LATAM_COUNTRIES:
                return answer[: max(int(len(answer) * 0.85), 20)]
            return answer[: max(int(len(answer) * 0.65), 20)]

        return answer


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible chat completion client (GPT, LatamGPT, etc.)."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        base_url_env: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt or (
            "Responde en espanol de forma breve y precisa a preguntas de "
            "conocimiento cultural latinoamericano."
        )

    def _resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return require_env(self.api_key_env)

    def _resolve_base_url(self) -> Optional[str]:
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return require_env(self.base_url_env)
        return None

    def generate(
        self,
        question: str,
        *,
        country: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        api_key = self._resolve_api_key()

        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        base_url = self._resolve_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        user_content = question
        if country or category:
            user_content = (
                f"Pais: {country or 'N/A'}. "
                f"Categoria: {category or 'N/A'}. "
                f"Pregunta: {question}"
            )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


class AnthropicClient(LLMClient):
    """Anthropic Claude chat client."""

    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        name: str = "Claude",
        model: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        question: str,
        *,
        country: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        api_key = self.api_key or require_env(self.api_key_env)

        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        user_content = question
        if country or category:
            user_content = (
                f"Pais: {country or 'N/A'}. "
                f"Categoria: {category or 'N/A'}. "
                f"Pregunta: {question}"
            )

        response = client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0.2,
            system=(
                "Responde en espanol de forma breve y precisa a preguntas de "
                "conocimiento cultural latinoamericano."
            ),
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()


DEFAULT_MODEL_NAMES = ("LatamGPT", "GPT", "Claude")


def get_model_clients(use_mock: Optional[bool] = None) -> Dict[str, LLMClient]:
    """
    Build the three evaluation clients.

    Uses mock clients only when ``use_mock=True`` or ``USE_MOCK_LLM=1``.
    Otherwise validates that all required ``.env`` variables are present.
    """
    load_env()

    if use_mock is None:
        use_mock = os.environ.get("USE_MOCK_LLM", "").lower() in {"1", "true", "yes"}

    if use_mock:
        return {name: MockLLMClient(name) for name in DEFAULT_MODEL_NAMES}

    require_env("OPENAI_API_KEY")
    require_env("ANTHROPIC_API_KEY")

    return {
        "GPT": OpenAICompatibleClient(
            name="GPT",
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key_env="OPENAI_API_KEY",
        ),
        "Claude": AnthropicClient(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        ),
    }
