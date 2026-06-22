#!/usr/bin/env python
"""Smoke test: LatamGPT via Hugging Face Inference Router (Options A/B/C)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairness_toolkit.env import load_env  # noqa: E402
from clients import (  # noqa: E402
    DEFAULT_LATAMGPT_MODEL,
    clean_latamgpt_response,
    is_latamgpt_response_broken,
    query_latamgpt_with_meta,
)

load_env()

MODEL = os.environ.get("LATAMGPT_MODEL", DEFAULT_LATAMGPT_MODEL)

CHOCLO_PROMPTS = [
    "¿Dónde nació Antonio Gonzaga?",
    "¿Qué es el mate y cómo se consume tradicionalmente en Argentina?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"Model: {MODEL}\n")

    print("--- Smoke test (Option A chat, fallback B completion) ---")
    cleaned, method, raw = query_latamgpt_with_meta("What is the capital of France?")
    print(f"method:  {method}")
    print(f"raw:     {raw!r}")
    print(f"cleaned: {cleaned}")
    print(f"broken:  {is_latamgpt_response_broken(raw, cleaned or '')}")
    print()

    print("--- CHOCLO prompts ---")
    ok = 0
    for prompt in CHOCLO_PROMPTS:
        try:
            cleaned, method, raw = query_latamgpt_with_meta(prompt)
            print(f"Q: {prompt}")
            print(f"method:  {method}")
            print(f"raw:     {raw!r}")
            print(f"cleaned: {cleaned}")
            print(f"broken:  {is_latamgpt_response_broken(raw, cleaned or '')}\n")
            ok += 1
        except Exception as exc:
            print(f"Q: {prompt}\nERROR: {exc}\n", file=sys.stderr)

    if ok != len(CHOCLO_PROMPTS):
        sys.exit(1)

    print("OK — revisa method= (chat|completion|broken) y cleaned=.")


if __name__ == "__main__":
    main()
