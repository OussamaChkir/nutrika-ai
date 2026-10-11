"""LLM access behind a small interface so the rest of the code (and the tests)
never depend on a specific provider."""

import logging
from typing import Protocol

from .schemas import LLMExplanation

log = logging.getLogger("nutrika_ai.llm")


class LLMError(Exception):
    """Any provider failure: network, quota, timeout, unparseable output."""


class LLMClient(Protocol):
    model: str | None

    def generate(self, *, system: str, prompt: str) -> LLMExplanation: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 20.0) -> None:
        # Imported here so tests and the no-key fallback mode do not need the SDK.
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=LLMExplanation,
                    temperature=0.2,
                    max_output_tokens=800,
                ),
            )
        except Exception as exc:  # SDK raises several exception types
            raise LLMError(f"gemini call failed: {exc}") from exc

        parsed = resp.parsed
        if not isinstance(parsed, LLMExplanation):
            raise LLMError("gemini returned output that does not match the schema")
        return parsed


class NullLLM:
    """Used when no API key is configured: every call fails, so the service
    degrades to the deterministic fallback explanation."""

    model = None

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        raise LLMError("no GEMINI_API_KEY configured")
