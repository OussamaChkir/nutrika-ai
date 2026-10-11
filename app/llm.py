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


def describe_bad_response(resp) -> str:
    """Why did Gemini answer 200 OK but give us nothing usable? Put the evidence in the log."""
    finish = None
    if getattr(resp, "candidates", None):
        finish = getattr(resp.candidates[0], "finish_reason", None)
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None) or ""
    return (
        f"finish_reason={finish} "
        f"output_tokens={getattr(usage, 'candidates_token_count', None)} "
        f"thinking_tokens={getattr(usage, 'thoughts_token_count', None)} "
        f"text_chars={len(text)} text_tail={text[-60:]!r}"
    )


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_output_tokens: int = 4096,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        # Imported here so tests and the no-key fallback mode do not need the SDK.
        from google import genai
        from google.genai import types

        self.model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        thinking = None
        if thinking_level is not None or thinking_budget is not None:
            kwargs = {}
            if thinking_level is not None:
                kwargs["thinking_level"] = thinking_level.upper()
            if thinking_budget is not None:
                kwargs["thinking_budget"] = thinking_budget
            thinking = types.ThinkingConfig(**kwargs)
        self._config_kwargs = dict(
            response_mime_type="application/json",
            response_schema=LLMExplanation,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
            # We pass no tools, so the SDK's automatic function calling is pure noise.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._types = types

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system, **self._config_kwargs
                ),
            )
        except Exception as exc:  # SDK raises several exception types
            raise LLMError(f"gemini call failed: {exc}") from exc

        parsed = resp.parsed
        if not isinstance(parsed, LLMExplanation):
            raise LLMError(
                "gemini returned output that does not match the schema: "
                + describe_bad_response(resp)
            )
        return parsed


class NullLLM:
    """Used when no API key is configured: every call fails, so the service
    degrades to the deterministic fallback explanation."""

    model = None

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        raise LLMError("no GEMINI_API_KEY configured")
