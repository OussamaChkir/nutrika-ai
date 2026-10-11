from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import GeminiClient, LLMError
from app.schemas import Highlight, LLMExplanation


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def client_with(response, **kwargs) -> tuple[GeminiClient, FakeModels]:
    client = GeminiClient("fake-key", "some-model", **kwargs)
    models = FakeModels(response)
    client._client = SimpleNamespace(models=models)
    return client, models


def truncated_response():
    """What Gemini returns when thinking tokens eat the output budget: 200 OK, cut-off JSON."""
    return SimpleNamespace(
        parsed=None,
        text='{"summary": "Ce produit obtient 32/100 car',
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        usage_metadata=SimpleNamespace(candidates_token_count=40, thoughts_token_count=760),
    )


def test_truncated_output_error_explains_why():
    client, _ = client_with(truncated_response())
    with pytest.raises(LLMError) as err:
        client.generate(system="s", prompt="p")
    message = str(err.value)
    assert "MAX_TOKENS" in message
    assert "thinking_tokens=760" in message


def test_valid_output_is_returned():
    good = LLMExplanation(
        summary="ok", highlights=[Highlight(topic="t", detail="d", tone="neutral")]
    )
    client, _ = client_with(SimpleNamespace(parsed=good))
    assert client.generate(system="s", prompt="p") == good


def test_provider_exception_becomes_llm_error():
    client, _ = client_with(RuntimeError("504 DEADLINE_EXCEEDED"))
    with pytest.raises(LLMError, match="DEADLINE_EXCEEDED"):
        client.generate(system="s", prompt="p")


def test_generation_config_carries_limits_and_thinking():
    client, models = client_with(
        SimpleNamespace(parsed=LLMExplanation(summary="x", highlights=[])),
        max_output_tokens=3000,
        thinking_level="low",
    )
    client.generate(system="sys", prompt="p")
    config = models.calls[0]["config"]
    assert config.max_output_tokens == 3000
    assert config.thinking_config.thinking_level.name == "LOW"
    assert config.system_instruction == "sys"
    assert config.automatic_function_calling.disable is True


def test_thinking_budget_zero_is_passed_through():
    client, models = client_with(
        SimpleNamespace(parsed=LLMExplanation(summary="x", highlights=[])), thinking_budget=0
    )
    client.generate(system="s", prompt="p")
    assert models.calls[0]["config"].thinking_config.thinking_budget == 0


def test_no_thinking_config_by_default():
    client, models = client_with(SimpleNamespace(parsed=LLMExplanation(summary="x", highlights=[])))
    client.generate(system="s", prompt="p")
    assert models.calls[0]["config"].thinking_config is None


def test_blank_thinking_settings_in_env_are_treated_as_unset():
    s = Settings(gemini_thinking_level="", gemini_thinking_budget=" ", _env_file=None)
    assert s.gemini_thinking_level is None and s.gemini_thinking_budget is None
    assert Settings(gemini_thinking_budget="0", _env_file=None).gemini_thinking_budget == 0
