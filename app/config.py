from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Get a free key at https://aistudio.google.com. Model names and free-tier
    # limits change, so check AI Studio and override GEMINI_MODEL if needed.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_s: float = 30.0

    # "Thinking" models spend output tokens (and seconds) reasoning before they answer.
    # The thinking tokens count against the output limit, so keep plenty of headroom and,
    # for this simple task, ask for as little thinking as the model allows.
    llm_max_output_tokens: int = 4096
    gemini_thinking_level: str | None = None   # Gemini 3 family: minimal | low | medium | high
    gemini_thinking_budget: int | None = None  # Gemini 2.5 family: 0 turns thinking off

    # Shared secret between Nutrika (server side) and this service.
    # When unset or blank, auth is disabled (local dev only).
    service_api_key: str | None = None

    @field_validator("gemini_thinking_level", "gemini_thinking_budget", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
