from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Get a free key at https://aistudio.google.com. Model names and free-tier
    # limits change, so check AI Studio and override GEMINI_MODEL if needed.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_s: float = 20.0

    # Shared secret between Nutrika (server side) and this service.
    # When unset, auth is disabled (local dev only).
    service_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
