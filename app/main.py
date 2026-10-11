import logging
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException

from .config import Settings, get_settings
from .llm import GeminiClient, LLMClient, NullLLM
from .schemas import ExplainRequest, ExplainResponse
from .service import explain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = FastAPI(title="Nutrika AI", version="0.1.0")

_llm_cache: dict[tuple, LLMClient] = {}


def get_llm(settings: Settings = Depends(get_settings)) -> LLMClient:
    if not settings.gemini_api_key:
        return NullLLM()
    key = (settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
    if key not in _llm_cache:
        _llm_cache[key] = GeminiClient(
            settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s
        )
    return _llm_cache[key]


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.service_api_key is None:
        return  # auth disabled (local dev)
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.service_api_key):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {"status": "ok", "llm_configured": bool(settings.gemini_api_key)}


@app.post("/v1/explain", response_model=ExplainResponse, dependencies=[Depends(require_api_key)])
def explain_endpoint(req: ExplainRequest, llm: LLMClient = Depends(get_llm)) -> ExplainResponse:
    # Plain `def`: FastAPI runs it in a worker thread, so the blocking SDK call
    # does not stall the event loop.
    return explain(req, llm)
