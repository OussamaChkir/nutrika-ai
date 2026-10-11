import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_llm

from .conftest import FakeLLM, good_explanation, make_request


@pytest.fixture
def client():
    app.dependency_overrides[get_llm] = lambda: FakeLLM(good_explanation())
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key=None, service_api_key=None, _env_file=None
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def payload():
    return make_request().model_dump(mode="json")


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "llm_configured": False}


def test_explain_ok(client):
    res = client.post("/v1/explain", json=payload())
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "llm"
    assert body["allergen_alert"]["matched"] == ["milk"]


@pytest.mark.parametrize(
    "patch",
    [
        {"product": {**payload()["product"], "barcode": "abc"}},
        {"product": {**payload()["product"], "score": 150}},
        {"locale": "de"},
    ],
)
def test_invalid_input_is_rejected(client, patch):
    res = client.post("/v1/explain", json={**payload(), **patch})
    assert res.status_code == 422


def test_api_key_is_enforced_when_configured(client):
    app.dependency_overrides[get_settings] = lambda: Settings(
        service_api_key="secret", _env_file=None
    )
    assert client.post("/v1/explain", json=payload()).status_code == 401
    assert client.post("/v1/explain", json=payload(), headers={"X-API-Key": "nope"}).status_code == 401
    ok = client.post("/v1/explain", json=payload(), headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
