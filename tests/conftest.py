import pytest

from app.llm import LLMError
from app.schemas import ExplainRequest, Highlight, LLMExplanation


def make_request(**overrides) -> ExplainRequest:
    product = {
        "barcode": "3017620422003",
        "name": "Hazelnut spread",
        "brand": "Example Brand",
        "score": 32,
        "grade": "D",
        "positives": ["Organic certified"],
        "negatives": ["Very high sugar (56.3g/100g)", "High saturated fat (10.6g/100g)"],
        "allergens": ["milk", "nuts"],
        "nova_group": 4,
        "nutrition": {"sugars": 56.3, "saturated_fat": 10.6},
        "ingredients_text": "Sugar, palm oil, hazelnuts 13%, skimmed milk powder",
    }
    return ExplainRequest.model_validate(
        {"product": product, "user": {"allergies": ["milk"]}, "locale": "en", **overrides}
    )


def good_explanation() -> LLMExplanation:
    return LLMExplanation(
        summary="This spread scores 32/100 mainly because of its very high sugar content.",
        highlights=[
            Highlight(topic="Sugar", detail="56.3 g of sugar per 100 g is very high.", tone="negative"),
            Highlight(topic="Label", detail="It is organic certified.", tone="positive"),
        ],
        swap_tip="Look for a version with less added sugar.",
    )


class FakeLLM:
    """Plays back a scripted list of responses (LLMExplanation or LLMError)."""

    model = "fake-model"

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        self.calls.append({"system": system, "prompt": prompt})
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def req() -> ExplainRequest:
    return make_request()


__all__ = ["FakeLLM", "LLMError", "good_explanation", "make_request"]
