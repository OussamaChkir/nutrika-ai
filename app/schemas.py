"""Request/response models.

`ProductFacts` mirrors what Nutrika already computes deterministically
(score, grade, positives, negatives, allergens). The LLM never produces these
values; it only explains them.
"""

from typing import Literal

from pydantic import BaseModel, Field

Locale = Literal["en", "fr", "ar"]
Grade = Literal["A", "B", "C", "D", "E"]
Tone = Literal["positive", "neutral", "negative"]


class Nutrition(BaseModel):
    """Per 100 g, same fields as the Nutrika Prisma `Product` model."""

    energy_kcal: float | None = None
    fat: float | None = None
    saturated_fat: float | None = None
    carbohydrates: float | None = None
    sugars: float | None = None
    fiber: float | None = None
    proteins: float | None = None
    salt: float | None = None


class ProductFacts(BaseModel):
    barcode: str = Field(pattern=r"^\d{6,14}$")
    name: str = Field(min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=120)
    score: int = Field(ge=0, le=100)
    grade: Grade
    positives: list[str] = Field(default_factory=list, max_length=20)
    negatives: list[str] = Field(default_factory=list, max_length=20)
    allergens: list[str] = Field(default_factory=list, max_length=30)
    dietary_tags: list[str] = Field(default_factory=list, max_length=20)
    nova_group: int | None = Field(default=None, ge=1, le=4)
    nutriscore_grade: str | None = Field(default=None, max_length=2)
    nutrition: Nutrition = Field(default_factory=Nutrition)
    # Free text from a public database / user contributions: treated as untrusted.
    ingredients_text: str | None = Field(default=None, max_length=4000)


class UserProfile(BaseModel):
    allergies: list[str] = Field(default_factory=list, max_length=30)


class ExplainRequest(BaseModel):
    product: ProductFacts
    user: UserProfile = Field(default_factory=UserProfile)
    locale: Locale = "en"


# --- What the LLM must return (kept constraint-free: Gemini's response schema
# --- support for min/max keywords varies, so limits are enforced in guardrails.py)


class Highlight(BaseModel):
    topic: str
    detail: str
    tone: Tone


class LLMExplanation(BaseModel):
    summary: str
    highlights: list[Highlight]
    swap_tip: str | None = None


# --- What the API returns


class AllergenAlert(BaseModel):
    """Computed in code from the user's profile, never by the LLM."""

    matched: list[str]


class ExplainResponse(BaseModel):
    summary: str
    highlights: list[Highlight]
    swap_tip: str | None = None
    allergen_alert: AllergenAlert | None = None
    locale: Locale
    source: Literal["llm", "fallback"]
    attempts: int
    prompt_version: str
    model: str | None = None
    latency_ms: int
