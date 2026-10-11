"""Deterministic checks on LLM output.

The LLM can phrase things, but it must not contradict facts Nutrika computed in code,
make medical claims, or imply a product is allergy-safe. Anything caught here is fed
back to the model once; if it still fails, the service returns a templated fallback.
"""

import re

from .schemas import ExplainRequest, LLMExplanation

MAX_SUMMARY = 400
MAX_TOPIC = 60
MAX_DETAIL = 240
MAX_TIP = 240
MAX_HIGHLIGHTS = 5

# Short, deliberately conservative lists (en / fr / ar). Extend as the eval set grows.
MEDICAL_CLAIMS = re.compile(
    r"\b(cures?|treats?|heals?|diagnos\w*|prevents? (?:cancer|diabetes|disease)|"
    r"gu[ée]rit|soigne|diagnostic|traite la maladie)\b|يعالج|يشفي|يشخص",
    re.IGNORECASE,
)
ALLERGY_SAFE_CLAIMS = re.compile(
    r"allergen[- ]?free|safe for (?:people|those|anyone) with|safe for allerg|"
    r"sans allerg[èe]ne|sans danger pour les allergiques|خالي من مسببات الحساسية",
    re.IGNORECASE,
)
SCORE_OUT_OF_100 = re.compile(r"\b(\d{1,3})\s*/\s*100\b")


def allergen_matches(user_allergies: list[str], product_allergens: list[str]) -> list[str]:
    """Same matching rule as Nutrika's product page: case-insensitive substring."""
    matched = []
    for allergen in product_allergens:
        name = allergen.lower()
        if any(u.strip() and u.strip().lower() in name for u in user_allergies):
            matched.append(allergen)
    return matched


def _all_text(exp: LLMExplanation) -> str:
    parts = [exp.summary, exp.swap_tip or ""]
    for h in exp.highlights:
        parts += [h.topic, h.detail]
    return "\n".join(parts)


def check_explanation(req: ExplainRequest, exp: LLMExplanation) -> list[str]:
    """Return a list of human-readable violations (empty list = output accepted)."""
    violations: list[str] = []
    text = _all_text(exp)

    if not exp.summary.strip():
        violations.append("summary is empty")
    if len(exp.summary) > MAX_SUMMARY:
        violations.append(f"summary is longer than {MAX_SUMMARY} characters")
    if not 1 <= len(exp.highlights) <= MAX_HIGHLIGHTS:
        violations.append(f"provide between 1 and {MAX_HIGHLIGHTS} highlights")
    if any(len(h.topic) > MAX_TOPIC or len(h.detail) > MAX_DETAIL for h in exp.highlights):
        violations.append("a highlight topic or detail is too long")
    if exp.swap_tip and len(exp.swap_tip) > MAX_TIP:
        violations.append(f"swap_tip is longer than {MAX_TIP} characters")

    for found in SCORE_OUT_OF_100.findall(text):
        if int(found) != req.product.score:
            violations.append(
                f"you wrote a score of {found}/100 but the score is {req.product.score}/100"
            )
            break

    if MEDICAL_CLAIMS.search(text):
        violations.append("remove medical claims (diagnose, treat, cure, prevent)")
    if ALLERGY_SAFE_CLAIMS.search(text):
        violations.append("never say a product is allergen-free or safe for allergies")

    return violations
