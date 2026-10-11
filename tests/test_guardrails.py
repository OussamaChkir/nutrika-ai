from app.guardrails import allergen_matches, check_explanation
from app.schemas import Highlight, LLMExplanation

from .conftest import good_explanation, make_request


def test_clean_explanation_passes():
    assert check_explanation(make_request(), good_explanation()) == []


def test_wrong_score_is_caught():
    exp = good_explanation()
    exp.summary = "This spread scores 80/100, which is great."
    violations = check_explanation(make_request(), exp)
    assert any("80/100" in v for v in violations)


def test_matching_score_is_allowed():
    exp = good_explanation()
    exp.summary = "It gets 32 / 100 overall."
    assert check_explanation(make_request(), exp) == []


def test_medical_claim_is_caught():
    exp = good_explanation()
    exp.highlights.append(Highlight(topic="Health", detail="This cures diabetes.", tone="positive"))
    assert any("medical" in v for v in check_explanation(make_request(), exp))


def test_allergy_safe_claim_is_caught():
    exp = good_explanation()
    exp.summary = "This product is allergen-free and safe for people with allergies."
    assert any("allergen" in v for v in check_explanation(make_request(), exp))


def test_french_medical_claim_is_caught():
    exp = good_explanation()
    exp.summary = "Ce produit guérit les maux d'estomac."
    assert any("medical" in v for v in check_explanation(make_request(), exp))


def test_length_limits():
    exp = LLMExplanation(summary="x" * 401, highlights=[])
    violations = check_explanation(make_request(), exp)
    assert any("summary" in v for v in violations)
    assert any("highlights" in v for v in violations)


def test_allergen_matches_substring_case_insensitive():
    assert allergen_matches(["Milk"], ["milk", "nuts"]) == ["milk"]
    assert allergen_matches(["nut"], ["may contain tree nuts"]) == ["may contain tree nuts"]
    assert allergen_matches([], ["milk"]) == []
    assert allergen_matches([" "], ["milk"]) == []
