import logging
import time

from .guardrails import allergen_matches, check_explanation
from .llm import LLMClient, LLMError
from .prompts import PROMPT_VERSION, build_system_prompt, build_user_prompt
from .schemas import (
    AllergenAlert,
    ExplainRequest,
    ExplainResponse,
    Highlight,
    LLMExplanation,
)

log = logging.getLogger("nutrika_ai.service")

MAX_ATTEMPTS = 2

# Used only when the LLM is unavailable or keeps failing the guardrails.
_FALLBACK_SUMMARY = {
    "en": "{name} scores {score}/100 (grade {grade}).",
    "fr": "{name} obtient {score}/100 (note {grade}).",
    "ar": "حصل {name} على {score}/100 (التصنيف {grade}).",
}


def fallback_explanation(req: ExplainRequest) -> LLMExplanation:
    p = req.product
    highlights = [Highlight(topic="", detail=n, tone="negative") for n in p.negatives[:3]]
    highlights += [Highlight(topic="", detail=n, tone="positive") for n in p.positives[:2]]
    if not highlights:
        highlights = [Highlight(topic="", detail="-", tone="neutral")]
    return LLMExplanation(
        summary=_FALLBACK_SUMMARY[req.locale].format(
            name=p.name, score=p.score, grade=p.grade
        ),
        highlights=highlights,
    )


def explain(req: ExplainRequest, llm: LLMClient) -> ExplainResponse:
    started = time.perf_counter()
    matched = allergen_matches(req.user.allergies, req.product.allergens)
    system = build_system_prompt(req.locale)

    result: LLMExplanation | None = None
    violations: list[str] = []
    attempts = 0

    while attempts < MAX_ATTEMPTS and result is None:
        attempts += 1
        try:
            candidate = llm.generate(
                system=system, prompt=build_user_prompt(req, violations or None)
            )
        except LLMError as exc:
            log.warning("llm_error barcode=%s attempt=%d error=%s", req.product.barcode, attempts, exc)
            continue
        violations = check_explanation(req, candidate)
        if violations:
            log.info("guardrail_violation barcode=%s attempt=%d violations=%s",
                     req.product.barcode, attempts, violations)
            continue
        result = candidate

    source = "llm" if result else "fallback"
    exp = result or fallback_explanation(req)
    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "explain barcode=%s locale=%s source=%s attempts=%d latency_ms=%d",
        req.product.barcode, req.locale, source, attempts, latency_ms,
    )

    return ExplainResponse(
        summary=exp.summary,
        highlights=exp.highlights,
        swap_tip=exp.swap_tip,
        allergen_alert=AllergenAlert(matched=matched) if matched else None,
        locale=req.locale,
        source=source,
        attempts=attempts,
        prompt_version=PROMPT_VERSION,
        model=llm.model if source == "llm" else None,
        latency_ms=latency_ms,
    )
