"""Evaluation harness for the /explain pipeline.

    python -m evals.run_evals            # offline: scripted stub, validates the harness
    python -m evals.run_evals --live     # real Gemini (needs GEMINI_API_KEY)

Metrics (written to evals/results.md):
  llm_success_rate    answered by the LLM and accepted by the guardrails (no fallback)
  first_try_rate      accepted on the first attempt, without a guardrail retry
  alert_correct_rate  allergen alert matches the expectation, computed in code
  injection_resisted  the prompt-injection case did not leak "100/100" or a medical claim
  latency p50 / p95   end to end, milliseconds

Grow golden.jsonl towards 50+ real products from your own scan history before quoting
numbers in a README or CV. The 6 cases here are synthetic placeholders.
"""

import argparse
import json
import statistics
from pathlib import Path

from app.config import get_settings
from app.guardrails import MEDICAL_CLAIMS
from app.llm import GeminiClient, NullLLM
from app.schemas import ExplainRequest, Highlight, LLMExplanation
from app.service import explain

HERE = Path(__file__).parent


class StubLLM:
    """Offline stand-in: a faithful explanation built from the facts."""

    model = "stub"

    def generate(self, *, system: str, prompt: str) -> LLMExplanation:
        facts = json.loads(prompt.split("\n")[1])
        return LLMExplanation(
            summary=f"{facts['name']} scores {facts['score']}/100.",
            highlights=[Highlight(topic="Overview", detail=f"Grade {facts['grade']}.", tone="neutral")],
        )


def percentile(values: list[int], pct: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="call the real Gemini API")
    args = parser.parse_args()

    settings = get_settings()
    if args.live:
        if not settings.gemini_api_key:
            raise SystemExit("--live needs GEMINI_API_KEY")
        llm = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
    else:
        llm = StubLLM()

    cases = [json.loads(line) for line in (HERE / "golden.jsonl").read_text().splitlines() if line.strip()]
    rows, latencies = [], []
    for case in cases:
        req = ExplainRequest.model_validate(case["request"])
        res = explain(req, llm)
        text = " ".join([res.summary, res.swap_tip or "", *[h.detail for h in res.highlights]])
        alert = sorted(res.allergen_alert.matched) if res.allergen_alert else []
        latencies.append(res.latency_ms)
        rows.append({
            "id": case["id"],
            "llm_ok": res.source == "llm",
            "first_try": res.source == "llm" and res.attempts == 1,
            "alert_ok": alert == sorted(case["expect_alert"]),
            "injection_ok": "injection" not in case["id"]
            or (f"{req.product.score}/100" in text and not MEDICAL_CLAIMS.search(text)),
        })

    n = len(rows)
    rate = lambda key: f"{100 * sum(r[key] for r in rows) / n:.0f}%"  # noqa: E731
    mode = f"live ({llm.model})" if args.live else "offline stub"
    lines = [
        f"# Eval results: {mode}",
        "",
        f"Cases: {n} · prompt: see app/prompts.py",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| llm_success_rate | {rate('llm_ok')} |",
        f"| first_try_rate | {rate('first_try')} |",
        f"| alert_correct_rate | {rate('alert_ok')} |",
        f"| injection_resisted | {rate('injection_ok')} |",
        f"| latency p50 (ms) | {int(statistics.median(latencies))} |",
        f"| latency p95 (ms) | {percentile(latencies, 0.95)} |",
        "",
        "| Case | LLM ok | First try | Alert ok | Injection ok |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {r['id']} | {r['llm_ok']} | {r['first_try']} | {r['alert_ok']} | {r['injection_ok']} |" for r in rows],
    ]
    out = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
