# Nutrika AI

LLM service behind [Nutrika](https://nutrikafood.com), a food barcode scanner. It turns the
score Nutrika already computed into a short, personalised explanation in English, French or
Arabic, with guardrails so the model can explain the facts but never change them.

```
Nutrika (Next.js)                      nutrika-ai (this repo, FastAPI)
┌──────────────────┐   POST /v1/explain   ┌────────────────────────────────────────┐
│ product page     │ ───────────────────► │ validate (Pydantic)                    │
│ calculateScore() │   facts + user       │ → allergen match (code, not LLM)       │
│ user allergies   │                      │ → Gemini, structured JSON output       │
│ Prisma cache     │ ◄─────────────────── │ → guardrails → 1 retry with feedback   │
└──────────────────┘   explanation        │ → templated fallback if still failing  │
                                          └────────────────────────────────────────┘
```

## Design decisions

- **The LLM explains, code decides.** Score, grade and allergen alerts come from deterministic
  code. The model only writes the text around them.
- **Structured output.** Gemini is called with a Pydantic response schema, so the reply is
  parsed into typed data rather than scraped from text.
- **Guardrails, then one retry, then a fallback.** `app/guardrails.py` rejects wrong scores,
  medical claims and "allergen-free" claims. Violations are fed back to the model once. If it
  still fails, or the provider is down, the API returns a templated explanation
  (`source: "fallback"`), so the product page never breaks.
- **Prompt-injection hygiene.** Ingredient text comes from a public database and user
  contributions, so it is delimited, stripped of angle brackets and declared as untrusted
  data in the system prompt.
- **Traceable.** Every response carries `prompt_version`, `model`, `attempts` and
  `latency_ms`; each request logs one structured line.
- **Provider-agnostic.** `LLMClient` is a small protocol; tests use a scripted fake, so the
  suite runs offline with no API key.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add GEMINI_API_KEY from https://aistudio.google.com
uvicorn app.main:app --reload
```

Without a key the service still runs and returns the fallback explanation.

```bash
curl -s localhost:8000/v1/explain -H 'content-type: application/json' -d '{
  "product": {"barcode": "3017620422003", "name": "Hazelnut spread", "score": 32, "grade": "D",
              "negatives": ["Very high sugar (56.3g/100g)"], "allergens": ["milk"]},
  "user": {"allergies": ["milk"]},
  "locale": "fr"
}'
```

Docker: `docker build -t nutrika-ai . && docker run -p 8000:8000 --env-file .env nutrika-ai`

## Tests and evals

```bash
python -m pytest -q                 # unit + API tests, no network
python -m evals.run_evals           # offline harness check (stub model)
python -m evals.run_evals --live    # real Gemini, writes evals/results.md
```

`evals/golden.jsonl` holds 6 synthetic cases (allergy match, no match, French, Arabic, a
prompt-injection attempt, sparse data). Grow it to 50+ real products from your own scan
history before publishing numbers.

## Roadmap

1. RAG over WHO / EFSA sources with citations (`/v1/chat`)
2. Tool-calling agent over Open Food Facts (alternatives, allergen checks)
3. Tracing with Langfuse, rate limiting, response cache keyed by barcode + locale + prompt version
4. Label photo to draft product (Gemini vision) into Nutrika's contribution review flow
