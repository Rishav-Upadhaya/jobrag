# JobRAG

[![CI](https://github.com/Rishav-Upadhaya/jobrag/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishav-Upadhaya/jobrag/actions/workflows/ci.yml)

Natural-language job search over 1,000 job postings (the LF Jobs dataset). A LangGraph pipeline classifies the question, extracts metadata filters, runs hybrid retrieval (dense vectors + Postgres full-text, fused with Reciprocal Rank Fusion), reranks, writes a grounded answer, and has an LLM judge check that answer against the retrieved context.

```
query ─▶ intent_classifier ─▶ reasoning_agent ─▶ retriever ─▶ synthesizer ─▶ judge ─▶ response
             │ off_topic / vague        │ filters      │                         │ fail
             ▼                          ▼              │                         └─▶ retry retriever
       polite rejection /        job_level, location,  │
       clarifying question       company, title, date  ▼
                                   SQL metadata pre-filter (jobs table)
                                     ├─ vector search:  BGE-M3 (1024-d) + pgvector HNSW, top 40
                                     └─ keyword search: tsvector + GIN, title/metadata weighted over body, top 40
                                   ─▶ RRF fusion (k=60) ─▶ max 2 chunks per job ─▶ Jina reranker ─▶ top 5
```

## Design decisions

- **Postgres for everything.** Job metadata, chunk text, 1024-d vectors (HNSW) and full-text vectors (GIN) live in one database. At ~1k jobs / 12k chunks this removes a separate vector service and lets metadata filters and both retrievers share one `job_id` whitelist.
- **Hybrid retrieval with RRF.** Dense search finds paraphrases ("ML" ≈ "machine learning"); full-text search catches exact titles, companies and locations that embeddings blur. Their scores are on unrelated scales (cosine vs `ts_rank_cd`), so they are merged by rank with Reciprocal Rank Fusion rather than by score.
- **Keyword terms are OR-ed.** `plainto_tsquery` ANDs every word, so "senior ML roles in New York" would require "roles". The query ORs terms and lets `ts_rank_cd` rank jobs matching more of them higher. Job titles carry weight A, other metadata B, description text C.
- **Explicit graph, not an agent loop.** Each LangGraph node has one job and returns only the state it changes, so routing (off-topic, vague, retry-on-judge-fail) is visible in code and in traces.
- **Local embeddings.** BGE-M3 runs on CPU via `sentence-transformers`; no embedding API cost or rate limit at ingestion time.
- **Plain SQL (psycopg2).** Retrieval queries are hand-written so index usage is explicit.

## Run it

Prerequisites: Docker, Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # set LLM_PROVIDER + model names + API keys, JINA_API_KEY
docker compose up -d db         # Postgres 16 + pgvector
uv sync
uv run uvicorn app.main:app --port 8000
```

The schema in `app/infrastructure/db/migrations/001_init.sql` is applied at startup and is idempotent, so existing databases pick up new columns and indexes on the next boot. `docker compose up -d` also starts the API, Adminer (`:8080`) and a small chat frontend (`:3000`).

`LLM_PROVIDER` is `gemini`, `openai` or `openrouter`. Use a pinned, non-reasoning model for the classifier and judge: they return short JSON under tight token caps, and reasoning models (including many behind `openrouter/free`) can spend the whole budget thinking and return nothing.

## Example

Load the dataset. `data/` is git-ignored: the LF Jobs spreadsheet is not redistributed here, and any `.xlsx`/`.csv` with the same columns works. Timing below is BGE-M3 on an Apple M-series CPU, where embedding dominates:

```bash
curl -X POST http://localhost:8000/api/load -F "file=@data/LF Jobs.xlsx" -F "overwrite=true"
```

```json
{"status": "ok", "jobs_loaded": 1000, "chunks_created": 12096, "dropped_rows": 0, "time_seconds": 623.8}
```

Query:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "remote python backend jobs"}'
```

Real response (Gemini 2.5 Flash for all LLM nodes; answer and chunk excerpts shortened):

```json
{
  "query": "remote python backend jobs",
  "intent": "valid",
  "answer": "Here are a few remote backend and Python-related roles that might be a good fit for you!\n\n**Principal Backend Software Engineer** at **Atlassian**\n· Location: Canada, Flexible / Remote, San Francisco, CA | Level: Senior Level\n…",
  "sources": [
    {"job_id": "LF0192", "job_title": "Senior Python Data Engineer", "company_name": "EPAM Systems", "job_location": "Nurota, Uzbekistan", "relevance_score": 0.61},
    {"job_id": "LF0233", "job_title": "Principal Backend Software Engineer", "company_name": "Atlassian", "job_location": "Canada, Flexible / Remote, San Francisco, CA", "relevance_score": 0.565},
    {"job_id": "LF0036", "job_title": "AI Researcher, 2025 Graduate U.S.", "company_name": "Atlassian", "job_location": "Flexible / Remote, San Francisco, CA", "relevance_score": 0.491},
    {"job_id": "LF0478", "job_title": "Staff Product Designer, Crypto", "company_name": "Robinhood", "job_location": "Bellevue, WA, Menlo Park, CA, New York, NY, Toronto, Canada", "relevance_score": 0.401}
  ],
  "judge": {"score": 0.9, "verdict": "pass", "relevance": 0.9, "quality": 0.9, "hallucination": false,
            "reasoning": "The answer accurately summarizes relevant remote Python and backend jobs from the context without any hallucinations."},
  "latency_ms": 19623
}
```

Other outcomes from the same run:

| Query | Result | Latency |
|---|---|---|
| `senior machine learning roles in New York` | filters `{job_level: Senior, job_title: Machine Learning Engineer, job_location: New York}` → Coinbase Staff ML Engineer; judge 1.0 | 8.4 s |
| `what is the weather today` | `intent: off_topic`, rejection message, no retrieval | 1.4 s |

`relevance_score` is the Jina rerank score when the reranker returned the chunk, otherwise the RRF score (≈0.016–0.033).

## Tests

```bash
uv run pytest                                   # unit tests, no services needed
docker run -d --rm -p 5433:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://postgres:test@localhost:5433/postgres uv run pytest   # + real-Postgres tests
```

- `tests/test_hybrid_search.py`: RRF ordering and de-duplication, per-job chunk cap, reranker success and failure fallback, keyword-only hits surviving fusion, metadata whitelist passed to both retrievers, keyword rank never leaking into `relevance_score`.
- `tests/test_keyword_search_db.py`: runs the real migration against pgvector and checks title-over-body ranking, OR semantics, metadata search, whitelist, stop-word-only queries, and migration idempotency.

## Known limitations

- **Latency is LLM-bound:** 8–20 s per query with four sequential LLM calls (classifier, reasoning, synthesis, judge) on free-tier Gemini, and far longer under 429 rate limiting (a single-word query took 220 s during testing). Merging the classifier and reasoning steps into one call is the obvious first cut.
- **Nodes fail open.** If the classifier, reasoning agent or judge errors or returns unparseable output, the pipeline continues with defaults (`intent=valid`, no filters, `verdict=pass`). That keeps answers flowing but means a broken judge looks like a passing one; the response says "Judge unavailable" when this happens.
- **Low-relevance backfill.** When fewer than five chunks clear the rerank threshold, the remainder is backfilled from fused order, which is how a product-design role appears in the Python example above.
- **No evaluation set yet.** Retrieval quality is checked by hand; recall@k and nDCG on a labelled query set are the next step.
- Single-node Postgres, no auth or rate limiting on the API.
