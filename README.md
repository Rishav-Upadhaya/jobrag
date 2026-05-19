# RAG Job Search

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for job search over the LF Jobs dataset. The README below documents the high-level architecture and engineering decisions, setup and installation steps, example usage, assumptions, and future work.

---

## 1. High-level architecture & engineering decisions

- **API layer — FastAPI (async)**: chosen for async request handling, Pydantic validation, and lightweight routing. FastAPI keeps endpoints fast and simple for concurrent query loads.
- **Orchestration — LangGraph state graph**: nodes model intent → retrieve → synthesise → judge flows. This explicit state graph makes retry logic, conditional edges, and testing of individual nodes straightforward.
- **Vector store — PostgreSQL + pgvector**: co-locates metadata and vectors; reliable, simple to run (docker-compose) and suitable for a dataset ~1k rows. HNSW indexes provide efficient cosine search without a separate vector DB service.
- **Embeddings — pluggable provider abstraction**: `BaseEmbedder` allows switching providers (Gemini / Cohere / HF) while enforcing batch size and vector dimension constraints. Keeps vendor lock-in minimal.
- **Retrieval — vector search (Jina reranker used for reranking)**: combines semantic matches and optional fusion steps. Keyword FTS is documented but currently not active in the codebase; vector search with metadata filters is the active path.
- **Reranker — cross-encoder (CPU-friendly model)**: reorders fused candidates for precision. CPU-friendly models (e.g., MiniLM cross-encoder) balance cost and quality.
- **Synthesis & judging — single LLM factory**: all LLM calls unify through `app/infrastructure/llm/gemini_client.py` to centralize provider configuration, rate limiting, and prompts per node (classifier, synthesizer, judge).
- **Schema separation — `jobs` vs `job_chunks`**: structured filters live in `jobs`, text+embeddings live in `job_chunks`. This keeps metadata filtering efficient and retrieval focused.
- **SQL via psycopg2 (no ORM)**: direct SQL gives predictable performance and explicit control over indexes and queries for hybrid search.

Rationale: choices prioritize reproducibility, traceability (explicit graph state), and predictable costs (postgre+CPU models) while keeping the architecture simple to run locally and on Docker.

---

## 2. Setup and installation

Prerequisites:
- Docker & Docker Compose
- Python 3.11+

Quick start (recommended):

```bash
# 1. Start Postgres with pgvector
docker compose up -d --build

# 2. Set up environment and install dependencies (preferred: `uv` CLI)
# Preferred: use `uv` to initialize and manage the virtual environment
uv init
uv venv .venv
uv add -r requirements.txt

# Alternative: using Python's built-in venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run DB migrations
psql $DATABASE_URL -f app/infrastructure/db/migrations/001_init.sql

# 4. Copy env and fill keys
cp .env.example .env
# edit .env to set DATABASE_URL and API keys

# 5. Start the API
uvicorn app.main:app --reload --port 8000
```

Notes:
- If you use Docker-only development, you may prefer running the API inside the project container defined in `docker-compose.yml`.
- Ensure `EMBEDDING_DIMENSION` in `.env` matches the provider and DB VECTOR dimension (default 768).

---

## 3. Example usage — requests & expected responses

1) Load dataset (ingestion)

Request:

```bash
curl -X POST http://localhost:8000/api/v1/load \
  -F "file=@data/LF Jobs.xlsx" \
  -F "overwrite=true"
```

Expected response (success):

```json
{
  "status": "ok",
  "jobs_loaded": 1000,
  "chunks_created": 3500,
  "dropped_rows": 0,
  "time_seconds": 12.3
}
```

2) Query the system

Request:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"senior machine learning roles in New York", "top_k":5}'
```

Expected response (valid intent):

```json
{
  "query": "senior machine learning roles in New York",
  "answer": "[Senior ML Engineer] at Acme · New York · Senior\n- Matches because...\n[Senior Data Scientist] at BetaCorp · NYC · Senior\n- Matches because...",
  "sources": [
    {"job_id":"LF001","job_title":"Senior ML Engineer","company_name":"Acme","job_level":"Senior","job_location":"New York","relevance_score":0.92},
    {"job_id":"LF021","job_title":"Senior Data Scientist","company_name":"BetaCorp","job_level":"Senior","job_location":"New York","relevance_score":0.88}
  ],
  "judge": {"score":0.85,"verdict":"pass","reasoning":"Relevant and faithful to context","relevance":0.88,"quality":0.82,"hallucination":false},
  "intent": "valid",
  "latency_ms": 980
}
```

If the classifier marks the query `vague`, the API returns a 200 with `intent: "vague"` and `clarification_question` instead of `answer`.

If the query is `off_topic`, the API returns an informative rejection message with `intent: "off_topic"`.

---

## 4. Assumptions made during development

- Small dataset (≈1k jobs): pgvector + CPU reranker is cost-effective and performant.
- Embedding dimension 768 is standard for chosen models; code enforces this dimension when upserting vectors.
- No authentication required for API (out of scope).
- All LLM calls are routed via `app/infrastructure/llm/gemini_client.py` and configured via environment variables.
- Reranker will run on CPU (no GPU dependency assumed).

---

## 5. Drawbacks and future enhancements

Drawbacks:
- Single-node Postgres with pgvector may not scale beyond modest datasets; large-scale production should use a managed vector DB or sharded solution.
- Free tier Jina reranking limits throughput and latency for high-concurrency workloads.
- Current judge/synthesizer prompts are designed for fidelity but may still allow subtle hallucinations — automated evaluation depends on quality of retrieved chunks.

Future enhancements:
- Add Premium Tiers reranking and batch LLM synthesis for latency-sensitive deployments.
- Add authentication and rate-limiting for a public API.
- Add monitoring (Prometheus / Grafana) and structured tracing for observability of LLM calls and DB latencies.
- Improve ingestion to support incremental updates and resumable embedding jobs.
- Add a lightweight UI for interactive browsing of retrieved jobs and sources.

---

## Observability & Frontend — what the screenshots show and how to interpret them

LangSmith traces (LangGraph waterfall)
- What you see: a vertical waterfall of nodes (for example `intent_classifier`, `reasoning_agent`, `retriever`, `synthesizer`, `judge`) with per-node timing, provider tags (which LLM/model was called), and expandable inputs/outputs (prompts, retrieved chunks, responses).
- Key signals visible in the trace:
  - Node durations: how long each node ran (LLM nodes vs non-LLM nodes). Treat each LLM call as a baseline of ~1–2 seconds for budgeting and SLOs — traces showing >>30s are likely slow-provider outliers or debug runs and can be ignored for baseline planning.
  - LLM prompt / response payloads: useful for spotting prompt drift, token inflation, or unwanted instructions that may cause hallucination.
  - Retrieved results summary: number of chunks returned, ranks/scores from the hybrid search and reranker, and whether any filters removed matches.
  - Judge outputs: per-dimension scores (relevance, quality) and `hallucination_flag` which indicate whether the synthesizer invented facts not present in context.
  - Retry loops & routing: repeated passes from `judge` → `retriever` indicate the system retried due to a failed judge verdict.

- Common issues you can spot and how to act:
  - Long latency concentrated in a single LLM node: check provider choice, network, or batching; consider switching provider or enabling batching/caching.
  - Many retries from judge with low judge_score: improve retriever quality (better embeddings, chunking, reranker) or relax judge thresholds for retries.
  - Empty or irrelevant `retrieved_chunks`: check ingestion (indexing), filters applied, and chunking strategy; ensure vector indexes (HNSW) are healthy.
  - Hallucination_flag=true: tighten the synthesizer prompt to require citations, increase context coverage, or surface provenance to the user.

Frontend UI snapshot (chat + sources view)
- What you see: a conversation UI with a history/left pane, the main chat area that shows the assistant's answer (job list entries formatted like `[Job] at [Company] · [Location] · [Level]` with short explanations), inline clarifying prompts (if the classifier asks for more detail), and a small judge badge (e.g., `PASS · 93%`) with a control to view matched sources/excerpts.

- What to inspect in the UI and why it matters:
  - Clarification prompts present: indicates the intent classifier found the query too vague — the system is correctly asking for user input before retrieving.
  - Matches listed vs sources panel: verify that every listed job in the answer is backed by a source excerpt in the sources view; mismatches point to synthesis hallucination.
  - Judge badge & score: low scores imply poor relevance or structure — use these as triggers for retriever tuning or prompt improvements.

Recommended observability metrics & thresholds (starting guidance)
- LLM latency per call: target mean ≈1.0–2.0s, p95 < 3s (treat anything consistently >5s as a problem).
- Retriever precision@5: aim for >0.7 for good user-facing relevance.
- Judge pass rate: target >0.75 (monitor drops after model changes).
- Hallucination rate: target <2% of queries (automated checks using judge + provenance sampling).
- Index staleness / ingestion lag: monitor time since last successful ingestion and number of failed chunk upserts.

Operational recommendations when traces / UI show problems
- If LLM nodes are slow: enable request batching, move to a faster provider, or cache repeated prompts/responses.
- If judge fails often: add more high-quality retrieved context (increase top_k, improve chunking overlap), retrain the reranker with hard negatives, or tighten the synthesizer prompt to forbid invention.
- If retriever returns poor results: re-evaluate embedding model (higher-quality embeddings such as OpenAI text-embedding models are a strong option), increase vector search recall, try alternative ANN backends, and add query reformulation before retrieval.
- If frontend shows missing provenance: surface matched chunk excerpts and a `view source` link per result so users (and the judge) can verify claims.

Notes on timing in traces
- While traces may sometimes show long per-node durations, use the 1–2s per-LLM-call baseline for planning and SLOs unless you intentionally run slower providers. Long trace durations in screenshots should not be treated as the expected performance baseline.

