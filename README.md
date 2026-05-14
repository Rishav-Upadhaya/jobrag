# Leapfrog Agent

## Project overview

Leapfrog Agent is a FastAPI-based retrieval-augmented generation system for job data. It lets users ask natural-language questions about roles, companies, and filters, then returns answers grounded in the LF Jobs dataset. The pipeline combines structured database filters, hybrid retrieval, and LLM synthesis so answers stay tied to actual listings.

## Architecture

The data model is intentionally split into two tables: `jobs` stores structured metadata for filtering, while `job_chunks` stores chunked job descriptions, embeddings, and keyword-search vectors. That separation keeps SQL filtering simple and lets semantic retrieval operate only on the chunk table.

Query execution follows a LangGraph flow: intent classification routes the request, valid queries go to hybrid retrieval, retrieval output is synthesized into an answer, and a judge node evaluates the result before the API response is finalized. The retry loop is controlled inside the graph, so the API layer only receives the completed state.

## Setup and installation

1. Start PostgreSQL, the backend, and Adminer:

   ```bash
	docker compose up --build
   ```

2. Open Adminer at http://localhost:8080 and connect with:

	- System: PostgreSQL
	- Server: db
	- Username: raguser
	- Password: ragpass
	- Database: ragdb

3. Run the database migration:

   ```bash
	psql postgresql://raguser:ragpass@localhost:5432/ragdb -f app/db/migrations/001_init.sql
   ```

4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Create your local environment file:

   ```bash
   cp .env.example .env
   ```

6. Fill in `.env` with your embedding settings, LLM provider, and API keys. The backend container already points at the compose database service.

## Running the API

Start the server with either of these commands:

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
make dev
```

The health check is available at `GET /health`.

**Note:** Initial data load for 1000 jobs takes 5-10 minutes on CPU (Docker cannot access Apple Silicon MPS). The embedding model is pre-cached in the Docker image so queries are fast after first load.

## Example requests

### Load data

Request:

```bash
curl -X POST http://localhost:8000/api/load \
	-F "file=@data/lf_jobs.xlsx" \
	-F "overwrite=true"
```

Expected response shape:

```json
{
	"status": "ok",
	"jobs_loaded": 123,
	"chunks_created": 456,
	"dropped_rows": 7,
	"time_seconds": 12.34
}
```

### Query jobs

Request:

```bash
curl -X POST http://localhost:8000/api/query \
	-H "Content-Type: application/json" \
	-d '{
		"query": "senior machine learning roles in New York",
		"top_k": 5,
		"filters": {
			"job_level": "Senior",
			"job_location": "New York"
		}
	}'
```

Expected response shape:

```json
{
	"query": "senior machine learning roles in New York",
	"answer": "...",
	"sources": [
		{
			"job_id": "LF0001",
			"job_title": "Machine Learning Engineer",
			"company_name": "Example Co",
			"job_level": "Senior",
			"job_location": "New York",
			"relevance_score": 0.91,
			"matched_chunk": "..."
		}
	],
	"judge": {
		"score": 0.88,
		"verdict": "pass",
		"reasoning": "...",
		"relevance": 0.9,
		"quality": 0.86,
		"hallucination": false
	},
	"intent": "valid",
	"latency_ms": 123.45,
	"clarification_question": null,
	"message": null,
	"status": null
}
```

If the query is vague, the response includes a `clarification_question` instead of an answer. If it is off topic, the response includes a rejection message and status information.

## LLM provider switching

Change `LLM_PROVIDER` in `.env` to switch providers:

```bash
LLM_PROVIDER=gemini
```

Supported values are `openai`, `gemini`, and `openrouter`. Make sure the matching API key is set as well: `OPENAI_API_KEY` for OpenAI, `GOOGLE_API_KEY` for Gemini, or `OPENROUTER_API_KEY` plus `OPENROUTER_BASE_URL` for OpenRouter. The synthesizer, classifier, and judge can also use separate model names through their respective environment variables.

## Embedding provider switching

To avoid Gemini embeddings and use a free local Hugging Face model instead, set:

```bash
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
```

This uses `sentence-transformers` locally, so there is no per-request embedding API cost once the model is downloaded.

## Running tests

Run the test suite with:

```bash
pytest tests/ -v
```

You can also use:

```bash
make test
```
