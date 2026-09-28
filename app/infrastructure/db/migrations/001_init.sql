-- ============================================================
-- 001_init.sql  —  Leapfrog Job Search RAG (BGE-M3 Refactor)
-- ============================================================
-- Architecture:
--   • jobs        → structured metadata only (location, level, category, etc.)
--   • job_chunks  → chunk text + BGE-M3 embeddings (vector(1024))
--
-- Hybrid search: SQL metadata pre-filter → dense vector search + Postgres
-- full-text search, fused with Reciprocal Rank Fusion, then reranked.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Table 1: one row per job, structured metadata for SQL filtering ──────────
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,       -- e.g. "LF0042"
    job_title        TEXT NOT NULL,
    company_name     TEXT NOT NULL,
    job_category     TEXT,
    publication_date TIMESTAMPTZ,           
    job_location     TEXT,
    job_level        TEXT,
    tags             TEXT[],
    enriched_text    TEXT,                   -- metadata prefix used for embedding
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Table 2: chunk text + Dense BGE-M3 Vectors ──────────────────────────────
CREATE TABLE IF NOT EXISTS job_chunks (
    id           SERIAL PRIMARY KEY,
    job_id       TEXT        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_index  INT         NOT NULL,
    chunk_text   TEXT        NOT NULL,
    -- BGE-M3 Dense Embedding (1024 dims)
    embedding    vector(1024),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (job_id, chunk_index)
);

-- ── Indexes ──────────────────────────────────────────────────────────────────

-- Metadata filter indexes
CREATE INDEX IF NOT EXISTS jobs_job_level_idx       ON jobs (LOWER(job_level));
CREATE INDEX IF NOT EXISTS jobs_job_category_idx    ON jobs (LOWER(job_category));
CREATE INDEX IF NOT EXISTS jobs_job_location_idx    ON jobs (LOWER(job_location));
CREATE INDEX IF NOT EXISTS jobs_company_name_idx    ON jobs (LOWER(company_name));
CREATE INDEX IF NOT EXISTS jobs_job_title_idx       ON jobs (LOWER(job_title));
CREATE INDEX IF NOT EXISTS jobs_pub_date_idx        ON jobs (publication_date DESC);

-- HNSW Vector index for fast cosine similarity search
-- Note: cosine similarity (1 - (a . b) / (||a|| ||b||)) is preferred for BGE models
CREATE INDEX IF NOT EXISTS job_chunks_embedding_hnsw_idx ON job_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS job_chunks_job_id_idx    ON job_chunks (job_id);

-- ── Keyword search (Postgres full-text) ─────────────────────────────────────
-- Job metadata is weighted above description text so a title match ("data
-- engineer") outranks a passing mention in a description. Generated columns
-- keep the vectors in sync on every upsert; ADD COLUMN IF NOT EXISTS lets
-- existing databases pick this up on the next startup.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(job_title, '')), 'A') ||
    setweight(to_tsvector('english',
        coalesce(company_name, '') || ' ' || coalesce(job_category, '') || ' ' ||
        coalesce(job_level, '')    || ' ' || coalesce(job_location, '')), 'B')
) STORED;

ALTER TABLE job_chunks ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', chunk_text), 'C')
) STORED;

CREATE INDEX IF NOT EXISTS jobs_search_tsv_idx       ON jobs       USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS job_chunks_search_tsv_idx ON job_chunks USING gin (search_tsv);
