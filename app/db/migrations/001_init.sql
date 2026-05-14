CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: one row per job, structured metadata for filtering
CREATE TABLE IF NOT EXISTS jobs (
	id               TEXT PRIMARY KEY,          -- LF####
	job_title        TEXT NOT NULL,
	company_name     TEXT NOT NULL,
	job_category     TEXT,
	publication_date TIMESTAMP,
	job_location     TEXT,
	job_level        TEXT,
	tags             TEXT[],
	created_at       TIMESTAMP DEFAULT NOW()
	-- NOTE: job_description is NOT stored here.
	-- Raw chunk text lives in job_chunks.chunk_text.
	-- Embeddings live in job_chunks.embedding.
);

-- Table 2: one row per JD chunk, embeddings + keyword search here
CREATE TABLE IF NOT EXISTS job_chunks (
	id           SERIAL PRIMARY KEY,
	job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
	chunk_index  INT  NOT NULL,
	chunk_text   TEXT NOT NULL,
	embedding    VECTOR(384) NOT NULL,
	ts_vector    tsvector GENERATED ALWAYS AS (
								 to_tsvector('english', chunk_text)
							 ) STORED,
	created_at   TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS jobs_job_level_idx ON jobs (job_level);
CREATE INDEX IF NOT EXISTS jobs_job_category_idx ON jobs (job_category);
CREATE INDEX IF NOT EXISTS jobs_job_location_idx ON jobs (job_location);
CREATE INDEX IF NOT EXISTS job_chunks_embedding_hnsw_idx ON job_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS job_chunks_ts_vector_gin_idx ON job_chunks USING gin (ts_vector);
CREATE INDEX IF NOT EXISTS job_chunks_job_id_idx ON job_chunks (job_id);
