from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from pgvector.psycopg2.vector import Vector

from app.utils.rrf import rrf_fusion
from app.utils.jina_reranker import get_jina_reranker
from app.config import settings

logger = logging.getLogger(__name__)


def _add_metadata_filters(
	where_clauses: list[str],
	params: list[Any],
	filters: dict,
	*,
	location_clause: str = "LOWER(j.job_location) LIKE LOWER(%s)",
) -> None:
	if filters.get("job_level"):
		where_clauses.append("LOWER(j.job_level) LIKE LOWER(%s)")
		params.append(f"%{filters['job_level']}%")
	if filters.get("job_category"):
		where_clauses.append("LOWER(j.job_category) LIKE LOWER(%s)")
		params.append(f"%{filters['job_category']}%")
	if filters.get("job_location"):
		where_clauses.append(location_clause)
		params.append(f"%{filters['job_location']}%")


def bulk_upsert_jobs(
	conn: psycopg2.extensions.connection,
	jobs: list[dict],
	*,
	commit: bool = True,
) -> None:
	"""Bulk upsert job metadata rows in a single statement."""
	if not jobs:
		return
	register_vector(conn)
	cursor = conn.cursor()

	try:
		query = """
		INSERT INTO jobs (id, job_title, company_name, job_category,
						  publication_date, job_location, job_level, tags)
		VALUES %s
		ON CONFLICT (id) DO UPDATE SET
			job_title = EXCLUDED.job_title,
			company_name = EXCLUDED.company_name,
			job_category = EXCLUDED.job_category,
			publication_date = EXCLUDED.publication_date,
			job_location = EXCLUDED.job_location,
			job_level = EXCLUDED.job_level,
			tags = EXCLUDED.tags
		"""
		rows = [
			(
				job.get("id"),
				job.get("job_title"),
				job.get("company_name"),
				job.get("job_category"),
				job.get("publication_date"),
				job.get("job_location"),
				job.get("job_level"),
				job.get("tags"),
			)
			for job in jobs
		]
		psycopg2.extras.execute_values(cursor, query, rows, page_size=500)
		if commit:
			conn.commit()
		logger.debug("Bulk upserted %s jobs", len(rows))
	except Exception as e:
		conn.rollback()
		logger.error("Error bulk upserting jobs: %s", e)
		raise
	finally:
		cursor.close()


def bulk_upsert_chunks(
	conn: psycopg2.extensions.connection,
	chunks: list[tuple[str, int, str, list[float]]],
) -> None:
	bulk_upsert_chunks_with_options(conn, chunks, commit=True)


def bulk_upsert_chunks_with_options(
	conn: psycopg2.extensions.connection,
	chunks: list[tuple[str, int, str, list[float]]],
	*,
	commit: bool = True,
) -> None:
	"""Bulk upsert chunk rows in a single statement."""
	if not chunks:
		return
	register_vector(conn)
	cursor = conn.cursor()

	try:
		query = """
		INSERT INTO job_chunks (job_id, chunk_index, chunk_text, embedding)
		VALUES %s
		ON CONFLICT DO NOTHING
		"""
		rows = [
			(job_id, chunk_index, chunk_text, Vector(embedding))
			for job_id, chunk_index, chunk_text, embedding in chunks
		]
		psycopg2.extras.execute_values(cursor, query, rows, page_size=500)
		if commit:
			conn.commit()
		logger.debug("Bulk upserted %s chunks", len(chunks))
	except Exception as e:
		conn.rollback()
		logger.error("Error bulk upserting chunks: %s", e)
		raise
	finally:
		cursor.close()


def upsert_job(conn: psycopg2.extensions.connection, job: dict) -> None:
	"""
	INSERT INTO jobs with ON CONFLICT (id) DO UPDATE.
	
	Args:
		conn: Database connection with pgvector registered
		job: Dict with keys: id, job_title, company_name, job_category,
			 publication_date, job_location, job_level, tags
	"""
	register_vector(conn)
	cursor = conn.cursor()
	
	try:
		query = """
		INSERT INTO jobs (id, job_title, company_name, job_category, 
						  publication_date, job_location, job_level, tags)
		VALUES (%(id)s, %(job_title)s, %(company_name)s, %(job_category)s,
				%(publication_date)s, %(job_location)s, %(job_level)s, %(tags)s)
		ON CONFLICT (id) DO UPDATE SET
			job_title = EXCLUDED.job_title,
			company_name = EXCLUDED.company_name,
			job_category = EXCLUDED.job_category,
			publication_date = EXCLUDED.publication_date,
			job_location = EXCLUDED.job_location,
			job_level = EXCLUDED.job_level,
			tags = EXCLUDED.tags
		"""
		cursor.execute(query, job)
		conn.commit()
		logger.debug(f"Upserted job {job['id']}")
	except Exception as e:
		conn.rollback()
		logger.error(f"Error upserting job {job.get('id', 'unknown')}: {e}")
		raise
	finally:
		cursor.close()


def upsert_chunk(
	conn: psycopg2.extensions.connection,
	job_id: str,
	chunk_index: int,
	chunk_text: str,
	embedding: list[float],
) -> None:
	"""
	INSERT INTO job_chunks with ON CONFLICT DO NOTHING.
	Chunks are immutable by (job_id, chunk_index).
	
	Args:
		conn: Database connection with pgvector registered
		job_id: Job ID (LF####)
		chunk_index: Chunk sequence number
		chunk_text: Raw chunk text
		embedding: Vector embedding (list of floats, should be 768-dim)
	"""
	register_vector(conn)
	cursor = conn.cursor()
	
	try:
		query = """
		INSERT INTO job_chunks (job_id, chunk_index, chunk_text, embedding)
		VALUES (%s, %s, %s, %s)
		ON CONFLICT DO NOTHING
		"""
		cursor.execute(query, (job_id, chunk_index, chunk_text, embedding))
		conn.commit()
		logger.debug(f"Upserted chunk {job_id}:{chunk_index}")
	except Exception as e:
		conn.rollback()
		logger.error(f"Error upserting chunk {job_id}:{chunk_index}: {e}")
		raise
	finally:
		cursor.close()


def vector_search(
	conn: psycopg2.extensions.connection,
	query_embedding: list[float],
	filters: dict,
	top_k: int = 20,
) -> list[dict]:
	"""
	Semantic search using cosine similarity on job_chunks.embedding.
	Apply metadata filters via JOIN to jobs table.
	
	Args:
		conn: Database connection with pgvector registered
		query_embedding: Query vector (list of floats)
		filters: Dict with optional keys: job_level, job_category, job_location
		top_k: Number of results to return
		
	Returns:
		List of dicts with keys: chunk_id, job_id, chunk_text, score,
		job_title, company_name, job_level, job_location, job_category
	"""
	register_vector(conn)
	cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
	
	try:
		# Build WHERE clause for filters
		where_clauses = []
		params: list[Any] = [Vector(query_embedding)]
		
		_add_metadata_filters(where_clauses, params, filters)
		
		where_clause = " AND ".join(where_clauses)
		if where_clause:
			where_clause = "WHERE " + where_clause
		
		query = f"""
		SELECT 
			jc.id as chunk_id,
			jc.job_id,
			jc.chunk_text,
			1 - (jc.embedding <=> %s) as score,
			j.job_title,
			j.company_name,
			j.job_level,
			j.job_location,
			j.job_category,
			j.publication_date,
			j.tags
		FROM job_chunks jc
		JOIN jobs j ON jc.job_id = j.id
		{where_clause}
		ORDER BY score DESC
		LIMIT %s
		"""
		params.append(top_k)
		cursor.execute(query, params)
		results = cursor.fetchall()
		logger.debug(f"Vector search returned {len(results)} results")
		return results
	except Exception as e:
		logger.error(f"Error in vector_search: {e}")
		raise
	finally:
		cursor.close()


def keyword_search(
	conn: psycopg2.extensions.connection,
	query_text: str,
	filters: dict,
	top_k: int = 20,
) -> list[dict]:
	"""
	Full-text search using tsvector and ts_rank.
	Apply metadata filters via JOIN to jobs table.
	
	Args:
		conn: Database connection with pgvector registered
		query_text: Query text for full-text search
		filters: Dict with optional keys: job_level, job_category, job_location
		top_k: Number of results to return
		
	Returns:
		List of dicts with keys: chunk_id, job_id, chunk_text, score,
		job_title, company_name, job_level, job_location, job_category
	"""
	register_vector(conn)
	cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
	
	try:
		# Build WHERE clause for filters
		where_clauses = []
		params: list[Any] = []
		
		_add_metadata_filters(where_clauses, params, filters)
		
		where_clause = " AND ".join(where_clauses)
		if where_clause:
			where_clause = "AND " + where_clause

		query = f"""
		SELECT 
			jc.id as chunk_id,
			jc.job_id,
			jc.chunk_text,
			ts_rank(
				setweight(to_tsvector('english', COALESCE(j.job_title,'')), 'A') ||
				setweight(to_tsvector('english', COALESCE(j.company_name,'')), 'A') ||
				setweight(jc.ts_vector, 'B'),
				websearch_to_tsquery('english', %s)
			) as score,
			j.job_title,
			j.company_name,
			j.job_level,
			j.job_location,
			j.job_category,
			j.publication_date,
			j.tags
		FROM job_chunks jc
		JOIN jobs j ON jc.job_id = j.id
		WHERE (
			setweight(to_tsvector('english', COALESCE(j.job_title,'')), 'A') ||
			setweight(to_tsvector('english', COALESCE(j.company_name,'')), 'A') ||
			setweight(jc.ts_vector, 'B')
		) @@ websearch_to_tsquery('english', %s)
		""" + where_clause + """
		ORDER BY score DESC
		LIMIT %s
		"""
		sql_params: list[Any] = [query_text, query_text, *params, top_k]
		cursor.execute(query, sql_params)
		results = cursor.fetchall()
		logger.debug(f"Keyword search returned {len(results)} results")
		return results
	except Exception as e:
		logger.error(f"Error in keyword_search: {e}")
		raise
	finally:
		cursor.close()


def deduplicate_by_job(
	chunks: list[dict],
	max_chunks_per_job: int = 2,
) -> list[dict]:
	"""
	Deduplicate chunks by job_id, keeping at most max_chunks_per_job per job.
	Preserves the order of fused_score (highest scoring chunks first).
	
	Args:
		chunks: List of chunk dicts with job_id field
		max_chunks_per_job: Maximum number of chunks to keep per job_id
		
	Returns:
		Filtered list of chunks preserving the input order
	"""
	seen_jobs: dict[str, int] = defaultdict(int)
	result = []
	for chunk in chunks:
		job_id = chunk.get("job_id")
		if seen_jobs[job_id] < max_chunks_per_job:
			seen_jobs[job_id] += 1
			result.append(chunk)
	return result


def rerank_chunks(
	chunks: list[dict],
	query: str,
	top_k: int | None = None,
	threshold: float | None = None,
) -> list[dict]:
	"""
	Rerank chunks using Jina Reranking API.
	
	Args:
		chunks: List of chunk dicts with chunk_text field
		query: The search query
		top_k: Number of top results to return (defaults to settings.TOP_K_RERANK)
		threshold: Minimum relevance score threshold (defaults to settings.JINA_RERANK_THRESHOLD)
		
	Returns:
		List of reranked chunks sorted by Jina relevance score
	"""
	if not chunks:
		return []
	
	if top_k is None:
		top_k = settings.TOP_K_RERANK
	
	if threshold is None:
		threshold = settings.JINA_RERANK_THRESHOLD
	
	try:
		# Extract chunk texts for reranking
		documents = [chunk.get("chunk_text", "") for chunk in chunks]
		
		if not documents:
			logger.warning("No documents to rerank")
			return chunks[:top_k]
		
		# Get Jina reranker and rerank
		reranker = get_jina_reranker()
		reranked_results = reranker.rerank(
			query=query,
			documents=documents,
			top_n=top_k,
			threshold=threshold,
		)
		
		if not reranked_results:
			logger.warning(f"No results from Jina reranker above threshold {threshold}, returning original chunks")
			return chunks[:top_k]
		
		reranked_chunks = []
		
		for result in reranked_results:
			index = result.get("index")
			if index is not None and index < len(chunks):
				chunk = chunks[index].copy()
				chunk["rerank_score"] = result.get("relevance_score", 0.0)
				reranked_chunks.append(chunk)
		
		logger.info(f"Reranked {len(chunks)} chunks, returning {len(reranked_chunks)} results")
		return reranked_chunks
	
	except Exception as e:
		logger.error(f"Error in rerank_chunks: {e}, falling back to original ranking")
		# Fallback to original ranking if reranking fails
		return chunks[:top_k]


def hybrid_search(
	conn: psycopg2.extensions.connection,
	query_text: str,
	query_embedding: list[float],
	filters: dict,
	top_k_vector: int = 20,
	top_k_keyword: int = 20,
	final_top_k: int = 5,
) -> list[dict]:
	"""
	Hybrid search combining vector and keyword search with RRF fusion.
	
	1. Vector search: cosine similarity on embeddings (top_k_vector results)
	2. Keyword search: full-text search with ts_rank (top_k_keyword results)
	3. Fuse rankings with Reciprocal Rank Fusion (RRF)
	4. Deduplicate chunks by job to limit repetition
	5. Rerank the strongest fused candidates when a reranker is configured
	6. Return top final_top_k results
	
	Args:
		conn: Database connection with pgvector registered
		query_text: Query text for full-text search
		query_embedding: Query vector embedding
		filters: Dict with optional keys: job_level, job_category, job_location
		top_k_vector: Number of vector search results to consider
		top_k_keyword: Number of keyword search results to consider
		final_top_k: Number of final results to return
		
	Returns:
		List of dicts with keys: chunk_id, job_id, chunk_text, score,
		job_title, company_name, job_level, job_location, job_category
	"""
	try:
		# Get vector search results
		vector_results = vector_search(conn, query_embedding, filters, top_k_vector)
		
		# Get keyword search results
		keyword_results = keyword_search(conn, query_text, filters, top_k_keyword)
		
		# Fuse with RRF
		fused = rrf_fusion([vector_results, keyword_results], k=60)
		
		# Deduplicate by job to avoid same job occupying multiple top-k slots.
		fused = deduplicate_by_job(fused, max_chunks_per_job=2)

		candidate_count = min(
			len(fused),
			max(final_top_k * 3, settings.TOP_K_RERANK, final_top_k),
		)
		candidates = fused[:candidate_count]

		if settings.JINA_API_KEY and candidates:
			reranked = rerank_chunks(
				candidates,
				query_text,
				top_k=min(len(candidates), max(final_top_k, settings.TOP_K_RERANK)),
				threshold=settings.JINA_RERANK_THRESHOLD,
			)
			if reranked:
				reranked_ids = {chunk.get("chunk_id") for chunk in reranked}
				backfill = [
					chunk for chunk in candidates
					if chunk.get("chunk_id") not in reranked_ids
				]
				return (reranked + backfill)[:final_top_k]

		return candidates[:final_top_k]
	except Exception as e:
		logger.error(f"Error in hybrid_search: {e}")
		raise
