"""Repository module — direct database access for Leapfrog Job Search.

Handles CRUD operations for jobs and chunks, metadata filtering, and semantic
vector search using pgvector.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


# ── Job upserts ───────────────────────────────────────────────────────────────

def bulk_upsert_jobs(
    conn: psycopg2.extensions.connection,
    jobs: list[dict],
    *,
    commit: bool = True,
) -> None:
    """Bulk upsert job metadata records.

    Args:
        conn:   Database connection
        jobs:   List of job dicts
        commit: Whether to commit after upsert
    """
    if not jobs:
        return
    cursor = conn.cursor()

    try:
        query = """
        INSERT INTO jobs (
            id, job_title, company_name, job_category,
            publication_date, job_location, job_level,
            tags, enriched_text
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            job_title        = EXCLUDED.job_title,
            company_name     = EXCLUDED.company_name,
            job_category     = EXCLUDED.job_category,
            publication_date = EXCLUDED.publication_date,
            job_location     = EXCLUDED.job_location,
            job_level        = EXCLUDED.job_level,
            tags             = EXCLUDED.tags,
            enriched_text    = EXCLUDED.enriched_text
        """
        rows = [
            (
                j.get("job_id"),
                j.get("job_title"),
                j.get("company_name"),
                j.get("job_category"),
                j.get("publication_date"),
                j.get("job_location"),
                j.get("job_level"),
                j.get("tags"),
                j.get("enriched_text"),
            )
            for j in jobs
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


# ── Chunk upserts ────────────────────────────────────────────────────────────

def bulk_upsert_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[tuple[str, int, str, list[float]]],
    *,
    commit: bool = True,
) -> None:
    """Bulk upsert chunk rows (job_id, chunk_index, chunk_text, embedding).

    Args:
        conn:   Database connection
        chunks: List of (job_id, chunk_index, chunk_text, embedding) tuples
        commit: Whether to commit after upsert
    """
    if not chunks:
        return
    register_vector(conn)
    cursor = conn.cursor()

    try:
        query = """
        INSERT INTO job_chunks (job_id, chunk_index, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (job_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding
        """
        psycopg2.extras.execute_values(cursor, query, chunks, page_size=500)
        if commit:
            conn.commit()
        logger.debug("Bulk upserted %s chunks", len(chunks))
    except Exception as e:
        conn.rollback()
        logger.error("Error bulk upserting chunks: %s", e)
        raise
    finally:
        cursor.close()


# ── SQL Pre-filtering ────────────────────────────────────────────────────────

def _add_metadata_filters(
    where_clauses: list[str], params: list[Any], filters: dict
) -> None:
    """Add WHERE clauses based on metadata filters."""
    if filters.get("job_level"):
        where_clauses.append("LOWER(j.job_level) LIKE LOWER(%s)")
        params.append(f"%{filters['job_level']}%")

    if filters.get("job_category"):
        where_clauses.append("LOWER(j.job_category) LIKE LOWER(%s)")
        params.append(f"%{filters['job_category']}%")

    if filters.get("job_location"):
        where_clauses.append("LOWER(j.job_location) LIKE LOWER(%s)")
        params.append(f"%{filters['job_location']}%")

    if filters.get("company_name"):
        where_clauses.append("LOWER(j.company_name) LIKE LOWER(%s)")
        params.append(f"%{filters['company_name']}%")

    if filters.get("job_title"):
        where_clauses.append("LOWER(j.job_title) LIKE LOWER(%s)")
        params.append(f"%{filters['job_title']}%")

    if filters.get("date_after"):
        where_clauses.append("j.publication_date >= %s")
        params.append(filters["date_after"])


def get_matching_job_ids(conn: psycopg2.extensions.connection, filters: dict) -> list[str] | None:
    """Perform a SQL pre-filter to find job IDs that match the metadata criteria.

    Returns None if no filters are active (meaning all jobs are valid).
    Returns an empty list if filters are active but no jobs match.
    """
    # Check if any filterable keys are present
    filter_keys = {"job_level", "job_category", "job_location",
                   "company_name", "job_title", "date_after"}
    active_filters = {k: v for k, v in filters.items() if k in filter_keys and v}

    if not active_filters:
        return None

    register_vector(conn)
    cursor = conn.cursor()
    try:
        where_clauses: list[str] = []
        params: list[Any] = []
        _add_metadata_filters(where_clauses, params, active_filters)

        where_sql = " AND ".join(where_clauses)
        query = f"SELECT id FROM jobs j WHERE {where_sql}"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        job_ids = [row[0] for row in rows]
        logger.info(
            "SQL pre-filter matched %d jobs for filters: %s",
            len(job_ids),
            active_filters,
        )
        return job_ids
    except Exception as e:
        logger.error("Error in get_matching_job_ids: %s", e)
        raise
    finally:
        cursor.close()


# ── Vector search (Local pgvector search) ───────────────────────────

def vector_search(
    conn: psycopg2.extensions.connection,
    query_embedding: list[float],
    job_id_whitelist: list[str] | None = None,
    top_k: int = 40,
) -> list[dict]:
    """Semantic search using pgvector and cosine distance.

    Args:
        conn:             Database connection
        query_embedding:  Dense vector from BGE-M3
        job_id_whitelist: Optional list of job IDs to restrict search to
        top_k:            Number of results to return

    Returns:
        List of dicts with chunk + job metadata fields.
    """
    register_vector(conn)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        whitelist_clause = ""
        params: list[Any] = [query_embedding]
        
        if job_id_whitelist is not None:
            if not job_id_whitelist:
                return []
            whitelist_clause = "AND jc.job_id = ANY(%s)"
            params.append(job_id_whitelist)
        
        params.append(top_k)

        query = f"""
        SELECT
            jc.id            AS chunk_id,
            jc.job_id,
            jc.chunk_text,
            1 - (jc.embedding <=> %s::vector) AS score,
            j.job_title,
            j.company_name,
            j.job_level,
            j.job_location,
            j.job_category,
            j.publication_date,
            j.tags
        FROM job_chunks jc
        JOIN jobs j ON jc.job_id = j.id
        WHERE 1=1 {whitelist_clause}
        ORDER BY score DESC
        LIMIT %s
        """
        cursor.execute(query, params)
        results = cursor.fetchall()
        logger.debug("Vector search returned %d results", len(results))
        return [dict(r) for r in results]
    except Exception as e:
        logger.error("Error in vector_search: %s", e)
        raise
    finally:
        cursor.close()