"""Database repository layer for job and job_chunk operations.

This module contains pure SQL operations for persisting and retrieving
job metadata and chunk vectors. It does not contain any business logic
or search algorithms.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from pgvector.psycopg2.vector import Vector

from app.config import settings

logger = logging.getLogger(__name__)


def _add_metadata_filters(
    where_clauses: list[str],
    params: list[Any],
    filters: dict,
    *,
    location_clause: str = "LOWER(j.job_location) LIKE LOWER(%s)",
) -> None:
    """Add metadata filter conditions to WHERE clause and parameters.

    Args:
        where_clauses: List of WHERE clause fragments
        params: List of SQL parameters
        filters: Dictionary of filter values
        location_clause: Optional custom location clause
    """
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
    """Bulk upsert job metadata rows in a single statement.

    Args:
        conn: Database connection with pgvector registered
        jobs: List of job dictionaries
        commit: Whether to commit transaction after upsert
    """
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
    """Bulk upsert chunk rows in a single statement.

    Args:
        conn: Database connection with pgvector registered
        chunks: List of (job_id, chunk_index, chunk_text, embedding) tuples
    """
    bulk_upsert_chunks_with_options(conn, chunks, commit=True)


def bulk_upsert_chunks_with_options(
    conn: psycopg2.extensions.connection,
    chunks: list[tuple[str, int, str, list[float]]],
    *,
    commit: bool = True,
) -> None:
    """Bulk upsert chunk rows in a single statement.

    Args:
        conn: Database connection with pgvector registered
        chunks: List of (job_id, chunk_index, chunk_text, embedding) tuples
        commit: Whether to commit transaction after upsert
    """
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
    """INSERT INTO jobs with ON CONFLICT (id) DO UPDATE.

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
    """INSERT INTO job_chunks with ON CONFLICT DO NOTHING.

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
    """Semantic search using cosine similarity on job_chunks.embedding.

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
    """Full-text search using tsvector and ts_rank.

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