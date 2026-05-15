"""Load endpoint — ingests job data from Excel/CSV.

New flow (post-LlamaCloud refactor):
  1. Parse Excel/CSV → DataFrame
  2. Preprocess → enriched job records (plain chunks + enriched_chunks)
  3. Upsert job metadata + plain chunks → PostgreSQL (FTS)
  4. Upload file directly to LlamaCloud (Managed Indexing handles embeddings)

No local embedding is performed. LlamaCloud owns the vector index.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg2 import Error as PsycopgError

from app.db.connection import get_conn
from app.db import repository
from app.models.response import LoadResponse
from app.pipeline.ingestion import parser, preprocessor
from app.pipeline.ingestion.llama_hooks import llama_search_hook

logger = logging.getLogger(__name__)

router = APIRouter()
LOAD_BATCH_SIZE = 20


def _iter_batches(rows: list, batch_size: int = LOAD_BATCH_SIZE):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@router.post("/load", response_model=LoadResponse)
async def load_file(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
) -> LoadResponse:
    """Ingest an Excel or CSV file of job listings.

    Steps:
      1. Parse file → validate columns
      2. Preprocess rows → enriched records for SQL
      3. Upsert jobs + plain chunks to PostgreSQL
      4. Upload raw file to LlamaCloud for managed vector indexing

    Args:
        file:      Uploaded Excel (.xlsx) or CSV file
        overwrite: If True, delete existing jobs and chunks before inserting
    """
    start_time = time.monotonic()
    tmp_path: str | None = None
    jobs_loaded = 0
    chunks_created = 0

    filename = file.filename or ""
    suffix = Path(filename).suffix

    try:
        # ── Step 1: Save upload to temp file, parse ───────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            df = parser.parse(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # ── Step 2: Preprocess → records for SQL ─────────────────────────────
        # We still preprocess to get structured metadata for the local DB
        preprocessed_rows = preprocessor.preprocess(df)
        dropped_rows = len(df) - len(preprocessed_rows)

        if not preprocessed_rows:
            return LoadResponse(
                status="ok",
                jobs_loaded=0,
                chunks_created=0,
                dropped_rows=dropped_rows,
                time_seconds=time.monotonic() - start_time,
            )

        logger.info(
            "Preprocessed %d rows for SQL (%d dropped)", len(preprocessed_rows), dropped_rows
        )

        # ── Step 3: Upsert metadata + plain chunks to PostgreSQL ──────────────
        conn = None
        try:
            with get_conn() as conn:
                conn.rollback()

                if overwrite:
                    with conn.cursor() as cursor:
                        cursor.execute("DELETE FROM job_chunks;")
                        cursor.execute("DELETE FROM jobs;")
                    logger.info("Overwrite mode: cleared existing jobs and chunks")

                for batch_index, batch in enumerate(_iter_batches(preprocessed_rows), 1):
                    # Upsert job metadata rows
                    repository.bulk_upsert_jobs(conn, batch, commit=False)
                    jobs_loaded += len(batch)

                    # Upsert plain chunk text rows for local vector search
                    chunk_texts: list[str] = []
                    job_indices: list[tuple[str, int]] = []
                    for job in batch:
                        job_id = job.get("job_id") or ""
                        # We embed the ENRICHED chunks for better semantic retrieval
                        for chunk_index, chunk_text in enumerate(job.get("enriched_chunks", [])):
                            chunk_texts.append(chunk_text)
                            job_indices.append((job_id, chunk_index))

                    if chunk_texts:
                        from app.pipeline.ingestion.embedder import get_embedder
                        logger.info(
                            "Batch %d: Embedding %d chunks locally using BGE-M3...", 
                            batch_index, len(chunk_texts)
                        )
                        embeddings = get_embedder().embed_batch(chunk_texts)
                        
                        # Note: We store the PLAIN chunk text in the DB for display/FTS,
                        # but the EMBEDDING comes from the enriched text.
                        chunk_rows = []
                        chunk_iter = 0
                        for job in batch:
                            job_id = job.get("job_id") or ""
                            for idx, plain_text in enumerate(job.get("chunks", [])):
                                emb = embeddings[chunk_iter]
                                chunk_rows.append((job_id, idx, plain_text, emb))
                                chunk_iter += 1

                        repository.bulk_upsert_chunks(conn, chunk_rows, commit=False)
                        chunks_created += len(chunk_rows)

                    conn.commit()
                    logger.info(
                        "SQL batch %d/%d: %d jobs committed with local embeddings", 
                        batch_index, (len(preprocessed_rows) + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE, len(batch)
                    )

        except PsycopgError as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception("Database error during load")
            raise HTTPException(
                status_code=500, detail="Database error during load"
            ) from exc

        # ── Step 4: Upload file to LlamaCloud for Managed Indexing ───────────
        # This triggers LlamaCloud's managed parsing and vector indexing.
        try:
            await llama_search_hook.index_file(tmp_path)
            logger.info("LlamaCloud managed indexing initiated for file: %s", filename)
        except Exception as exc:
            # We log but don't fail, as SQL data is already committed.
            logger.error("LlamaCloud indexing failed: %s", exc)

        return LoadResponse(
            status="ok",
            jobs_loaded=jobs_loaded,
            chunks_created=chunks_created,
            dropped_rows=dropped_rows,
            time_seconds=time.monotonic() - start_time,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        await file.close()
