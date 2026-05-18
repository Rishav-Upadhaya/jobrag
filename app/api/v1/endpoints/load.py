from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg2 import Error as PsycopgError

from app.infrastructure.db.connection import get_conn
from app.infrastructure.db import repository
from app.schemas.response import LoadResponse
from app.ingestion import parser, preprocessor
from app.infrastructure.indexing.llama_cloud import llama_search_hook

logger = logging.getLogger(__name__)

router = APIRouter()
LOAD_BATCH_SIZE = 20


def _iter_batches(rows: list, batch_size: int = LOAD_BATCH_SIZE):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@router.post("", response_model=LoadResponse)
async def load_file(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
) -> LoadResponse:
    start_time = time.monotonic()
    tmp_path: str | None = None
    jobs_loaded = 0
    chunks_created = 0

    filename = file.filename or ""
    suffix = Path(filename).suffix

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            df = parser.parse(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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

        conn = None
        try:
            with get_conn() as conn:
                conn.rollback()

                if overwrite:
                    with conn.cursor() as cursor:
                        cursor.execute("DELETE FROM job_chunks;")
                        cursor.execute("DELETE FROM jobs;")
                    logger.info("Cleared existing jobs and chunks (overwrite mode)")

                for batch in _iter_batches(preprocessed_rows):
                    repository.bulk_upsert_jobs(conn, batch, commit=False)
                    jobs_loaded += len(batch)

                    chunk_texts: list[str] = []
                    job_indices: list[tuple[str, int]] = []
                    for job in batch:
                        job_id = job.get("job_id") or ""
                        for chunk_index, chunk_text in enumerate(job.get("enriched_chunks", [])):
                            chunk_texts.append(chunk_text)
                            job_indices.append((job_id, chunk_index))

                    if chunk_texts:
                        from app.infrastructure.embeddings.bge_m3 import get_embedder
                        embeddings = get_embedder().embed_batch(chunk_texts)
                        
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

        try:
            await llama_search_hook.index_file(tmp_path)
            logger.info("LlamaCloud indexing initiated for file: %s", filename)
        except Exception as exc:
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
