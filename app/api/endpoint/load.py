from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg2 import Error as PsycopgError

from app.db import vectorstore
from app.db.connection import get_conn
from app.models.response import LoadResponse
from app.pipeline.ingestion import embedder, parser, preprocessor

logger = logging.getLogger(__name__)

router = APIRouter()
LOAD_BATCH_SIZE = 100


def _iter_batches(rows: list[dict[str, object]], batch_size: int = LOAD_BATCH_SIZE):
	for start in range(0, len(rows), batch_size):
		yield rows[start : start + batch_size]


@router.post("/load", response_model=LoadResponse)
async def load_file(file: UploadFile = File(...), overwrite: bool = Form(False)) -> LoadResponse:
	start = time.monotonic()
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
		embedding_client = embedder.get_embedder()
		total_batches = max(1, (len(preprocessed_rows) + LOAD_BATCH_SIZE - 1) // LOAD_BATCH_SIZE)

		try:
			conn = None
			with get_conn() as conn:
				# Reset pooled connection state in case a prior request left a transaction open.
				conn.rollback()
				if overwrite:
					with conn.cursor() as cursor:
						cursor.execute("DELETE FROM job_chunks;")
						cursor.execute("DELETE FROM jobs;")

				# Gather all jobs and chunks first
				jobs_payload: list[dict[str, object]] = []
				all_chunk_texts: list[str] = []
				all_chunk_meta: list[tuple[str, int]] = []  # (job_id, chunk_index)

				for job in preprocessed_rows:
					jobs_payload.append(
						{
							"id": job["job_id"],
							"job_title": job.get("job_title"),
							"company_name": job.get("company_name"),
							"job_category": job.get("job_category"),
							"publication_date": job.get("publication_date"),
							"job_location": job.get("job_location"),
							"job_level": job.get("job_level"),
							"tags": job.get("tags"),
						}
					)
					for chunk_index, chunk_text in enumerate(job.get("chunks", [])):
						all_chunk_texts.append(chunk_text)
						all_chunk_meta.append((job["job_id"], chunk_index))

				logger.info(
					"Total jobs: %s, total chunks: %s",
					len(jobs_payload),
					len(all_chunk_texts),
				)

				# Embed all chunks in one batch
				embeddings: list[list[float]] = []
				if all_chunk_texts:
					embeddings = embedding_client.embed_batch(all_chunk_texts)
					if len(embeddings) != len(all_chunk_texts):
						raise ValueError(
							f"Embedding count mismatch: {len(embeddings)} != {len(all_chunk_texts)}"
						)

				# Prepare chunk rows with embeddings
				chunk_rows: list[tuple[str, int, str, list[float]]] = []
				for (job_id, chunk_index), chunk_text, embedding in zip(all_chunk_meta, all_chunk_texts, embeddings):
					chunk_rows.append((job_id, chunk_index, chunk_text, embedding))

				# Upsert jobs and chunks in batches
				for batch_index, job_batch in enumerate(_iter_batches(jobs_payload), start=1):
					logger.info("Upserting job batch %s/%s", batch_index, total_batches)
					vectorstore.bulk_upsert_jobs(conn, job_batch, commit=False)
					jobs_loaded += len(job_batch)

					# Corresponding chunks for this job batch
					batch_job_ids = {job["id"] for job in job_batch}
					chunk_batch = [row for row in chunk_rows if row[0] in batch_job_ids]
					if chunk_batch:
						vectorstore.bulk_upsert_chunks_with_options(conn, chunk_batch, commit=False)
						chunks_created += len(chunk_batch)

					conn.commit()

		except PsycopgError as exc:
			if conn is not None:
				try:
					conn.rollback()
				except Exception:
					pass
				logger.exception("Database error during load")
				raise HTTPException(status_code=500, detail="Database error during load") from exc
		except Exception as exc:
			logger.exception("Processing error during load")
			if conn is not None:
				try:
					conn.rollback()
				except Exception:
					pass
				raise HTTPException(status_code=500, detail="Error during load processing") from exc

		time_seconds = time.monotonic() - start
		return LoadResponse(
			status="ok",
			jobs_loaded=jobs_loaded,
			chunks_created=chunks_created,
			dropped_rows=dropped_rows,
			time_seconds=time_seconds,
		)
	finally:
		if tmp_path and os.path.exists(tmp_path):
			os.unlink(tmp_path)
		await file.close()
