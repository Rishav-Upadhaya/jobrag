from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


def normalize_scalar(value: Any) -> Any:
	if value is None:
		return None
	try:
		if pd.isna(value):
			return None
	except TypeError:
		pass
	return value


def normalize_tags(value: Any) -> list[str] | None:
	if isinstance(value, list):
		tags = [str(item).strip() for item in value if str(item).strip()]
		return tags or None
	value = normalize_scalar(value)
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	parts = re.split(r"\s*[;,|]\s*", text)
	tags = [part for part in (piece.strip() for piece in parts) if part]
	return tags or None


def clean_html(text: str) -> str:
	if not text:
		return ""
	soup = BeautifulSoup(text, "html.parser")
	cleaned = soup.get_text(separator=" ")
	return " ".join(cleaned.split())


def build_enriched_prefix(
	job_title: str | None,
	job_level: str | None,
	job_category: str | None,
	job_location: str | None,
	company_name: str | None,
) -> str:
	"""Build a metadata prefix prepended to every chunk at ingest time.

	This ensures that LlamaCloud embeds structured metadata alongside the JD
	text, so semantic searches for e.g. "Senior ML Engineer at Leapfrog" will
	match even if the JD body never repeats those exact words.

	Format: "<level> | <category> | <location> | <company> | <title>"
	"""
	parts = [
		job_level or "",
		job_category or "",
		job_location or "",
		company_name or "",
		job_title or "",
	]
	non_empty = [p.strip() for p in parts if p and p.strip()]
	return " | ".join(non_empty)


def chunk_text(text: str) -> list[str]:
	if not text:
		return []
	return _SPLITTER.split_text(text)


def preprocess(df: pd.DataFrame) -> list[dict[str, Any]]:
	"""Preprocess raw DataFrame into structured job records with enriched chunks.

	Each record contains:
	- Structured metadata fields (for SQL storage + filtering)
	- enriched_text: full metadata prefix (stored in jobs table for reference)
	- chunks: list of raw JD text chunks (stored in job_chunks, text-only)
	- enriched_chunks: list of "prefix + chunk" strings sent to LlamaCloud

	The enriched_chunks are what LlamaCloud indexes/embeds; the plain chunks
	are what pgvector FTS searches.
	"""
	results: list[dict[str, Any]] = []
	dropped_rows = 0

	for _, row in df.iterrows():
		raw_description = row.get("Job Description", "")
		cleaned_description = clean_html(
			"" if normalize_scalar(raw_description) is None else str(raw_description)
		)
		if not cleaned_description:
			dropped_rows += 1
			continue

		chunks = chunk_text(cleaned_description)
		if not chunks:
			dropped_rows += 1
			continue

		tags = normalize_tags(row.get("Tags"))

		job_title    = normalize_scalar(row.get("Job Title"))
		company_name = normalize_scalar(row.get("Company Name"))
		job_category = normalize_scalar(row.get("Job Category"))
		job_location = normalize_scalar(row.get("Job Location"))
		job_level    = normalize_scalar(row.get("Job Level"))

		prefix = build_enriched_prefix(
			job_title=job_title,
			job_level=job_level,
			job_category=job_category,
			job_location=job_location,
			company_name=company_name,
		)

		# enriched_chunks = what LlamaCloud will embed
		# Plain chunks    = stored in job_chunks.chunk_text for FTS
		enriched_chunks = [
			f"{prefix}\n\n{chunk}" if prefix else chunk
			for chunk in chunks
		]

		results.append(
			{
				"job_id":           normalize_scalar(row.get("ID")),
				"job_title":        job_title,
				"company_name":     company_name,
				"job_category":     job_category,
				"publication_date": normalize_scalar(row.get("Publication Date")),
				"job_location":     job_location,
				"job_level":        job_level,
				"tags":             tags,
				"enriched_text":    prefix,   # stored in jobs.enriched_text
				"chunks":           chunks,   # plain text → job_chunks + FTS
				"enriched_chunks":  enriched_chunks,  # prefix+text → LlamaCloud
			}
		)

	logger.info(
		"Preprocessed %s rows; dropped %s empty descriptions",
		len(results),
		dropped_rows,
	)
	return results
