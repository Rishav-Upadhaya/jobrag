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


def chunk_text(text: str) -> list[str]:
	if not text:
		return []
	return _SPLITTER.split_text(text)


def preprocess(df: pd.DataFrame) -> list[dict[str, Any]]:
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

		results.append(
			{
				"job_id": normalize_scalar(row.get("ID")),
				"job_title": normalize_scalar(row.get("Job Title")),
				"company_name": normalize_scalar(row.get("Company Name")),
				"job_category": normalize_scalar(row.get("Job Category")),
				"publication_date": normalize_scalar(row.get("Publication Date")),
				"job_location": normalize_scalar(row.get("Job Location")),
				"job_level": normalize_scalar(row.get("Job Level")),
				"tags": tags,
				"chunks": chunks,
			}
		)

	logger.info(
		"Preprocessed %s rows; dropped %s empty descriptions",
		len(results),
		dropped_rows,
	)
	return results
