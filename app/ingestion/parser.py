from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
	"ID",
	"Job Category",
	"Job Title",
	"Company Name",
	"Publication Date",
	"Job Location",
	"Job Level",
	"Tags",
	"Job Description",
]


def parse(file_path: str) -> pd.DataFrame:
	path = Path(file_path)
	if path.suffix.lower() == ".xlsx":
		df = pd.read_excel(path)
	elif path.suffix.lower() == ".csv":
		df = pd.read_csv(path)
	else:
		raise ValueError(f"Unsupported file type: {path.suffix}")

	missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

	df["Publication Date"] = pd.to_datetime(
		df["Publication Date"], errors="coerce"
	)

	string_columns = df.select_dtypes(include=["object", "string"]).columns
	if len(string_columns) > 0:
		df[string_columns] = df[string_columns].apply(
			lambda series: series.astype("string").str.strip()
		)

	logger.info("Parsed %s rows from %s", len(df), path.name)
	return df
