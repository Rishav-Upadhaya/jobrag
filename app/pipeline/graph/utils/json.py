"""JSON parsing helpers for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or a fenced LLM response."""
    cleaned = text.strip()
    if not cleaned:
        return {}

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def clamp_score(value: object, default: float = 0.0) -> float:
    """Convert a value to a 0.0-1.0 score."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return min(max(score, 0.0), 1.0)
