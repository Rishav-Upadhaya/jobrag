
from datetime import datetime
from typing import Any


def filter_preferences_for_context(data: Any) -> dict[str, Any] | str:
    if data is None:
        return ""

    if hasattr(data, "__dict__"):
        data_dict = vars(data)
    elif isinstance(data, dict):
        data_dict = data
    else:
        return str(data)

    filtered: dict[str, Any] = {}

    for key, value in data_dict.items():
        if key.endswith("_id") or key == "id":
            continue

        if isinstance(value, datetime) or value in (None, "", 0):
            continue

        filtered[key] = value

    return filtered


def build_context_block_from_chunks(
    chunks: list[dict],
    max_tokens: int = 3000,
) -> tuple[str, int]:
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    parts = []
    total_tokens = 0
    included = 0

    for chunk in chunks:
        block = (
            f"---\n"
            f"Job ID: {chunk.get('job_id')} | Title: {chunk.get('job_title')} "
            f"| Company: {chunk.get('company_name')}\n"
            f"Location: {chunk.get('job_location')} | Level: {chunk.get('job_level')}\n"
            f"Excerpt: {chunk.get('chunk_text', '')}\n"
            f"---"
        )
        chunk_tokens = estimate_tokens(block)

        if total_tokens + chunk_tokens > max_tokens:
            break

        parts.append(block)
        total_tokens += chunk_tokens
        included += 1

    return "\n\n".join(parts), included
