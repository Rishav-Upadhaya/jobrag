"""Context and data filtering utilities."""

from datetime import datetime
from typing import Any


def filter_preferences_for_context(data: Any) -> dict[str, Any] | str:
    """
    Filter user preferences and context to exclude IDs and datetime fields.
    
    Removes fields ending with '_id', 'id', datetime objects, and empty values.
    
    Args:
        data: Object to filter (dict, object with __dict__, or scalar)
        
    Returns:
        Dictionary with filtered fields, or string representation if not object-like
    """
    if data is None:
        return ""

    # Convert to dict if it's an object with __dict__
    if hasattr(data, "__dict__"):
        data_dict = vars(data)
    elif isinstance(data, dict):
        data_dict = data
    else:
        return str(data)

    filtered: dict[str, Any] = {}

    for key, value in data_dict.items():
        # Skip ID fields
        if key.endswith("_id") or key == "id":
            continue

        # Skip datetime fields, None, empty strings, and zeros
        if isinstance(value, datetime) or value in (None, "", 0):
            continue

        filtered[key] = value

    return filtered


def build_context_block_from_chunks(
    chunks: list[dict],
    max_tokens: int = 3000,
) -> tuple[str, int]:
    """
    Build LLM context block from retrieved chunks with token budget.
    
    Includes full chunks until token budget is reached.
    Never truncates mid-chunk — either include full chunk or skip it.
    
    Args:
        chunks: List of retrieved chunk dicts with metadata
        max_tokens: Maximum tokens allowed for context block
        
    Returns:
        Tuple of (context_block: str, chunks_included: int)
    """
    def estimate_tokens(text: str) -> int:
        """Rough estimation: 1 token ≈ 4 characters."""
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
