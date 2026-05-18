
from typing import Any


def build_sources_from_chunks(retrieved_chunks: list[dict]) -> list[dict[str, Any]]:
    seen_job_ids: set[str] = set()
    sources: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        job_id = str(chunk.get("job_id", ""))
        if not job_id or job_id in seen_job_ids:
            continue

        sources.append({
            "job_id": job_id,
            "job_title": chunk.get("job_title", ""),
            "company_name": chunk.get("company_name", ""),
            "job_level": chunk.get("job_level", ""),
            "job_location": chunk.get("job_location", ""),
            "relevance_score": (
                chunk.get("rerank_score")
                or chunk.get("fused_score")
                or chunk.get("score", 0.0)
            ),
            "matched_chunk": chunk.get("chunk_text", "")[:200],
        })
        seen_job_ids.add(job_id)

    return sources


def fallback_answer_from_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "No matching jobs found for this query."

    lines = ["Found relevant roles from the indexed job data:"]
    for source in sources[:5]:
        lines.append(
            f"- **{source.get('job_title') or 'Unknown role'}** at "
            f"**{source.get('company_name') or 'Unknown company'}** "
            f"· {source.get('job_location') or 'N/A'} · "
            f"{source.get('job_level') or 'N/A'}"
        )
    return "\n".join(lines)
