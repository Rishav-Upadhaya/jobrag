from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def rrf_fusion(
	results_lists: list[list[dict]],
	k: int = 60,
) -> list[dict]:
	"""
	Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.
	
	The RRF score for a result is: sum(1 / (k + rank_i)) across all lists.
	Results not present in a list are assigned rank infinity (score 0).
	
	De-duplication is done via "chunk_id" field — each unique chunk_id is scored
	only once across all lists.
	
	Args:
		results_lists: List of result lists, each containing dicts with "chunk_id" field
		k: RRF parameter (default 60 per standard)
		
	Returns:
		List of dicts sorted by fused RRF score (descending), preserving all fields
		from the original results
	"""
	# De-duplicate and aggregate scores
	chunk_scores: dict[int, float] = defaultdict(float)
	chunk_data: dict[int, dict] = {}
	
	for rank_list in results_lists:
		for rank, result in enumerate(rank_list, start=1):
			chunk_id = result.get("chunk_id")
			if chunk_id is None:
				logger.warning("Result missing 'chunk_id' field, skipping")
				continue
			
			# RRF score: 1 / (k + rank)
			rrf_score = 1.0 / (k + rank)
			chunk_scores[chunk_id] += rrf_score
			
			# Store the result data if not already stored
			# (in case of duplicates, keep the first occurrence)
			if chunk_id not in chunk_data:
				chunk_data[chunk_id] = result.copy()
	
	# Sort by fused RRF score (descending)
	sorted_chunk_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
	
	# Build final result list with fused score
	fused_results = []
	for chunk_id in sorted_chunk_ids:
		result = chunk_data[chunk_id].copy()
		result["fused_score"] = chunk_scores[chunk_id]
		fused_results.append(result)
	
	logger.debug(f"RRF fusion: combined {len(results_lists)} lists, {len(fused_results)} unique chunks")
	return fused_results
