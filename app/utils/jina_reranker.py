from __future__ import annotations

import logging
from typing import TypedDict

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerResult(TypedDict, total=False):
	document: str
	relevance_score: float
	index: int


class RerankerRequest(BaseModel):
	base_url: str = "https://api.jina.ai/v1/rerank"
	model: str
	query: str
	documents: list[str]
	top_n: int = Field(gt=0)

	@field_validator("documents")
	@classmethod
	def validate_documents(cls, value: list[str]) -> list[str]:
		if not value:
			raise ValueError("documents list cannot be empty")
		return value


class JinaReranker:
	def __init__(self, rerank_model: str | None = None) -> None:
		self.api_key = settings.JINA_API_KEY
		if not self.api_key:
			raise ValueError("JINA_API_KEY environment variable is required for Jina reranker")
		
		self.session = requests.Session()
		headers = {"Content-Type": "application/json"}
		if self.api_key:
			headers["Authorization"] = f"Bearer {self.api_key}"
		self.session.headers.update(headers)
		self.rerank_model = rerank_model or settings.JINA_RERANKER_MODEL

	def _make_request(self, request: RerankerRequest) -> list[RerankerResult]:
		try:
			response = self.session.post(
				request.base_url,
				json=request.model_dump(exclude={"base_url"}),
				timeout=30,
			)
			response.raise_for_status()
			payload = response.json()
			results = payload.get("results", [])
			if not isinstance(results, list):
				logger.error(f"unexpected response structure from jina api: {payload}")
				return []
			parsed_results: list[RerankerResult] = []
			for item in results:
				if not isinstance(item, dict):
					continue
				try:
					parsed_results.append(
						RerankerResult(
							document=str(item.get("document", "")),
							relevance_score=float(item.get("relevance_score", 0.0)),
							index=int(item.get("index", 0)),
						)
					)
				except (TypeError, ValueError):
					continue
			return parsed_results
		except requests.RequestException as error:
			logger.error(f"jina api request failed: {error}")
			raise

	def rerank(
		self,
		query: str,
		documents: list[str],
		top_n: int = 5,
		threshold: float = 0.5,
	) -> list[RerankerResult]:
		"""
		Rerank documents using Jina Reranking API.
		
		Args:
			query: The search query
			documents: List of documents/chunks to rerank
			top_n: Number of top results to return
			threshold: Minimum relevance score threshold (0.0-1.0)
			
		Returns:
			List of reranked results with relevance scores
		"""
		try:
			request_data = RerankerRequest(
				model=self.rerank_model,
				query=query,
				documents=documents,
				top_n=top_n,
			)
		except ValidationError as error:
			logger.error(f"invalid jina rerank request parameters: {error}")
			raise

		results = self._make_request(request=request_data)
		filtered_results = [
			result for result in results 
			if result.get("relevance_score", 0.0) >= threshold
		]

		logger.info(
			f"reranked {len(documents)} documents, "
			f"returning {len(filtered_results)} above threshold {threshold}"
		)

		return filtered_results


# Singleton instance
_jina_reranker: JinaReranker | None = None


def get_jina_reranker() -> JinaReranker:
	"""Get or create the JinaReranker singleton instance."""
	global _jina_reranker
	if _jina_reranker is None:
		_jina_reranker = JinaReranker()
	return _jina_reranker
