from __future__ import annotations

import logging
import os
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Iterable

try:
	import cohere
except ImportError:  # pragma: no cover - optional dependency
	cohere = None

try:
	from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency
	GoogleGenerativeAIEmbeddings = None

try:
	from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
	SentenceTransformer = None

from app.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 64
HF_BATCH_SIZE = 128


def _batch_iter(texts: list[str], batch_size: int = BATCH_SIZE) -> Iterable[list[str]]:
	for idx in range(0, len(texts), batch_size):
		yield texts[idx : idx + batch_size]


class BaseEmbedder(ABC):
	@abstractmethod
	def embed_batch(self, texts: list[str]) -> list[list[float]]:
		raise NotImplementedError

	@staticmethod
	def _validate_dimensions(embeddings: list[list[float]], dimension: int) -> None:
		for embedding in embeddings:
			if len(embedding) != dimension:
				raise ValueError(
					f"Embedding dimension mismatch: {len(embedding)} != {dimension}"
				)


class GeminiEmbedder(BaseEmbedder):
	def __init__(self, model: str, api_key: str | None, dimension: int) -> None:
		if GoogleGenerativeAIEmbeddings is None:
			raise ImportError(
				"langchain-google-genai is required for Gemini embeddings"
			)
		if not api_key:
			raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings")
		self._client = GoogleGenerativeAIEmbeddings(
			model=model,
			api_key=api_key,
			output_dimensionality=dimension,
		)
		self._dimension = dimension

	def _embed_text(self, text: str) -> list[float]:
		embedding = self._client.embed_query(text)
		return [float(value) for value in embedding]

	def embed_batch(self, texts: list[str]) -> list[list[float]]:
		embeddings: list[list[float]] = []
		for batch in _batch_iter(texts):
			batch_embeddings = self._client.embed_documents(batch)
			batch_embeddings = [list(map(float, row)) for row in batch_embeddings]
			self._validate_dimensions(batch_embeddings, self._dimension)
			embeddings.extend(batch_embeddings)
		return embeddings


class CohereEmbedder(BaseEmbedder):
	def __init__(self, model: str, api_key: str | None, dimension: int) -> None:
		if cohere is None:
			raise ImportError("cohere is required for Cohere embeddings")
		if not api_key:
			raise ValueError("COHERE_API_KEY is required for Cohere embeddings")
		self._client = cohere.Client(api_key)
		self._model = model
		self._dimension = dimension

	def embed_batch(self, texts: list[str]) -> list[list[float]]:
		embeddings: list[list[float]] = []
		for batch in _batch_iter(texts):
			response = self._client.embed(texts=batch, model=self._model)
			batch_embeddings = [list(map(float, row)) for row in response.embeddings]
			self._validate_dimensions(batch_embeddings, self._dimension)
			embeddings.extend(batch_embeddings)
		return embeddings


class HFEmbedder(BaseEmbedder):
	def __init__(self, model: str, dimension: int) -> None:
		if SentenceTransformer is None:
			raise ImportError("sentence-transformers is required for HuggingFace embeddings")
		# Determine device: use CUDA if available, else MPS for Apple Silicon, else CPU
		try:
			import torch
			device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
		except Exception:
			device = "cpu"
		self._model = SentenceTransformer(model, device=device)
		self._dimension = dimension
		logger.info("HFEmbedder using device: %s", device)
		# Use half precision only on CUDA devices
		if device == "cuda":
			self._model.half()

	def embed_batch(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []
		# Use configurable batch size and optimize for CPU/GPU
		batch_embeddings = self._model.encode(
			texts,
			batch_size=settings.EMBEDDING_BATCH_SIZE,
			show_progress_bar=False,
			convert_to_numpy=True,
			normalize_embeddings=True,
		)
		# batch_embeddings is a numpy array; convert to list of floats
		batch_list = [list(map(float, row)) for row in batch_embeddings]
		self._validate_dimensions(batch_list, self._dimension)
		return batch_list


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
	provider = settings.EMBEDDING_PROVIDER.lower()
	model = settings.EMBEDDING_MODEL
	dimension = settings.EMBEDDING_DIMENSION

	if provider == "gemini":
		return GeminiEmbedder(model, settings.GOOGLE_API_KEY, dimension)
	if provider == "cohere":
		api_key = os.getenv("COHERE_API_KEY")
		return CohereEmbedder(model, api_key, dimension)
	if provider == "huggingface":
		return HFEmbedder(model, dimension)

	raise ValueError(f"Unsupported embedding provider: {provider}")
