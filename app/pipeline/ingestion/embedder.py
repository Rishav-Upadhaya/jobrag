"""Embedder module using BAAI/bge-m3.

This module provides local embedding capabilities using the SentenceTransformers
library. The BGE-M3 model is used for its superior performance and 1024-dim
dense vectors.
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Embedder using BAAI/bge-m3 from HuggingFace."""

    _instance: BGEEmbedder | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            model_name = "BAAI/bge-m3"
            logger.info("Loading SentenceTransformer model: %s...", model_name)
            try:
                # Use CPU for now as default; can be tuned for GPU if available
                self._model = SentenceTransformer(model_name)
                logger.info("Model %s loaded successfully.", model_name)
            except Exception as e:
                logger.error("Failed to load model %s: %s", model_name, e)
                raise
        return self._model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of 1024-dimensional float vectors
        """
        if not texts:
            return []
        
        # BGE-M3 returns a numpy array by default; convert to list of lists
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query."""
        return self.embed_batch([query])[0]


def get_embedder() -> BGEEmbedder:
    """Returns a singleton instance of the BGEEmbedder."""
    return BGEEmbedder()
