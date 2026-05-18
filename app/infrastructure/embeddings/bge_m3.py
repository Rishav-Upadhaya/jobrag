
from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEEmbedder:
    def __init__(self) -> None:
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
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
        return self.embed_batch([query])[0]


_embedder: BGEEmbedder | None = None

def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder
