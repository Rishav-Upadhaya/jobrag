from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from the new endpoint package
from app.api.endpoint.load import router as load_router
from app.api.endpoint.query import router as query_router

# Database utilities remain unchanged
from app.db.connection import close_pool, initialize_pool, initialize_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_pool()
        initialize_schema()
        # Pre-warm embedding model on startup
        logger.info("Pre-warming embedding model...")
        from app.pipeline.ingestion.embedder import get_embedder
        embedder = get_embedder()
        # Run one dummy embed to fully initialize the model
        embedder.embed_batch(["warmup"])
        logger.info("Embedding model ready")
    except Exception:
        logger.exception("Database pool warmup failed during startup")
    yield
    close_pool()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(load_router, prefix="/api")
app.include_router(query_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
