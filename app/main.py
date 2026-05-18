from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.logging import setup_logging
from app.infrastructure.db.connection import close_pool, initialize_pool, initialize_schema

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_pool()
        initialize_schema()
        logger.info("Database pool and schema ready")
    except Exception:
        logger.exception("Database pool warmup failed during startup")
    yield
    close_pool()

app = FastAPI(title="Leapfrog Job Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
