
from __future__ import annotations

import logging
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_cloud.client import AsyncLlamaCloud
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.types import Page
from llama_index.core.schema import (
    Document as LIDocument,
)
from llama_index.core.schema import (
    ImageDocument,
    TextNode,
)
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Parsing Models ────────────────────────────────────────────────────────────



# ── Search Models ─────────────────────────────────────────────────────────────

class LlamaSearchResult(BaseModel):
    chunk_id: str
    job_id: str
    chunk_index: int
    chunk_text: str
    job_title: str
    company_name: str
    job_level: str
    job_location: str
    job_category: str
    publication_date: str
    score: float


# ── Hooks ─────────────────────────────────────────────────────────────────────

class LlamaCloudHook:
    def __init__(self) -> None:
        self.api_key = settings.LLAMA_CLOUD_API_KEY
        self._client: AsyncLlamaCloud | None = None
        self._pipeline_id: str | None = None

    @property
    def client(self) -> AsyncLlamaCloud:
        if self._client is None:
            self._client = AsyncLlamaCloud(token=self.api_key)
        return self._client

    async def ensure_pipeline(self) -> str:
        if self._pipeline_id:
            return self._pipeline_id

        project_name = settings.LLAMA_CLOUD_PROJECT_NAME
        pipeline_name = settings.LLAMA_CLOUD_INDEX_NAME

        try:
            from llama_cloud.types import PipelineCreate
            projects = await self.client.projects.list_projects()
            project = next((p for p in projects if p.name == project_name), None)
            if not project:
                project = projects[0] if projects else None
                if not project:
                    raise RuntimeError("No LlamaCloud projects found.")

            pipelines = await self.client.pipelines.search_pipelines(
                project_id=project.id,
                pipeline_name=pipeline_name
            )
            existing = pipelines[0] if pipelines else None
            
            if existing:
                self._pipeline_id = existing.id
                return self._pipeline_id

            # Create if missing
            req = PipelineCreate(name=pipeline_name, pipeline_type="MANAGED")
            new_pipeline = await self.client.pipelines.create_pipeline(
                project_id=project.id,
                request=req,
            )
            self._pipeline_id = new_pipeline.id
            return self._pipeline_id
        except Exception as e:
            logger.error(f"Failed to ensure LlamaCloud pipeline: {e}")
            raise


class LlamaSearchHook(LlamaCloudHook):

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 40,
        job_id_whitelist: list[str] | None = None,
    ) -> list[LlamaSearchResult]:
        try:
            pipeline_id = await self.ensure_pipeline()
        except Exception:
            return []

        filters = None
        if job_id_whitelist:
            filters = {
                "any": [
                    {"key": "job_id", "value": jid, "operator": "=="}
                    for jid in job_id_whitelist
                ]
            }

        try:
            response = await self.client.pipelines.retrieve(
                pipeline_id=pipeline_id,
                query=query,
                similarity_top_k=top_k,
                filters=filters
            )

            results = []
            for node in response:
                meta = node.metadata or {}
                results.append(LlamaSearchResult(
                    chunk_id=node.id,
                    job_id=meta.get("job_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    chunk_text=node.text,
                    job_title=meta.get("job_title", ""),
                    company_name=meta.get("company_name", ""),
                    job_level=meta.get("job_level", ""),
                    job_location=meta.get("job_location", ""),
                    job_category=meta.get("job_category", ""),
                    publication_date=meta.get("publication_date", ""),
                    score=node.score or 0.0,
                ))
            return results
        except Exception as e:
            logger.error(f"LlamaCloud search hook failed: {e}")
            return []

    async def index_file(self, file_path: str) -> str:
        try:
            pipeline_id = await self.ensure_pipeline()
            with open(file_path, "rb") as f:
                job = await self.client.pipelines.create_batch_pipeline_job(
                    pipeline_id=pipeline_id,
                    upload_files=[f]
                )
            return job.id
        except Exception as e:
            logger.error(f"Failed to index file to LlamaCloud: {e}")
            raise


llama_search_hook = LlamaSearchHook()
