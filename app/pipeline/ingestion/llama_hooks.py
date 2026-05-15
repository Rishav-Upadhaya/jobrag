"""LlamaCloud Hooks.

Provides unified hooks for Parsing (LlamaParse) and Search (LlamaCloud Managed Index).
Based on the user-provided reference patterns.
"""

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

from app.config import settings

logger = logging.getLogger(__name__)


# ── Parsing Models ────────────────────────────────────────────────────────────

class LlamaParsedContent(BaseModel):
    markdown_documents: list[LIDocument] | None
    text_documents: list[LIDocument] | None
    image_documents: list[ImageDocument] | None
    pages: list[Page] | None
    raw_result: Any


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
    """Base hook for LlamaCloud services."""
    
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.LLAMA_CLOUD_API_KEY
        self._client: AsyncLlamaCloud | None = None
        self._pipeline_id: str | None = None

    @property
    def client(self) -> AsyncLlamaCloud:
        if self._client is None:
            self._client = AsyncLlamaCloud(token=self.api_key)
        return self._client

    async def ensure_pipeline(self) -> str:
        """Resolve or create the managed pipeline ID."""
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


class LlamaParseHook(LlamaCloudHook):
    """Hook for document parsing using LlamaParse."""

    def __init__(
        self,
        api_key: str | None = None,
        num_workers: int | None = None,
        verbose: bool | None = None,
        language: str | None = None,
        parser_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(api_key)
        self.num_workers = num_workers if num_workers is not None else 4
        self.verbose = verbose if verbose is not None else True
        self.language = language or "en"
        self.parser_options = parser_options or {}

    async def parse_document_async(
        self,
        path: str | Path,
        *,
        include_markdown: bool = True,
        include_text: bool = True,
        split_markdown_by_page: bool = True,
        split_text_by_page: bool = False,
        parser_overrides: dict[str, Any] | None = None,
    ) -> LlamaParsedContent:
        parser = self._build_parser(parser_overrides)
        file_path = str(Path(path))
        try:
            result = await parser.aparse(file_path)
        except Exception as error:
            logger.error(f"Failed to parse document asynchronously {file_path}: {error}")
            raise
        return self._extract_content(
            result,
            include_markdown=include_markdown,
            include_text=include_text,
            split_markdown_by_page=split_markdown_by_page,
            split_text_by_page=split_text_by_page,
        )

    def _build_parser(self, parser_overrides: dict[str, Any] | None) -> LlamaParse:
        parser_args: dict[str, Any] = {
            "api_key": self.api_key,
            "num_workers": self.num_workers,
            "verbose": self.verbose,
            "language": self.language,
            **self.parser_options,
        }
        if parser_overrides:
            parser_args.update(parser_overrides)
        parser_args = {k: v for k, v in parser_args.items() if v is not None}
        return LlamaParse(**parser_args)

    def _extract_content(
        self,
        result: Any,
        *,
        include_markdown: bool,
        include_text: bool,
        split_markdown_by_page: bool,
        split_text_by_page: bool,
    ) -> LlamaParsedContent:
        markdown_docs = []
        if include_markdown and hasattr(result, "get_markdown_documents"):
            markdown_docs = list(result.get_markdown_documents(split_by_page=split_markdown_by_page))

        text_docs = []
        if include_text and hasattr(result, "get_text_documents"):
            text_docs = list(result.get_text_documents(split_by_page=split_text_by_page))

        pages = []
        if hasattr(result, "pages"):
            pages = list(getattr(result, "pages"))

        return LlamaParsedContent(
            markdown_documents=markdown_docs,
            text_documents=text_docs,
            image_documents=None,
            pages=pages,
            raw_result=result,
        )


class LlamaSearchHook(LlamaCloudHook):
    """Hook for managed retrieval using LlamaCloud Index."""

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
        """Upload and index a file in the managed pipeline."""
        pipeline_id = await self.ensure_pipeline()
        try:
            with open(file_path, "rb") as f:
                file_obj = await self.client.files.create(file=f, purpose="index")
            
            await self.client.pipelines.files.create(
                pipeline_id=pipeline_id,
                file_id=file_obj.id
            )
            return file_obj.id
        except Exception as e:
            logger.error(f"LlamaCloud index hook failed: {e}")
            raise


# Exported instances
llama_parse_hook = LlamaParseHook()
llama_search_hook = LlamaSearchHook()
