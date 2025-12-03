"""Ingestion utilities for raw documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from pydantic import BaseModel, Field

from rag.models import DocumentChunk

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


class RawDocument(BaseModel):
    path: Path
    text: str = Field(..., min_length=1)


SUPPORTED_EXTENSIONS = {".txt"}


def load_text_documents(input_dir: Path) -> List[RawDocument]:
    documents: List[RawDocument] = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(RawDocument(path=path, text=text))
    logging.info("Loaded %s raw documents from %s", len(documents), input_dir)
    if logfire:
        logfire.info("ingestion.load_text_documents", count=len(documents), directory=str(input_dir))
    return documents


def to_chunks(raw_documents: Iterable[RawDocument], chunk_size: int = 600, overlap: int = 50) -> List[DocumentChunk]:
    documents_list = list(raw_documents)
    chunks: List[DocumentChunk] = []
    for doc in documents_list:
        text = doc.text
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunk_id = f"{doc.path.stem}-{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=str(doc.path),
                    metadata={"chunk_index": idx},
                )
            )
            idx += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    logging.info("Created %s chunks", len(chunks))
    if logfire:
        logfire.info("ingestion.to_chunks", chunks=len(chunks), documents=len(documents_list))
    return chunks
