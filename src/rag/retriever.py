"""Retriever wrapper for the vector store."""
from __future__ import annotations

import logging
from typing import List

from .models import DocumentChunk, RetrievalResult
from .vector_store import TfidfVectorStore

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


def build_retriever(documents: List[DocumentChunk], top_k: int = 5) -> TfidfVectorStore:
    store = TfidfVectorStore()
    store.index(documents)
    logging.info("Retriever built with %s documents", len(documents))
    return store


def retrieve(query: str, store: TfidfVectorStore, top_k: int = 5) -> RetrievalResult:
    docs, scores, latency_ms = store.search(query, top_k=top_k)
    if logfire:
        logfire.info("retrieval", query=query, top_k=top_k, latency_ms=latency_ms)
    return RetrievalResult(query=query, documents=docs, scores=scores, latency_ms=latency_ms)
