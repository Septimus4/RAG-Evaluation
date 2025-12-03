"""Simple TF-IDF vector store with Logfire instrumentation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import DocumentChunk

try:  # pragma: no cover - optional dependency
    import logfire
except Exception:  # pragma: no cover
    logfire = None


@dataclass
class TfidfVectorStore:
    """In-memory vector store for prototyping and tests."""

    vectorizer: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(stop_words="english"))
    matrix: np.ndarray | None = None
    documents: List[DocumentChunk] = field(default_factory=list)

    def index(self, documents: List[DocumentChunk]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing")
        logging.info("Indexing %s documents", len(documents))
        corpus = [doc.text for doc in documents]
        start = time.perf_counter()
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.documents = documents
        duration_ms = (time.perf_counter() - start) * 1000
        if logfire:
            logfire.info("vector_store.index", count=len(documents), latency_ms=duration_ms)
        logging.debug("Indexed %s documents in %.2f ms", len(documents), duration_ms)

    def search(self, query: str, top_k: int = 5) -> tuple[list[DocumentChunk], list[float], float]:
        if self.matrix is None or not self.documents:
            raise RuntimeError("Vector store is empty. Run index() first.")
        start = time.perf_counter()
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        duration_ms = (time.perf_counter() - start) * 1000
        if logfire:
            logfire.info("vector_store.search", top_k=top_k, latency_ms=duration_ms)
        docs = [self.documents[i] for i in top_indices]
        scores = [float(sims[i]) for i in top_indices]
        return docs, scores, duration_ms
