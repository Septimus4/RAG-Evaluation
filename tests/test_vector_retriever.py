from pathlib import Path

from data_pipeline.ingestion import RawDocument, to_chunks
from rag.retriever import build_retriever, retrieve


def test_tfidf_retriever_search():
    docs = [
        RawDocument(path=Path("a.txt"), text="player scored points"),
        RawDocument(path=Path("b.txt"), text="rebounds and assists"),
    ]
    chunks = to_chunks(docs, chunk_size=50, overlap=10)
    store = build_retriever(chunks, top_k=2)
    result = retrieve("points", store, top_k=1)
    assert result.documents
    assert result.scores
    assert result.latency_ms >= 0
