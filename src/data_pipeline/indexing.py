"""CLI-friendly indexing entrypoint."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from rag.retriever import build_retriever
from rag.vector_store import TfidfVectorStore

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None

from .ingestion import RawDocument, load_text_documents, to_chunks


app = typer.Typer(add_completion=False)


@app.command()
def build_index(input_dir: Path = typer.Option(Path("inputs"), help="Directory with raw text files."),
                top_k: int = typer.Option(5, help="Retriever default top-k."),
                output_path: Optional[Path] = typer.Option(None, help="Persisted index path (not used for TF-IDF prototype).")):
    raw_docs = load_text_documents(input_dir)
    chunks = to_chunks(raw_docs)
    retriever = build_retriever(chunks, top_k=top_k)
    logging.info("Index ready with %s documents", len(retriever.documents))
    if logfire:
        logfire.info("indexing.build_index", documents=len(retriever.documents))
    typer.echo("Index built in memory. Persisting is not required for TF-IDF prototype.")


def get_retriever_from_dir(input_dir: Path = Path("inputs")) -> TfidfVectorStore:
    raw_docs = load_text_documents(input_dir)
    if not raw_docs:
        # Ensure inputs directory exists and seed a starter file so pipeline runs
        input_dir.mkdir(parents=True, exist_ok=True)
        starter = input_dir / "starter.txt"
        try:
            starter.write_text("SportSee RAG starter corpus for evaluation.", encoding="utf-8")
        except Exception:
            pass
        raw_docs = [RawDocument(path=starter, text=starter.read_text(encoding="utf-8"))]
    chunks = to_chunks(raw_docs)
    return build_retriever(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
