# Architecture

## High-level components

- `src/data_pipeline/`
  - Document ingestion and chunking.
  - TF‑IDF indexing used by the prototype retriever.

- `src/rag/`
  - Pipeline entrypoint: `run_rag_pipeline`.
  - Retriever wrapper and vector store implementation.
  - LLM client/config.

- `src/db/`
  - SQLAlchemy models/schema.
  - SQL tool for conservative NL→SQL.

- `src/evaluation/`
  - Datasets and RAGAS evaluation harness.
  - Results stored under `src/evaluation/results/`.

- `src/api/`
  - Minimal API wrapper (FastAPI).

## Data flow

1. Ingestion loads documents from `inputs/`.
2. Chunking creates overlapping `DocumentChunk` items.
3. TF‑IDF store indexes chunks in memory.
4. For each query:
   - Retrieve top-k chunks (default 5).
   - Either:
     - Use LLM with retrieved context (text questions), or
     - Use SQL tool for numeric/stat/statistics questions.
5. Evaluation calls the same pipeline and writes CSV + summaries.
