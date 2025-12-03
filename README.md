# SportSee RAG Evaluation Stack

This repository contains a modular Retrieval Augmented Generation (RAG) assistant for SportSee basketball analytics. The project demonstrates how to combine unstructured match reports with structured match statistics in SQLite, instrument the pipeline with Pydantic Logfire, and evaluate quality with RAGAS.

## Architecture

- **Data pipeline (`src/data_pipeline`)** – text ingestion, chunking, and TF-IDF indexing used for the prototype retriever.
- **RAG core (`src/rag`)** – Pydantic models, vector store wrapper, retriever helpers, LLM client, and the `run_rag_pipeline` orchestration entrypoint.
- **Structured data (`src/db`)** – SQLAlchemy schema for players, matches, stats, and reports; Pydantic validators; Excel ingestion CLI; SQL tool for validated NL→SQL queries.
- **Evaluation (`src/evaluation`)** – sample dataset and `evaluate_ragas.py` script for running the pipeline and persisting metrics.
- **UI prototype** – the original Streamlit app (`MistralChat.py`) remains available for reference.
- **Observability** – optional Pydantic Logfire hooks across ingestion, retrieval, LLM, SQL tool, and evaluation steps.

## Setup

1. **Prerequisites**
   - Python 3.11+
   - SQLite (default) or PostgreSQL URL provided via `DATABASE_URL`.
2. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configuration**
   Copy `.env.example` to `.env` and fill in:
   - `MISTRAL_API_KEY` (or leave empty to use mock LLM responses)
   - `DATABASE_URL`
   - `LOGFIRE_TOKEN` if Logfire is available

## Data preparation

The structured schema lives in `src/db/schema.py`. To validate and ingest Excel exports into SQLite:

```bash
python -m db.load_excel_to_db \
  --players-path /path/to/players.xlsx \
  --matches-path /path/to/matches.xlsx \
  --stats-path /path/to/stats.xlsx \
  --reports-path /path/to/reports.xlsx \
  --database-url sqlite:///./sportsee.db
```

Use `--dry-run` to validate without writing data. After ingestion, inspect tables with any SQLite browser or `sqlite3 sportsee.db "SELECT COUNT(*) FROM players;"`.

## Running the assistant

The `run_rag_pipeline` function is the canonical entrypoint for the instrumented pipeline:

```python
from pathlib import Path
from rag.pipeline import run_rag_pipeline
from db.sql_tool import SQLTool

answer = run_rag_pipeline(
    "Who led the rebounds last night?",
    data_dir=Path("inputs"),
    sql_tool=SQLTool(database_url="sqlite:///./sportsee.db"),
)
print(answer.answer)
```

- Purely textual questions rely on the retriever + LLM.
- Numeric/statistical questions trigger the SQL tool and return structured summaries.

## Evaluation

A starter dataset lives at `src/evaluation/datasets/sample_questions.json`. Run the evaluation harness:

```bash
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/sample_questions.json
```

The script saves per-question outputs to `src/evaluation/results/` and writes RAGAS metrics when the dependency is available.

## Limitations and future work

- The TF-IDF vector store is a lightweight placeholder; swap for a persistent FAISS/Chroma index for production.
- SQL generation is intentionally conservative and should be expanded with richer NL→SQL prompting and guardrails.
- Evaluation datasets are illustrative; populate them with real SportSee questions and ground truth to obtain meaningful metrics.
- Logfire hooks are optional and only active when the dependency and token are provided.
