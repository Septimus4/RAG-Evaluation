# SportSee RAG Evaluation Stack

This repository contains a modular Retrieval Augmented Generation (RAG) assistant for SportSee basketball analytics. The project demonstrates how to combine unstructured match reports with structured match statistics in SQLite, instrument the pipeline with Pydantic Logfire, and evaluate quality with RAGAS.

## Architecture

- **Data pipeline (`src/data_pipeline`)** – text ingestion, chunking, and TF-IDF indexing used for the prototype retriever.
- **RAG core (`src/rag`)** – Pydantic models, vector store wrapper, retriever helpers, LLM client, and the `run_rag_pipeline` orchestration entrypoint.
- **Structured data (`src/db`)** – SQLAlchemy schema for players, matches, stats, and reports; Pydantic validators; Excel ingestion CLI; SQL tool for validated NL→SQL queries.
- **Evaluation (`src/evaluation`)** – sample dataset and `evaluate_ragas.py` script for running the pipeline and persisting metrics.
- **UI prototype** – the original Streamlit app (`MistralChat.py`) remains available for reference.
- **Observability** – optional Pydantic Logfire hooks across ingestion, retrieval, LLM, SQL tool, and evaluation steps.

### Diagram

```mermaid
flowchart LR
   Q[User / Dataset Question] --> API[REST API /query]
   API --> PIPE[run_rag_pipeline]
   subgraph RAG[RAG Pipeline]
      PIPE --> RET[Retriever (TF-IDF prototype)]
      RET --> CTX[Retrieved Context Chunks]
      PIPE --> LLM[LLM (Mistral or mock)]
      PIPE --> SQL[SQL Tool]
   end
   SQL --> DB[(SQLite / SQLAlchemy)]
   CTX --> LLM
   PIPE --> OUT[AnswerPayload]
   subgraph OBS[Observability]
      LOG[Logfire spans + logs]
   end
   PIPE -.-> LOG
   RET -.-> LOG
   SQL -.-> LOG
   LLM -.-> LOG
   subgraph EVAL[Evaluation]
      EV[python -m evaluation.evaluate_ragas] --> CSV[(results/*.csv + last_run_summary.md)]
   end
   Q --> EV
```

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

Optional extras (UI prototype + legacy FAISS indexer + OCR fallbacks):

```bash
pip install -r requirements-optional.txt
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

To run additional datasets (noisy/robustness):

```bash
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/noisy_questions.json
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/robustness_questions.json
```

## Documentation

- Project docs live in [docs/README.md](docs/README.md).
- Aggregated evaluation metrics report: [REPORT_RAGAS_EVALUATION_METRICS.md](REPORT_RAGAS_EVALUATION_METRICS.md).

## REST API

A minimal API is provided for integration testing and demonstrations.

Run locally:

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` → liveness check
- `POST /query` → run the RAG pipeline

Example calls:

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/query \
   -H 'Content-Type: application/json' \
   -d '{"question":"What does the starter note say?"}'
```

## Limitations and future work

- The TF-IDF vector store is a lightweight placeholder; swap for a persistent FAISS/Chroma index for production.
- SQL generation is intentionally conservative and should be expanded with richer NL→SQL prompting and guardrails.
- Evaluation datasets are illustrative; populate them with real SportSee questions and ground truth to obtain meaningful metrics.
- Logfire hooks are optional and only active when the dependency and token are provided.
