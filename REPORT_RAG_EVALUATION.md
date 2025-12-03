# RAG System Setup and Evaluation Report

## Introduction
This report summarizes the refactored SportSee RAG assistant that combines unstructured match commentary with structured statistics. The goal is to enable traceable question answering, cover numeric/statistical intents via SQL, and measure quality with RAGAS.

## Methodology
- **Data sources:** text archives placed under `inputs/` (ingested as `.txt`) and Excel exports for players, matches, stats, and reports.
- **Pipeline configuration:** TF-IDF retriever for prototypes, optional Mistral LLM client (mocked when no API key), and Pydantic Logfire hooks across ingestion, retrieval, LLM, SQL tool, and evaluation.
- **Database schema:** tables `players`, `matches`, `stats`, and `reports` defined in `src/db/schema.py` with Pydantic validation models in `src/db/models.py`.
- **Evaluation design:** starter dataset in `src/evaluation/datasets/sample_questions.json`; `evaluate_ragas.py` runs the pipeline per question and records outputs plus optional RAGAS metrics.

## Results
- Baseline and enhanced evaluations should be written to `src/evaluation/results/` via `python -m evaluation.evaluate_ragas`. Metrics are recorded when RAGAS is available; otherwise, the script still captures model answers for later comparison.
- SQL tool execution is logged (when Logfire is configured) to differentiate structured answers from text-only retrievals.

## Discussion
- **Structured data benefits:** numeric queries (averages, best/worst, rebounds/points) route to the SQL tool to avoid hallucinating stats. The tool enforces row limits and parameterized queries to keep execution safe.
- **Limitations:** TF-IDF retrieval is non-semantic and should be replaced with production-grade embeddings. SQL generation currently uses conservative templates and does not attempt complex joins; additional few-shot examples and schema hints will improve coverage. Evaluation dataset is illustrative and must be expanded with SportSee-specific ground truth to measure progress.
- **Robustness:** Pydantic models surface validation errors early during ingestion, but NL→SQL interpretation can still mis-handle ambiguous time ranges or misspellings. Mixed questions (stats + narrative) require both SQL and RAG context; orchestration logic is present but should be hardened with more guardrails and tests.

## Conclusion
The repository now contains a reproducible skeleton for a traceable SportSee RAG assistant. Key next steps are to replace the prototype TF-IDF index with the chosen vector store, enrich the NL→SQL prompt with realistic examples, and grow the evaluation dataset so before/after metrics capture the impact of structured data integration.
