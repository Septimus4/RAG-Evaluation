# Auto-assessment — Evaluate LLM Performance

This is a self-assessment report for the SportSee RAG Evaluation Stack. It answers whether each requested element is implemented, where it lives in the repo, and how it works.

## Scope and assumptions

- Scope: current codebase state on branch codex/turn-rag-prototype-into-robust-system.
- Evaluation is run on a limited local corpus under [inputs/](inputs/), so retrieval/context-related metrics may be pessimistic compared to a richer datasource.

## Checklist (implementation status)

Legend: **Yes** = implemented and used; **Partial** = present but incomplete or not consistently enforced; **No** = missing.

| Requirement | Status | Where | How / short comment |
|---|---|---|---|
| The environment reproduction file contains only the dependencies actually used in the project. | Partial | [requirements.txt](requirements.txt), [requirements-optional.txt](requirements-optional.txt) | Core deps are in requirements.txt; optional/UI/legacy deps are separated. Some packages are “supporting” (e.g., tooling/tests) and may not be required for runtime-only installs. |
| The RAG setup script includes all necessary steps, clearly separated and modular, implemented as functions or classes. | Yes | [scripts/setup_system.py](scripts/setup_system.py) | Provides setup_database(), setup_retriever(), smoke_test() and a CLI main() that orchestrates them. |
| The script includes structured logging to trace main steps, errors, and performance. | Yes | [src/observability/logfire_setup.py](src/observability/logfire_setup.py), [scripts/setup_system.py](scripts/setup_system.py) | Uses Logfire spans/events via span()/info(); also bridges stdlib logging to Logfire for unified tracing. |
| Running the script produces an operational system in one run or through clearly defined functions. | Yes | [scripts/setup_system.py](scripts/setup_system.py) | Single entrypoint runs retriever setup and a smoke-test query; optional DB ingest step is included. |
| Evaluation and data-preparation scripts are readable and commented. | Partial | [src/evaluation/evaluate_ragas.py](src/evaluation/evaluate_ragas.py), [src/db/load_excel_to_db.py](src/db/load_excel_to_db.py), [scripts/setup_system.py](scripts/setup_system.py) | Docstrings and clear structure exist; inline comments are present but not extensive across all modules. |
| Query sets are varied. | Yes | [src/evaluation/datasets/](src/evaluation/datasets/) | Includes sample, noisy, and robustness datasets. |
| Query sets are realistic and cover simple, complex, and noisy cases. | Partial | [src/evaluation/datasets/](src/evaluation/datasets/) | Covers multiple categories/types; realism is limited by small corpus and illustrative ground truth. |
| The script uses RAGAS and Pydantic to automate evaluation and secure data flows. | Yes | [src/evaluation/evaluate_ragas.py](src/evaluation/evaluate_ragas.py), [src/rag/models.py](src/rag/models.py) | RAGAS compute path is enabled when dependencies + API key exist; Pydantic models validate pipeline payloads. |
| Validation with Pydantic controls input/output schema and response coherence. | Partial | [src/rag/models.py](src/rag/models.py), [src/api/app.py](src/api/app.py) | Input schemas (QueryRequest/QueryResponse, DocumentChunk, AnswerPayload) are validated. “Coherence” is not fully enforced (no automatic factual consistency checks beyond RAGAS faithfulness). |
| RAGAS metrics are justified. | Yes | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md) | Explains answer_relevancy, context_precision, faithfulness and their intent (answer quality, noise, hallucination proxy). |
| Test results are presented in a summary table and interpreted using expected thresholds or performance targets. | Yes | [src/evaluation/results/summary_sample_questions_2025-12-12T23:41:15.627731+00:00.md](src/evaluation/results/summary_sample_questions_2025-12-12T23:41:15.627731+00:00.md), [src/evaluation/results/summary_noisy_questions_2025-12-13T00:11:09.581161+00:00.md](src/evaluation/results/summary_noisy_questions_2025-12-13T00:11:09.581161+00:00.md), [src/evaluation/results/summary_robustness_questions_2025-12-13T00:21:05.495375+00:00.md](src/evaluation/results/summary_robustness_questions_2025-12-13T00:21:05.495375+00:00.md) | Each dataset summary includes mean/min/max tables and compares means against heuristic targets (targets are explicitly labeled as initial). |
| Robustness tests are documented. | Yes | [tests/test_robustness.py](tests/test_robustness.py), [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md), [src/evaluation/datasets/robustness_questions.json](src/evaluation/datasets/robustness_questions.json) | Documents robustness motivations and includes automated tests + a robustness dataset. |
| Robustness tests are justified based on business use cases. | Partial | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md) | Justification exists (messy inputs, guardrail bypass, partial data). Could be tightened with explicit SportSee user stories and failure costs. |
| Structured logging covers key steps and errors in the RAG/LLM pipeline. | Yes | [src/rag/pipeline.py](src/rag/pipeline.py), [src/observability/logfire_setup.py](src/observability/logfire_setup.py), [src/evaluation/evaluate_ragas.py](src/evaluation/evaluate_ragas.py) | Spans cover pipeline, retrieval, SQL routing, LLM generation, evaluation run boundaries. |
| The report explains the methodology. | Yes | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md), [docs/evaluation.md](docs/evaluation.md) | Methodology includes data sources, evaluation design, metrics, targets, and interpretation guidance. |
| Methodological choices are justified. | Partial | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md) | Justifies metric selection and robustness modes; could add deeper rationale for dataset construction and target thresholds. |
| Limits and potential biases of the evaluation are identified and analyzed. | Yes | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md), [docs/reports/rag_evaluation_metrics.md](docs/reports/rag_evaluation_metrics.md) | Notes limited corpus effects, proxy labels (sources as relevance), model variability; additional bias notes can be added as datasets evolve. |
| Results are interpreted in relation to business issues. | Partial | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md) | Provides business-aligned interpretation (wrong analytics decisions, hallucinated stats). Could be expanded into concrete KPIs/SLA and user impact. |
| Recommendations for improving the infrastructure are concrete and actionable. | Yes | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md) | Includes specific steps: embedding retriever, evidence packing/dedup, SQL hardening, dataset expansion, CI. |
| The report is clearly and professionally structured. | Yes | [REPORT_RAG_EVALUATION.md](REPORT_RAG_EVALUATION.md), [REPORT_RAGAS_EVALUATION_METRICS.md](REPORT_RAGAS_EVALUATION_METRICS.md) | Uses sections for methodology, metrics, results, limitations, recommendations. |
| The README clearly describes the target architecture and includes a diagram. | Yes | [README.md](README.md) | Architecture section + Mermaid diagram included. |
| The REST API is documented with endpoints, request/response formats, and example calls. | Yes | [src/api/app.py](src/api/app.py), [README.md](README.md), [docs/running.md](docs/running.md) | Documents GET /health and POST /query plus curl examples; request/response models are defined with Pydantic. |
| Scripts, files, and folders are organized and explained for easy onboarding. | Partial | [README.md](README.md), [docs/README.md](docs/README.md), [docs/architecture.md](docs/architecture.md) | Repo structure is explained; some legacy paths (e.g., indexer.py vs TF‑IDF path) may still confuse newcomers. |
| Deployment, execution, and evaluation procedures are detailed. | Yes | [README.md](README.md), [docs/setup.md](docs/setup.md), [docs/running.md](docs/running.md), [docs/evaluation.md](docs/evaluation.md) | Step-by-step commands included for setup, API run, and evaluation runs. |
| The procedure is fully reproducible. | Partial | [scripts/setup_system.py](scripts/setup_system.py), [docs/setup.md](docs/setup.md) | Reproducible given the same inputs and API access. External LLM calls introduce nondeterminism; CI pinning + caching could improve repeatability. |
| Accessibility for non-specialist technical teams is accounted for. | Partial | [docs/](docs/) | Docs focus on “how to run” and “what’s where”; could add glossary, simpler onboarding path, and a minimal “day-1 checklist”. |

## Methodology (summary)

- System under test: TF‑IDF retrieval + LLM generation + optional SQL tool routing.
  - Pipeline entrypoint: [src/rag/pipeline.py](src/rag/pipeline.py)
  - Retrieval/indexing: [src/data_pipeline/indexing.py](src/data_pipeline/indexing.py), [src/rag/retriever.py](src/rag/retriever.py)
  - Schema validation: Pydantic models in [src/rag/models.py](src/rag/models.py)
- Evaluation method:
  - Run dataset questions through the pipeline and record per-question outputs.
  - Compute retrieval-oriented metrics (recall@5, precision@5, MRR, nDCG) and context proxies.
  - Compute RAGAS metrics when API key is available.
  - Aggregated view: [REPORT_RAGAS_EVALUATION_METRICS.md](REPORT_RAGAS_EVALUATION_METRICS.md)

## Limits and potential biases (practical)

- Limited corpus in [inputs/](inputs/) makes context_precision / contamination metrics volatile.
- Using “expected source file paths” as relevance proxies does not guarantee passage-level evidence.
- Live model calls (Mistral) can vary between runs; RAGAS warning about “1 generation instead of 3” reduces sampling and can increase noise.
- Some RAGAS scoring can hit token limits on certain prompts (seen during robustness scoring), producing partial/noisy metric estimates.

## Recommendations (next steps)

- Make evaluation more deterministic: add a mock/scoring mode or fixed-seed sampling where possible.
- Add an env-driven retrieval limit: e.g., RAG_TOP_K to scale down retrieval for limited datasources and speed up evaluation.
- Replace TF‑IDF with an embedding-based retriever for production and re-run the same datasets to quantify gains.
- Expand query sets with real SportSee user queries and verified ground truth; version datasets to track regressions.
- Add a short onboarding “for non-specialists” guide (glossary + minimal runbook).
