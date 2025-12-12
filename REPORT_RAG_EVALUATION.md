# RAG System Setup and Evaluation Report

## Introduction
This report summarizes the refactored SportSee RAG assistant that combines unstructured match commentary with structured statistics. The goal is to enable traceable question answering, cover numeric/statistical intents via SQL, and measure quality with RAGAS.

## Methodology
- **Data sources:** text archives placed under `inputs/` (ingested as `.txt`) and Excel exports for players, matches, stats, and reports.
- **Pipeline configuration:** TF-IDF retriever for prototypes, optional Mistral LLM client (mocked when no API key), and Pydantic Logfire hooks across ingestion, retrieval, LLM, SQL tool, and evaluation.
- **Database schema:** tables `players`, `matches`, `stats`, and `reports` defined in `src/db/schema.py` with Pydantic validation models in `src/db/models.py`.
- **Evaluation design:** starter dataset in `src/evaluation/datasets/sample_questions.json`; `evaluate_ragas.py` runs the pipeline per question and records outputs plus optional RAGAS metrics.

### Metrics chosen (and why)

- **Retrieval metrics (custom):** `recall@5`, `precision@5`, `mrr`, `ndcg` are used to quantify whether the retriever surfaces the expected sources early and consistently. These metrics map well to a business goal of “getting the right evidence into the context window”.
- **RAGAS metrics:**
	- `answer_relevancy`: checks if the answer addresses the question.
	- `faithfulness`: checks whether the answer is supported by the retrieved context (hallucination risk proxy).
	- `context_precision`: checks whether retrieved context is relevant (noise proxy).

### Targets (initial, heuristic)

These thresholds are **starter targets** to make results interpretable; they should be calibrated with business stakeholders and a realistic question set:

- Retrieval: `recall@5 ≥ 0.70`, `precision@5 ≥ 0.30`, `ndcg ≥ 0.60`
- RAGAS: `answer_relevancy ≥ 0.60`, `faithfulness ≥ 0.50`, `context_precision ≥ 0.30`

## Robustness tests

Robustness is evaluated by combining:

- **Noisy inputs:** typos, shorthand queries, and ambiguous requests (`src/evaluation/datasets/noisy_questions.json`).
- **Prompt-injection / policy pressure:** attempts to force the system to ignore context or reveal internal SQL (`src/evaluation/datasets/robustness_questions.json`).
- **Operational resilience:** empty/partial corpora and missing DB configuration (covered by tests and by pipeline fallbacks).

Rationale: these are common real-world failure modes for analytics assistants (messy user input, attempts to bypass guardrails, partial data availability).

## Results
- Baseline and enhanced evaluations should be written to `src/evaluation/results/` via `python -m evaluation.evaluate_ragas`. Metrics are recorded when RAGAS is available; otherwise, the script still captures model answers for later comparison.
- SQL tool execution is logged (when Logfire is configured) to differentiate structured answers from text-only retrievals.

### How to read the results

- The per-run markdown summary at `src/evaluation/results/last_run_summary.md` includes:
	- dataset distribution (categories/types)
	- summary tables (mean/min/max)
	- a quick comparison against the heuristic targets above

## Discussion
- **Structured data benefits:** numeric queries (averages, best/worst, rebounds/points) route to the SQL tool to avoid hallucinating stats. The tool enforces row limits and parameterized queries to keep execution safe.
- **Limitations:** TF-IDF retrieval is non-semantic and should be replaced with production-grade embeddings. SQL generation currently uses conservative templates and does not attempt complex joins; additional few-shot examples and schema hints will improve coverage. Evaluation dataset is illustrative and must be expanded with SportSee-specific ground truth to measure progress.
- **Robustness:** Pydantic models surface validation errors early during ingestion, but NL→SQL interpretation can still mis-handle ambiguous time ranges or misspellings. Mixed questions (stats + narrative) require both SQL and RAG context; orchestration logic is present but should be hardened with more guardrails and tests.

### Limits and biases

- **Synthetic/placeholder ground truth:** some expected answers are illustrative; this can bias metrics and makes absolute scores less meaningful.
- **Retriever mismatch:** the prototype TF-IDF retriever is sensitive to wording and may underperform on paraphrases compared to embedding retrieval.
- **Context-source labeling:** evaluation currently checks expected *sources* (file paths), which is a proxy for evidence correctness; it does not guarantee that the exact supporting passage was retrieved.
- **Model variability:** if an API key is provided, the live model introduces non-determinism; without a key, mock answers do not reflect real generation quality.

### Business interpretation

- **If retrieval metrics are below targets:** users will see answers grounded in weak evidence, increasing the risk of wrong analytics decisions.
- **If faithfulness is low:** hallucinated numbers/claims become likely, which is unacceptable for sports analytics dashboards and reporting.
- **If context_precision is low:** the system wastes context window space on irrelevant chunks, harming both quality and latency.

### Recommendations (actionable)

1. Replace TF-IDF with an embedding retriever (FAISS/Chroma) and evaluate again on the same datasets to quantify improvements.
2. Improve evidence packing: pass only the most relevant chunks and deduplicate sources to increase context precision.
3. Harden SQL tool: implement a stricter intent parser (and/or schema-aware prompt) so numeric questions reliably use structured data.
4. Expand evaluation datasets with real SportSee questions + verified ground truth; version datasets and track regressions over time.
5. Add CI to run `pytest` and a lightweight evaluation smoke test on each change.

## Conclusion
The repository now contains a reproducible skeleton for a traceable SportSee RAG assistant. Key next steps are to replace the prototype TF-IDF index with the chosen vector store, enrich the NL→SQL prompt with realistic examples, and grow the evaluation dataset so before/after metrics capture the impact of structured data integration.
