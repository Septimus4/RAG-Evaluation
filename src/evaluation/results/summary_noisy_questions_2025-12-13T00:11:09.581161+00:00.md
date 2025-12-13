# RAG Evaluation — Last Run Summary

- Dataset: `src/evaluation/datasets/noisy_questions.json`
- Model: `mistral-small-latest`
- Per-question results CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/ragas_results_2025-12-13T00:11:09.581161+00:00.csv`
- Per-question metrics CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/rag_metrics_2025-12-13T00:11:09.581161+00:00.csv`
- RAGAS metrics CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/ragas_metrics_2025-12-13T00:11:09.581161+00:00.csv`

## Dataset Overview

This run contains **5** questions.

Category counts:

| category | count |
|---|---:|
| noisy | 4 |
| robustness | 1 |

Type counts:

| type | count |
|---|---:|
| sql-stats | 3 |
| pdf-text | 1 |
| hybrid | 1 |

## Retrieval Metrics

The table below summarizes retrieval-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| recall@5 | 0.7000 | 0.0000 | 1.0000 |
| precision@5 | 0.4000 | 0.0000 | 0.5000 |
| mrr | 0.8000 | 0.0000 | 1.0000 |
| ndcg | 0.7226 | 0.0000 | 1.0000 |
| retrieval_latency_ms | 1.1480 | 0.9400 | 1.4300 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |
| retrieved_context_count | 4.0000 | 4.0000 | 4.0000 |

## Context Metrics

The table below summarizes context-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| context_window_utilization_chars | 1317.0000 | 1317.0000 | 1317.0000 |
| context_contamination_rate | 0.6333 | 0.5000 | 1.0000 |
| deduplication_rate | 0.5000 | 0.5000 | 0.5000 |

## Generation Metrics

The table below summarizes generation-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_length | 315.6000 | 105.0000 | 726.0000 |
| response_latency_ms | - | - | - |

## Targets & Interpretation

The targets below are **initial, heuristic thresholds** to help interpret results.
They should be adjusted once you define business KPIs and gather more realistic queries.

Retrieval targets (based on `rag_metrics_*.csv` means):

- recall@5: mean=0.7000 (target≥0.70) → OK
- precision@5: mean=0.4000 (target≥0.30) → OK
- ndcg: mean=0.7226 (target≥0.60) → OK

## RAGAS Metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.4142 | 0.0000 | 0.7777 |
| context_precision | 0.5500 | 0.0000 | 1.0000 |
| faithfulness | 0.8882 | 0.5000 | 1.0000 |

RAGAS targets (based on `ragas_metrics_*.csv` means):

- answer_relevancy: mean=0.4142 (target≥0.60) → Needs work
- faithfulness: mean=0.8882 (target≥0.50) → OK
- context_precision: mean=0.5500 (target≥0.30) → OK
