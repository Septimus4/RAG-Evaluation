# RAG Evaluation — Last Run Summary

- Dataset: `src/evaluation/datasets/sample_questions.json`
- Model: `mistral-small-latest`
- Per-question results CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/ragas_results_2025-12-12T23:41:15.627731+00:00.csv`
- Per-question metrics CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/rag_metrics_2025-12-12T23:41:15.627731+00:00.csv`
- RAGAS metrics CSV: `/workspaces/RAG-Evaluation/src/evaluation/results/ragas_metrics_2025-12-12T23:41:15.627731+00:00.csv`

## Dataset Overview

This run contains **22** questions.

Category counts:

| category | count |
|---|---:|
| numeric | 10 |
| factual | 3 |
| reasoning | 3 |
| multi-hop | 2 |
| paraphrase | 2 |
| summarization | 1 |
| simple | 1 |

Type counts:

| type | count |
|---|---:|
| sql-stats | 12 |
| pdf-text | 6 |
| hybrid | 3 |
| txt | 1 |

## Retrieval Metrics

The table below summarizes retrieval-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| recall@5 | 0.6591 | 0.0000 | 1.0000 |
| precision@5 | 0.3636 | 0.0000 | 0.5000 |
| mrr | 0.5455 | 0.0000 | 1.0000 |
| ndcg | 0.5598 | 0.0000 | 1.0000 |
| retrieval_latency_ms | 1.0695 | 0.8000 | 2.7500 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |
| retrieved_context_count | 4.0000 | 4.0000 | 4.0000 |

## Context Metrics

The table below summarizes context-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| context_window_utilization_chars | 1317.0000 | 1317.0000 | 1317.0000 |
| context_contamination_rate | 0.6591 | 0.5000 | 1.0000 |
| deduplication_rate | 0.5000 | 0.5000 | 0.5000 |

## Generation Metrics

The table below summarizes generation-related metrics across questions.

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_length | 320.9091 | 96.0000 | 1353.0000 |
| response_latency_ms | - | - | - |

## Targets & Interpretation

The targets below are **initial, heuristic thresholds** to help interpret results.
They should be adjusted once you define business KPIs and gather more realistic queries.

Retrieval targets (based on `rag_metrics_*.csv` means):

- recall@5: mean=0.6591 (target≥0.70) → Needs work
- precision@5: mean=0.3636 (target≥0.30) → OK
- ndcg: mean=0.5598 (target≥0.60) → Needs work

## RAGAS Metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.3792 | 0.0000 | 1.0000 |
| context_precision | 0.2311 | 0.0000 | 1.0000 |
| faithfulness | 0.7292 | 0.2000 | 1.0000 |

RAGAS targets (based on `ragas_metrics_*.csv` means):

- answer_relevancy: mean=0.3792 (target≥0.60) → Needs work
- faithfulness: mean=0.7292 (target≥0.50) → OK
- context_precision: mean=0.2311 (target≥0.30) → Needs work
