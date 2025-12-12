# RAG Evaluation — Last Run Summary

- Dataset: `src/evaluation/datasets/sample_questions.json`
- Model: `mistral-small-latest`
- Per-question results CSV: `src/evaluation/results/ragas_results_2025-12-12T17:45:30.666124+00:00.csv`
- Per-question metrics CSV: `src/evaluation/results/rag_metrics_2025-12-12T17:45:30.668209+00:00.csv`
- RAGAS metrics CSV: `src/evaluation/results/ragas_metrics_2025-12-12T17:49:36.277045+00:00.csv`

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
| retrieval_latency_ms | 1.2186 | 0.8000 | 5.1600 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |

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
| answer_length | 342.6818 | 96.0000 | 1639.0000 |
| response_latency_ms | - | - | - |

## RAGAS Metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.3458 | 0.0000 | 1.0000 |
| context_precision | 0.0000 | 0.0000 | 0.0000 |
| faithfulness | 0.1786 | 0.0000 | 1.0000 |

