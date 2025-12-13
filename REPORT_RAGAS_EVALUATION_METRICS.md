# RAGAS Evaluation — Key Metrics

This report aggregates the latest evaluation run for each dataset.
Artifacts are under `src/evaluation/results/`.

## Datasets included

| dataset | timestamp | rag_metrics | ragas_metrics |
|---|---|---|---|
| noisy_questions | 2025-12-13T00:11:09.581161+00:00 | rag_metrics_2025-12-13T00:11:09.581161+00:00.csv | ragas_metrics_2025-12-13T00:11:09.581161+00:00.csv |
| robustness_questions | 2025-12-13T00:21:05.495375+00:00 | rag_metrics_2025-12-13T00:21:05.495375+00:00.csv | ragas_metrics_2025-12-13T00:21:05.495375+00:00.csv |
| sample_questions | 2025-12-12T23:41:15.627731+00:00 | rag_metrics_2025-12-12T23:41:15.627731+00:00.csv | ragas_metrics_2025-12-12T23:41:15.627731+00:00.csv |

## noisy_questions

### Retrieval metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| recall@5 | 0.7000 | 0.0000 | 1.0000 |
| precision@5 | 0.4000 | 0.0000 | 0.5000 |
| mrr | 0.8000 | 0.0000 | 1.0000 |
| ndcg | 0.7226 | 0.0000 | 1.0000 |
| retrieval_latency_ms | 1.1480 | 0.9400 | 1.4300 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieved_context_count | 4.0000 | 4.0000 | 4.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |

### Context metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| context_window_utilization_chars | 1317.0000 | 1317.0000 | 1317.0000 |
| context_contamination_rate | 0.6333 | 0.5000 | 1.0000 |
| deduplication_rate | 0.5000 | 0.5000 | 0.5000 |

### RAGAS metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.4142 | 0.0000 | 0.7777 |
| context_precision | 0.5500 | 0.0000 | 1.0000 |
| faithfulness | 0.8882 | 0.5000 | 1.0000 |

## robustness_questions

### Retrieval metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| recall@5 | 0.7000 | 0.0000 | 1.0000 |
| precision@5 | 0.4000 | 0.0000 | 0.5000 |
| mrr | 0.6000 | 0.0000 | 1.0000 |
| ndcg | 0.5750 | 0.0000 | 1.0000 |
| retrieval_latency_ms | 1.0540 | 0.9200 | 1.2300 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieved_context_count | 4.0000 | 4.0000 | 4.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |

### Context metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| context_window_utilization_chars | 1317.0000 | 1317.0000 | 1317.0000 |
| context_contamination_rate | 0.6333 | 0.5000 | 1.0000 |
| deduplication_rate | 0.5000 | 0.5000 | 0.5000 |

### RAGAS metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.4647 | 0.1069 | 1.0000 |
| context_precision | 0.1000 | 0.0000 | 0.5000 |
| faithfulness | 0.7381 | 0.0000 | 1.0000 |

## sample_questions

### Retrieval metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| recall@5 | 0.6591 | 0.0000 | 1.0000 |
| precision@5 | 0.3636 | 0.0000 | 0.5000 |
| mrr | 0.5455 | 0.0000 | 1.0000 |
| ndcg | 0.5598 | 0.0000 | 1.0000 |
| retrieval_latency_ms | 1.0695 | 0.8000 | 2.7500 |
| retrieval_fail_rate | 0.0000 | 0.0000 | 0.0000 |
| retrieved_context_count | 4.0000 | 4.0000 | 4.0000 |
| retrieval_diversity_ratio | 0.5000 | 0.5000 | 0.5000 |

### Context metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| context_window_utilization_chars | 1317.0000 | 1317.0000 | 1317.0000 |
| context_contamination_rate | 0.6591 | 0.5000 | 1.0000 |
| deduplication_rate | 0.5000 | 0.5000 | 0.5000 |

### RAGAS metrics

| metric | mean | min | max |
|---|---:|---:|---:|
| answer_relevancy | 0.3792 | 0.0000 | 1.0000 |
| context_precision | 0.2311 | 0.0000 | 1.0000 |
| faithfulness | 0.7292 | 0.2000 | 1.0000 |

## Notes

- With a small/limited corpus in `inputs/`, context-related metrics can be noisy and may under-represent real performance.
- RAGAS may warn about generation counts (provider returns 1 generation). The evaluation continues with 1 and the scores remain usable.
