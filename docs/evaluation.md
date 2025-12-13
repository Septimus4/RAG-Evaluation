# Evaluation (RAGAS)

## Datasets

Datasets live in `src/evaluation/datasets/`:

- `sample_questions.json`
- `noisy_questions.json`
- `robustness_questions.json`

## Run evaluation

```bash
source .venv/bin/activate
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/sample_questions.json
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/noisy_questions.json
python -m evaluation.evaluate_ragas --dataset src/evaluation/datasets/robustness_questions.json
```

Outputs are written to `src/evaluation/results/`:

- `ragas_results_<timestamp>.csv` (per-question outputs)
- `rag_metrics_<timestamp>.csv` (custom retrieval/context metrics)
- `ragas_metrics_<timestamp>.csv` (RAGAS metrics; requires API key)
- `summary_<dataset>_<timestamp>.md` (dataset-specific summary)
- `last_run_summary.md` (overwritten each run; convenience pointer)

## Limited datasource notes

This repo uses a small local corpus by default (`inputs/`). With limited data, it is normal to see:

- Lower context precision (retrieved chunks may not match expected sources).
- Higher contamination rate (retrieved sources differ from expected sources).
- High variance across questions.

If you want to reduce retrieval load, lower `top_k` in `src/rag/retriever.py` (or add an env-driven override).

## About RAGAS warnings

You may see:

- `LLM returned 1 generations instead of requested 3...`

This is typically due to the provider/model returning a single generation even when the evaluator requests multiple samples. RAGAS falls back to 1 and continues.

If you see max-token truncation errors during RAGAS scoring, consider using a larger model/context window or reducing prompt/context size.
