# Troubleshooting

## RAGAS metrics are missing

If `ragas_metrics_<timestamp>.csv` is not written, check:

- `.env` exists and contains `MISTRAL_API_KEY`.
- `ragas` is installed (`pip show ragas`).

## Slow evaluation

Evaluation calls the live LLM for each question and for RAGAS metrics, so it can take minutes.

Ways to speed up:

- Reduce dataset size.
- Use a smaller model.
- Disable RAGAS metrics by unsetting API keys (custom retrieval metrics will still be written).

## Token limit / truncation

RAGAS scoring may raise token-limit related exceptions for some prompts. When that happens, you may still get partial results.

Mitigations:

- Reduce retrieved context size (smaller chunks, fewer chunks, or smaller input docs).
- Use a model with a larger context window.
