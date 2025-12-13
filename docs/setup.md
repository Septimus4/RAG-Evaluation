# Setup

## Prerequisites

- Python 3.11+
- (Optional) A Mistral API key for LLM + RAGAS metrics

## Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-optional.txt
```

## Environment variables

Create a `.env` file at the repo root.

Minimum recommended keys:

- `MISTRAL_API_KEY=...`
- `DATABASE_URL=sqlite:///./sportsee.db`

Optional:

- `LOGFIRE_TOKEN=...`

Notes:

- The evaluation runner loads `.env` automatically.
- If `MISTRAL_API_KEY` is missing, the core pipeline may still work in mock/test contexts, but RAGAS LLM-based metrics will be skipped.
