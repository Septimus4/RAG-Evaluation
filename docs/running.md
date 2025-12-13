# Running

## Run the pipeline (Python)

```python
from pathlib import Path
from rag.pipeline import run_rag_pipeline

answer = run_rag_pipeline("What does the starter note say?", data_dir=Path("inputs"))
print(answer.answer)
```

## Run the API

```bash
source .venv/bin/activate
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
```

Query:

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the starter note say?"}'
```
