"""Minimal REST API exposing the RAG pipeline.

Endpoints:
- GET /health
- POST /query

Run (from repo root):
  PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from db.sql_tool import SQLTool
from rag.llm import LLMConfig
from rag.pipeline import run_rag_pipeline


app = FastAPI(title="SportSee RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    used_sql: bool
    model: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    model = req.model or os.environ.get("RAG_MODEL") or "mistral-small-latest"
    llm_config = LLMConfig(model=model)

    database_url = os.environ.get("DATABASE_URL")
    sql_tool = SQLTool(database_url=database_url) if database_url else SQLTool()

    inputs_dir = Path(os.environ.get("RAG_INPUT_DIR", "inputs"))
    payload = run_rag_pipeline(req.question, data_dir=inputs_dir, sql_tool=sql_tool, llm_config=llm_config)

    return QueryResponse(answer=payload.answer, used_sql=payload.used_sql, model=payload.model)
