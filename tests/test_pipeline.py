from pathlib import Path
import os

from rag.pipeline import run_rag_pipeline
from rag.models import AnswerPayload
from db.sql_tool import SQLTool


def test_pipeline_returns_answer_text_only(monkeypatch):
    # Tighten ingestion caps for tests
    monkeypatch.setenv("INGEST_MAX_FILES", "10")
    monkeypatch.setenv("INGEST_MAX_BYTES", "20000000")
    answer = run_rag_pipeline("Test question", data_dir=Path("inputs"))
    assert isinstance(answer, AnswerPayload)
    assert answer.answer


def test_pipeline_routes_to_sql(monkeypatch):
    monkeypatch.setenv("INGEST_MAX_FILES", "10")
    monkeypatch.setenv("INGEST_MAX_BYTES", "20000000")
    ans = run_rag_pipeline(
        "Average rebounds for player X in the last 5 games?",
        data_dir=Path("inputs"),
        sql_tool=SQLTool(),
    )
    assert isinstance(ans, AnswerPayload)
    assert ans.used_sql is True
