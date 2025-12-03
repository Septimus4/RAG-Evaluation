from pathlib import Path

from rag.pipeline import run_rag_pipeline
from rag.models import AnswerPayload


def test_pipeline_returns_answer():
    answer = run_rag_pipeline("Test question", data_dir=Path("inputs"))
    assert isinstance(answer, AnswerPayload)
    assert answer.answer
