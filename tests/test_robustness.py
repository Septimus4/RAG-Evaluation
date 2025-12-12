import json
from pathlib import Path

import pandas as pd

from rag.llm import LLMConfig
from rag.pipeline import run_rag_pipeline
from evaluation.evaluate_ragas import run_dataset


def test_pipeline_runs_with_empty_input_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INGEST_MAX_FILES", "5")
    monkeypatch.setenv("INGEST_MAX_BYTES", "5000000")
    empty_dir = tmp_path / "empty_inputs"
    empty_dir.mkdir()
    answer = run_rag_pipeline("What does the starter note say?", data_dir=empty_dir)
    assert answer.answer


def test_run_dataset_writes_context_count_metric(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INGEST_MAX_FILES", "5")
    monkeypatch.setenv("INGEST_MAX_BYTES", "5000000")

    dataset_path = tmp_path / "tiny_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "What does the starter note say?",
                    "expected_answer": "A brief instruction or note from starter.txt.",
                    "reference_context": ["inputs/starter.txt"],
                    "category": "simple",
                    "type": "txt",
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "results"
    output_dir.mkdir()

    _ = run_dataset(dataset_path, output_dir, LLMConfig(model="mistral-small-latest"))

    metrics_files = sorted(output_dir.glob("rag_metrics_*.csv"))
    assert metrics_files, "Expected a rag_metrics_*.csv file"
    metrics_df = pd.read_csv(metrics_files[0])
    assert "retrieved_context_count" in metrics_df.columns
