"""Run RAGAS evaluation against a dataset."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import run_rag_pipeline
from rag.llm import LLMConfig
from rag.models import AnswerPayload

try:  # pragma: no cover
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
except Exception:  # pragma: no cover
    evaluate = None
    answer_relevancy = context_precision = faithfulness = None

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


app = typer.Typer(add_completion=False)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def run_dataset(dataset_path: Path, output_dir: Path, llm_config: LLMConfig) -> Path:
    data = load_dataset(dataset_path)
    outputs: List[Dict[str, Any]] = []
    for item in data:
        answer: AnswerPayload = run_rag_pipeline(item["question"], llm_config=llm_config)
        outputs.append({
            "question": item["question"],
            "model_answer": answer.answer,
            "expected_answer": item.get("expected_answer"),
            "category": item.get("category"),
            "type": item.get("type"),
        })
    df = pd.DataFrame(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"ragas_results_{datetime.utcnow().isoformat()}.csv"
    df.to_csv(result_path, index=False)
    if evaluate and answer_relevancy:
        # Minimal evaluation example when ragas is installed
        try:
            evaluation = evaluate(
                dataset={
                    "question": df["question"].tolist(),
                    "answer": df["model_answer"].tolist(),
                    "contexts": [[] for _ in range(len(df))],
                    "ground_truth": df["expected_answer"].tolist(),
                },
                metrics=[answer_relevancy, context_precision, faithfulness],
            )
            metrics_df = evaluation.to_pandas()
            metrics_path = output_dir / f"ragas_metrics_{datetime.utcnow().isoformat()}.csv"
            metrics_df.to_csv(metrics_path, index=False)
        except Exception as exc:  # pragma: no cover
            logging.warning("RAGAS evaluation failed: %s", exc)
    if logfire:
        logfire.info("evaluation.run_dataset", dataset=str(dataset_path), rows=len(df))
    return result_path


@app.command()
def main(
    dataset: Path = typer.Option(Path(__file__).parent / "datasets" / "sample_questions.json"),
    output_dir: Path = typer.Option(Path(__file__).parent / "results"),
    model: str = typer.Option("mistral-small-latest"),
):
    logging.basicConfig(level=logging.INFO)
    llm_config = LLMConfig(model=model)
    result_path = run_dataset(dataset, output_dir, llm_config)
    typer.echo(f"Results written to {result_path}")


if __name__ == "__main__":
    app()
