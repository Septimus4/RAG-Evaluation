"""Run RAGAS evaluation against a dataset."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rag.llm import LLMConfig
from rag.models import AnswerPayload
from rag.pipeline import run_rag_pipeline
from rag.retriever import retrieve
from data_pipeline.indexing import get_retriever_from_dir

from observability.logfire_setup import configure_logfire, info, span

try:  # pragma: no cover
    from ragas import evaluate
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.embeddings import (
        BaseRagasEmbedding,
        BaseRagasEmbeddings,
        HuggingfaceEmbeddings as LegacyHuggingfaceEmbeddings,
    )
    from ragas.embeddings.huggingface_provider import (
        HuggingFaceEmbeddings as ModernHuggingFaceEmbeddings,
    )
    from ragas.run_config import RunConfig
except Exception:  # pragma: no cover
    evaluate = None
    answer_relevancy = context_precision = faithfulness = None
    BaseRagasLLM = None  # type: ignore
    BaseRagasEmbeddings = None  # type: ignore
    BaseRagasEmbedding = None  # type: ignore
    LegacyHuggingfaceEmbeddings = None  # type: ignore
    ModernHuggingFaceEmbeddings = None  # type: ignore
    RunConfig = None  # type: ignore


if BaseRagasEmbeddings is not None and BaseRagasEmbedding is not None:
    class _ModernEmbeddingAdapter(BaseRagasEmbeddings):
        """Bridge modern RAGAS embeddings into the legacy interface."""

        def __init__(self, delegate: BaseRagasEmbedding):
            super().__init__()
            self._delegate = delegate
            if RunConfig is not None:
                self.set_run_config(RunConfig())

        def embed_query(self, text: str) -> List[float]:
            return self._delegate.embed_text(text)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._delegate.embed_texts(texts)

        async def aembed_query(self, text: str) -> List[float]:
            return await self._delegate.aembed_text(text)

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            return await self._delegate.aembed_texts(texts)


app = typer.Typer(add_completion=False)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _recall_precision_k(retrieved: list[str], expected: list[str], k: int = 5) -> tuple[float, float]:
    if not expected:
        return 0.0, 0.0
    exp = set(expected)
    topk = _dedupe_preserve_order(retrieved)[:k]
    hit = len(set(topk) & exp)
    recall = hit / max(len(exp), 1)
    precision = hit / max(len(topk), 1)
    return recall, precision


def _mrr_ndcg(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    if not expected or not retrieved:
        return 0.0, 0.0
    exp = set(expected)
    retrieved = _dedupe_preserve_order(retrieved)
    # MRR: first relevant rank reciprocal
    mrr = 0.0
    for i, s in enumerate(retrieved, start=1):
        if s in exp:
            mrr = 1.0 / i
            break
    # nDCG: binary relevance
    import math
    dcg = 0.0
    idcg = 0.0
    for i, s in enumerate(retrieved, start=1):
        rel = 1.0 if s in exp else 0.0
        if rel:
            dcg += 1.0 / math.log2(i + 1)
    # ideal: all relevant first
    for i in range(1, min(len(exp), len(retrieved)) + 1):
        idcg += 1.0 / math.log2(i + 1)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return mrr, ndcg


def run_dataset(dataset_path: Path, output_dir: Path, llm_config: LLMConfig) -> Path:
    data = load_dataset(dataset_path)
    outputs: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    # Prepare retriever for retrieval-only metrics
    retriever = get_retriever_from_dir(Path("inputs"))
    with span("evaluation.run_dataset", dataset=str(dataset_path), model=llm_config.model, rows=len(data)):
        for item in data:
            # Retrieve for retrieval metrics
            retrieval = retrieve(item["question"], retriever)
            retrieved_sources = [doc.source for doc in retrieval.documents]
            expected_sources = item.get("reference_context") or []
            r_at_k, p_at_k = _recall_precision_k(retrieved_sources, expected_sources, k=5)
            mrr, ndcg = _mrr_ndcg(retrieved_sources, expected_sources)
            # Diversity vs redundancy: unique sources ratio
            unique_ratio = (len(set(retrieved_sources)) / max(len(retrieved_sources), 1)) if retrieved_sources else 0.0
            retrieval_fail = 1.0 if not retrieved_sources else 0.0
            # Context integrity approximations
            ctx_size = sum(len(doc.text) for doc in retrieval.documents)
            # Simple contamination proxy: 1 - jaccard between retrieved sources and expected sources
            contamination = 1.0 - _jaccard(set(retrieved_sources), set(expected_sources))
            dedup_rate = 1.0 - unique_ratio
            # Run full pipeline for generation metrics
            answer: AnswerPayload = run_rag_pipeline(item["question"], llm_config=llm_config)
            outputs.append({
                "question": item["question"],
                "model_answer": answer.answer,
                "expected_answer": item.get("expected_answer"),
                "category": item.get("category"),
                "type": item.get("type"),
            })
            metrics.append({
                "question": item["question"],
                # Retrieval Quality
                "recall@5": round(r_at_k, 4),
                "precision@5": round(p_at_k, 4),
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
                "retrieval_latency_ms": round(retrieval.latency_ms or 0.0, 2),
                "retrieval_fail_rate": retrieval_fail,
                "retrieval_diversity_ratio": round(unique_ratio, 4),
                # Context Integrity
                "context_window_utilization_chars": ctx_size,
                "context_contamination_rate": round(contamination, 4),
                "deduplication_rate": round(dedup_rate, 4),
                # Generation Quality (partial)
                "answer_length": len(answer.answer or ""),
                "response_latency_ms": None,  # available inside pipeline logs but not returned; keep placeholder
            })
    df = pd.DataFrame(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"ragas_results_{datetime.now(timezone.utc).isoformat()}.csv"
    df.to_csv(result_path, index=False)
    # Write custom metrics report
    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / f"rag_metrics_{datetime.now(timezone.utc).isoformat()}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    ragas_metrics_path: Optional[Path] = None
    if evaluate and answer_relevancy and Dataset:
        # Minimal evaluation example when ragas is installed
        try:
            ragas_llm = _build_ragas_llm()
            ragas_embeddings = _build_ragas_embeddings()
            if ragas_llm is None or ragas_embeddings is None:
                logging.info("Skipping RAGAS metrics (missing API key or embeddings provider).")
            else:
                hf_dataset = Dataset.from_dict({
                    "question": df["question"].tolist(),
                    "answer": df["model_answer"].tolist(),
                    "contexts": [[] for _ in range(len(df))],
                    "ground_truth": df["expected_answer"].fillna("").tolist(),
                })
                evaluation = evaluate(
                    hf_dataset,
                    metrics=[answer_relevancy, context_precision, faithfulness],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                if hasattr(evaluation, "to_pandas"):
                    ragas_df = evaluation.to_pandas()
                else:  # pragma: no cover - new ragas API surface
                    ragas_df = evaluation.to_dataframe()
                ragas_metrics_path = output_dir / f"ragas_metrics_{datetime.now(timezone.utc).isoformat()}.csv"
                ragas_df.to_csv(ragas_metrics_path, index=False)
        except Exception as exc:  # pragma: no cover
            logging.warning("RAGAS evaluation failed: %s", exc)

    report_path = output_dir / "last_run_summary.md"
    try:
        write_markdown_summary(
            result_path=result_path,
            metrics_path=metrics_path,
            ragas_metrics_path=ragas_metrics_path,
            dataset_path=dataset_path,
            model=llm_config.model,
            output_path=report_path,
        )
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to write markdown summary: %s", exc)

    info("evaluation.completed", dataset=str(dataset_path), rows=len(df), model=llm_config.model)
    return result_path


@app.command()
def main(
    dataset: Path = typer.Option(Path(__file__).parent / "datasets" / "sample_questions.json"),
    output_dir: Path = typer.Option(Path(__file__).parent / "results"),
    model: str = typer.Option("mistral-small-latest"),
):
    logging.basicConfig(level=logging.INFO)
    configure_logfire()
    llm_config = LLMConfig(model=model)
    result_path = run_dataset(dataset, output_dir, llm_config)
    typer.echo(f"Results written to {result_path}")
    typer.echo(f"Summary written to {output_dir / 'last_run_summary.md'}")


def _build_ragas_llm() -> Optional["BaseRagasLLM"]:
    if BaseRagasLLM is None:
        return None
    # Keep evaluation behavior consistent with the main pipeline: allow `.env` keys.
    try:
        from dotenv import load_dotenv

        dotenv_path = ROOT / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()
    except Exception:
        pass
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
    model_name = os.environ.get("RAGAS_MODEL", "mistral-small-latest")
    try:
        from openai import OpenAI  # type: ignore
        from ragas.llms import llm_factory

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        return llm_factory(model_name, client=client)
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to initialise RAGAS LLM client: %s", exc)
        return None


def _build_ragas_embeddings() -> Optional["BaseRagasEmbeddings"]:
    if BaseRagasEmbeddings is None:
        return None

    _ensure_metadata_pathfinder_invalidate_caches_is_classmethod()

    model_name = os.environ.get("RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        if LegacyHuggingfaceEmbeddings is not None:
            return LegacyHuggingfaceEmbeddings(model_name=model_name)
    except Exception:
        # Fall back to modern embedding providers below
        pass
    if BaseRagasEmbedding is None or ModernHuggingFaceEmbeddings is None:
        return None
    try:
        delegate = ModernHuggingFaceEmbeddings(model=model_name)
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to initialise HuggingFace embeddings for RAGAS: %s", exc)
        return None
    if isinstance(delegate, BaseRagasEmbeddings):
        return delegate
    if BaseRagasEmbeddings is not None and BaseRagasEmbedding is not None:
        return _ModernEmbeddingAdapter(delegate)
    return None


def _ensure_metadata_pathfinder_invalidate_caches_is_classmethod() -> None:
    """Work around a Python 3.12 build where MetadataPathFinder.invalidate_caches is not a @classmethod.

    Some dependency stacks (notably HuggingFace) call `MetadataPathFinder.invalidate_caches()` with no args.
    If the method isn't a classmethod, that raises:
    `TypeError: ... missing 1 required positional argument: 'cls'`.

    This patch is safe and local to the current interpreter process.
    """

    try:
        from importlib.metadata import MetadataPathFinder
    except Exception:  # pragma: no cover
        return
    raw = getattr(MetadataPathFinder, "__dict__", {}).get("invalidate_caches")
    if raw is None:
        return
    if isinstance(raw, classmethod):
        return
    try:
        MetadataPathFinder.invalidate_caches = classmethod(raw)  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        return


def write_markdown_summary(
    *,
    result_path: Path,
    metrics_path: Path,
    ragas_metrics_path: Optional[Path],
    dataset_path: Path,
    model: str,
    output_path: Path,
) -> None:
    results_df = pd.read_csv(result_path)
    metrics_df = pd.read_csv(metrics_path)

    def _fmt_num(v: float) -> str:
        try:
            return f"{float(v):.4f}"
        except Exception:
            return "-"

    def _describe_numeric(series: pd.Series) -> Dict[str, str]:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return {"mean": "-", "min": "-", "max": "-"}
        return {
            "mean": _fmt_num(series.mean()),
            "min": _fmt_num(series.min()),
            "max": _fmt_num(series.max()),
        }

    # Basic dataset descriptors
    row_count = int(len(results_df))
    categories = (
        results_df["category"].fillna("(missing)").value_counts().to_dict()
        if "category" in results_df.columns
        else {}
    )
    types = (
        results_df["type"].fillna("(missing)").value_counts().to_dict()
        if "type" in results_df.columns
        else {}
    )

    retrieval_cols = [
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg",
        "retrieval_latency_ms",
        "retrieval_fail_rate",
        "retrieval_diversity_ratio",
    ]
    context_cols = [
        "context_window_utilization_chars",
        "context_contamination_rate",
        "deduplication_rate",
    ]
    generation_cols = [
        "answer_length",
        "response_latency_ms",
    ]

    def _section_table(cols: List[str]) -> str:
        rows = []
        for c in cols:
            if c not in metrics_df.columns:
                continue
            stats = _describe_numeric(metrics_df[c])
            rows.append((c, stats["mean"], stats["min"], stats["max"]))
        if not rows:
            return "(no columns found)\n"
        out = "| metric | mean | min | max |\n|---|---:|---:|---:|\n"
        out += "\n".join(f"| {m} | {mean} | {mn} | {mx} |" for (m, mean, mn, mx) in rows)
        out += "\n"
        return out

    lines: List[str] = []
    lines.append("# RAG Evaluation — Last Run Summary")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Per-question results CSV: `{result_path}`")
    lines.append(f"- Per-question metrics CSV: `{metrics_path}`")
    if ragas_metrics_path is not None:
        lines.append(f"- RAGAS metrics CSV: `{ragas_metrics_path}`")
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"This run contains **{row_count}** questions.")
    if categories:
        lines.append("")
        lines.append("Category counts:")
        lines.append("")
        lines.append("| category | count |\n|---|---:|")
        for k, v in categories.items():
            lines.append(f"| {k} | {int(v)} |")
    if types:
        lines.append("")
        lines.append("Type counts:")
        lines.append("")
        lines.append("| type | count |\n|---|---:|")
        for k, v in types.items():
            lines.append(f"| {k} | {int(v)} |")
    lines.append("")

    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("The table below summarizes retrieval-related metrics across questions.")
    lines.append("")
    lines.append(_section_table(retrieval_cols))

    lines.append("## Context Metrics")
    lines.append("")
    lines.append("The table below summarizes context-related metrics across questions.")
    lines.append("")
    lines.append(_section_table(context_cols))

    lines.append("## Generation Metrics")
    lines.append("")
    lines.append("The table below summarizes generation-related metrics across questions.")
    lines.append("")
    lines.append(_section_table(generation_cols))

    if ragas_metrics_path is not None:
        try:
            ragas_df = pd.read_csv(ragas_metrics_path)
            lines.append("## RAGAS Metrics")
            lines.append("")
            numeric_cols = [c for c in ragas_df.columns if c.lower() not in {"question"}]
            if numeric_cols:
                # Describe each numeric column (some ragas outputs include strings)
                out = "| metric | mean | min | max |\n|---|---:|---:|---:|\n"
                rows = []
                for c in numeric_cols:
                    s = pd.to_numeric(ragas_df[c], errors="coerce")
                    if s.notna().any():
                        stats = _describe_numeric(s)
                        rows.append((c, stats["mean"], stats["min"], stats["max"]))
                if rows:
                    out += "\n".join(f"| {m} | {mean} | {mn} | {mx} |" for (m, mean, mn, mx) in rows)
                    out += "\n"
                    lines.append(out)
        except Exception:
            pass

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
