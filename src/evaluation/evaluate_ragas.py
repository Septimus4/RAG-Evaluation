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

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None

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


def _configure_logfire():
    if not logfire:
        return
    token = os.environ.get("LOGFIRE_TOKEN")
    if not token:
        return
    if getattr(logfire, "_rag_configured", False):  # type: ignore[attr-defined]
        return
    try:
        logfire.configure(token=token)
        setattr(logfire, "_rag_configured", True)
    except Exception as exc:  # pragma: no cover - logfire optional
        logging.debug("Unable to configure logfire: %s", exc)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def _recall_precision_k(retrieved: list[str], expected: list[str], k: int = 5) -> tuple[float, float]:
    if not expected:
        return 0.0, 0.0
    exp = set(expected)
    topk = retrieved[:k]
    hit = len([s for s in topk if s in exp])
    recall = hit / len(exp)
    precision = hit / max(len(topk), 1)
    return recall, precision


def _mrr_ndcg(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    if not expected or not retrieved:
        return 0.0, 0.0
    exp = set(expected)
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
            dcg += rel / math.log2(i + 1)
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
    if evaluate and answer_relevancy and Dataset:
        # Minimal evaluation example when ragas is installed
        try:
            ragas_llm = _build_ragas_llm()
            ragas_embeddings = _build_ragas_embeddings()
            if ragas_llm is None or ragas_embeddings is None:
                raise RuntimeError("RAGAS dependencies unavailable")
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
                metrics_df = evaluation.to_pandas()
            else:  # pragma: no cover - new ragas API surface
                metrics_df = evaluation.to_dataframe()
            metrics_path2 = output_dir / f"ragas_metrics_{datetime.now(timezone.utc).isoformat()}.csv"
            metrics_df.to_csv(metrics_path2, index=False)
        except Exception as exc:  # pragma: no cover
            logging.warning("RAGAS evaluation failed: %s", exc)
    if logfire:
        logfire.info(
            "evaluation.run_dataset",
            dataset=str(dataset_path),
            rows=len(df),
        )
    return result_path


@app.command()
def main(
    dataset: Path = typer.Option(Path(__file__).parent / "datasets" / "sample_questions.json"),
    output_dir: Path = typer.Option(Path(__file__).parent / "results"),
    model: str = typer.Option("mistral-small-latest"),
):
    logging.basicConfig(level=logging.INFO)
    _configure_logfire()
    llm_config = LLMConfig(model=model)
    result_path = run_dataset(dataset, output_dir, llm_config)
    typer.echo(f"Results written to {result_path}")


def _build_ragas_llm() -> Optional["BaseRagasLLM"]:
    if BaseRagasLLM is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
    model_name = os.environ.get("RAGAS_MODEL", "mistral-small-latest")
    try:
        from openai import OpenAI  # type: ignore
        from ragas.llms import llm_factory
    except Exception as exc:  # pragma: no cover
        logging.warning("OpenAI client unavailable for RAGAS metrics: %s", exc)
        return None
        # Removed duplicated metrics block
        hallucination_risk = None
        groundedness = None
        if _st_model and np is not None:
            try:
                q_emb = _st_model.encode(item["question"], normalize_embeddings=True)
                doc_texts = [d.text for d in ctx_docs]
                if doc_texts:
                    d_embs = _st_model.encode(doc_texts, normalize_embeddings=True)
                    sims = [float(np.dot(q_emb, d)) for d in d_embs]
                    query_doc_sim = float(np.mean(sims)) if sims else 0.0
                    # Redundancy: avg pairwise similarity among docs
                    if len(d_embs) > 1:
                        pair_sims = []
                        for i in range(len(d_embs)):
                            for j in range(i + 1, len(d_embs)):
                                pair_sims.append(float(np.dot(d_embs[i], d_embs[j])))
                        redundancy = float(np.mean(pair_sims)) if pair_sims else 0.0
                        diversity = 1.0 - redundancy
                    # Hallucination risk: low similarity between answer and docs
                    a_emb = _st_model.encode(answer.answer, normalize_embeddings=True)
                    ans_sims = [float(np.dot(a_emb, d)) for d in d_embs]
                    if ans_sims:
                        max_ans_sim = max(ans_sims)
                        hallucination_risk = float(1.0 - max_ans_sim)
                        groundedness = float(np.mean(ans_sims))
            except Exception:
                pass
        # Context integrity
        total_ctx_chars = sum(len(d.text) for d in ctx_docs)
        # Assume nominal 16k token window ~ 64k chars; utilization ratio
        context_utilization = total_ctx_chars / 64000.0
        contamination_rate = 1.0 - precision_at_k if k else 0.0
        dedup_rate = 0.0
        if ctx_docs:
            seen_hashes = {}
            dup = 0
            for d in ctx_docs:
                h = hash(d.text)
                dup += 1 if h in seen_hashes else 0
                seen_hashes[h] = True
            dedup_rate = dup / k
        # Passage freshness: use file mtime ages if sources are files
        freshness_days = None
        try:
            ages = []
            for s in retrieved_sources:
                p = Path(s)
                if p.exists():
                    ages.append((datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds() / 86400.0)
            if ages:
                freshness_days = float(np.mean(ages)) if np is not None else sum(ages) / len(ages)
        except Exception:
            pass
        # Generation metrics
        faithfulness_proxy = groundedness if groundedness is not None else 0.0
        gold = item.get("expected_answer") or ""
        acc_exact = 1.0 if gold and gold.strip().lower() == (answer.answer or "").strip().lower() else 0.0
        acc_substr = 1.0 if gold and gold.strip().lower() in (answer.answer or "").strip().lower() else 0.0
        answer_len = len(answer.answer or "")
        metrics_rows.append({
            "question": item["question"],
            "latency_ms": round(e2e_ms, 2),
            # Retrieval Quality
            "recall@k": round(recall_at_k, 4),
            "precision@k": round(precision_at_k, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
            "query_doc_similarity": round(query_doc_sim or 0.0, 4),
            "redundancy": round(redundancy or 0.0, 4),
            "diversity": round(diversity or 0.0, 4),
            "retrieval_fail_rate": 1.0 if k == 0 else 0.0,
            # Context Integrity
            "context_utilization": round(context_utilization, 4),
            "contamination_rate": round(contamination_rate, 4),
            "hallucination_risk": round(hallucination_risk or 0.0, 4),
            "freshness_days": round(freshness_days or 0.0, 2),
            "dedup_rate": round(dedup_rate, 4),
            # Generation Quality
            "faithfulness_proxy": round(faithfulness_proxy, 4),
            "groundedness_proxy": round(groundedness or 0.0, 4),
            "accuracy_exact": acc_exact,
            "accuracy_substr": acc_substr,
            "answer_length": answer_len,
        })
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


if __name__ == "__main__":
    app()
