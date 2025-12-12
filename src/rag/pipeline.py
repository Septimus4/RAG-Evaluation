"""High-level RAG pipeline orchestrator."""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from data_pipeline.indexing import get_retriever_from_dir
from db.sql_tool import SQLTool
from observability.logfire_setup import configure_logfire, info, span
from rag.llm import LLMConfig, LLMService
from rag.models import AnswerPayload, LLMInput
from rag.retriever import retrieve


def should_use_sql(query: str) -> bool:
    lowered = query.lower()
    numeric_keywords = ["average", "percent", "percentage", "stats", "rebounds", "points", "score", "assists", "best"]
    return any(word in lowered for word in numeric_keywords)


def run_rag_pipeline(question: str, data_dir: Path = Path("inputs"), sql_tool: Optional[SQLTool] = None,
                     llm_config: Optional[LLMConfig] = None) -> AnswerPayload:
    start_total = time.perf_counter()
    # Load environment and configure optional Logfire
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Configure observability as early as possible so downstream logs/spans inherit
    # the correct OpenTelemetry resource (notably `service.name`).
    configure_logfire()

    if os.environ.get("RAG_DEBUG_LOG", "0") == "1":
        import logging as _logging
        _logging.basicConfig(level=_logging.DEBUG)
        _logging.debug("RAG_DEBUG_LOG enabled")

    run_id = os.environ.get("RAG_RUN_ID") or uuid.uuid4().hex
    query_len = len(question or "")
    query_preview = (question or "")[:120]
    include_query = os.environ.get("LOGFIRE_LOG_QUERY", "0") == "1"
    span_attrs = {
        "run_id": run_id,
        "data_dir": str(data_dir),
        "query_len": query_len,
        "query_preview": query_preview,
    }
    if include_query:
        span_attrs["query"] = question

    with span("rag.pipeline", **span_attrs):
        retriever = get_retriever_from_dir(data_dir)
        llm = LLMService(llm_config or LLMConfig())

        use_sql = should_use_sql(question)
        tool = sql_tool
        if tool is None and use_sql:
            tool = SQLTool()

        if tool and use_sql:
            with span("rag.sql", run_id=run_id):
                sql_result = tool.run_query(question)
                summary = tool.summarize(sql_result)
            answer_text = f"Structured stats answer: {summary}"
            payload = AnswerPayload(
                answer=answer_text,
                context_documents=[],
                reasoning="SQL tool selected for numeric query.",
                used_sql=True,
                model=llm_config.model if llm_config else None,
            )
            info("pipeline.used_sql", run_id=run_id, rows=len(sql_result.rows))
            return payload

        with span("rag.retrieve", run_id=run_id):
            retrieval = retrieve(question, retriever)
        llm_input = LLMInput(query=question, context=retrieval.documents)
        with span("rag.llm.generate", run_id=run_id, model=llm.config.model):
            answer = llm.generate(llm_input)

        info(
            "pipeline.rag_answer",
            run_id=run_id,
            context=len(retrieval.documents),
            retrieval_latency_ms=retrieval.latency_ms,
            used_sql=False,
            model=llm.config.model,
        )

        total_ms = (time.perf_counter() - start_total) * 1000
        logging.getLogger(__name__).info("Pipeline completed in %.2f ms", total_ms)
        return answer
