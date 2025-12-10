"""High-level RAG pipeline orchestrator."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from data_pipeline.indexing import get_retriever_from_dir
from db.sql_tool import SQLTool
from rag.llm import LLMConfig, LLMService
from rag.models import AnswerPayload, LLMInput
from rag.retriever import retrieve

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


def should_use_sql(query: str) -> bool:
    lowered = query.lower()
    numeric_keywords = ["average", "percent", "percentage", "stats", "rebounds", "points", "score", "assists", "best"]
    return any(word in lowered for word in numeric_keywords)


def _configure_logfire_if_needed():
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
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.debug("Logfire configuration skipped: %s", exc)


def run_rag_pipeline(question: str, data_dir: Path = Path("inputs"), sql_tool: Optional[SQLTool] = None,
                     llm_config: Optional[LLMConfig] = None) -> AnswerPayload:
    start_total = time.perf_counter()
    # Load environment and configure optional Logfire
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    _configure_logfire_if_needed()
    if os.environ.get("RAG_DEBUG_LOG", "0") == "1":
        import logging as _logging
        _logging.basicConfig(level=_logging.DEBUG)
        _logging.debug("RAG_DEBUG_LOG enabled")

    retriever = get_retriever_from_dir(data_dir)
    llm = LLMService(llm_config or LLMConfig())

    use_sql = should_use_sql(question)
    tool = sql_tool
    if tool is None and use_sql:
        tool = SQLTool()

    if tool and use_sql:
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
        if logfire:
            logfire.info("pipeline.used_sql", query=question, rows=len(sql_result.rows))
        return payload

    retrieval = retrieve(question, retriever)
    llm_input = LLMInput(query=question, context=retrieval.documents)
    answer = llm.generate(llm_input)
    if logfire:
        logfire.info(
            "pipeline.rag_answer",
            query=question,
            context=len(retrieval.documents),
            latency_ms=retrieval.latency_ms,
            used_sql=False,
        )
    total_ms = (time.perf_counter() - start_total) * 1000
    try:
        import logging as _logging
        _logging.info("Pipeline completed in %.2f ms", total_ms)
    except Exception:
        pass
    return answer
