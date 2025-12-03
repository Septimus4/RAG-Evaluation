"""High-level RAG pipeline orchestrator."""
from __future__ import annotations

import logging
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


def run_rag_pipeline(question: str, data_dir: Path = Path("inputs"), sql_tool: Optional[SQLTool] = None,
                     llm_config: Optional[LLMConfig] = None) -> AnswerPayload:
    retriever = get_retriever_from_dir(data_dir)
    llm = LLMService(llm_config or LLMConfig())

    if sql_tool and should_use_sql(question):
        sql_result = sql_tool.run_query(question)
        summary = sql_tool.summarize(sql_result)
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
    return answer
