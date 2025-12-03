"""Core data models for the RAG pipeline."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class DocumentChunk(BaseModel):
    """Validated text chunk to embed and index."""

    id: str = Field(..., description="Stable identifier for the chunk")
    text: str = Field(..., min_length=1, description="Raw text of the chunk")
    source: str = Field(..., description="Human readable source name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class RetrievalResult(BaseModel):
    """Result of a retrieval call."""

    query: str
    documents: List[DocumentChunk]
    scores: List[float]
    latency_ms: float


class LLMInput(BaseModel):
    """Payload sent to the LLM."""

    query: str
    context: List[DocumentChunk]


class AnswerPayload(BaseModel):
    """Structured answer returned by the pipeline."""

    answer: str
    context_documents: List[DocumentChunk]
    reasoning: Optional[str] = None
    used_sql: bool = False
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    model: Optional[str] = None


class SQLQuery(BaseModel):
    """Validated SQL tool input."""

    natural_query: str
    limit: int = Field(default=50, le=200)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    player: Optional[str] = None
    team: Optional[str] = None


class SQLResultRow(BaseModel):
    columns: list[str]
    values: list[Any]


class SQLResult(BaseModel):
    """Output of SQL tool execution."""

    query: str
    rows: List[SQLResultRow]
    latency_ms: float
