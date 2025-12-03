"""LLM client wrapper with optional Mistral support."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from pydantic import BaseModel

from .models import AnswerPayload, LLMInput

try:  # pragma: no cover
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except Exception:  # pragma: no cover
    MistralClient = None
    ChatMessage = None

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


class LLMConfig(BaseModel):
    model: str = "mistral-small-latest"
    temperature: float = 0.1
    api_key: Optional[str] = None


class LLMService:
    """Lightweight LLM wrapper to make swapping providers easy."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key or os.environ.get("MISTRAL_API_KEY")
        self.client = None
        if MistralClient and api_key:
            try:
                self.client = MistralClient(api_key=api_key)
            except Exception as exc:  # pragma: no cover
                logging.warning("Falling back to mock LLM: %s", exc)
        if self.client is None:
            logging.warning("Mistral client unavailable; using mock responses.")

    def generate(self, payload: LLMInput) -> AnswerPayload:
        start = time.perf_counter()
        if self.client and ChatMessage:
            messages = [
                ChatMessage(role="system", content="You are a basketball analyst. Use the context to answer."),
                ChatMessage(role="user", content=self._build_prompt(payload)),
            ]
            response = self.client.chat(model=self.config.model, messages=messages, temperature=self.config.temperature)
            text = response.choices[0].message.content
        else:
            text = self._mock_response(payload)
        latency_ms = (time.perf_counter() - start) * 1000
        if logfire:
            logfire.info(
                "llm.generate",
                model=self.config.model,
                latency_ms=latency_ms,
                used_mock=self.client is None,
                context_count=len(payload.context),
            )
        return AnswerPayload(
            answer=text,
            context_documents=list(payload.context),
            reasoning="Mocked" if self.client is None else None,
            model=self.config.model,
        )

    def _build_prompt(self, payload: LLMInput) -> str:
        context_block = "\n\n".join(f"[source={doc.source}] {doc.text}" for doc in payload.context)
        return f"Context:\n{context_block}\n\nQuestion: {payload.query}\nAnswer concisely and cite sources."

    def _mock_response(self, payload: LLMInput) -> str:
        snippet = payload.context[0].text[:120] + "..." if payload.context else "No context available."
        return f"[MOCK ANSWER] {payload.query}\nContext snippet: {snippet}"
