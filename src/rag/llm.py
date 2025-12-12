"""LLM client wrapper with optional Mistral support."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from pydantic import BaseModel

from .models import AnswerPayload, LLMInput

from observability.logfire_setup import configure_logfire, info, span

try:  # pragma: no cover
    from mistralai import Mistral
except Exception:  # pragma: no cover
    MistralClient = None
    ChatMessage = None



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
        if Mistral and api_key:
            try:
                self.client = Mistral(api_key=api_key)
            except Exception as exc:  # pragma: no cover
                logging.warning("Falling back to mock LLM: %s", exc)
        if self.client is None:
            logging.warning("Mistral client unavailable; using mock responses.")

    def generate(self, payload: LLMInput) -> AnswerPayload:
        start = time.perf_counter()
        configure_logfire()
        used_mock = False
        text: str
        with span("llm.generate", model=self.config.model, context_count=len(payload.context)):
            if self.client:
                try:
                    response = self.client.chat.complete(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are a basketball analyst. Use the context to answer."},
                            {"role": "user", "content": self._build_prompt(payload)},
                        ],
                        temperature=self.config.temperature,
                    )
                    text = self._extract_response_text(response)
                    if not text:
                        logging.debug("Empty response from Mistral; falling back to mock output")
                        text = self._mock_response(payload)
                        used_mock = True
                except Exception as exc:
                    logging.warning("Mistral chat failed; falling back to mock responses: %s", exc)
                    self.client = None
                    text = self._mock_response(payload)
                    used_mock = True
            else:
                text = self._mock_response(payload)
                used_mock = True
        latency_ms = (time.perf_counter() - start) * 1000
        info(
            "llm.generate",
            model=self.config.model,
            latency_ms=latency_ms,
            used_mock=used_mock,
            context_count=len(payload.context),
        )
        return AnswerPayload(
            answer=text,
            context_documents=list(payload.context),
            reasoning="Mocked" if used_mock else None,
            model=self.config.model,
        )

    def _build_prompt(self, payload: LLMInput) -> str:
        context_block = "\n\n".join(f"[source={doc.source}] {doc.text}" for doc in payload.context)
        return f"Context:\n{context_block}\n\nQuestion: {payload.query}\nAnswer concisely and cite sources."

    def _mock_response(self, payload: LLMInput) -> str:
        snippet = payload.context[0].text[:120] + "..." if payload.context else "No context available."
        return f"[MOCK ANSWER] {payload.query}\nContext snippet: {snippet}"

    def _extract_response_text(self, response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            return ""
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if not message:
            return ""
        content = getattr(message, "content", None)
        return self._stringify_content(content)

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value
            if isinstance(text_value, (list, tuple)):
                return "".join(
                    chunk.get("text") if isinstance(chunk, dict) else str(chunk)
                    for chunk in text_value
                    if (isinstance(chunk, dict) and chunk.get("text")) or isinstance(chunk, str)
                )
            return ""
        if isinstance(content, (list, tuple)):
            parts = []
            for chunk in content:
                if isinstance(chunk, str):
                    parts.append(chunk)
                    continue
                if isinstance(chunk, dict):
                    text_value = chunk.get("text")
                else:
                    text_value = getattr(chunk, "text", None)
                if text_value:
                    parts.append(text_value)
            return "".join(parts)
        return ""
