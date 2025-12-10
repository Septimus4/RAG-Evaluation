from rag.llm import LLMConfig, LLMService
from rag.models import DocumentChunk, LLMInput


def test_llm_mock_response_without_api_key():
    service = LLMService(LLMConfig(api_key=None))
    payload = LLMInput(query="Who won?", context=[DocumentChunk(id="1", text="Some context", source="x", metadata={})])
    answer = service.generate(payload)
    assert answer.answer
    assert answer.model == service.config.model
