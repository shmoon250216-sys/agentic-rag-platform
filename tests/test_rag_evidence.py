import pytest

from app.core.config import Settings
from app.graph.workflow import AgentWorkflow
from app.llm.client import OpenAICompatibleLLMClient
from app.rag.reranker import NoopReranker
from app.rag.retriever import KnowledgeBaseRetriever, retrieval_cache
from app.schemas.chat import ChatRequest, SourceChunk


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("async_store", [False, True])
async def test_full_evidence_reaches_model_on_fresh_and_cached_search(
    monkeypatch, streaming, async_store
):
    # The answer is deliberately beyond the UI preview boundary.
    evidence = "住宿制度的适用范围和材料说明。" * 20 + "L2员工在杭州每晚住宿上限为550元。"
    assert evidence.index("550") > 220
    source = SourceChunk(doc_id="policy", title="住宿标准", snippet=evidence, score=1)

    class Store:
        calls = 0

        def search(self, query, top_k):
            self.calls += 1
            return [source]

    class AsyncStore(Store):
        async def search(self, query, top_k):
            return super().search(query, top_k)

    store = AsyncStore() if async_store else Store()
    retriever = KnowledgeBaseRetriever(store=store, reranker=NoopReranker())
    monkeypatch.setattr(retriever.settings, "cache_enabled", True)
    prompts = []
    llm = OpenAICompatibleLLMClient(
        Settings(llm_api_key="test-only", llm_base_url="https://unused.invalid")
    )

    async def complete(*, system_prompt, user_prompt):
        prompts.append(user_prompt)
        return "550元"

    async def stream_complete(*, system_prompt, user_prompt):
        prompts.append(user_prompt)
        yield "550元"

    # Capture the actual prompt builder without making a network request.
    monkeypatch.setattr(llm, "_complete", complete)
    monkeypatch.setattr(llm, "_stream_complete", stream_complete)
    workflow = AgentWorkflow(llm=llm)
    workflow.retriever = retriever
    request = ChatRequest(message="L2员工到杭州出差住宿报销上限是多少？")
    retrieval_cache.clear()
    try:
        for attempt in range(2):
            if streaming:
                events = [
                    event async for event in workflow.stream(request, f"evidence-{attempt}")
                ]
                response = next(payload for event, payload in events if event == "done")
            else:
                response = await workflow.run(request, f"evidence-{attempt}")
            assert response.sources[0].snippet == evidence
            assert evidence in prompts[-1]
            assert "550元" in prompts[-1]
        assert len(prompts) == 2
        assert store.calls == 1
    finally:
        retrieval_cache.clear()
