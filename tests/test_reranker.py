import httpx
import pytest

from app.rag.reranker import HTTPReranker, NoopReranker
from app.rag.retriever import KnowledgeBaseRetriever, retrieval_cache
from app.schemas.chat import SourceChunk


def candidates() -> list[SourceChunk]:
    return [
        SourceChunk(doc_id="a", title="宽泛规则", snippet="员工可以申请差旅预支", score=1),
        SourceChunk(
            doc_id="b",
            title="试用期限制",
            snippet="试用期员工的差旅预支不得超过三千元",
            score=0.8,
        ),
    ]


@pytest.mark.asyncio
async def test_http_reranker_applies_model_order_and_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["query"] == "试用期员工最多预支多少"
        assert payload["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.96},
                    {"index": 0, "relevance_score": 0.31},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    reranker = HTTPReranker(
        url="https://rerank.test/v1/rerank",
        model="test-reranker",
        api_key="test-key",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    results = await reranker.rerank(
        "试用期员工最多预支多少", candidates(), top_k=2
    )

    assert [result.doc_id for result in results] == ["b", "a"]
    assert results[0].score == 0.96


@pytest.mark.asyncio
async def test_noop_reranker_preserves_rrf_order() -> None:
    results = await NoopReranker().rerank("query", candidates(), top_k=1)

    assert [result.doc_id for result in results] == ["a"]


@pytest.mark.asyncio
async def test_retriever_falls_back_to_rrf_order_when_reranker_is_unavailable() -> None:
    class Store:
        def search(self, query: str, top_k: int):
            del query
            return candidates()[:top_k]

    class FailingReranker:
        async def rerank(self, query, candidates, *, top_k):
            del query, candidates, top_k
            raise httpx.ConnectError("reranker unavailable")

    retrieval_cache.clear()
    retriever = KnowledgeBaseRetriever(store=Store(), reranker=FailingReranker())
    retriever.settings.rerank_fail_open = True

    results = await retriever.search("unique fail-open query", top_k=1)

    assert [result.doc_id for result in results] == ["a"]
