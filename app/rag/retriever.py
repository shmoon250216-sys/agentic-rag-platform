import inspect

import httpx

from app.core.cache import TTLCache, make_cache_key
from app.core.config import get_settings
from app.rag.factory import get_knowledge_base
from app.rag.reranker import NoopReranker, Reranker, get_reranker
from app.schemas.chat import SourceChunk

settings = get_settings()
retrieval_cache = TTLCache(
    max_entries=settings.cache_max_entries,
    ttl_seconds=settings.cache_ttl_seconds,
)


class KnowledgeBaseRetriever:
    """Retriever backed by the configured knowledge base."""

    def __init__(self, store=None, reranker: Reranker | None = None) -> None:
        self.store = store or get_knowledge_base()
        self.settings = get_settings()
        self.reranker = reranker or get_reranker(self.settings)

    async def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        cache_key = make_cache_key(
            "rag-search",
            {
                "store": type(self.store).__name__,
                "query": query,
                "top_k": top_k,
                "reranker": type(self.reranker).__name__,
                "rerank_model": self.settings.rerank_model,
            },
        )
        if get_settings().cache_enabled:
            cached = retrieval_cache.get(cache_key)
            if cached is not None:
                return cached

        candidate_count = top_k
        if not isinstance(self.reranker, NoopReranker):
            candidate_count = max(top_k, self.settings.rerank_candidate_count)

        result = self.store.search(query, top_k=candidate_count)
        if inspect.isawaitable(result):
            result = await result

        try:
            result = await self.reranker.rerank(query, result, top_k=top_k)
        except (ValueError, httpx.HTTPError):
            if not self.settings.rerank_fail_open:
                raise
            result = list(result[:top_k])

        result = [source.model_copy(update={"snippet": source.snippet[:220]}) for source in result]

        if get_settings().cache_enabled:
            retrieval_cache.set(cache_key, result)
        return result
