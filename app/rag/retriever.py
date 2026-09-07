import inspect

from app.core.cache import TTLCache, make_cache_key
from app.core.config import get_settings
from app.rag.factory import get_knowledge_base
from app.schemas.chat import SourceChunk

settings = get_settings()
retrieval_cache = TTLCache(
    max_entries=settings.cache_max_entries,
    ttl_seconds=settings.cache_ttl_seconds,
)


class KnowledgeBaseRetriever:
    """Retriever backed by the configured knowledge base."""

    def __init__(self, store=None) -> None:
        self.store = store or get_knowledge_base()

    async def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        cache_key = make_cache_key(
            "rag-search",
            {
                "store": type(self.store).__name__,
                "query": query,
                "top_k": top_k,
            },
        )
        if get_settings().cache_enabled:
            cached = retrieval_cache.get(cache_key)
            if cached is not None:
                return cached

        result = self.store.search(query, top_k=top_k)
        if inspect.isawaitable(result):
            result = await result

        if get_settings().cache_enabled:
            retrieval_cache.set(cache_key, result)
        return result
