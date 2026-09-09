from functools import lru_cache

from app.core.config import get_settings
from app.rag.embedding import get_embedding_model
from app.rag.redis_store import RedisKnowledgeBase
from app.rag.store import InMemoryKnowledgeBase


@lru_cache
def get_knowledge_base():
    settings = get_settings()
    embedding_model = get_embedding_model(settings)
    if settings.rag_backend.lower() == "redis":
        if (
            embedding_model.dimensions is not None
            and embedding_model.dimensions != settings.rag_vector_dimensions
        ):
            raise ValueError(
                "RAG_VECTOR_DIMENSIONS must match EMBEDDING_DIMENSIONS when Redis is enabled: "
                f"{settings.rag_vector_dimensions} != {embedding_model.dimensions}"
            )
        return RedisKnowledgeBase(
            redis_url=settings.redis_url,
            index_name=settings.rag_index_name,
            vector_dimensions=settings.rag_vector_dimensions,
            embedding_model=embedding_model,
            connect_timeout=settings.redis_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            candidate_multiplier=settings.rag_candidate_multiplier,
            rrf_k=settings.rag_rrf_k,
            bm25_weight=settings.rag_bm25_weight,
            vector_weight=settings.rag_vector_weight,
        )
    store = InMemoryKnowledgeBase(
        embedding_model=embedding_model,
        candidate_multiplier=settings.rag_candidate_multiplier,
        rrf_k=settings.rag_rrf_k,
        bm25_weight=settings.rag_bm25_weight,
        vector_weight=settings.rag_vector_weight,
    )
    store.seed_defaults()
    return store
