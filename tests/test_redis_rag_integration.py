import os
from uuid import uuid4

import pytest
from redis.exceptions import ResponseError

from app.rag.embedding import HashEmbeddingModel
from app.rag.redis_store import RedisKnowledgeBase, RedisRagDefaults


REDIS_INTEGRATION_URL = os.getenv("REDIS_INTEGRATION_URL")


@pytest.mark.asyncio
@pytest.mark.skipif(not REDIS_INTEGRATION_URL, reason="Redis Stack integration URL is not set")
async def test_document_survives_redis_store_recreation() -> None:
    suffix = uuid4().hex
    index_name = f"idx:test-rag:{suffix}"
    defaults = RedisRagDefaults(
        chunk_prefix=f"test:rag:{suffix}:chunk:",
        doc_prefix=f"test:rag:{suffix}:doc:",
        seed_key=f"test:rag:{suffix}:seeded",
    )
    model = HashEmbeddingModel(dimensions=32)
    first_store = RedisKnowledgeBase(
        redis_url=REDIS_INTEGRATION_URL,
        index_name=index_name,
        vector_dimensions=32,
        embedding_model=model,
        defaults=defaults,
    )
    second_store = RedisKnowledgeBase(
        redis_url=REDIS_INTEGRATION_URL,
        index_name=index_name,
        vector_dimensions=32,
        embedding_model=model,
        defaults=defaults,
    )
    first_store_closed = False

    try:
        created = await first_store.add_document(
            "报销制度",
            "员工报销超过一千元时，需要直属主管审批并上传发票。",
        )
        await first_store.close()
        first_store_closed = True

        documents = await second_store.list_documents()
        results = await second_store.search("超过一千元报销需要谁审批", top_k=3)

        assert any(document.doc_id == created.doc_id for document in documents)
        assert any(source.doc_id == created.doc_id for source in results)
    finally:
        if not first_store_closed:
            await first_store.close()
        keys = []
        for pattern in (f"{defaults.doc_prefix}*", f"{defaults.chunk_prefix}*"):
            async for key in second_store.redis.scan_iter(pattern):
                keys.append(key)
        if keys:
            await second_store.redis.delete(*keys)
        await second_store.redis.delete(defaults.seed_key)
        try:
            await second_store.redis.ft(index_name).dropindex()
        except ResponseError:
            pass
        await second_store.close()
