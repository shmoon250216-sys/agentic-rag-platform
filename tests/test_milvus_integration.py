"""Requires an actual Milvus Standalone; CI starts it, no mock or Milvus Lite."""
import os
from uuid import uuid4

import pytest

from app.rag.embedding import HashEmbeddingModel
from app.rag.milvus_store import MilvusConfigurationError, MilvusKnowledgeBase

URI = os.getenv("MILVUS_TEST_URI")
pytestmark = pytest.mark.skipif(not URI, reason="MILVUS_TEST_URI not set")


async def test_real_milvus_persistence_hybrid_and_lifecycle():
    from pymilvus import MilvusClient
    name = "ci_rag_" + uuid4().hex
    config = dict(uri=URI, token=os.getenv("MILVUS_TEST_TOKEN", ""), collection_name=name,
                  embedding_model=HashEmbeddingModel(dimensions=32), embedding_revision="hash-v1",
                  timeout_seconds=60, bm25_weight=1, vector_weight=0.1)
    first, second = MilvusKnowledgeBase(**config), MilvusKnowledgeBase(**config)
    admin = MilvusClient(uri=URI, token=config["token"])
    try:
        doc = await first.add_document("住宿报销制度", "住宿报销标准为每日五百元，必须提供发票。" * 70)
        await first.add_document("信息安全制度", "密码不得向其他人透露，离开座位需要锁屏。" * 30)
        await first.close()
        assert doc in await second.list_documents()
        assert (await second.search("住宿报销发票", 1))[0].doc_id == doc.doc_id
        assert (await second.health())["connected"]
        mismatch = MilvusKnowledgeBase(**{**config, "embedding_revision": "other-model"})
        try:
            with pytest.raises(MilvusConfigurationError):
                await mismatch.ensure_available()
        finally:
            await mismatch.close()
        assert await second.delete_document(doc.doc_id)
        assert doc not in await second.list_documents()
        rows = admin.query(collection_name=name, filter=f'doc_id == "{doc.doc_id}"',
                           output_fields=["chunk_id"], consistency_level="Strong")
        assert rows == []
        assert not await second.delete_document(doc.doc_id)
    finally:
        await first.close()
        await second.close()
        if admin.has_collection(collection_name=name):
            admin.drop_collection(collection_name=name)
        admin.close()
