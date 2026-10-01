"""Two-process check around an actual server restart (used by GitHub Actions).

Only creates/deletes its own unique test collection; never touches the user KB.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from pymilvus import MilvusClient

from app.rag.embedding import HashEmbeddingModel
from app.rag.milvus_store import MilvusKnowledgeBase

STATE = Path("milvus-restart-result.json")


async def main():
    uri = os.getenv("MILVUS_TEST_URI", "http://127.0.0.1:19530")
    token = os.getenv("MILVUS_TEST_TOKEN", "")
    mode = sys.argv[1]
    state = {"collection": "restart_check_" + uuid4().hex} if mode == "write" else json.loads(
        STATE.read_text(encoding="utf-8"))
    name = state["collection"]
    backend = MilvusKnowledgeBase(uri=uri, token=token, collection_name=name,
                                 embedding_model=HashEmbeddingModel(dimensions=32),
                                 embedding_revision="hash-v1", timeout_seconds=60)
    try:
        if mode == "write":
            doc = await backend.add_document("数据库重启验收", "住宿报销须提供发票并经主管审批。" * 60)
            admin = MilvusClient(uri=uri, token=token)
            try:
                admin.flush(collection_name=name, timeout=60)
            finally:
                admin.close()
            state.update(doc_id=doc.doc_id, chunk_count=doc.chunk_count, written=True)
        elif mode == "read":
            documents = await backend.list_documents()
            assert any(d.doc_id == state["doc_id"] and d.chunk_count == state["chunk_count"]
                       for d in documents)
            assert (await backend.search("住宿报销发票", 1))[0].doc_id == state["doc_id"]
            assert await backend.delete_document(state["doc_id"])
            assert await backend.list_documents() == []
            state.update(server_restart_verified=True, retrieval_verified=True,
                         deletion_verified=True)
            admin = MilvusClient(uri=uri, token=token)
            try:
                admin.drop_collection(collection_name=name)
            finally:
                admin.close()
        else:
            raise ValueError("Expected write or read")
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(state, ensure_ascii=False))
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
