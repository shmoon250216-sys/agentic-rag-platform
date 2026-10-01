"""Contract/failure tests using the real SDK schema and a small fake transport."""
import asyncio
import copy
import threading

import pytest

from app.core.errors import AppError
from app.rag.embedding import HashEmbeddingModel
from app.rag.milvus_store import MilvusConfigurationError, MilvusKnowledgeBase


class Iterator:
    def __init__(self, rows):
        self.rows, self.closed = rows, False

    def next(self):
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.rows, self.schema, self.iterator = {}, None, None
        self.calls = []
        self.fail_insert = False

    def has_collection(self, **kwargs):
        return self.schema is not None

    def create_collection(self, schema, index_params, **kwargs):
        self.schema = schema.to_dict()
        self.indexes = index_params

    def describe_collection(self, **kwargs):
        return self.schema

    def list_indexes(self, **kwargs):
        return ["dense", "sparse"]

    def describe_index(self, index_name, **kwargs):
        return {"field_name": index_name, "metric_type":
                "COSINE" if index_name == "dense" else "BM25"}

    def load_collection(self, **kwargs):
        pass

    def insert(self, data, **kwargs):
        self.rows.update({r["chunk_id"]: r for r in data})
        if self.fail_insert:
            raise RuntimeError("secret connection failure")
        return {"insert_count": len(data)}

    def query_iterator(self, **kwargs):
        self.iterator = Iterator([{k: r[k] for k in kwargs["output_fields"]}
                                  for r in self.rows.values() if r["chunk_index"] == 0])
        return self.iterator

    def query(self, filter, **kwargs):
        return [r for r in self.rows.values() if r["doc_id"] in filter][:1]

    def delete(self, filter, **kwargs):
        self.rows = {k: r for k, r in self.rows.items() if r["doc_id"] not in filter}

    def search(self, anns_field, **kwargs):
        self.calls.append((anns_field, kwargs, threading.get_ident()))
        rows = list(self.rows.values())
        if anns_field == "dense":
            rows.reverse()
        return [[{"chunk_id": r["chunk_id"], "distance": 1.0/(i+1), "entity": r}
                 for i, r in enumerate(rows)]]

    def close(self):
        pass


def store(client=None, **kwargs):
    return MilvusKnowledgeBase(uri="http://127.0.0.1:19530", token="",
                               collection_name="test_chunks", embedding_revision="hash-v1",
                               embedding_model=HashEmbeddingModel(dimensions=32),
                               client=client or Client(), **kwargs)


async def test_crud_reopen_search_and_delete_all_chunks():
    client = Client()
    first = store(client)
    doc = await first.add_document("差旅制度", "报销需要主管审批，住宿标准按城市划分。" * 80)
    assert doc.chunk_count > 1
    reopened = store(client)
    assert await reopened.list_documents() == [doc]
    assert client.iterator.closed
    results = await reopened.search("住宿报销", 2)
    assert len(results) == 2 and all(r.doc_id == doc.doc_id for r in results)
    assert [call[0] for call in client.calls] == ["sparse", "dense"]
    assert all(call[2] != threading.get_ident() for call in client.calls)
    assert await reopened.delete_document(doc.doc_id)
    assert not client.rows
    assert not await reopened.delete_document(doc.doc_id)
    assert not await reopened.delete_document('" or true')


async def test_native_schema_and_index_configuration():
    client = Client()
    await store(client).ensure_available()
    fields = {f["name"]: f for f in client.schema["fields"]}
    assert fields["dense"]["params"]["dim"] == 32
    assert fields["text"]["params"]["enable_analyzer"]
    assert client.schema["functions"][0]["input_field_names"] == ["text"]
    assert "HNSW" in str(client.indexes)


@pytest.mark.parametrize("change", ["dimension", "revision", "field", "metric"])
async def test_existing_collection_mismatch_fails_closed(change):
    client = Client()
    await store(client).ensure_available()
    client.schema = copy.deepcopy(client.schema)
    if change == "dimension":
        next(f for f in client.schema["fields"] if f["name"] == "dense")["params"]["dim"] = 64
    elif change == "revision":
        client.schema["description"] = "old embedding"
    elif change == "field":
        client.schema["fields"] = [f for f in client.schema["fields"] if f["name"] != "sparse"]
    else:
        client.describe_index = lambda **kw: {"field_name": kw["index_name"], "metric_type": "L2"}
    with pytest.raises(MilvusConfigurationError):
        await store(client).ensure_available()


async def test_failed_insert_cleanup_and_safe_error():
    client = Client()
    client.fail_insert = True
    with pytest.raises(AppError) as exc:
        await store(client).add_document("制度", "报销需要提供发票以及主管审批。" * 3)
    assert exc.value.status_code == 503
    assert "secret" not in exc.value.message
    assert not client.rows


async def test_embeddings_validated_before_any_write():
    client = Client()
    backend = store(client)
    backend.embedding_model.embed = lambda _: [float("nan")] * 32
    with pytest.raises(ValueError, match="Invalid embedding"):
        await backend.add_document("制度", "发票审批报销" * 30)
    assert not client.rows


async def test_health_and_empty_search():
    backend = store()
    assert (await backend.health())["persistent"] is True
    assert await backend.search(" ") == []
    backend._client.describe_collection = lambda **kw: (_ for _ in ()).throw(RuntimeError())
    assert (await backend.health())["connected"] is False


async def test_blocking_sdk_does_not_block_event_loop():
    import time
    backend = store()
    original = backend._client.load_collection
    backend._client.load_collection = lambda **kw: (time.sleep(.08), original(**kw))
    task = asyncio.create_task(backend.ensure_available())
    await asyncio.sleep(.01)
    assert not task.done()
    await task
