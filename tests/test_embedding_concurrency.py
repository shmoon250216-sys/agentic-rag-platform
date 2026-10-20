"""Gate-based concurrency tests: no remote model or timing speedup claims."""
import asyncio
import threading

import aiosqlite
import pytest

from app.memory.long_term_store import SQLiteLongTermMemoryStore
from app.rag.embedding import HashEmbeddingModel
from app.schemas.memory import MemoryType
from test_milvus_store import store


class GatedEmbedding(HashEmbeddingModel):
    def __init__(self):
        super().__init__(dimensions=32)
        self.wait_seconds = 3
        self.block_text = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.thread_id = None

    def embed(self, text):
        if self.block_text and self.block_text in text:
            self.thread_id = threading.get_ident()
            self.entered.set()
            try:
                if not self.release.wait(self.wait_seconds):
                    raise RuntimeError("Test gate timed out")
            finally:
                self.finished.set()
        return super().embed(text)


@pytest.mark.parametrize('operation', ['add', 'search', 'backfill'])
async def test_memory_embedding_leaves_event_loop_responsive(tmp_path, operation):
    model = GatedEmbedding()
    memory = SQLiteLongTermMemoryStore(f'sqlite:///{tmp_path / "memory.db"}', model)
    await memory.add('user', MemoryType.project_fact, '历史项目事实')
    if operation == 'backfill':
        async with aiosqlite.connect(memory.database_path) as db:
            await db.execute('UPDATE user_memories SET embedding_json = NULL')
            await db.commit()
    model.block_text = {'add': '新项目事实', 'search': '查询', 'backfill': '历史项目事实'}[operation]
    task = asyncio.create_task(memory.add('user', MemoryType.project_fact, '新项目事实')
                               if operation == 'add' else memory.search('user', '查询'))
    try:
        assert await asyncio.to_thread(model.entered.wait, 1)
        assert model.thread_id != threading.get_ident()
        # A completely separate async operation completes before the embedding is released.
        assert await asyncio.wait_for(memory.list_for_user('user'), 1)
        assert not task.done()
    finally:
        model.release.set()
        await task


@pytest.mark.parametrize('operation', ['upload', 'query'])
async def test_slow_embedding_does_not_hold_milvus_client_lock(operation):
    backend = store()
    original = await backend.add_document('原制度', '报销需要主管审批。' * 10)
    model = GatedEmbedding()
    backend.embedding_model = model
    model.block_text = '慢任务'
    task = asyncio.create_task(backend.add_document('慢任务', '新制度内容。' * 150)
                               if operation == 'upload' else backend.search('慢任务'))
    try:
        assert await asyncio.to_thread(model.entered.wait, 1)
        documents = await asyncio.wait_for(backend.list_documents(), 1)
        results = await asyncio.wait_for(backend.search('报销'), 1)
        assert documents == [original]
        assert results[0].doc_id == original.doc_id
        assert not task.done()
        # The upload is not visible before all its embeddings are ready.
        assert all(row['doc_id'] == original.doc_id for row in backend._client.rows.values())
    finally:
        model.release.set()
        await task
        await backend.close()


async def test_cancel_during_embedding_never_starts_insert():
    backend = store()
    model = GatedEmbedding()
    backend.embedding_model = model
    model.block_text = '慢任务'
    task = asyncio.create_task(backend.add_document('慢任务', '待入库内容。' * 5))
    try:
        assert await asyncio.to_thread(model.entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        model.release.set()
        assert await asyncio.to_thread(model.finished.wait, 1)
    assert not backend._client.rows
    await backend.close()
