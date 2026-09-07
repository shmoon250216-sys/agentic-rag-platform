import pytest

from app.memory.long_term_store import SQLiteLongTermMemoryStore
from app.memory.manager import LongTermMemoryManager
from app.schemas.memory import MemoryType


@pytest.mark.asyncio
async def test_long_term_memory_store_adds_searches_and_dedupes(tmp_path) -> None:
    store = SQLiteLongTermMemoryStore(f"sqlite:///{tmp_path / 'memory.db'}")

    first = await store.add(
        user_id="user-1",
        memory_type=MemoryType.preference,
        content="用户偏好：希望使用中文详细解释",
        importance=0.8,
    )
    second = await store.add(
        user_id="user-1",
        memory_type=MemoryType.preference,
        content="用户偏好：希望使用中文详细解释",
        importance=0.7,
    )

    results = await store.search(user_id="user-1", query="中文解释")

    assert first.memory_id == second.memory_id
    assert len(await store.list_for_user("user-1")) == 1
    assert results[0].content == "用户偏好：希望使用中文详细解释"


@pytest.mark.asyncio
async def test_long_term_memory_search_uses_vector_signal(tmp_path) -> None:
    store = SQLiteLongTermMemoryStore(f"sqlite:///{tmp_path / 'memory.db'}")
    await store.add(
        user_id="user-1",
        memory_type=MemoryType.project_fact,
        content="项目事实：LangGraph Supervisor 负责路由 RAG 工具和闲聊分支",
        importance=0.7,
    )
    await store.add(
        user_id="user-1",
        memory_type=MemoryType.preference,
        content="用户偏好：回答时先给结论再解释原因",
        importance=0.7,
    )

    results = await store.search(user_id="user-1", query="Supervisor 路由", limit=1)

    assert results[0].memory_type == MemoryType.project_fact


def test_memory_manager_extracts_useful_long_term_memory() -> None:
    manager = LongTermMemoryManager()

    candidates = manager.extract_candidates("我希望以后都用中文详细解释，我的目标是把报销制度接入知识库")

    assert {candidate.memory_type for candidate in candidates} == {
        MemoryType.preference,
        MemoryType.goal,
    }


def test_memory_manager_rejects_sensitive_text() -> None:
    manager = LongTermMemoryManager()

    candidates = manager.extract_candidates("我的 api key 是 sk-test1234567890")

    assert candidates == []
