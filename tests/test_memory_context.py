import asyncio
import json

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.errors import AppError
from app.graph.workflow import AgentWorkflow
from app.llm.client import FakeLLMClient, OpenAICompatibleLLMClient
from app.memory.context import build_context, format_context
from app.memory.manager import LongTermMemoryManager
from app.memory.store import MessageRecord
from app.memory.sqlite_store import SQLiteSessionStore
from app.memory.long_term_store import SQLiteLongTermMemoryStore
from app.schemas.chat import ChatRequest, RouteName, SourceChunk
from app.schemas.memory import MemoryType
from app.services.chat_service import ChatService


@pytest.fixture
def memory_settings(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "sqlite_url", f"sqlite:///{tmp_path / 'context.db'}")
    monkeypatch.setattr(settings, "cache_enabled", False)
    return settings


class CaptureLLM(FakeLLMClient):
    def __init__(self):
        self.prompts = []

    async def chat(self, message):
        self.prompts.append(message)
        return "已收到本轮问题"

    async def answer_with_context(self, message, sources):
        self.prompts.append(message)
        return "依据制度需要部门负责人审批"


class CaptureRetriever:
    def __init__(self):
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return [
            SourceChunk(doc_id="policy", title="住宿例外", snippet="超标须部门负责人审批", score=1)
        ]


def service_with_capture():
    service = ChatService()
    service.workflow = AgentWorkflow(llm=CaptureLLM())
    service.workflow.retriever = CaptureRetriever()
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_followup_retrieval_and_actual_model_context(memory_settings, streaming):
    service = service_with_capture()
    first = await service.chat(ChatRequest(user_id="alice", message="住宿超过报销标准能报销吗？"))
    request = ChatRequest(user_id="alice", session_id=first.session_id, message="那需要谁审批？")
    if streaming:
        events = [event async for event in service.stream_chat(request)]
        response = events[-1][1]
        assert events[-1][0] == "done"
    else:
        response = await service.chat(request)
    assert response.route.route == RouteName.rag
    assert "住宿超过报销标准" in service.workflow.retriever.queries[-1]
    assert "那需要谁审批" in service.workflow.retriever.queries[-1]
    prompt = service.workflow.llm.prompts[-1]
    assert "住宿超过报销标准" in prompt
    assert "依据制度需要部门负责人审批" in prompt  # Actual previous assistant turn.
    assert prompt.count("那需要谁审批") == 1  # Current turn is not duplicated into history.
    checkpoints = await service.list_checkpoints(first.session_id)
    saved = await service.get_checkpoint(checkpoints[-1].checkpoint_id)
    assert saved.state["query"] == service.workflow.retriever.queries[-1]


@pytest.mark.asyncio
async def test_person_reference_has_history_and_does_not_leak_between_sessions(memory_settings):
    service = service_with_capture()
    first = await service.chat(ChatRequest(user_id="alice", message="张三是我的同学。"))
    await service.chat(
        ChatRequest(user_id="alice", session_id=first.session_id, message="他叫什么名字？")
    )
    assert "张三" in service.workflow.llm.prompts[-1]
    await service.chat(ChatRequest(user_id="bob", message="他叫什么名字？"))
    assert "张三" not in service.workflow.llm.prompts[-1]


@pytest.mark.asyncio
async def test_cross_user_session_denied_before_message_write(memory_settings):
    service = service_with_capture()
    response = await service.chat(ChatRequest(user_id="alice", message="你好"))
    with pytest.raises(AppError) as error:
        await service.chat(
            ChatRequest(user_id="bob", session_id=response.session_id, message="偷看")
        )
    assert error.value.status_code == 403
    assert len(await service.list_messages(response.session_id)) == 2


def test_context_limits_summary_sources_and_new_topic(memory_settings):
    messages = [MessageRecord("user", f"历史事实{i}：" + "中" * 1000) for i in range(20)]
    messages.insert(1, MessageRecord("assistant", "错误猜测：报销永远不需要审批"))
    context = build_context(messages, "新的问题是什么？")
    assert len(context.history) <= 6
    assert sum(len(item["content"]) for item in context.history) <= 6000
    assert len(context.summary) <= 1200
    assert "错误猜测" not in context.summary
    assert context.query == "新的问题是什么？"
    prompt = format_context(
        {
            "message": "问题",
            "history": context.history,
            "summary": context.summary,
            "memories": ["长" * 10000] * 10,
        }
    )
    assert len(prompt) < 10000


@pytest.mark.asyncio
async def test_preferences_update_survive_restart_and_are_always_recalled(memory_settings):
    manager = LongTermMemoryManager()
    await manager.remember_from_message("alice", "我希望以后用中文详细解释。")
    await manager.remember_from_message("alice", "以后请简短回答。")
    await manager.remember_from_message("alice", "这次请详细解释代码。")
    restarted = LongTermMemoryManager()
    memories = await restarted.relevant_memories("alice", "住宿标准是多少")
    contents = [m.content for m in memories]
    assert "回答语言：中文" in contents
    assert "回答详略：简洁" in contents
    assert not any("详细" in content for content in contents)
    assert await restarted.relevant_memories("bob", "住宿标准是多少") == []


@pytest.mark.asyncio
async def test_cross_session_memory_in_prompt(memory_settings):
    service = service_with_capture()
    await service.chat(ChatRequest(user_id="alice", message="以后请用英文回答。"))
    service = service_with_capture()  # Persisted store, new workflow and session.
    await service.chat(ChatRequest(user_id="alice", message="你好"))
    assert "回答语言：英文" in service.workflow.llm.prompts[-1]


@pytest.mark.asyncio
async def test_memory_capacity_expiry_and_user_isolation(memory_settings, monkeypatch):
    monkeypatch.setattr(memory_settings, "memory_max_items", 5)
    store = SQLiteLongTermMemoryStore()
    for i in range(8):
        await store.add("alice", MemoryType.project_fact, f"项目事实{i}")
    await store.add("bob", MemoryType.project_fact, "另一个用户的项目事实")
    assert len(await store.list_for_user("alice")) == 5
    async with aiosqlite.connect(store.database_path) as db:
        await db.execute(
            "UPDATE user_memories SET updated_at = '2000-01-01' WHERE user_id = 'alice'"
        )
        await db.commit()
    assert await store.search("alice", "项目事实") == []
    assert len(await store.list_for_user("bob")) == 1


@pytest.mark.asyncio
async def test_session_retention_digest_and_snapshot_retention(memory_settings, monkeypatch):
    monkeypatch.setattr(memory_settings, "session_max_messages", 10)
    monkeypatch.setattr(memory_settings, "session_max_checkpoints", 3)
    service = service_with_capture()
    session_id = None
    for i in range(8):
        response = await service.chat(
            ChatRequest(user_id="alice", session_id=session_id, message=f"讨论话题{i}")
        )
        session_id = response.session_id
    session = await SQLiteSessionStore().get_or_create("alice", session_id)
    assert len(session.messages) == 10
    assert "讨论话题0" in session.summary
    assert len(session.summary) <= 1200
    assert len(await service.list_checkpoints(session_id)) == 3


@pytest.mark.asyncio
async def test_same_session_requests_are_serialized(memory_settings):
    service = service_with_capture()
    first = await service.chat(ChatRequest(user_id="alice", message="起始问题"))
    await asyncio.gather(
        *[
            service.chat(
                ChatRequest(user_id="alice", session_id=first.session_id, message=f"后续消息{i}")
            )
            for i in range(3)
        ]
    )
    messages = await service.list_messages(first.session_id)
    assert [m.role for m in messages] == ["user", "assistant"] * 4


@pytest.mark.asyncio
async def test_history_does_not_replay_a_tool_action(memory_settings):
    service = service_with_capture()
    first = await service.chat(ChatRequest(user_id="alice", message="计算 12 + 30"))
    response = await service.chat(
        ChatRequest(user_id="alice", session_id=first.session_id, message="那为什么呢？")
    )
    assert response.route.route == RouteName.chat
    assert response.tool_result is None


def test_sensitive_and_transient_information_not_promoted(memory_settings):
    manager = LongTermMemoryManager()
    assert manager.extract_candidates("我的密码是123456，请记住") == []
    assert manager.extract_candidates("这次我希望用中文详细回答") == []
    candidates = manager.extract_candidates("我的项目使用FastAPI。今天吃什么？")
    assert len(candidates) == 1
    assert "吃什么" not in candidates[0].content


@pytest.mark.asyncio
async def test_http_payload_contains_context_in_both_modes(memory_settings, monkeypatch):
    import httpx

    payloads = []

    def handler(request):
        data = json.loads(request.content)
        payloads.append(data)
        if data.get("stream"):
            return httpx.Response(
                200, text='data: {"choices":[{"delta":{"content":"完成"}}]}\n\ndata: [DONE]\n\n'
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "完成"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(memory_settings, "llm_api_key", "test-only-key")
    llm = OpenAICompatibleLLMClient(memory_settings)
    prompt = format_context(
        {
            "message": "他是谁",
            "history": [{"role": "user", "content": "张三是同学"}],
            "memories": ["回答语言：中文"],
        }
    )
    await llm.chat(prompt)
    assert "完成" == "".join([text async for text in llm.stream_chat(prompt)])
    for payload in payloads:
        assert "张三" in payload["messages"][1]["content"]
        assert "回答语言" in payload["messages"][1]["content"]


def test_http_rejects_cross_user_session_in_normal_and_stream(memory_settings, monkeypatch):
    from app.api import routes
    from app.main import create_app

    monkeypatch.setattr(routes, "chat_service", service_with_capture())
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}
    first = client.post(
        "/api/v1/chat", headers=headers, json={"user_id": "alice", "message": "你好"}
    ).json()
    for path in ("/api/v1/chat", "/api/v1/chat/stream"):
        response = client.post(
            path,
            headers=headers,
            json={"user_id": "bob", "session_id": first["session_id"], "message": "你好"},
        )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_legacy_database_migration_and_preference_replacement(memory_settings):
    path = memory_settings.sqlite_url.removeprefix("sqlite:///")
    async with aiosqlite.connect(path) as db:
        await db.execute("""CREATE TABLE user_memories (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, memory_type TEXT,
            content TEXT, importance REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, memory_type, content))""")
        await db.execute("""INSERT INTO user_memories(user_id,memory_type,content,importance)
            VALUES ('alice','preference','用户偏好：我希望使用中文详细解释',0.8)""")
        await db.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, user_id TEXT, title TEXT, created_at TEXT, updated_at TEXT)"
        )
        await db.execute(
            "INSERT INTO sessions VALUES ('old-session','alice','旧会话',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
        )
        await db.commit()
    manager = LongTermMemoryManager()
    await manager.remember_from_message("alice", "以后请简短回答")
    contents = [m.content for m in await manager.relevant_memories("alice", "你好")]
    assert set(contents) == {"回答语言：中文", "回答详略：简洁"}
    session = await SQLiteSessionStore().get_or_create("alice", "old-session")
    assert session.summary == ""


@pytest.mark.asyncio
async def test_real_local_retrieval_uses_followup_topic(memory_settings):
    from app.rag.store import InMemoryKnowledgeBase
    from app.rag.retriever import KnowledgeBaseRetriever

    store = InMemoryKnowledgeBase()
    target = store.add_document("住宿报销制度", "住宿超过报销标准时，必须由部门负责人审批。" * 3)
    store.add_document("采购管理", "采购合同需要采购负责人审批。" * 3)
    service = service_with_capture()
    service.workflow.retriever = KnowledgeBaseRetriever(store=store)
    first = await service.chat(ChatRequest(user_id="alice", message="住宿超过报销标准怎么办？"))
    second = await service.chat(
        ChatRequest(user_id="alice", session_id=first.session_id, message="那需要谁审批？")
    )
    assert second.sources[0].doc_id == target.doc_id
    assert (
        await service.workflow.checkpointer.aget_tuple(
            {"configurable": {"thread_id": first.session_id}}
        )
        is None
    )


@pytest.mark.asyncio
async def test_concurrent_first_use_schema_and_slot_updates(memory_settings):
    stores = [SQLiteLongTermMemoryStore() for _ in range(3)]
    await asyncio.gather(
        *[
            store.add("alice", MemoryType.preference, f"回答详略：{value}")
            for store, value in zip(stores, ["详细", "简洁", "详细"])
        ]
    )
    memories = await stores[0].list_for_user("alice")
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_manual_preferences_share_update_policy(memory_settings):
    from app.schemas.memory import MemoryCreate

    service = service_with_capture()
    await service.add_memory("alice", MemoryCreate(content="以后请用中文，并且详细解释"))
    await service.add_memory("alice", MemoryCreate(content="简短回答"))
    contents = [m.content for m in await service.list_memories("alice")]
    assert set(contents) == {"回答语言：中文", "回答详略：简洁"}
