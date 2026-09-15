from collections.abc import AsyncIterator
import asyncio
from weakref import WeakValueDictionary
from uuid import uuid4

from app.memory.context import build_context

from app.graph.workflow import AgentWorkflow
from app.core.errors import AppError
from app.memory.checkpoint_store import SQLiteGraphCheckpointStore
from app.memory.manager import LongTermMemoryManager
from app.memory.sqlite_store import SQLiteSessionStore
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.memory import MemoryCreate, MemoryType


class ChatService:
    def __init__(self) -> None:
        self.sessions = SQLiteSessionStore()
        self.checkpoints = SQLiteGraphCheckpointStore()
        self.memories = LongTermMemoryManager()
        self.workflow = AgentWorkflow()
        self._locks = WeakValueDictionary()

    def _lock(self, session_id):
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request = request.model_copy(update={"session_id": request.session_id or str(uuid4())})
        async with self._lock(request.session_id):
            return await self._chat(request)

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        session = await self.sessions.get_or_create(request.user_id, request.session_id)
        context = build_context(session.messages, request.message, session.summary)
        await self.sessions.append(session.session_id, "user", request.message)
        await self.memories.remember_from_message(request.user_id, request.message)
        memories = await self.memories.relevant_memories(request.user_id, context.query)
        response = await self.workflow.run(
            request,
            session.session_id,
            memories=[memory.content for memory in memories],
            context=context,
        )
        await self.sessions.append(session.session_id, "assistant", response.answer)
        await self.checkpoints.append(
            session_id=session.session_id,
            route=response.route.route.value,
            state={
                "user_id": request.user_id,
                "session_id": session.session_id,
                "message": request.message,
                "query": context.query,
                "history": context.history,
                "summary": context.summary,
                "memories": [memory.content for memory in memories],
                "route": response.route,
                "sources": response.sources,
                "tool_result": response.tool_result,
                "answer": response.answer,
            },
        )
        return response

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[tuple[str, object]]:
        request = request.model_copy(update={"session_id": request.session_id or str(uuid4())})
        async with self._lock(request.session_id):
            async for event in self._stream_chat(request):
                yield event

    async def _stream_chat(self, request: ChatRequest) -> AsyncIterator[tuple[str, object]]:
        session = await self.sessions.get_or_create(request.user_id, request.session_id)
        context = build_context(session.messages, request.message, session.summary)
        await self.sessions.append(session.session_id, "user", request.message)
        await self.memories.remember_from_message(request.user_id, request.message)
        memories = await self.memories.relevant_memories(request.user_id, context.query)
        memory_contents = [memory.content for memory in memories]

        final_response: ChatResponse | None = None
        async for event, payload in self.workflow.stream(
            request,
            session.session_id,
            memories=memory_contents,
            context=context,
        ):
            if event == "done":
                final_response = payload
            if event != "done":
                yield event, payload

        if final_response is None:
            return

        await self.sessions.append(session.session_id, "assistant", final_response.answer)
        await self.checkpoints.append(
            session_id=session.session_id,
            route=final_response.route.route.value,
            state={
                "user_id": request.user_id,
                "session_id": session.session_id,
                "message": request.message,
                "query": context.query,
                "history": context.history,
                "summary": context.summary,
                "memories": memory_contents,
                "route": final_response.route,
                "sources": final_response.sources,
                "tool_result": final_response.tool_result,
                "answer": final_response.answer,
                "streaming": True,
            },
        )

        yield "done", final_response

    async def list_sessions(self, user_id: str | None = None):
        return await self.sessions.list_sessions(user_id=user_id)

    async def list_messages(self, session_id: str):
        return await self.sessions.list_messages(session_id)

    async def list_checkpoints(self, session_id: str):
        return await self.checkpoints.list_for_session(session_id)

    async def get_checkpoint(self, checkpoint_id: int):
        return await self.checkpoints.get(checkpoint_id)

    async def list_memories(self, user_id: str):
        return await self.memories.store.list_for_user(user_id)

    async def add_memory(self, user_id: str, request: MemoryCreate):
        if not self.memories.is_safe_to_store(request.content):
            raise AppError(
                code="MEMORY_REJECTED",
                message="Sensitive information should not be stored as long-term memory",
                status_code=400,
            )
        if request.memory_type == MemoryType.preference:
            candidates = self.memories.extract_candidates("我希望以后" + request.content)
            canonical = [
                c for c in candidates if c.content.startswith(("回答语言：", "回答详略："))
            ]
            if canonical:
                saved = None
                for candidate in canonical:
                    saved = await self.memories.store.add(
                        user_id, candidate.memory_type, candidate.content, request.importance
                    )
                return saved
        return await self.memories.store.add(
            user_id=user_id,
            memory_type=request.memory_type,
            content=request.content,
            importance=request.importance,
        )

    async def delete_memory(self, memory_id: int, user_id: str | None = None) -> bool:
        return await self.memories.store.delete(memory_id, user_id=user_id)
