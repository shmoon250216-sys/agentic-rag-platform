from collections.abc import AsyncIterator

from app.graph.workflow import AgentWorkflow
from app.core.errors import AppError
from app.memory.checkpoint_store import SQLiteGraphCheckpointStore
from app.memory.manager import LongTermMemoryManager
from app.memory.sqlite_store import SQLiteSessionStore
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.memory import MemoryCreate


class ChatService:
    def __init__(self) -> None:
        self.sessions = SQLiteSessionStore()
        self.checkpoints = SQLiteGraphCheckpointStore()
        self.memories = LongTermMemoryManager()
        self.workflow = AgentWorkflow()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session = await self.sessions.get_or_create(request.user_id, request.session_id)
        await self.sessions.append(session.session_id, "user", request.message)
        await self.memories.remember_from_message(request.user_id, request.message)
        memories = await self.memories.relevant_memories(request.user_id, request.message)
        response = await self.workflow.run(
            request,
            session.session_id,
            memories=[memory.content for memory in memories],
        )
        await self.sessions.append(session.session_id, "assistant", response.answer)
        await self.checkpoints.append(
            session_id=session.session_id,
            route=response.route.route.value,
            state={
                "user_id": request.user_id,
                "session_id": session.session_id,
                "message": request.message,
                "memories": [memory.content for memory in memories],
                "route": response.route,
                "sources": response.sources,
                "tool_result": response.tool_result,
                "answer": response.answer,
            },
        )
        return response

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[tuple[str, object]]:
        session = await self.sessions.get_or_create(request.user_id, request.session_id)
        await self.sessions.append(session.session_id, "user", request.message)
        await self.memories.remember_from_message(request.user_id, request.message)
        memories = await self.memories.relevant_memories(request.user_id, request.message)
        memory_contents = [memory.content for memory in memories]

        final_response: ChatResponse | None = None
        async for event, payload in self.workflow.stream(
            request,
            session.session_id,
            memories=memory_contents,
        ):
            if event == "done":
                final_response = payload
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
                "memories": memory_contents,
                "route": final_response.route,
                "sources": final_response.sources,
                "tool_result": final_response.tool_result,
                "answer": final_response.answer,
                "streaming": True,
            },
        )

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
        return await self.memories.store.add(
            user_id=user_id,
            memory_type=request.memory_type,
            content=request.content,
            importance=request.importance,
        )

    async def delete_memory(self, memory_id: int, user_id: str | None = None) -> bool:
        return await self.memories.store.delete(memory_id, user_id=user_id)
