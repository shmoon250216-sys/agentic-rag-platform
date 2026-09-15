import inspect
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.api.deps import verify_token
from app.core.config import get_settings
from app.core.errors import AppError
from app.evaluation.quality_gate import QualityGateReport, evaluate_quality_gate
from app.evaluation.runner import EvaluationRunner
from app.llm.client import llm_response_cache
from app.rag.factory import get_knowledge_base
from app.rag.document_parser import DocumentParseError, parse_uploaded_document
from app.rag.redis_store import RedisKnowledgeBase
from app.rag.retriever import retrieval_cache
from app.schemas.checkpoint import GraphCheckpointDetailResponse, GraphCheckpointListResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import DocumentCreate, DocumentCreateResponse, DocumentListResponse
from app.schemas.memory import MemoryCreate, MemoryCreateResponse, MemoryListResponse
from app.schemas.session import MessageListResponse, SessionListResponse
from app.schemas.tool import ToolCallResult, ToolInvokeRequest, ToolListResponse
from app.services.chat_service import ChatService
from app.tools.registry import ToolRegistry

router = APIRouter()
chat_service = ChatService()
tool_registry = ToolRegistry()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict[str, object]:
    """Report whether dependencies required by the selected RAG backend are ready."""
    rag_status = await _rag_status(require_available=True)
    return {"status": "ready", "rag": rag_status}


@router.get("/api/v1/system/info", dependencies=[Depends(verify_token)])
async def system_info() -> dict[str, object]:
    """Expose non-sensitive runtime information for the local operations panel."""
    settings = get_settings()
    return {
        "app": {
            "name": settings.app_name,
            "environment": settings.app_env,
            "version": "0.1.0",
        },
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        },
        "rag": await _rag_status(),
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
        "memory": {
            "session_store": "sqlite",
            "graph_checkpointer": "langgraph-memory + sqlite-audit",
        },
    }


@router.get("/api/v1/cache/stats", dependencies=[Depends(verify_token)])
async def cache_stats() -> dict[str, dict[str, int]]:
    return {
        "retrieval": retrieval_cache.stats(),
        "llm_response": llm_response_cache.stats(),
    }


@router.delete("/api/v1/cache", dependencies=[Depends(verify_token)])
async def clear_cache() -> dict[str, bool]:
    retrieval_cache.clear()
    llm_response_cache.clear()
    return {"cleared": True}


@router.get(
    "/api/v1/quality/gate",
    response_model=QualityGateReport,
    dependencies=[Depends(verify_token)],
)
async def quality_gate() -> QualityGateReport:
    report = await EvaluationRunner().run()
    return evaluate_quality_gate(report)


@router.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest) -> ChatResponse:
    return await chat_service.chat(request)


@router.post("/api/v1/chat/stream", dependencies=[Depends(verify_token)])
async def stream_chat(request: ChatRequest) -> EventSourceResponse:
    if request.session_id:
        # Reject a mismatched session before HTTP/SSE headers are sent.
        await chat_service.sessions.get_or_create(request.user_id, request.session_id)
    async def event_generator() -> AsyncIterator[dict[str, str]]:
        yield _sse_event("start", {"message": "workflow_started"})
        async for event, payload in chat_service.stream_chat(request):
            if event == "route":
                yield _sse_event("route", payload.model_dump(mode="json"))
            elif event == "token":
                yield _sse_event("token", {"text": payload})
            elif event == "done":
                yield _sse_event("done", payload.model_dump(mode="json"))

    return EventSourceResponse(event_generator())


@router.get(
    "/api/v1/sessions",
    response_model=SessionListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_sessions(user_id: str | None = None) -> SessionListResponse:
    return SessionListResponse(sessions=await chat_service.list_sessions(user_id=user_id))


@router.get(
    "/api/v1/sessions/{session_id}/messages",
    response_model=MessageListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_messages(session_id: str) -> MessageListResponse:
    return MessageListResponse(messages=await chat_service.list_messages(session_id))


@router.get(
    "/api/v1/sessions/{session_id}/checkpoints",
    response_model=GraphCheckpointListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_checkpoints(session_id: str) -> GraphCheckpointListResponse:
    return GraphCheckpointListResponse(
        checkpoints=await chat_service.list_checkpoints(session_id),
    )


@router.get(
    "/api/v1/checkpoints/{checkpoint_id}",
    response_model=GraphCheckpointDetailResponse,
    dependencies=[Depends(verify_token)],
)
async def get_checkpoint(checkpoint_id: int) -> GraphCheckpointDetailResponse:
    checkpoint = await chat_service.get_checkpoint(checkpoint_id)
    if checkpoint is None:
        raise AppError(
            code="CHECKPOINT_NOT_FOUND",
            message="Graph checkpoint not found",
            status_code=404,
        )
    return GraphCheckpointDetailResponse(checkpoint=checkpoint)


@router.get(
    "/api/v1/users/{user_id}/memories",
    response_model=MemoryListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_memories(user_id: str) -> MemoryListResponse:
    return MemoryListResponse(memories=await chat_service.list_memories(user_id))


@router.post(
    "/api/v1/users/{user_id}/memories",
    response_model=MemoryCreateResponse,
    dependencies=[Depends(verify_token)],
)
async def create_memory(user_id: str, request: MemoryCreate) -> MemoryCreateResponse:
    return MemoryCreateResponse(memory=await chat_service.add_memory(user_id, request))


@router.delete("/api/v1/users/{user_id}/memories/{memory_id}", dependencies=[Depends(verify_token)])
async def delete_memory(user_id: str, memory_id: int) -> dict[str, bool]:
    return {"deleted": await chat_service.delete_memory(memory_id, user_id=user_id)}


@router.get(
    "/api/v1/tools",
    response_model=ToolListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_tools() -> ToolListResponse:
    return ToolListResponse(tools=tool_registry.list_tools())


@router.post(
    "/api/v1/tools/{tool_name}/invoke",
    response_model=ToolCallResult,
    dependencies=[Depends(verify_token)],
)
async def invoke_tool(tool_name: str, request: ToolInvokeRequest) -> ToolCallResult:
    return await tool_registry.call_tool(tool_name, request.arguments)


@router.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_documents() -> DocumentListResponse:
    documents = await _maybe_await(get_knowledge_base().list_documents())
    return DocumentListResponse(
        documents=documents,
        total_chunks=sum(document.chunk_count for document in documents),
    )


@router.post(
    "/api/v1/documents",
    response_model=DocumentCreateResponse,
    dependencies=[Depends(verify_token)],
)
async def create_document(request: DocumentCreate) -> DocumentCreateResponse:
    document = await _maybe_await(
        get_knowledge_base().add_document(request.title, request.content)
    )
    _clear_document_caches()
    return DocumentCreateResponse(document=document)


@router.post(
    "/api/v1/documents/upload",
    response_model=DocumentCreateResponse,
    dependencies=[Depends(verify_token)],
)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=120),
) -> DocumentCreateResponse:
    settings = get_settings()
    filename = file.filename or ""
    try:
        data = await file.read(settings.document_upload_max_bytes + 1)
    finally:
        await file.close()

    if len(data) > settings.document_upload_max_bytes:
        raise AppError(
            code="DOCUMENT_FILE_TOO_LARGE",
            message="文件不能超过 10MB",
            status_code=413,
        )

    try:
        parsed = parse_uploaded_document(
            filename,
            data,
            max_characters=settings.document_extract_max_chars,
            max_pdf_pages=settings.document_pdf_max_pages,
            max_docx_uncompressed_bytes=settings.document_docx_max_uncompressed_bytes,
        )
    except DocumentParseError as exc:
        raise AppError(code=exc.code, message=exc.message, status_code=400) from exc

    document_title = title.strip() if title and title.strip() else parsed.title
    document = await _maybe_await(
        get_knowledge_base().add_document(document_title, parsed.content)
    )
    _clear_document_caches()
    return DocumentCreateResponse(document=document)


@router.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(verify_token)])
async def delete_document(doc_id: str) -> dict[str, bool]:
    deleted = await _maybe_await(get_knowledge_base().delete_document(doc_id))
    if deleted:
        _clear_document_caches()
    return {"deleted": deleted}


def _sse_event(event: str, payload: dict) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(payload, ensure_ascii=False),
    }


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _rag_status(require_available: bool = False) -> dict[str, object]:
    store = get_knowledge_base()
    if isinstance(store, RedisKnowledgeBase):
        if require_available:
            await store.ensure_available()
        return await store.health()
    return {"backend": "memory", "connected": True, "persistent": False}


def _clear_document_caches() -> None:
    retrieval_cache.clear()
    llm_response_cache.clear()
