from app.rag.store import InMemoryKnowledgeBase
from app.rag.text_splitter import split_text


def test_split_text_creates_overlapping_chunks() -> None:
    chunks = split_text("a" * 1200, chunk_size=500, overlap=80)

    assert len(chunks) == 3
    assert chunks[0][-80:] == chunks[1][:80]


def test_knowledge_base_adds_and_searches_document() -> None:
    store = InMemoryKnowledgeBase()
    summary = store.add_document(
        title="Redis RAG 说明",
        content="Redis Stack 可以保存文本片段和向量索引，用来支持 RAG 检索增强生成。",
    )

    results = store.search("Redis 向量检索", top_k=2)

    assert summary.chunk_count == 1
    assert results
    assert results[0].title == "Redis RAG 说明"


def test_knowledge_base_uses_vector_signal_for_ranking() -> None:
    store = InMemoryKnowledgeBase()
    store.add_document(
        title="LangGraph 工作流",
        content="Supervisor 节点负责请求路由，并把问题分发到 RAG、工具和闲聊分支。",
    )
    store.add_document(
        title="生活记录",
        content="今天晚饭准备吃米饭和青菜，饭后可以散步。",
    )

    results = store.search("Supervisor 路由 LangGraph", top_k=1)

    assert results[0].title == "LangGraph 工作流"


def test_knowledge_base_deletes_document() -> None:
    store = InMemoryKnowledgeBase()
    summary = store.add_document(
        title="临时文档",
        content="这是一段用于测试删除能力的临时知识库内容，长度满足最小要求。",
    )

    assert store.delete_document(summary.doc_id) is True
    assert store.list_documents() == []
