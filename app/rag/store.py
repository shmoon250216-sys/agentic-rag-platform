from dataclasses import dataclass
from uuid import uuid4

from app.rag.embedding import EmbeddingModel, HashEmbeddingModel, cosine_similarity, tokenize
from app.rag.text_splitter import split_text
from app.schemas.chat import SourceChunk
from app.schemas.document import DocumentSummary


@dataclass
class StoredChunk:
    doc_id: str
    title: str
    chunk_id: str
    content: str
    embedding: list[float]


@dataclass
class StoredDocument:
    doc_id: str
    title: str
    content: str
    chunks: list[StoredChunk]


class InMemoryKnowledgeBase:
    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self.embedding_model = embedding_model or HashEmbeddingModel()

    def add_document(self, title: str, content: str) -> DocumentSummary:
        doc_id = str(uuid4())
        chunks = [
            StoredChunk(
                doc_id=doc_id,
                title=title,
                chunk_id=f"{doc_id}:{index}",
                content=chunk,
                embedding=self.embedding_model.embed(f"{title}\n{chunk}"),
            )
            for index, chunk in enumerate(split_text(content))
        ]
        document = StoredDocument(doc_id=doc_id, title=title, content=content, chunks=chunks)
        self._documents[doc_id] = document
        return self._to_summary(document)

    def list_documents(self) -> list[DocumentSummary]:
        return [self._to_summary(document) for document in self._documents.values()]

    def delete_document(self, doc_id: str) -> bool:
        return self._documents.pop(doc_id, None) is not None

    def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        query_embedding = self.embedding_model.embed(query)
        scored: list[tuple[float, StoredChunk]] = []
        for document in self._documents.values():
            for chunk in document.chunks:
                searchable_text = f"{chunk.title}\n{chunk.content}"
                lexical_score = _lexical_score(query, query_terms, searchable_text)
                vector_score = cosine_similarity(query_embedding, chunk.embedding)
                score = _hybrid_score(vector_score, lexical_score)
                if score > 0:
                    scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SourceChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                snippet=chunk.content[:220],
                score=round(min(score, 1.0), 4),
            )
            for score, chunk in scored[:top_k]
        ]

    def seed_defaults(self) -> None:
        if self._documents:
            return
        self.add_document(
            title="项目架构说明",
            content=(
                "系统通过 Supervisor 将用户请求路由到 RAG、工具、闲聊和兜底分支。"
                "当前后端使用 FastAPI 接收请求，使用 Pydantic 校验数据，"
                "使用 LangGraph StateGraph 编排 Agent 工作流。"
            ),
        )
        self.add_document(
            title="学习路线",
            content=(
                "先完成可运行闭环，再逐步接入 LangGraph、Redis、MCP 和评测体系。"
                "RAG 阶段需要完成文档上传、文本切分、检索召回和来源引用。"
            ),
        )

    def _to_summary(self, document: StoredDocument) -> DocumentSummary:
        return DocumentSummary(
            doc_id=document.doc_id,
            title=document.title,
            chunk_count=len(document.chunks),
            content_length=len(document.content),
        )


def _lexical_score(query: str, query_terms: set[str], content: str) -> float:
    lowered_content = content.lower()
    content_terms = tokenize(content)
    overlap = query_terms & content_terms
    if not overlap:
        return 0

    lexical_score = len(overlap) / max(len(query_terms), 1)
    exact_boost = 0.25 if query.lower() in lowered_content else 0
    title_boost = 0.1 if any(term in lowered_content for term in query_terms if len(term) >= 2) else 0
    return lexical_score + exact_boost + title_boost


def _hybrid_score(vector_score: float, lexical_score: float) -> float:
    normalized_vector_score = max(vector_score, 0.0)
    return 0.7 * normalized_vector_score + 0.3 * lexical_score


knowledge_base = InMemoryKnowledgeBase()
knowledge_base.seed_defaults()
