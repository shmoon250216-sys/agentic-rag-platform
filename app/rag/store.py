from dataclasses import dataclass
from uuid import uuid4

from app.rag.embedding import (
    EmbeddingModel,
    HashEmbeddingModel,
    cosine_similarity,
    tokenize,
    tokenize_for_search,
)
from app.rag.ranking import BM25Ranker, RankedItem, normalize_scores, reciprocal_rank_fusion
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
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        *,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 0.1,
    ) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

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
        query_terms = tokenize_for_search(query)
        if not query_terms:
            return []

        chunks = [
            chunk
            for document in self._documents.values()
            for chunk in document.chunks
        ]
        if not chunks:
            return []

        candidate_count = min(max(top_k * self.candidate_multiplier, top_k), len(chunks))
        query_embedding = self.embedding_model.embed(query)
        vector_ranking = sorted(
            (
                RankedItem(
                    item=chunk.chunk_id,
                    score=max(cosine_similarity(query_embedding, chunk.embedding), 0.0),
                )
                for chunk in chunks
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        vector_ranking = [result for result in vector_ranking if result.score > 0][
            :candidate_count
        ]

        corpus = [tokenize_for_search(f"{chunk.title}\n{chunk.content}") for chunk in chunks]
        bm25_scores = BM25Ranker(corpus).scores(query_terms)
        bm25_ranking = sorted(
            (
                RankedItem(item=chunk.chunk_id, score=score)
                for chunk, score in zip(chunks, bm25_scores)
                if score > 0
            ),
            key=lambda result: result.score,
            reverse=True,
        )[:candidate_count]

        fused = normalize_scores(
            reciprocal_rank_fusion(
                [bm25_ranking, vector_ranking],
                rrf_k=self.rrf_k,
                weights=[self.bm25_weight, self.vector_weight],
            )
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return [
            SourceChunk(
                doc_id=chunks_by_id[result.item].doc_id,
                title=chunks_by_id[result.item].title,
                snippet=chunks_by_id[result.item].content[:2000],
                score=round(result.score, 4),
            )
            for result in fused[:top_k]
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
