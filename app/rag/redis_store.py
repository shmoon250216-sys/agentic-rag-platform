import asyncio
from array import array
from dataclasses import dataclass
from uuid import uuid4

import redis.asyncio as redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError

from app.core.errors import AppError
from app.rag.embedding import EmbeddingModel, HashEmbeddingModel, tokenize
from app.rag.ranking import RankedItem, normalize_scores, reciprocal_rank_fusion
from app.rag.text_splitter import split_text
from app.schemas.chat import SourceChunk
from app.schemas.document import DocumentSummary


@dataclass(frozen=True)
class RedisRagDefaults:
    chunk_prefix: str = "rag:chunk:"
    doc_prefix: str = "rag:doc:"
    seed_key: str = "rag:seeded"


class RedisBackendUnavailable(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="RAG_BACKEND_UNAVAILABLE",
            message="Redis RAG 知识库暂时不可用，请检查 Redis Stack 服务",
            status_code=503,
        )


class RedisIndexConfigurationError(AppError):
    def __init__(self, actual: int, expected: int) -> None:
        super().__init__(
            code="RAG_INDEX_DIMENSION_MISMATCH",
            message=(
                "Redis 向量索引维度与当前 Embedding 配置不一致："
                f"索引为 {actual} 维，配置为 {expected} 维，请重建索引"
            ),
            status_code=503,
        )


class RedisKnowledgeBase:
    """Redis Stack backed knowledge base using hashes plus RediSearch vectors."""

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        vector_dimensions: int,
        embedding_model: EmbeddingModel | None = None,
        defaults: RedisRagDefaults | None = None,
        connect_timeout: float = 2,
        socket_timeout: float = 5,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 0.1,
    ) -> None:
        self.redis = redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
            health_check_interval=30,
        )
        self.index_name = index_name
        self.embedding_model = embedding_model or HashEmbeddingModel(dimensions=vector_dimensions)
        self.vector_dimensions = vector_dimensions
        self.defaults = defaults or RedisRagDefaults()
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def add_document(self, title: str, content: str) -> DocumentSummary:
        try:
            await self._ensure_ready()
            return await self._add_document(title, content)
        except RedisError as exc:
            raise RedisBackendUnavailable() from exc

    async def list_documents(self) -> list[DocumentSummary]:
        try:
            await self._ensure_ready()
            summaries: list[DocumentSummary] = []
            async for key in self.redis.scan_iter(f"{self.defaults.doc_prefix}*"):
                data = await self.redis.hgetall(key)
                summaries.append(
                    DocumentSummary(
                        doc_id=_decode(data[b"doc_id"]),
                        title=_decode(data[b"title"]),
                        chunk_count=int(_decode(data[b"chunk_count"])),
                        content_length=int(_decode(data[b"content_length"])),
                    )
                )
            return sorted(summaries, key=lambda item: item.title)
        except RedisError as exc:
            raise RedisBackendUnavailable() from exc

    async def delete_document(self, doc_id: str) -> bool:
        try:
            await self._ensure_ready()
            keys: list[bytes | str] = [f"{self.defaults.doc_prefix}{doc_id}"]
            async for key in self.redis.scan_iter(f"{self.defaults.chunk_prefix}{doc_id}:*"):
                keys.append(key)
            deleted = await self.redis.delete(*keys)
            return deleted > 0
        except RedisError as exc:
            raise RedisBackendUnavailable() from exc

    async def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        try:
            await self._ensure_ready()
            query_terms = tokenize(query)
            if not query_terms:
                return []

            candidate_count = max(top_k * self.candidate_multiplier, top_k)
            vector_query = (
                Query(
                    f"(*)=>[KNN {candidate_count} "
                    "@embedding $vector AS vector_distance]"
                )
                .sort_by("vector_distance")
                .return_fields("doc_id", "chunk_id", "title", "content", "vector_distance")
                .paging(0, candidate_count)
                .dialect(2)
            )
            lexical_query = (
                Query(_build_full_text_query(query_terms))
                .with_scores()
                .scorer("BM25STD")
                .language("chinese")
                .return_fields("doc_id", "chunk_id", "title", "content")
                .paging(0, candidate_count)
                .dialect(2)
            )
            vector_result, lexical_result = await asyncio.gather(
                self.redis.ft(self.index_name).search(
                    vector_query,
                    {"vector": _vector_to_bytes(self._embed(query))},
                ),
                self.redis.ft(self.index_name).search(lexical_query),
            )
        except RedisError as exc:
            raise RedisBackendUnavailable() from exc

        candidates: dict[str, SourceChunk] = {}
        vector_ranking: list[RankedItem[str]] = []
        for doc in vector_result.docs:
            chunk_id = _decode(doc.chunk_id)
            vector_score = max(0.0, 1.0 - float(_decode(doc.vector_distance)))
            if vector_score <= 0:
                continue
            candidates[chunk_id] = _source_from_redis_document(doc)
            vector_ranking.append(RankedItem(item=chunk_id, score=vector_score))

        bm25_ranking: list[RankedItem[str]] = []
        for doc in lexical_result.docs:
            chunk_id = _decode(doc.chunk_id)
            bm25_score = float(_decode(doc.score))
            if bm25_score <= 0:
                continue
            candidates[chunk_id] = _source_from_redis_document(doc)
            bm25_ranking.append(RankedItem(item=chunk_id, score=bm25_score))

        fused = normalize_scores(
            reciprocal_rank_fusion(
                [bm25_ranking, vector_ranking],
                rrf_k=self.rrf_k,
                weights=[self.bm25_weight, self.vector_weight],
            )
        )
        return [
            candidates[result.item].model_copy(update={"score": round(result.score, 4)})
            for result in fused[:top_k]
        ]

    async def health(self) -> dict[str, object]:
        try:
            await self.redis.ping()
        except RedisError:
            return {"backend": "redis", "connected": False, "index": self.index_name}
        return {"backend": "redis", "connected": True, "index": self.index_name}

    async def ensure_available(self) -> None:
        try:
            await self.redis.ping()
        except RedisError as exc:
            raise RedisBackendUnavailable() from exc

    async def close(self) -> None:
        await self.redis.aclose()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        async with self._ready_lock:
            if self._ready:
                return
            await self._ensure_index()
            seeded = await self.redis.exists(self.defaults.seed_key)
            if not seeded:
                await self._seed_defaults()
                await self.redis.set(self.defaults.seed_key, "1")
            self._ready = True

    async def _ensure_index(self) -> None:
        try:
            index_info = await self.redis.ft(self.index_name).info()
        except ResponseError:
            pass
        else:
            actual_dimensions = _extract_vector_dimensions(index_info, "embedding")
            if actual_dimensions is not None and actual_dimensions != self.vector_dimensions:
                raise RedisIndexConfigurationError(
                    actual=actual_dimensions,
                    expected=self.vector_dimensions,
                )
            return

        await self.redis.ft(self.index_name).create_index(
            fields=[
                TagField("doc_id"),
                TagField("chunk_id"),
                TextField("title"),
                TextField("content"),
                NumericField("content_length"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimensions,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ],
            definition=IndexDefinition(
                prefix=[self.defaults.chunk_prefix],
                index_type=IndexType.HASH,
                language="chinese",
            ),
        )

    async def _add_document(self, title: str, content: str) -> DocumentSummary:
        doc_id = str(uuid4())
        chunks = split_text(content)
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(
            f"{self.defaults.doc_prefix}{doc_id}",
            mapping={
                "doc_id": doc_id,
                "title": title,
                "chunk_count": len(chunks),
                "content_length": len(content),
            },
        )
        for index, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}:{index}"
            pipe.hset(
                f"{self.defaults.chunk_prefix}{chunk_id}",
                mapping={
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "content": chunk,
                    "content_length": len(chunk),
                    "embedding": _vector_to_bytes(self._embed(f"{title}\n{chunk}")),
                },
            )
        await pipe.execute()
        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            chunk_count=len(chunks),
            content_length=len(content),
        )

    async def _seed_defaults(self) -> None:
        await self._add_document(
            title="项目架构说明",
            content=(
                "系统通过 Supervisor 将用户请求路由到 RAG、工具、闲聊和兜底分支。"
                "当前后端使用 FastAPI 接收请求，使用 Pydantic 校验数据，"
                "使用 LangGraph StateGraph 编排 Agent 工作流。"
            ),
        )
        await self._add_document(
            title="学习路线",
            content=(
                "先完成可运行闭环，再逐步接入 LangGraph、Redis、MCP 和评测体系。"
                "RAG 阶段需要完成文档上传、文本切分、检索召回和来源引用。"
            ),
        )

    def _embed(self, text: str) -> list[float]:
        vector = self.embedding_model.embed(text)
        if len(vector) != self.vector_dimensions:
            raise RuntimeError(
                "Embedding vector dimension does not match Redis index dimension: "
                f"{len(vector)} != {self.vector_dimensions}"
            )
        return vector


def _vector_to_bytes(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _build_full_text_query(query_terms: set[str]) -> str:
    terms = sorted(query_terms, key=lambda term: (-len(term), term))
    return " | ".join(f'"{term}"' for term in terms)


def _source_from_redis_document(document: object) -> SourceChunk:
    return SourceChunk(
        doc_id=_decode(document.doc_id),
        title=_decode(document.title),
        snippet=_decode(document.content)[:2000],
        score=0.0,
    )


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _extract_vector_dimensions(index_info: dict, field_name: str) -> int | None:
    attributes = index_info.get("attributes") or index_info.get(b"attributes") or []
    for attribute in attributes:
        if isinstance(attribute, dict):
            values = {_decode(key): value for key, value in attribute.items()}
        else:
            values = {
                _decode(attribute[index]): attribute[index + 1]
                for index in range(0, len(attribute) - 1, 2)
            }
        identifier = values.get("identifier") or values.get("attribute")
        if _decode(identifier) != field_name:
            continue
        dimension = values.get("dim")
        return int(_decode(dimension)) if dimension is not None else None
    return None
