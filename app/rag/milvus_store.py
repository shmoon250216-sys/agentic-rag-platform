"""Milvus Standalone backend: native BM25 + HNSW/COSINE, application-side RRF.

The blocking SDK and embedding adapter run in worker threads. A client lock
serializes initialization and writes; it is not a distributed transaction lock.
"""
import asyncio
import hashlib
import json
import logging
import math
import re
import threading
from uuid import UUID, uuid4

from app.core.errors import AppError
from app.rag.embedding import EmbeddingModel
from app.rag.ranking import RankedItem, normalize_scores, reciprocal_rank_fusion
from app.rag.text_splitter import split_text
from app.schemas.chat import SourceChunk
from app.schemas.document import DocumentSummary

logger = logging.getLogger(__name__)
OUTPUT_FIELDS = ["doc_id", "title", "content"]


class MilvusConfigurationError(AppError):
    def __init__(self) -> None:
        super().__init__(
            "MILVUS_SCHEMA_MISMATCH",
            "Milvus collection schema, indexes or embedding revision mismatch; "
            "use a new collection and re-import documents.", 503,
        )


class MilvusKnowledgeBase:
    def __init__(
        self, *, uri: str, token: str, collection_name: str,
        embedding_model: EmbeddingModel, embedding_revision: str,
        timeout_seconds: float = 15, candidate_multiplier: int = 4,
        rrf_k: int = 60, bm25_weight: float = 1.0, vector_weight: float = 0.1,
        client=None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", collection_name):
            raise ValueError("Invalid Milvus collection name")
        if not embedding_revision.strip():
            raise ValueError("MILVUS_EMBEDDING_REVISION must identify the embedding model")
        if timeout_seconds <= 0 or not 1 <= candidate_multiplier <= 100:
            raise ValueError("Invalid Milvus timeout or candidate multiplier")
        if rrf_k < 1 or min(bm25_weight, vector_weight) < 0:
            raise ValueError("Invalid RRF configuration")
        if not math.isfinite(bm25_weight + vector_weight) or bm25_weight + vector_weight <= 0:
            raise ValueError("At least one finite positive RRF weight is required")
        self.uri, self.token = uri, token
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dimensions = embedding_model.dimensions
        self.embedding_revision = embedding_revision
        self.timeout = timeout_seconds
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k, self.weights = rrf_k, [bm25_weight, vector_weight]
        self._client = client
        self._ready = False
        self._lock = threading.RLock()

    @property
    def cache_namespace(self) -> str:
        identity = f"{self.uri}|{self.collection_name}|{self.description}"
        return "milvus:" + hashlib.sha256(identity.encode()).hexdigest()

    @property
    def description(self) -> str:
        return json.dumps({"schema": 1, "embedding": self.embedding_revision,
                           "dimensions": self.vector_dimensions}, sort_keys=True)

    async def _run(self, function, *args):
        try:
            return await asyncio.to_thread(self._locked, function, *args)
        except (AppError, ValueError):
            raise
        except Exception as exc:
            # Do not return connection strings/tokens or SDK exception text to clients.
            logger.warning("Milvus operation failed (%s)", type(exc).__name__)
            raise AppError("RAG_BACKEND_UNAVAILABLE",
                           "Milvus operation failed; check service health and configuration.",
                           503) from exc

    def _locked(self, function, *args):
        with self._lock:
            self._ensure_ready()
            return function(*args)

    def _ensure_ready(self) -> None:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        if self._client is None:
            self._client = MilvusClient(uri=self.uri, token=self.token, timeout=self.timeout)
        c = self._client
        name = self.collection_name
        if self._ready:
            return
        if not c.has_collection(collection_name=name, timeout=self.timeout):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                               description=self.description)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=36)
            schema.add_field("title", DataType.VARCHAR, max_length=1024)
            schema.add_field("content", DataType.VARCHAR, max_length=16384)
            schema.add_field("text", DataType.VARCHAR, max_length=17408,
                             enable_analyzer=True,
                             analyzer_params={"tokenizer": "jieba"})
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("chunk_count", DataType.INT64)
            schema.add_field("content_length", DataType.INT64)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.vector_dimensions)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(Function(name="text_bm25", input_field_names=["text"],
                                         output_field_names=["sparse"],
                                         function_type=FunctionType.BM25))
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index("dense", index_type="HNSW", metric_type="COSINE",
                              params={"M": 16, "efConstruction": 200})
            indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            # Another process may have created it between has_collection and create.
            try:
                c.create_collection(collection_name=name, schema=schema, index_params=indexes,
                                    consistency_level="Strong", timeout=self.timeout)
            except Exception:
                if not c.has_collection(collection_name=name, timeout=self.timeout):
                    raise
        detail = c.describe_collection(collection_name=name, timeout=self.timeout)
        fields = {f["name"]: f for f in detail["fields"]}
        required = {"chunk_id", "doc_id", "title", "content", "text", "chunk_index",
                    "chunk_count", "content_length", "dense", "sparse"}
        if (detail.get("description") != self.description or not required <= fields.keys()
                or int(fields["dense"].get("params", {}).get("dim", 0)) != self.vector_dimensions):
            raise MilvusConfigurationError()
        found = {}
        for index_name in c.list_indexes(collection_name=name, timeout=self.timeout):
            index = c.describe_index(collection_name=name, index_name=index_name,
                                     timeout=self.timeout)
            found[index.get("field_name")] = index.get("metric_type")
        if found.get("dense") != "COSINE" or found.get("sparse") != "BM25":
            raise MilvusConfigurationError()
        c.load_collection(collection_name=name, timeout=self.timeout)
        self._ready = True

    async def ensure_available(self) -> None:
        await self._run(lambda: self._client.describe_collection(
            collection_name=self.collection_name, timeout=self.timeout))

    async def health(self) -> dict[str, object]:
        status = {"backend": "milvus", "persistent": True,
                  "collection": self.collection_name, "dimensions": self.vector_dimensions,
                  "vector_index": "HNSW", "metric": "COSINE", "lexical": "BM25/jieba"}
        try:
            await self.ensure_available()
            return {**status, "connected": True}
        except AppError as exc:
            return {**status, "connected": False, "error_code": exc.code}

    def _embedding(self, text: str) -> list[float]:
        vector = self.embedding_model.embed(text)
        if len(vector) != self.vector_dimensions or not all(math.isfinite(v) for v in vector):
            raise ValueError("Invalid embedding dimension or non-finite value")
        return vector

    async def add_document(self, title: str, content: str) -> DocumentSummary:
        return await self._run(self._add_document, title, content)

    def _add_document(self, title: str, content: str) -> DocumentSummary:
        pieces = split_text(content)
        if not pieces or not content.strip():
            raise ValueError("Document must contain text")
        if len(title.encode("utf-8")) > 1024:
            raise ValueError("Document title exceeds Milvus byte limit")
        doc_id = str(uuid4())
        # Compute and validate every vector before the first database write.
        rows = [{"chunk_id": f"{doc_id}:{i}", "doc_id": doc_id, "title": title,
                 "content": chunk, "text": f"{title}\n{chunk}", "chunk_index": i,
                 "chunk_count": len(pieces), "content_length": len(content),
                 "dense": self._embedding(f"{title}\n{chunk}")}
                for i, chunk in enumerate(pieces)]
        try:
            result = self._client.insert(collection_name=self.collection_name, data=rows,
                                         timeout=self.timeout)
            if result.get("insert_count") != len(rows):
                raise RuntimeError("Incomplete Milvus insert")
        except Exception:
            # Best-effort cleanup of a failed upload; no cross-system atomicity claim.
            try:
                self._client.delete(collection_name=self.collection_name,
                                    filter=f'doc_id == "{doc_id}"', timeout=self.timeout)
            except Exception:
                logger.warning("Milvus failed-upload cleanup unsuccessful for %s", doc_id)
            raise
        return DocumentSummary(doc_id=doc_id, title=title, chunk_count=len(rows),
                               content_length=len(content))

    async def list_documents(self) -> list[DocumentSummary]:
        return await self._run(self._list_documents)

    def _list_documents(self) -> list[DocumentSummary]:
        iterator = self._client.query_iterator(
            collection_name=self.collection_name, filter="chunk_index == 0",
            output_fields=["doc_id", "title", "chunk_count", "content_length"],
            batch_size=256, consistency_level="Strong", timeout=self.timeout)
        documents = []
        try:
            while batch := iterator.next():
                documents.extend(DocumentSummary(**row) for row in batch)
        finally:
            iterator.close()
        return sorted(documents, key=lambda doc: doc.doc_id)

    async def delete_document(self, doc_id: str) -> bool:
        try:
            canonical = str(UUID(doc_id))
        except ValueError:
            return False
        return await self._run(self._delete_document, canonical)

    def _delete_document(self, doc_id: str) -> bool:
        condition = f'doc_id == "{doc_id}"'
        existing = self._client.query(collection_name=self.collection_name, filter=condition,
                                      output_fields=["chunk_id"], limit=1,
                                      consistency_level="Strong", timeout=self.timeout)
        if not existing:
            return False
        self._client.delete(collection_name=self.collection_name, filter=condition,
                            timeout=self.timeout)
        return True

    async def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        if not query.strip() or top_k <= 0:
            return []
        return await self._run(self._search, query, top_k)

    def _search(self, query: str, top_k: int) -> list[SourceChunk]:
        limit = min(top_k * self.candidate_multiplier, 1000)
        shared = dict(collection_name=self.collection_name, limit=limit,
                      output_fields=OUTPUT_FIELDS, consistency_level="Strong", timeout=self.timeout)
        bm25 = self._client.search(data=[query], anns_field="sparse",
                                   search_params={"metric_type": "BM25", "params": {}}, **shared)[0]
        dense = self._client.search(data=[self._embedding(query)], anns_field="dense",
                                    search_params={"metric_type": "COSINE",
                                                   "params": {"ef": max(64, limit)}}, **shared)[0]
        rankings, entities = [], {}
        for hits in (bm25, dense):
            ranking = []
            for hit in hits:
                if hit["distance"] <= 0:
                    continue
                entities[hit["id"]] = hit["entity"]
                ranking.append(RankedItem(item=hit["id"], score=hit["distance"]))
            rankings.append(ranking)
        fused = normalize_scores(reciprocal_rank_fusion(
            rankings, rrf_k=self.rrf_k, weights=self.weights))
        return [SourceChunk(doc_id=entities[r.item]["doc_id"], title=entities[r.item]["title"],
                            snippet=entities[r.item]["content"][:2000], score=r.score)
                for r in fused[:top_k]]

    async def close(self) -> None:
        def shutdown():
            with self._lock:
                if self._client is not None:
                    self._client.close()
                    self._client = None
                self._ready = False
        await asyncio.to_thread(shutdown)
