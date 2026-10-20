"""Local originals and durable upload receipts; vectors remain in the selected KB."""
import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from app.core.errors import AppError
from app.rag.document_parser import (
    DocumentParseError, SUPPORTED_DOCUMENT_EXTENSIONS, parse_uploaded_document,
)
from app.schemas.document import DocumentCreateResponse

# Bump when parser/chunking semantics change and uploads need re-indexing.
PIPELINE_REVISION = "parser-v1:split-500-80-v1"


async def call_store(method, *args):
    if inspect.iscoroutinefunction(method):
        return await method(*args)
    return await asyncio.to_thread(method, *args)


class DocumentUploads:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings.document_storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "uploads.sqlite3"
        backend = settings.rag_backend.lower()
        target = {
            "milvus": [settings.milvus_uri, settings.milvus_collection],
            "redis": [settings.redis_url, settings.rag_index_name],
            "memory": ["local"],
        }.get(backend, [backend])
        identity = [backend, target, settings.embedding_provider, settings.embedding_base_url,
                    settings.embedding_model, settings.embedding_dimensions,
                    settings.milvus_embedding_revision, PIPELINE_REVISION]
        self.scope = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS uploads (
                scope TEXT NOT NULL, file_hash TEXT NOT NULL,
                filename TEXT NOT NULL, extension TEXT NOT NULL,
                status TEXT NOT NULL, doc_id TEXT, document TEXT,
                PRIMARY KEY (scope, file_hash))""")

    @contextmanager
    def mutation(self):
        """One local ingest/delete at a time, across workers; OS releases on exit/crash."""
        with (self.root / "uploads.lock").open("a+b") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise AppError("DOCUMENT_UPLOAD_BUSY", "另一份文档正在入库或删除，请稍后重试", 409) from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def records(self):
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                "SELECT * FROM uploads WHERE scope = ?", (self.scope,))]

    def path_for(self, row):
        return self.root / "originals" / (row["file_hash"] + row["extension"])

    def save_original(self, row, data):
        path = self.path_for(row)
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def remember(self, row, document=None):
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("""INSERT OR REPLACE INTO uploads
                (scope, file_hash, filename, extension, status, doc_id, document)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.scope, row["file_hash"], row["filename"], row["extension"],
                 "ready" if document else "indexing", document.doc_id if document else None,
                 document.model_dump_json() if document else None))

    def forget(self, row):
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("DELETE FROM uploads WHERE scope = ? AND file_hash = ?",
                       (self.scope, row["file_hash"]))
            referenced = db.execute("SELECT 1 FROM uploads WHERE file_hash = ? AND extension = ?",
                                    (row["file_hash"], row["extension"])).fetchone()
        if not referenced:
            self.path_for(row).unlink(missing_ok=True)

    def with_originals(self, documents):
        names = {row["doc_id"]: row["filename"] for row in self.records()
                 if row["status"] == "ready" and self.path_for(row).is_file()}
        return [doc.model_copy(update={"original_filename": names.get(doc.doc_id)})
                for doc in documents]

    async def upload(self, store, filename, data, title=None):
        safe_name = Path(filename.replace("\\", "/")).name
        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise AppError("DOCUMENT_TYPE_UNSUPPORTED", "仅支持 PDF 和 DOCX 文件", 400)
        digest = hashlib.sha256(data).hexdigest()
        with self.mutation():
            row = next((r for r in self.records() if r["file_hash"] == digest), None)
            if row:
                if row["status"] != "ready":
                    raise AppError("DOCUMENT_UPLOAD_INTERRUPTED",
                                   "此前入库被中断，请核对知识库并修复上传记录后重试，避免重复计费", 409)
                documents = await call_store(store.list_documents)
                existing = next((d for d in documents if d.doc_id == row["doc_id"]), None)
                if existing:
                    # Restore a missing original from the identical uploaded bytes, without parsing.
                    if not self.path_for(row).is_file():
                        self.save_original(row, data)
                    return DocumentCreateResponse(
                        document=existing.model_copy(update={"original_filename": row["filename"]}),
                        reused=True)
                self.forget(row)  # A deleted/reset KB must not produce a false dedup hit.
            try:
                parsed = await asyncio.to_thread(
                    parse_uploaded_document, safe_name, data,
                    max_characters=self.settings.document_extract_max_chars,
                    max_pdf_pages=self.settings.document_pdf_max_pages,
                    max_docx_uncompressed_bytes=self.settings.document_docx_max_uncompressed_bytes)
            except DocumentParseError as exc:
                raise AppError(exc.code, exc.message, 400) from exc
            row = {"file_hash": digest, "filename": safe_name, "extension": extension}
            self.remember(row)
            document = None
            try:
                self.save_original(row, data)
                document = await call_store(store.add_document,
                                            title.strip() if title and title.strip() else parsed.title,
                                            parsed.content)
                self.remember(row, document)
            except Exception:
                # Compensate an ordinary failure. A hard process crash leaves an indexing receipt
                # for manual reconciliation; this is not a distributed exactly-once transaction.
                if document is not None:
                    await call_store(store.delete_document, document.doc_id)
                self.forget(row)
                raise
            return DocumentCreateResponse(
                document=document.model_copy(update={"original_filename": safe_name}), reused=False)

    async def delete(self, store, doc_id):
        with self.mutation():
            deleted = await call_store(store.delete_document, doc_id)
            rows = [row for row in self.records() if row["doc_id"] == doc_id]
            for row in rows:
                self.forget(row)
            return deleted or bool(rows)

    async def original(self, store, doc_id):
        with self.mutation():
            documents = await call_store(store.list_documents)
            if any(doc.doc_id == doc_id for doc in documents):
                for row in self.records():
                    if row["doc_id"] == doc_id and row["status"] == "ready":
                        path = self.path_for(row)
                        if path.is_file():
                            return row["filename"], path.read_bytes(), row["extension"]
            raise AppError("DOCUMENT_ORIGINAL_NOT_FOUND", "该文档没有保存原始文件或已被删除", 404)
