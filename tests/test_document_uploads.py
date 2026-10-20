from io import BytesIO
from unittest.mock import Mock

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import AppError
from app.main import create_app
from app.rag.embedding import HashEmbeddingModel
from app.rag.store import InMemoryKnowledgeBase
from app.services.document_uploads import DocumentUploads


def docx_bytes(text="住宿报销需要提供酒店发票和行程说明，经主管审批后按公司标准报销。"):
    doc = Document()
    doc.add_paragraph(text)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def setup_uploads(tmp_path):
    settings = Settings(document_storage_dir=str(tmp_path / "originals"),
                        rag_backend="memory", embedding_provider="hash")
    model = HashEmbeddingModel(32)
    model.embed = Mock(wraps=model.embed)
    store = InMemoryKnowledgeBase(embedding_model=model)
    return settings, store, model


async def test_same_bytes_renamed_and_after_reopen_skip_parser_and_embeddings(setup_uploads, monkeypatch):
    settings, store, model = setup_uploads
    data = docx_bytes()
    first = await DocumentUploads(settings).upload(store, "制度.docx", data, "首次标题")
    assert not first.reused
    calls = model.embed.call_count
    assert calls > 0
    monkeypatch.setattr("app.services.document_uploads.parse_uploaded_document",
                        Mock(side_effect=AssertionError("Duplicate must not parse")))
    second = await DocumentUploads(settings).upload(store, "改名.docx", data, "新标题")
    assert second.reused
    assert second.document.doc_id == first.document.doc_id
    assert second.document.title == "首次标题"
    assert model.embed.call_count == calls
    archive = DocumentUploads(settings)
    name, saved, extension = await archive.original(store, first.document.doc_id)
    assert (name, saved, extension) == ("制度.docx", data, ".docx")
    assert len(store.list_documents()) == 1


async def test_same_filename_changed_bytes_indexes_new_document(setup_uploads):
    settings, store, model = setup_uploads
    archive = DocumentUploads(settings)
    first = await archive.upload(store, "制度.docx", docx_bytes())
    calls = model.embed.call_count
    second = await archive.upload(store, "制度.docx", docx_bytes("新的差旅制度规定每晚住宿上限为六百元，必须保存原始发票并按时提交。"))
    assert not second.reused
    assert second.document.doc_id != first.document.doc_id
    assert model.embed.call_count > calls


async def test_delete_cleans_receipt_and_original_and_allows_reupload(setup_uploads):
    settings, store, _ = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    first = await archive.upload(store, "../nested/制度.docx", data)
    row = archive.records()[0]
    assert row["filename"] == "制度.docx"
    assert archive.path_for(row).is_file()
    assert await archive.delete(store, first.document.doc_id)
    assert not archive.path_for(row).exists()
    assert archive.records() == []
    second = await archive.upload(store, "制度.docx", data)
    assert not second.reused
    assert second.document.doc_id != first.document.doc_id


async def test_reset_backend_does_not_reuse_stale_receipt(setup_uploads):
    settings, store, model = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    first = await archive.upload(store, "制度.docx", data)
    empty_store = InMemoryKnowledgeBase(embedding_model=model)
    second = await DocumentUploads(settings).upload(empty_store, "制度.docx", data)
    assert not second.reused
    assert second.document.doc_id != first.document.doc_id


async def test_index_configuration_changes_do_not_reuse_old_vectors(setup_uploads):
    settings, store, model = setup_uploads
    data = docx_bytes()
    await DocumentUploads(settings).upload(store, "制度.docx", data)
    calls = model.embed.call_count
    changed = settings.model_copy(update={"embedding_model": "another-model"})
    result = await DocumentUploads(changed).upload(store, "制度.docx", data)
    assert not result.reused
    assert model.embed.call_count > calls
    assert len(store.list_documents()) == 2


async def test_index_failure_does_not_poison_retry(setup_uploads, monkeypatch):
    settings, store, _ = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(store, "add_document", Mock(side_effect=RuntimeError("embedding failed")))
        with pytest.raises(RuntimeError, match="embedding failed"):
            await archive.upload(store, "制度.docx", data)
    assert archive.records() == []
    assert list((archive.root / "originals").iterdir()) == []
    assert not (await archive.upload(store, "制度.docx", data)).reused


async def test_invalid_document_is_not_archived(setup_uploads):
    settings, store, model = setup_uploads
    archive = DocumentUploads(settings)
    with pytest.raises(AppError):
        await archive.upload(store, "broken.pdf", b"not a PDF")
    assert archive.records() == []
    assert not (archive.root / "originals").exists()
    assert model.embed.call_count == 0


async def test_missing_original_restored_without_embedding(setup_uploads, monkeypatch):
    settings, store, model = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    await archive.upload(store, "制度.docx", data)
    archive.path_for(archive.records()[0]).unlink()
    monkeypatch.setattr("app.services.document_uploads.parse_uploaded_document",
                        Mock(side_effect=AssertionError("must not reparse")))
    calls = model.embed.call_count
    result = await archive.upload(store, "制度.docx", data)
    assert result.reused
    assert model.embed.call_count == calls
    assert (await archive.original(store, result.document.doc_id))[1] == data


async def test_simultaneous_upload_cannot_double_index(setup_uploads):
    import asyncio
    settings, store, model = setup_uploads
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowStore:
        list_documents = store.list_documents

        async def add_document(self, title, content):
            entered.set()
            await release.wait()
            return store.add_document(title, content)

    data = docx_bytes()
    first = asyncio.create_task(DocumentUploads(settings).upload(SlowStore(), "制度.docx", data))
    await asyncio.wait_for(entered.wait(), 10)
    try:
        with pytest.raises(AppError) as error:
            await DocumentUploads(settings).upload(store, "制度.docx", data)
        assert error.value.status_code == 409
        assert error.value.code == "DOCUMENT_UPLOAD_BUSY"
    finally:
        release.set()
        await first
    assert len(store.list_documents()) == 1
    assert model.embed.call_count == 1


async def test_interrupted_receipt_does_not_blindly_rebill(setup_uploads):
    import hashlib
    settings, store, model = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    with archive.mutation():
        archive.remember({"file_hash": hashlib.sha256(data).hexdigest(),
                          "filename": "制度.docx", "extension": ".docx"})
    with pytest.raises(AppError) as error:
        await archive.upload(store, "制度.docx", data)
    assert error.value.code == "DOCUMENT_UPLOAD_INTERRUPTED"
    assert model.embed.call_count == 0


def test_api_reuses_and_downloads_original_with_auth(tmp_path, monkeypatch):
    from app.api import routes
    settings = Settings(document_storage_dir=str(tmp_path / "archive"), rag_backend="memory")
    store = InMemoryKnowledgeBase(embedding_model=HashEmbeddingModel(32))
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_knowledge_base", lambda: store)
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}
    data = docx_bytes()
    first = client.post("/api/v1/documents/upload", headers=headers,
                        files={"file": ("policy.docx", data)})
    assert first.status_code == 200
    doc_id = first.json()["document"]["doc_id"]
    second = client.post("/api/v1/documents/upload", headers=headers,
                         files={"file": ("renamed.docx", data)})
    assert second.status_code == 200 and second.json()["reused"]
    url = f"/api/v1/documents/{doc_id}/original"
    assert client.get(url).status_code == 401
    original = client.get(url, headers=headers)
    assert original.status_code == 200 and original.content == data
    assert "attachment" in original.headers["content-disposition"]
    listed = client.get("/api/v1/documents", headers=headers).json()
    assert listed["documents"][0]["original_filename"] == "policy.docx"
    assert client.delete(f"/api/v1/documents/{doc_id}", headers=headers).json()["deleted"]
    assert client.get(url, headers=headers).status_code == 404

async def test_receipt_write_failure_rolls_back_index_and_original(setup_uploads, monkeypatch):
    settings, store, _ = setup_uploads
    archive = DocumentUploads(settings)
    original_remember = archive.remember

    def fail_success_receipt(row, document=None):
        if document is not None:
            raise OSError("disk full")
        return original_remember(row)

    monkeypatch.setattr(archive, "remember", fail_success_receipt)
    with pytest.raises(OSError, match="disk full"):
        await archive.upload(store, "policy.docx", docx_bytes())
    assert store.list_documents() == []
    assert archive.records() == []
    assert list((archive.root / "originals").iterdir()) == []


async def test_shared_original_survives_deletion_from_one_index_scope(setup_uploads):
    settings, store, _ = setup_uploads
    archive = DocumentUploads(settings)
    data = docx_bytes()
    first = await archive.upload(store, "policy.docx", data)
    other = DocumentUploads(settings.model_copy(update={"embedding_model": "other"}))
    second = await other.upload(store, "policy.docx", data)
    assert await archive.delete(store, first.document.doc_id)
    assert (await other.original(store, second.document.doc_id))[1] == data


def test_upload_lock_is_shared_across_processes(setup_uploads):
    import subprocess
    import sys
    settings, _, _ = setup_uploads
    archive = DocumentUploads(settings)
    code = '''
import sys
from app.core.config import Settings
from app.core.errors import AppError
from app.services.document_uploads import DocumentUploads
try:
    with DocumentUploads(Settings(document_storage_dir=sys.argv[1])).mutation():
        print("incorrectly_acquired")
except AppError as exc:
    print(exc.code)
'''
    with archive.mutation():
        result = subprocess.run([sys.executable, "-c", code, str(archive.root)],
                                capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == "DOCUMENT_UPLOAD_BUSY"
