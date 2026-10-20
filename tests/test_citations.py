from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppError
from app.rag.evidence import citation_warnings, content_hash, format_evidence, label_sources
from app.rag.retriever import KnowledgeBaseRetriever, retrieval_cache
from app.rag.reranker import NoopReranker
from test_milvus_store import Client, store


async def test_live_evidence_identity_change_delete_and_invalid_path():
    client = Client()
    kb = store(client)
    doc = await kb.add_document("住宿规定", "住宿上限550元，凭发票和水单报销。" * 45)
    results = await kb.search("住宿", 3)
    second = results[1]
    assert second.chunk_index is not None
    resolved = await kb.get_evidence(doc.doc_id, second.chunk_id)
    assert resolved.snippet == second.snippet
    assert resolved.content_sha256 == content_hash(second.snippet)
    assert resolved.evidence_url == second.evidence_url
    for cid in ['" or true', "other:0", doc.doc_id + ":bad"]:
        with pytest.raises(AppError):
            await kb.get_evidence(doc.doc_id, cid)
    await kb.delete_document(doc.doc_id)
    with pytest.raises(AppError) as exc:
        await kb.get_evidence(doc.doc_id, second.chunk_id)
    assert exc.value.status_code == 404


async def test_production_and_strategy_comparison_share_results(monkeypatch):
    kb = store()
    await kb.add_document("制度", "住宿报销发票说明。" * 80)
    comparisons = await kb.search_strategies("报销", 3)
    assert await kb.search("报销", 3) == comparisons["rrf"]
    retriever = KnowledgeBaseRetriever(kb, NoopReranker())
    monkeypatch.setattr(retriever.settings, "cache_enabled", False)
    labelled = await retriever.search("报销", 3)
    assert [s.citation_id for s in labelled] == [f"S{i}" for i in range(1, len(labelled)+1)]
    assert labelled[0].chunk_id == comparisons["rrf"][0].chunk_id
    text = format_evidence(labelled)
    assert labelled[0].snippet in text and "[S1]" in text
    assert citation_warnings("根据条款 [S1]", labelled) == []
    assert "未知引用编号：S99" in citation_warnings("根据条款 [S99]", labelled)
    assert citation_warnings("没有编号", labelled)
    retrieval_cache.clear()


async def test_evidence_api_auth_hash_mismatch_and_deleted(monkeypatch):
    from app.api import routes
    from app.main import app

    kb = store()
    doc = await kb.add_document("制度", "住宿每晚550元，需要提交酒店水单。" * 3)
    source = (await kb.search("住宿", 1))[0]
    monkeypatch.setattr(routes, "get_knowledge_base", lambda: kb)
    headers = {"Authorization": f"Bearer {routes.get_settings().api_token}"}
    with TestClient(app) as api:
        assert api.get(source.evidence_url).status_code == 401
        response = api.get(source.evidence_url, headers=headers)
        assert response.status_code == 200
        assert response.json()["snippet"] == source.snippet
        mismatch = urlsplit(source.evidence_url).path + "?sha256=" + "0" * 64
        assert api.get(mismatch, headers=headers).status_code == 409
        await kb.delete_document(doc.doc_id)
        assert api.get(source.evidence_url, headers=headers).status_code == 404


def test_old_sources_without_identity_remain_compatible():
    from app.schemas.chat import SourceChunk
    source = SourceChunk(doc_id="old", title="旧数据", snippet="旧内容", score=.5)
    assert "[S1]" in format_evidence(label_sources([source]))
    assert source.evidence_url is None


async def test_citations_persist_after_session_store_reopen(tmp_path):
    from app.memory.sqlite_store import SQLiteSessionStore
    from app.rag.evidence import fragment_source
    db = f"sqlite:///{tmp_path / 'conversation.db'}"
    first = SQLiteSessionStore(db)
    session = await first.get_or_create("u", "s")
    source = fragment_source(doc_id="d", title="制度", chunk_id="d:0",
                             chunk_index=0, content="住宿标准", score=1)
    source = label_sources([source])[0]
    await first.append(session.session_id, "assistant", "依据 [S1]",
                       evidence={"sources": [source.model_dump()],
                                 "citation_warnings": []})
    second = SQLiteSessionStore(db)
    messages = await second.list_messages("s")
    assert messages[0].sources == [source]
    # Evidence snapshots do not silently expand the next turn's history payload.
    context = await second.get_or_create("u", "s")
    assert context.messages[0].content == "依据 [S1]"
