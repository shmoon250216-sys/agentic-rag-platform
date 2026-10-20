"""Stable fragment identity and structural citation checks, not a truth judge."""
import hashlib
import re
from urllib.parse import quote

from app.schemas.chat import SourceChunk


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fragment_source(*, doc_id, title, chunk_id, chunk_index, content, score,
                    resolvable=False) -> SourceChunk:
    digest = content_hash(content)
    url = None
    if resolvable:
        url = (f"/api/v1/documents/{quote(doc_id, safe='')}/evidence/"
               f"{quote(chunk_id, safe='')}?sha256={digest}")
    return SourceChunk(doc_id=doc_id, title=title, snippet=content, score=score,
                       chunk_id=chunk_id, chunk_index=chunk_index,
                       content_sha256=digest, evidence_url=url)


def label_sources(sources: list[SourceChunk]) -> list[SourceChunk]:
    # Labels are local to this answer; stable identity is chunk_id + content hash.
    return [s.model_copy(update={"citation_id": f"S{i}"})
            for i, s in enumerate(sources, 1)]


def format_evidence(sources: list[SourceChunk]) -> str:
    return "\n\n".join(
        f"[{s.citation_id or f'S{i}'}] 来源标题：{s.title}\n"
        f"文档ID：{s.doc_id}\n分片ID：{s.chunk_id or '未提供'}\n"
        f"内容SHA256：{s.content_sha256 or '未提供'}\n片段：{s.snippet}"
        for i, s in enumerate(sources, 1)
    )


def citation_warnings(answer: str, sources: list[SourceChunk]) -> list[str]:
    allowed = {s.citation_id or f"S{i}" for i, s in enumerate(sources, 1)}
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    warnings = [f"未知引用编号：{label}" for label in sorted(cited - allowed)]
    if sources and not cited:
        warnings.append("回答未提供 [S编号] 引用；请核对证据片段。")
    return warnings
