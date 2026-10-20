"""Evidence-level retrieval evaluation; independent of routing and answer generation."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.text_splitter import split_text


def normalize(text: str) -> str:
    return " ".join(text.split())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusDocument(StrictModel):
    key: str
    title: str
    version: str
    content: str


class Evidence(StrictModel):
    id: str
    document: str
    quote: str


class RetrievalCase(StrictModel):
    id: str
    split: Literal["dev", "test"]
    category: str
    question: str
    history: list[str] = Field(default_factory=list)
    evidence_ids: list[str]
    answerable: bool = True


class RetrievalDataset(StrictModel):
    version: str
    synthetic: Literal[True]
    documents: list[CorpusDocument]
    evidence: list[Evidence]
    cases: list[RetrievalCase]

    @model_validator(mode="after")
    def validate_labels(self):
        for items, attribute in ((self.documents, "key"), (self.evidence, "id"),
                                 (self.cases, "id")):
            values = [getattr(item, attribute) for item in items]
            if len(values) != len(set(values)):
                raise ValueError("Duplicate dataset IDs")
        documents = {d.key: d for d in self.documents}
        evidence = {e.id: e for e in self.evidence}
        for e in self.evidence:
            if e.document not in documents or not e.quote.strip():
                raise ValueError(f"Invalid evidence {e.id}")
            # Ground truth is an exact body passage, never a title substring.
            if not any(normalize(e.quote) in piece
                       for piece in split_text(documents[e.document].content)):
                raise ValueError(f"Evidence {e.id} is not contained in a complete chunk")
        splits = {}
        for c in self.cases:
            if (c.answerable != bool(c.evidence_ids)
                    or len(set(c.evidence_ids)) != len(c.evidence_ids)
                    or not set(c.evidence_ids) <= evidence.keys()):
                raise ValueError(f"Invalid labels for {c.id}")
            for eid in c.evidence_ids:
                # Paraphrases and follow-ups about the same fact stay in one split.
                if eid in splits and splits[eid] != c.split:
                    raise ValueError(f"Evidence family {eid} leaks across dev/test")
                splits[eid] = c.split
        return self


def load_dataset(path: Path) -> tuple[RetrievalDataset, str]:
    data = path.read_bytes()
    return RetrievalDataset.model_validate_json(data), hashlib.sha256(data).hexdigest()


def score_sources(sources, case: RetrievalCase, evidence: dict[str, Evidence],
                  document_keys: dict[str, str], ks=(1, 3, 5)):
    """Score distinct gold evidence units, not duplicate overlapping chunks."""
    seen_chunks = set()
    matches = []
    retained = []
    for source in sources:
        identity = source.chunk_id or (source.doc_id, source.snippet)
        if identity in seen_chunks:
            continue
        seen_chunks.add(identity)
        doc_key = document_keys.get(source.doc_id)
        hit = {eid for eid in case.evidence_ids
               if evidence[eid].document == doc_key
               and normalize(evidence[eid].quote) in normalize(source.snippet)}
        matches.append(hit)
        retained.append({
            "document": doc_key, "chunk_id": source.chunk_id,
            "content_sha256": source.content_sha256, "score": source.score,
            "matched_evidence": sorted(hit),
        })
    if not case.answerable:
        return {"metrics": None, "returned_any": bool(retained), "results": retained}
    metrics = {}
    for k in ks:
        found = set().union(*matches[:k]) if matches[:k] else set()
        metrics[f"hit@{k}"] = float(bool(found))
        metrics[f"recall@{k}"] = len(found) / len(case.evidence_ids)
        metrics[f"complete@{k}"] = float(found == set(case.evidence_ids))
    rank = next((i for i, hit in enumerate(matches, 1) if hit), None)
    metrics[f"mrr@{max(ks)}"] = 1 / rank if rank and rank <= max(ks) else 0.0
    return {"metrics": metrics, "results": retained}


def aggregate(rows):
    """Failures score zero in the denominator, and are also reported separately."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    names = {name for row in rows for name in row.get("strategies", {})}
    if rows:
        names |= {"bm25", "vector", "rrf"}
    result = {}
    for name in sorted(names):
        metric_names = {f"{metric}@{k}" for metric in ("hit", "recall", "complete")
                        for k in (1, 3, 5)} | {"mrr@5"}
        result[name] = {
            "answerable_cases": len(answerable),
            "metrics": ({
                metric: round(sum(
                    (row.get("strategies", {}).get(name, {}).get("metrics") or {}).get(metric, 0)
                    for row in answerable) / max(len(answerable), 1), 4)
                for metric in sorted(metric_names)
            } if answerable else None),
            "error_cases": sum(name not in row.get("strategies", {})
                               or "error" in row["strategies"][name] for row in rows),
            "unanswerable_diagnostics": {
                "cases": len(unanswerable),
                "returned_any": sum(bool(row.get("strategies", {}).get(name, {}).get(
                    "returned_any")) for row in unanswerable),
                "note": "Retrieval candidates do not prove answerability; not refusal accuracy.",
            },
        }
    durations = sorted(row["shared_retrieval_ms"] for row in rows
                       if "shared_retrieval_ms" in row)
    return {
        "cases": len(rows),
        "request_errors": sum("error" in row for row in rows),
        "strategies": result,
        "shared_retrieval_latency_ms": {
            "mean": round(mean(durations), 2) if durations else None,
            "p95": durations[math.ceil(len(durations) * .95) - 1] if durations else None,
            "note": "One shared BM25+dense+RRF call; not separate strategy latency.",
        },
    }


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
