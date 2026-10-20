from pathlib import Path

import pytest

from app.evaluation.retrieval import (
    Evidence, RetrievalCase, RetrievalDataset, aggregate, load_dataset, score_sources,
)
from app.rag.evidence import fragment_source


def source(doc="uuid-a", cid="a:0", text="住宿500元；提前审批"):
    return fragment_source(doc_id=doc, title="相同标题", chunk_id=cid, chunk_index=0,
                           content=text, score=1)


def test_true_recall_deduplicates_evidence_and_checks_document_identity():
    case = RetrievalCase(id="multi", split="dev", category="multi",
                         question="预算和审批？", evidence_ids=["amount", "approval"])
    evidence = {
        "amount": Evidence(id="amount", document="a", quote="住宿500元"),
        "approval": Evidence(id="approval", document="b", quote="提前审批"),
    }
    scores = score_sources([
        source(), source(cid="a:1"),
        source(doc="uuid-b", cid="b:0", text="提前审批"),
    ], case, evidence, {"uuid-a": "a", "uuid-b": "b"})
    assert scores["metrics"]["hit@1"] == 1
    assert scores["metrics"]["recall@1"] == .5
    assert scores["metrics"]["complete@1"] == 0
    assert scores["metrics"]["recall@3"] == 1
    assert scores["results"][0]["matched_evidence"] == ["amount"]
    # Same wording in a different document cannot satisfy the evidence label.
    wrong = score_sources([source(doc="uuid-wrong")], case, evidence, {"uuid-wrong": "other"})
    assert wrong["metrics"]["hit@3"] == 0


def test_mrr_and_duplicate_chunks():
    case = RetrievalCase(id="q", split="dev", category="direct",
                         question="额度", evidence_ids=["e"])
    scored = score_sources([source(text="无关"), source(), source(cid="a:1")],
                           case, {"e": Evidence(id="e", document="a", quote="住宿500元")},
                           {"uuid-a": "a"})
    # First two entries have identical chunk IDs; duplicated candidate cannot add a hit.
    assert scored["metrics"]["mrr@5"] == .5


def test_no_answer_is_not_fabricated_as_success_and_errors_stay_in_denominator():
    unknown = RetrievalCase(id="u", split="dev", category="unanswerable",
                            question="缺少依据", evidence_ids=[], answerable=False)
    assert score_sources([source()], unknown, {}, {})["metrics"] is None
    rows = [
        {"answerable": True, "strategies": {"rrf": {"metrics": {"hit@1": 1}}}},
        {"answerable": True, "strategies": {}, "error": "TimeoutError"},
    ]
    summary = aggregate(rows)
    assert summary["strategies"]["rrf"]["metrics"]["hit@1"] == .5
    assert summary["strategies"]["rrf"]["error_cases"] == 1
    assert aggregate([{"answerable": False, "strategies": {"rrf": {"metrics": None}}}]
                     )["strategies"]["rrf"]["metrics"] is None


def test_dataset_is_grounded_and_split_by_evidence_family():
    path = Path(__file__).resolve().parents[1] / "examples/retrieval_benchmark_v1.json"
    dataset, fingerprint = load_dataset(path)
    assert len(dataset.cases) == 56 and len(fingerprint) == 64
    assert sum(not c.answerable for c in dataset.cases) == 4
    data = dataset.model_dump()
    data["cases"][1]["split"] = "test"
    with pytest.raises(ValueError, match="leaks"):
        RetrievalDataset.model_validate(data)
    data = dataset.model_dump()
    data["evidence"][0]["quote"] = "这段标签不存在于资料中"
    with pytest.raises(ValueError, match="complete chunk"):
        RetrievalDataset.model_validate(data)
