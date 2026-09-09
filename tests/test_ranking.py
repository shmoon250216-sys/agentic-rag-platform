import pytest

from app.rag.ranking import BM25Ranker, RankedItem, normalize_scores, reciprocal_rank_fusion


def test_bm25_prefers_document_containing_rare_query_term() -> None:
    ranker = BM25Ranker(
        [
            ["报销", "制度", "员工", "申请"],
            ["报销", "制度", "x9", "专用", "凭证"],
        ]
    )

    scores = ranker.scores(["x9"])

    assert scores[0] == 0
    assert scores[1] > 0


def test_rrf_uses_rank_positions_instead_of_raw_score_scales() -> None:
    bm25 = [
        RankedItem("a", 18.0),
        RankedItem("b", 12.0),
        RankedItem("c", 8.0),
    ]
    vectors = [
        RankedItem("b", 0.91),
        RankedItem("d", 0.86),
        RankedItem("a", 0.80),
    ]

    fused = normalize_scores(reciprocal_rank_fusion([bm25, vectors], rrf_k=60))

    assert [result.item for result in fused[:2]] == ["b", "a"]
    assert fused[0].score == pytest.approx(1.0)


def test_rrf_rejects_non_positive_rank_constant() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_fusion([], rrf_k=0)
