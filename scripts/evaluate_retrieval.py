import json
from pathlib import Path

from app.rag.document_parser import parse_uploaded_document
from app.rag.embedding import (
    HashEmbeddingModel,
    cosine_similarity,
    tokenize_for_search,
)
from app.rag.ranking import BM25Ranker, RankedItem, reciprocal_rank_fusion
from app.rag.text_splitter import split_text


ROOT = Path(__file__).resolve().parents[1]
QUESTION_PATH = ROOT / "examples" / "evaluation_questions.json"
OUTPUT_PATH = ROOT / "docs" / "retrieval-evaluation-results.json"


def _rank_positions(ranking: list[RankedItem[int]], chunks: list[str], topic: str) -> int | None:
    for rank, result in enumerate(ranking, start=1):
        if topic in chunks[result.item]:
            return rank
    return None


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    total = max(len(ranks), 1)
    return {
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / total, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / total, 4),
        "mrr": round(sum(1 / rank for rank in ranks if rank is not None) / total, 4),
    }


def main() -> None:
    dataset = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    document_path = ROOT / "examples" / "documents" / dataset["document"]
    parsed = parse_uploaded_document(
        document_path.name,
        document_path.read_bytes(),
        max_characters=200_000,
        max_pdf_pages=300,
        max_docx_uncompressed_bytes=50 * 1024 * 1024,
    )
    chunks = split_text(parsed.content)
    embedding_model = HashEmbeddingModel(dimensions=256)
    chunk_embeddings = [
        embedding_model.embed(f"{parsed.title}\n{chunk}") for chunk in chunks
    ]
    corpus = [tokenize_for_search(f"{parsed.title}\n{chunk}") for chunk in chunks]
    bm25 = BM25Ranker(corpus)

    ranks_by_strategy: dict[str, list[int | None]] = {
        "bm25": [],
        "vector": [],
        "rrf": [],
    }
    case_results: list[dict[str, object]] = []
    for case in dataset["questions"]:
        query = case["question"]
        topic = case["expected_topic"]
        evidence = case["expected_evidence"]
        bm25_ranking = sorted(
            (
                RankedItem(index, score)
                for index, score in enumerate(bm25.scores(tokenize_for_search(query)))
                if score > 0
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        query_embedding = embedding_model.embed(query)
        vector_ranking = sorted(
            (
                RankedItem(index, max(cosine_similarity(query_embedding, embedding), 0.0))
                for index, embedding in enumerate(chunk_embeddings)
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        vector_ranking = [result for result in vector_ranking if result.score > 0]
        rrf_ranking = reciprocal_rank_fusion(
            [bm25_ranking[:12], vector_ranking[:12]],
            rrf_k=60,
            weights=[1.0, 0.1],
        )

        positions = {
            "bm25": _rank_positions(bm25_ranking, chunks, evidence),
            "vector": _rank_positions(vector_ranking, chunks, evidence),
            "rrf": _rank_positions(rrf_ranking, chunks, evidence),
        }
        for strategy, rank in positions.items():
            ranks_by_strategy[strategy].append(rank)
        case_results.append(
            {
                "id": case["id"],
                "question": query,
                "expected_topic": topic,
                "expected_evidence": evidence,
                "rank": positions,
            }
        )

    report = {
        "dataset": {
            "document": dataset["document"],
            "synthetic": True,
            "page_count": parsed.page_count,
            "character_count": len(parsed.content),
            "chunk_count": len(chunks),
            "question_count": len(dataset["questions"]),
            "embedding": "deterministic hash embedding; not a semantic production model",
            "rrf_weights": {"bm25": 1.0, "vector": 0.1},
        },
        "metrics": {
            strategy: _metrics(ranks) for strategy, ranks in ranks_by_strategy.items()
        },
        "cases": case_results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
