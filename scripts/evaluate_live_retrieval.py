"""Run bounded real-Milvus retrieval evaluation against an isolated collection.

Uses configured Embedding credentials on synthetic corpus text only. No chat LLM.
"""
import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from app.core.config import get_settings
from app.evaluation.retrieval import aggregate, load_dataset, score_sources, write_report
from app.memory.context import resolve_followup
from app.rag.embedding import get_embedding_model
from app.rag.milvus_store import MilvusKnowledgeBase
from app.rag.reranker import NoopReranker, get_reranker

ROOT = Path(__file__).resolve().parents[1]


class CountedEmbedding:
    def __init__(self, model):
        self.model = model
        self.dimensions = model.dimensions
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return self.model.embed(text)


async def evaluate(args):
    from pymilvus import MilvusClient

    settings = get_settings()
    dataset, dataset_hash = load_dataset(args.dataset)
    if settings.rag_backend != "milvus":
        raise ValueError("Set RAG_BACKEND=milvus; no silent memory backend fallback")
    if settings.embedding_provider in {"hash", "local", "fake"} and not args.allow_hash:
        raise ValueError("Real evaluation rejects Hash Embedding; --allow-hash is CI-only")
    cases = [c for c in dataset.cases if args.split == "all" or c.split == args.split]
    if args.max_cases:
        cases = cases[:args.max_cases]
    if not cases:
        raise ValueError("No selected cases")
    model = CountedEmbedding(get_embedding_model(settings))
    if model.dimensions != settings.rag_vector_dimensions:
        raise ValueError("Embedding/Milvus dimensions mismatch")
    # Never use, clear or drop the application collection.
    collection = "eval_rag_" + uuid4().hex
    backend = MilvusKnowledgeBase(
        uri=settings.milvus_uri, token=settings.milvus_token, collection_name=collection,
        embedding_model=model,
        embedding_revision=(f"{settings.embedding_provider}:{settings.embedding_model}:"
                            f"{settings.milvus_embedding_revision}"),
        timeout_seconds=settings.milvus_timeout_seconds,
        candidate_multiplier=settings.rag_candidate_multiplier, rrf_k=settings.rag_rrf_k,
        bm25_weight=settings.rag_bm25_weight, vector_weight=settings.rag_vector_weight)
    reranker = get_reranker(settings)
    use_rerank = not isinstance(reranker, NoopReranker)
    candidate_count = max(5, settings.rerank_candidate_count) if use_rerank else 5
    rows = []
    report = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": ("real_milvus_hash_contract" if args.allow_hash
                 and settings.embedding_provider in {"hash", "local", "fake"}
                 else "real_milvus_semantic_retrieval"),
        "dataset_sha256": dataset_hash, "dataset_version": dataset.version,
        "synthetic": True, "collection": collection, "selected_split": args.split,
        "selected_cases": len(cases),
        "configuration": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.milvus_embedding_revision,
            "dimensions": model.dimensions, "rrf_k": settings.rag_rrf_k,
            "weights": [settings.rag_bm25_weight, settings.rag_vector_weight],
            "per_route_candidate_limit": min(candidate_count * settings.rag_candidate_multiplier, 1000),
            "rerank_enabled": use_rerank, "rerank_model": settings.rerank_model if use_rerank else None,
            "application_result_cache": False, "pymilvus": version("pymilvus"),
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ("app/rag/milvus_store.py", "app/rag/ranking.py",
                                         "app/evaluation/retrieval.py")},
        },
        "scope": [
            "Same Milvus BM25/HNSW/COSINE/RRF implementation as production search.",
            "No route/answer-quality/abstention evaluation; no chat LLM called.",
            "Empty-gold questions are diagnostics, excluded from Hit/Recall/MRR.",
            "Rerank errors reported explicitly, never counted as successful reranking.",
            "No tuning performed on held-out test cases. Synthetic labels need human review.",
        ],
        "cases": rows,
    }
    admin = None
    try:
        admin = await asyncio.to_thread(
            MilvusClient, uri=settings.milvus_uri, token=settings.milvus_token,
            timeout=settings.milvus_timeout_seconds)
        if await asyncio.to_thread(admin.has_collection, collection_name=collection):
            # UUID collision: no cleanup of a pre-existing collection.
            admin.close()
            admin = None
            raise ValueError("Evaluation collection unexpectedly exists")
        doc_keys = {}
        report["documents"] = []
        for doc in dataset.documents:
            item = await backend.add_document(doc.title, doc.content)
            doc_keys[item.doc_id] = doc.key
            report["documents"].append({
                "key": doc.key, "version": doc.version, "doc_id": item.doc_id,
                "content_sha256": hashlib.sha256(doc.content.encode()).hexdigest(),
                "chunk_count": item.chunk_count})
        report["ingest_embedding_calls"] = model.calls
        evidence = {e.id: e for e in dataset.evidence}
        for index, case in enumerate(cases, 1):
            query = resolve_followup(
                case.question, [SimpleNamespace(role="user", content=h) for h in case.history])
            row = {"id": case.id, "split": case.split, "category": case.category,
                   "question": case.question, "retrieval_query": query,
                   "answerable": case.answerable, "gold_evidence": case.evidence_ids,
                   "strategies": {}}
            started = perf_counter()
            try:
                strategies = await backend.search_strategies(query, candidate_count)
                row["shared_retrieval_ms"] = round((perf_counter() - started) * 1000, 2)
                for name, sources in strategies.items():
                    row["strategies"][name] = score_sources(
                        sources[:5], case, evidence, doc_keys)
                if use_rerank:
                    rerank_started = perf_counter()
                    try:
                        sources = await reranker.rerank(query, strategies["rrf"], top_k=5)
                        row["strategies"]["rerank"] = score_sources(
                            sources, case, evidence, doc_keys)
                    except Exception as exc:
                        row["strategies"]["rerank"] = {"error": type(exc).__name__}
                    row["rerank_ms"] = round((perf_counter() - rerank_started) * 1000, 2)
            except Exception as exc:
                row["error"] = type(exc).__name__  # never dump HTTP bodies/credentials
            rows.append(row)
            write_report(args.output, report)
            print(f"{index}/{len(cases)} {case.id}: {'error' if 'error' in row else 'recorded'}",
                  flush=True)
        report["summary"] = aggregate(rows)
        report["by_split"] = {split: aggregate([r for r in rows if r["split"] == split])
                              for split in ("dev", "test")}
        report["by_category"] = {category: aggregate([r for r in rows if r["category"] == category])
                                 for category in sorted({r["category"] for r in rows})}
        report["status"] = ("completed_with_errors" if any(
            "error" in r or any("error" in v for v in r["strategies"].values()) for r in rows)
                            else "completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
    finally:
        report["embedding_calls"] = model.calls
        try:
            await backend.close()
            if admin is not None and not args.keep_collection:
                # Only this run's randomly named evaluation collection is eligible.
                if not collection.startswith("eval_rag_") or collection == settings.milvus_collection:
                    raise ValueError("Unsafe cleanup target")
                if await asyncio.to_thread(admin.has_collection, collection_name=collection):
                    await asyncio.to_thread(admin.drop_collection, collection_name=collection)
                report["collection_cleaned"] = True
            else:
                report["collection_cleaned"] = False
        except Exception as exc:
            report["cleanup_error_type"] = type(exc).__name__
        finally:
            if admin is not None:
                admin.close()
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_report(args.output, report)
    print(f"Report: {args.output}; status={report['status']}", flush=True)
    return 0 if report["status"] == "completed" and "cleanup_error_type" not in report else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "examples/retrieval_benchmark_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/live-retrieval-results.json")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--keep-collection", action="store_true")
    parser.add_argument("--allow-hash", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
