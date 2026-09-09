from collections.abc import Callable, Sequence
from typing import Protocol

import httpx

from app.core.config import Settings
from app.schemas.chat import SourceChunk


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: Sequence[SourceChunk],
        *,
        top_k: int,
    ) -> list[SourceChunk]:
        raise NotImplementedError


class NoopReranker:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[SourceChunk],
        *,
        top_k: int,
    ) -> list[SourceChunk]:
        del query
        return list(candidates[:top_k])


class HTTPReranker:
    """Adapter for `/rerank` APIs returning index and relevance_score fields."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 15,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        if not url:
            raise ValueError("RERANK_URL is required when RERANK_PROVIDER=http")
        if not model:
            raise ValueError("RERANK_MODEL is required when RERANK_PROVIDER=http")
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.client_factory = client_factory or httpx.AsyncClient

    async def rerank(
        self,
        query: str,
        candidates: Sequence[SourceChunk],
        *,
        top_k: int,
    ) -> list[SourceChunk]:
        if not candidates or top_k <= 0:
            return []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.client_factory(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                headers=headers,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [candidate.snippet for candidate in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
            )
            response.raise_for_status()
            payload = response.json()

        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Rerank response must contain a results list")

        reranked: list[SourceChunk] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict) or "index" not in result:
                raise ValueError("Each rerank result must contain an index")
            index = int(result["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError("Rerank response contains an invalid or duplicate index")
            raw_score = result.get("relevance_score", result.get("score"))
            if raw_score is None:
                raise ValueError("Each rerank result must contain a relevance score")
            seen.add(index)
            reranked.append(
                candidates[index].model_copy(
                    update={"score": round(max(0.0, min(float(raw_score), 1.0)), 4)}
                )
            )
            if len(reranked) >= top_k:
                break

        return reranked


def get_reranker(settings: Settings) -> Reranker:
    provider = settings.rerank_provider.lower()
    if provider in {"none", "off", "disabled"}:
        return NoopReranker()
    if provider == "http":
        return HTTPReranker(
            url=settings.rerank_url,
            model=settings.rerank_model,
            api_key=settings.rerank_api_key,
            timeout=settings.rerank_timeout_seconds,
        )
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")
