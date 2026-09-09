import hashlib
import math
import re
from collections.abc import Callable
from typing import Protocol

import httpx

from app.core.config import Settings


class EmbeddingModel(Protocol):
    dimensions: int | None

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingModel:
    """Local deterministic embedding model for development.

    It is not a semantic model. It gives every text a stable vector so the RAG
    pipeline can exercise embedding, vector storage, and cosine retrieval before
    a real embedding provider is configured.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


class OpenAICompatibleEmbeddingModel:
    """Embedding adapter for OpenAI-compatible `/embeddings` APIs."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = 30,
        dimensions: int | None = None,
        client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dimensions = dimensions
        self.client_factory = client_factory or httpx.Client

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for OpenAI-compatible embeddings")

        payload: dict[str, object] = {"model": self.model, "input": text}
        with self.client_factory(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        vector = data["data"][0]["embedding"]
        normalized = _normalize([float(value) for value in vector])
        if self.dimensions is not None and len(normalized) != self.dimensions:
            raise RuntimeError(
                "Embedding API returned an unexpected vector dimension: "
                f"{len(normalized)} != {self.dimensions}"
            )
        return normalized


def get_embedding_model(settings: Settings) -> EmbeddingModel:
    provider = settings.embedding_provider.lower()
    if provider in {"hash", "local", "fake"}:
        return HashEmbeddingModel(dimensions=settings.embedding_dimensions)
    if provider == "openai-compatible":
        return OpenAICompatibleEmbeddingModel(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout=settings.embedding_timeout_seconds,
            dimensions=settings.embedding_dimensions,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def tokenize_for_search(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    chinese_bigrams = [
        lowered[index : index + 2]
        for index in range(len(lowered) - 1)
        if all("\u4e00" <= char <= "\u9fff" for char in lowered[index : index + 2])
    ]
    return words + chinese_chars + chinese_bigrams


def tokenize(text: str) -> set[str]:
    return set(tokenize_for_search(text))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
