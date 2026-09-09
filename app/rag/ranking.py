import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class RankedItem(Generic[T]):
    item: T
    score: float


class BM25Ranker:
    """Small BM25 implementation used by the in-memory development backend."""

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._term_frequencies = [Counter(document) for document in corpus]
        self._document_lengths = [len(document) for document in corpus]
        self._document_count = len(corpus)
        self._average_document_length = (
            sum(self._document_lengths) / self._document_count if self._document_count else 0.0
        )

        document_frequencies: Counter[str] = Counter()
        for document in corpus:
            document_frequencies.update(set(document))
        self._inverse_document_frequencies = {
            term: math.log(
                1.0 + (self._document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query_terms: Iterable[str]) -> list[float]:
        if not self._document_count:
            return []

        unique_query_terms = set(query_terms)
        return [
            self._score_document(unique_query_terms, index)
            for index in range(self._document_count)
        ]

    def _score_document(self, query_terms: set[str], index: int) -> float:
        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        average_length = max(self._average_document_length, 1.0)
        length_normalization = 1.0 - self.b + self.b * document_length / average_length

        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = self._inverse_document_frequencies.get(term, 0.0)
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * length_normalization
            score += inverse_frequency * numerator / denominator
        return score


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedItem[T]]],
    *,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RankedItem[T]]:
    """Fuse ranked lists without assuming their raw scores share a scale."""

    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than zero")
    selected_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(selected_weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if not any(weight > 0 for weight in selected_weights):
        raise ValueError("at least one RRF weight must be greater than zero")

    fused_scores: dict[T, float] = {}
    best_raw_score: dict[T, float] = {}
    for ranking, weight in zip(rankings, selected_weights):
        if weight <= 0:
            continue
        seen: set[T] = set()
        for rank, result in enumerate(ranking, start=1):
            if result.item in seen:
                continue
            seen.add(result.item)
            fused_scores[result.item] = fused_scores.get(result.item, 0.0) + weight / (
                rrf_k + rank
            )
            best_raw_score[result.item] = max(
                best_raw_score.get(result.item, float("-inf")), result.score
            )

    return [
        RankedItem(item=item, score=score)
        for item, score in sorted(
            fused_scores.items(),
            key=lambda pair: (pair[1], best_raw_score[pair[0]]),
            reverse=True,
        )
    ]


def normalize_scores(ranking: Sequence[RankedItem[T]]) -> list[RankedItem[T]]:
    if not ranking:
        return []
    maximum = max(result.score for result in ranking)
    if maximum <= 0:
        return [RankedItem(item=result.item, score=0.0) for result in ranking]
    return [RankedItem(item=result.item, score=result.score / maximum) for result in ranking]
