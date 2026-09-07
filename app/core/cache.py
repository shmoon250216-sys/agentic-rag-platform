import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """Small in-process TTL cache for local hot data."""

    def __init__(self, max_entries: int = 256, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

    def get(self, key: str) -> Any | None:
        entry = self._items.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at <= time.monotonic():
            self.misses += 1
            self.evictions += 1
            self._items.pop(key, None)
            return None

        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._prune_expired()
        if key not in self._items and len(self._items) >= self.max_entries:
            oldest_key = next(iter(self._items))
            self._items.pop(oldest_key, None)
            self.evictions += 1

        self._items[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        self.sets += 1

    def clear(self) -> None:
        self._items.clear()
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

    def stats(self) -> dict[str, int]:
        self._prune_expired()
        return {
            "size": len(self._items),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
        }

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._items.pop(key, None)
            self.evictions += 1


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"
