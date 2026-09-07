import time

from app.core.cache import TTLCache, make_cache_key


def test_ttl_cache_tracks_hits_and_misses() -> None:
    cache = TTLCache(max_entries=2, ttl_seconds=60)

    assert cache.get("missing") is None
    cache.set("answer", "cached")

    assert cache.get("answer") == "cached"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_ttl_cache_expires_entries() -> None:
    cache = TTLCache(max_entries=2, ttl_seconds=0)
    cache.set("answer", "cached")
    time.sleep(0.001)

    assert cache.get("answer") is None
    assert cache.stats()["size"] == 0


def test_cache_key_is_stable_for_equivalent_payloads() -> None:
    left = make_cache_key("demo", {"b": 2, "a": 1})
    right = make_cache_key("demo", {"a": 1, "b": 2})

    assert left == right
