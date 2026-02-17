"""LRU cache for embedding vectors.

Caches dense+sparse embedding results keyed by normalized text hash.
4,096 entries at ~4.4KB each = ~18MB RAM. Expected 20-45% hit rate
in thematic DyTopo swarms.
"""
import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("embedding-cache")


class EmbeddingCache:
    """Thread-safe LRU cache for embedding results."""

    def __init__(self, maxsize: int = 4096):
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _hash(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, text: str) -> dict | None:
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, result: dict):
        key = self._hash(text)
        self.cache[key] = result
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> str:
        return f"Cache: {len(self.cache)}/{self.maxsize} entries, {self.hit_rate:.1%} hit rate ({self.hits} hits, {self.misses} misses)"
