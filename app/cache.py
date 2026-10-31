import hashlib
from typing import Optional
from time import time


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live).
    
    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0


    def _make_key(self, query: str) -> str:
        """Create a cache key from the normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    # 'What is Python?' and 'what is python?' should map to the same cache key

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
        # Check TTL
        if time.time() - entry["timestamp"] < self.ttl:
            self.hits += 1
            return entry["response"]
        else:
            # Expired - remove it
            del self._cache[key]

        self.misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Cache a response."""
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
