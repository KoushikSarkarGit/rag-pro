import logging
import json
import time
from datetime import datetime, UTC
from threading import Lock
from typing import Any
from app.config import get_settings


settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation (ELK, Datadog, etc.)."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        # Merge any extra data attached to the record
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)

def get_logger(name: str = "production-api") -> logging.Logger:
    """Create a structured JSON logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)

    return logger


class MetricsCollector:
    """Collect in-process request counts, latency, cache, and token metrics."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.total_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._total_latency_ms = 0.0
        self._cache_hits = 0
        self._lock = Lock()

    def record_request(
        self,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        error: bool = False,
    ) -> None:
        """Record one completed or failed request."""
        latency = max(0.0, float(latency_ms))
        input_count = max(0, int(input_tokens))
        output_count = max(0, int(output_tokens))

        with self._lock:
            self.total_requests += 1
            self.total_errors += int(error)
            self.total_input_tokens += input_count
            self.total_output_tokens += output_count
            self._total_latency_ms += latency
            self._cache_hits += int(cache_hit)

    @property
    def summary(self) -> dict[str, int | float | str]:
        """Return a snapshot matching the API's ``MetricsResponse`` schema."""
        with self._lock:
            requests = self.total_requests
            errors = self.total_errors
            cache_hits = self._cache_hits

            error_rate = errors / requests if requests else 0.0
            cache_hit_rate = cache_hits / requests if requests else 0.0
            average_latency = self._total_latency_ms / requests if requests else 0.0

            return {
                "total_requests": requests,
                "total_errors": errors,
                "error_rate": f"{error_rate:.1%}",
                "avg_latency_ms": round(average_latency, 2),
                "cache_hit_rate": f"{cache_hit_rate:.1%}",
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            }


class RequestTimer:
    """Measure elapsed wall-clock duration for a ``with`` block in ms."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    def __enter__(self) -> "RequestTimer":
        self._started_at = time.perf_counter()
        self._stopped_at = None
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._started_at is not None:
            self._stopped_at = time.perf_counter()
        # Do not suppress exceptions raised inside the timed block.
        return False

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds, including while the timed block is running."""
        if self._started_at is None:
            return 0.0
        end = (
            self._stopped_at
            if self._stopped_at is not None
            else time.perf_counter()
        )
        return (end - self._started_at) * 1000


