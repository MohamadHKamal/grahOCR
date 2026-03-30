"""Circuit breaker — monitors failure rates and auto-disables degraded paths.

Implements a sliding window circuit breaker pattern:
- CLOSED: Normal operation, tracking failures.
- OPEN: Path is disabled, all traffic rerouted. Auto-recovers after timeout.
- HALF_OPEN: Allowing limited traffic to test recovery.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from threading import Lock

from graphocr.core.config import get_settings
from graphocr.core.exceptions import CircuitBreakerOpenError
from graphocr.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Sliding window circuit breaker for processing paths.

    Thread-safe. Tracks success/failure in a time window and opens
    the circuit when the failure rate exceeds the threshold.
    """

    def __init__(
        self,
        name: str,
        window_seconds: int = 300,
        failure_threshold: float = 0.15,
        min_calls: int = 50,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 10,
    ):
        self.name = name
        self.window_seconds = window_seconds
        self.failure_threshold = failure_threshold
        self.min_calls = min_calls
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._calls: deque[tuple[float, bool]] = deque()  # (timestamp, success)
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._opened_at > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("circuit_half_open", breaker=self.name)
            return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._calls.append((time.time(), True))
            self._prune()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    logger.info("circuit_closed", breaker=self.name)

    def record_failure(self) -> None:
        """Record a failed call. May open the circuit."""
        with self._lock:
            self._calls.append((time.time(), False))
            self._prune()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open -> back to open
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.warning("circuit_reopened", breaker=self.name)
                return

            # Check if we should open
            if len(self._calls) >= self.min_calls:
                failures = sum(1 for _, success in self._calls if not success)
                rate = failures / len(self._calls)
                if rate >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.time()
                    logger.warning(
                        "circuit_opened",
                        breaker=self.name,
                        failure_rate=rate,
                        window_calls=len(self._calls),
                    )

    def check(self) -> None:
        """Check if the circuit allows a call. Raises if OPEN."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.recovery_timeout}s."
            )

    @property
    def failure_rate(self) -> float:
        """Current failure rate in the window."""
        with self._lock:
            self._prune()
            if not self._calls:
                return 0.0
            failures = sum(1 for _, success in self._calls if not success)
            return failures / len(self._calls)

    @property
    def metrics(self) -> dict:
        """Current circuit breaker metrics."""
        with self._lock:
            self._prune()
            total = len(self._calls)
            failures = sum(1 for _, s in self._calls if not s)
            return {
                "name": self.name,
                "state": self._state.value,
                "total_calls": total,
                "failures": failures,
                "failure_rate": failures / total if total > 0 else 0.0,
            }

    def _prune(self) -> None:
        """Remove calls outside the time window."""
        cutoff = time.time() - self.window_seconds
        while self._calls and self._calls[0][0] < cutoff:
            self._calls.popleft()


class CircuitBreakerRegistry:
    """Registry of circuit breakers for different processing paths."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, **kwargs) -> CircuitBreaker:
        if name not in self._breakers:
            settings = get_settings()
            cb_config = settings.monitoring.get("circuit_breaker", {})
            self._breakers[name] = CircuitBreaker(
                name=name,
                window_seconds=kwargs.get("window_seconds", cb_config.get("window_seconds", 300)),
                failure_threshold=kwargs.get("failure_threshold", cb_config.get("failure_rate_threshold", 0.15)),
                min_calls=kwargs.get("min_calls", cb_config.get("min_calls_in_window", 50)),
                recovery_timeout=kwargs.get("recovery_timeout", cb_config.get("recovery_timeout_seconds", 60)),
            )
        return self._breakers[name]

    def all_metrics(self) -> list[dict]:
        return [b.metrics for b in self._breakers.values()]


# Global registry
circuit_breakers = CircuitBreakerRegistry()
