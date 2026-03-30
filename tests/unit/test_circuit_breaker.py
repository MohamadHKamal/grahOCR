"""Tests for the circuit breaker."""

import time

import pytest

from graphocr.core.exceptions import CircuitBreakerOpenError
from graphocr.layer3_inference.circuit_breaker import CircuitBreaker, CircuitState

pytestmark = [pytest.mark.unit, pytest.mark.layer3]


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", min_calls=5, failure_threshold=0.5)
        assert cb.state == CircuitState.CLOSED

    def test_opens_on_high_failure_rate(self):
        cb = CircuitBreaker("test", min_calls=5, failure_threshold=0.5, window_seconds=60)
        # Record 5 failures
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", min_calls=5, failure_threshold=0.5, window_seconds=60)
        for _ in range(4):
            cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_check_raises_when_open(self):
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError):
            cb.check()

    def test_transitions_to_half_open(self):
        """With recovery_timeout=0, reading state auto-transitions OPEN -> HALF_OPEN."""
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, recovery_timeout=0, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        # With recovery_timeout=0, reading .state immediately transitions to HALF_OPEN
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_recovery_timeout(self):
        """With a non-zero timeout, circuit stays OPEN until timeout expires."""
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, recovery_timeout=60, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_closes_on_success(self):
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, recovery_timeout=0, half_open_max_calls=2, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, recovery_timeout=0, window_seconds=60)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        # After failure in half-open, it reopens — but with timeout=0,
        # reading .state again immediately transitions back to HALF_OPEN
        assert cb.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_metrics(self):
        cb = CircuitBreaker("test", min_calls=2, failure_threshold=0.5, window_seconds=60)
        cb.record_success()
        cb.record_failure()
        m = cb.metrics
        assert m["name"] == "test"
        assert m["total_calls"] == 2
        assert m["failures"] == 1
        assert 0.49 < m["failure_rate"] < 0.51

    def test_failure_rate_property(self):
        cb = CircuitBreaker("test", window_seconds=60)
        assert cb.failure_rate == 0.0
        cb.record_failure()
        assert cb.failure_rate == 1.0
        cb.record_success()
        assert cb.failure_rate == 0.5
