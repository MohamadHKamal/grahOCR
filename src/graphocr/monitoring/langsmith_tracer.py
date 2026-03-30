"""LangSmith tracing integration for full pipeline observability.

Every agent step, OCR call, Neo4j query, and DSPy optimization
is traced through LangSmith for monitoring and debugging.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger

logger = get_logger(__name__)


def configure_langsmith() -> None:
    """Configure LangSmith environment variables for tracing."""
    settings = get_settings()

    if not settings.langsmith_api_key:
        logger.info("langsmith_disabled", reason="No API key configured")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info(
        "langsmith_configured",
        project=settings.langsmith_project,
        tracing=settings.langsmith_tracing,
    )


def get_trace_metadata(
    document_id: str,
    processing_path: str = "",
    agent_role: str = "",
    round_number: int = 0,
) -> dict[str, Any]:
    """Build metadata dict for LangSmith trace tagging.

    This metadata is attached to every LangSmith run so traces
    can be filtered and analyzed by document, path, agent, etc.
    """
    return {
        "document_id": document_id,
        "processing_path": processing_path,
        "agent_role": agent_role,
        "round_number": round_number,
        "pipeline": "graphocr",
    }


class AccuracyTracker:
    """Tracks pipeline accuracy for circuit breaker and monitoring.

    Maintains a rolling window of accuracy measurements.
    """

    def __init__(self, window_size: int = 1000):
        self._results: list[tuple[float, bool]] = []  # (timestamp, correct)
        self._window_size = window_size

    def record(self, correct: bool) -> None:
        """Record a single result."""
        import time
        self._results.append((time.time(), correct))
        if len(self._results) > self._window_size * 2:
            self._results = self._results[-self._window_size:]

    @property
    def accuracy(self) -> float:
        """Current rolling accuracy."""
        recent = self._results[-self._window_size:]
        if not recent:
            return 1.0
        return sum(1 for _, c in recent if c) / len(recent)

    @property
    def sample_count(self) -> int:
        return len(self._results)

    def detect_decay(self, window_minutes: int = 60, slope_threshold: float = -0.001) -> bool:
        """Detect gradual accuracy decay (not just sudden failures).

        Uses linear regression over the time window to detect a
        downward trend in accuracy.
        """
        import time

        cutoff = time.time() - (window_minutes * 60)
        recent = [(t, c) for t, c in self._results if t >= cutoff]

        if len(recent) < 50:
            return False

        # Simple linear regression on accuracy over time
        n = len(recent)
        x_vals = [r[0] for r in recent]
        y_vals = [1.0 if r[1] else 0.0 for r in recent]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return False

        slope = numerator / denominator
        return slope < slope_threshold


# Global tracker
accuracy_tracker = AccuracyTracker()
