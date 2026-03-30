"""Langfuse tracing integration for pipeline observability at scale.

Parallel to langsmith_tracer.py — wraps agent calls with Langfuse trace
spans and records per-document accuracy scores for real-time monitoring.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger

logger = get_logger(__name__)

_langfuse_client = None


def configure_langfuse() -> None:
    """Initialize the Langfuse client from settings."""
    global _langfuse_client
    settings = get_settings()
    langfuse_config = settings.monitoring.get("langfuse", {})

    if not langfuse_config.get("enabled", False):
        logger.info("langfuse_disabled")
        return

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=langfuse_config.get("public_key", ""),
            secret_key=langfuse_config.get("secret_key", ""),
            host=langfuse_config.get("host", "https://cloud.langfuse.com"),
        )
        logger.info("langfuse_configured", host=langfuse_config.get("host"))
    except Exception as e:
        logger.warning("langfuse_init_failed", error=str(e))


def create_trace(
    document_id: str,
    processing_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Create a new Langfuse trace for a document processing run.

    Returns a trace object, or None if Langfuse is not configured.
    """
    if not _langfuse_client:
        return None

    try:
        trace = _langfuse_client.trace(
            name="document_processing",
            metadata={
                "document_id": document_id,
                "processing_path": processing_path,
                "pipeline": "graphocr",
                **(metadata or {}),
            },
            tags=[processing_path] if processing_path else [],
        )
        return trace
    except Exception as e:
        logger.warning("langfuse_trace_failed", error=str(e))
        return None


@contextmanager
def trace_agent_span(
    trace: Any | None,
    agent_name: str,
    document_id: str,
    round_number: int = 0,
    input_data: dict | None = None,
) -> Generator[Any | None, None, None]:
    """Context manager that wraps an agent call in a Langfuse span.

    Usage:
        with trace_agent_span(trace, "extractor", doc_id) as span:
            result = await extractor_node(state)
            if span:
                span.end(output={"fields": len(result.fields)})
    """
    if not trace:
        yield None
        return

    try:
        span = trace.span(
            name=agent_name,
            metadata={
                "document_id": document_id,
                "round_number": round_number,
            },
            input=input_data or {},
        )
        start = time.time()
        yield span
        latency_ms = (time.time() - start) * 1000
        span.update(metadata={"latency_ms": round(latency_ms, 1)})
    except Exception as e:
        logger.warning("langfuse_span_failed", agent=agent_name, error=str(e))
        yield None


def record_score(
    trace: Any | None,
    name: str,
    value: float,
    comment: str = "",
) -> None:
    """Record a score on a Langfuse trace (e.g., extraction confidence, accuracy)."""
    if not trace:
        return

    try:
        trace.score(name=name, value=value, comment=comment)
    except Exception as e:
        logger.warning("langfuse_score_failed", name=name, error=str(e))


def flush() -> None:
    """Flush pending Langfuse events."""
    if _langfuse_client:
        try:
            _langfuse_client.flush()
        except Exception:
            pass
