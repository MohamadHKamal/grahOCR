"""Custom exception hierarchy."""

from __future__ import annotations


class GraphOCRError(Exception):
    """Base exception for the pipeline."""


class OCRExtractionError(GraphOCRError):
    """OCR engine failed to extract text."""


class SpatialAssemblyError(GraphOCRError):
    """Failed to merge/order OCR outputs spatially."""


class GraphValidationError(GraphOCRError):
    """Neo4j knowledge graph validation failed."""

    def __init__(self, violations: list[dict], message: str = "Graph validation failed"):
        self.violations = violations
        super().__init__(message)


class AgentConsensusError(GraphOCRError):
    """Agents could not reach consensus after max rounds."""


class SelfHealingError(GraphOCRError):
    """VLM re-scan or feedback loop failed."""


class CircuitBreakerOpenError(GraphOCRError):
    """Circuit breaker is open — processing path temporarily disabled."""


class DSPyOptimizationError(GraphOCRError):
    """DSPy optimization run failed."""


class DataResidencyError(GraphOCRError):
    """Data would leave its jurisdiction."""
