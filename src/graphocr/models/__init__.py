"""Data models — re-export key types for convenience."""

from graphocr.models.claim import ClaimLineItem, InsuranceClaim, MedicationEntry
from graphocr.models.document import DocumentBatch, PageImage, RawDocument
from graphocr.models.extraction import ExtractionResult, FieldExtraction
from graphocr.models.failure import (
    Challenge,
    FailureClassification,
    FailureReport,
    GraphViolation,
)
from graphocr.models.token import BoundingBox, SpatialToken

__all__ = [
    "BoundingBox",
    "Challenge",
    "ClaimLineItem",
    "DocumentBatch",
    "ExtractionResult",
    "FailureClassification",
    "FailureReport",
    "FieldExtraction",
    "GraphViolation",
    "InsuranceClaim",
    "MedicationEntry",
    "PageImage",
    "RawDocument",
    "SpatialToken",
]
