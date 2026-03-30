"""Extraction result models with provenance tracking."""

from __future__ import annotations

from pydantic import BaseModel, Field

from graphocr.core.types import ProcessingPath, ValidationStatus


class FieldExtraction(BaseModel):
    """A single extracted field with full provenance."""

    field_name: str
    value: str
    source_tokens: list[str] = Field(default_factory=list)  # token_ids
    confidence: float = 0.0
    validation_status: ValidationStatus = ValidationStatus.VALID
    neo4j_checks_passed: list[str] = Field(default_factory=list)
    neo4j_checks_failed: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Complete extraction output for a single claim document."""

    claim_id: str
    document_id: str
    fields: dict[str, FieldExtraction] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    processing_path: ProcessingPath = ProcessingPath.CHEAP_RAIL
    agent_consensus: dict[str, str] = Field(default_factory=dict)  # agent -> verdict
    rounds_taken: int = 1
    self_healing_applied: bool = False
    escalated: bool = False
