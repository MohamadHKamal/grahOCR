"""Failure classification and reporting models.

This module defines Pydantic models used throughout the GraphOCR pipeline
to represent, classify, and report failures that occur during document
processing. It covers the full failure lifecycle: initial classification,
graph-based rule violations, adversarial challenges, and post-mortem reports.

Classes:
    FailureClassification: Categorises a detected pipeline failure with
        severity scoring and a suggested remediation strategy.
    GraphViolation: Records a constraint violation surfaced by the Neo4j
        knowledge graph during field validation.
    Challenge: Represents an adversarial hypothesis raised by the Challenger
        agent questioning an extracted value.
    FailureReport: Post-mortem artefact capturing root cause, resolution,
        and optional DSPy training eligibility for a resolved failure.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from uuid_extensions import uuid7

from graphocr.core.types import FailureType


class FailureClassification(BaseModel):
    """Classification of a detected failure in the pipeline.

    Attributes:
        failure_id: Unique identifier (UUID v7) for this failure instance.
        failure_type: Enumerated category of the failure (e.g. OCR, layout).
        affected_tokens: List of token IDs impacted by the failure.
        severity: Normalised severity score in the range [0.0, 1.0].
        suggested_remedy: Recommended remediation action. One of
            ``"vlm_rescan"``, ``"context_reinjection"``, or ``"escalate"``.
        evidence: Free-text evidence supporting the classification.
    """

    failure_id: str = Field(default_factory=lambda: str(uuid7()))
    failure_type: FailureType
    affected_tokens: list[str] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0)
    suggested_remedy: str
    evidence: str = ""


class GraphViolation(BaseModel):
    """A rule violation detected by the Neo4j knowledge graph.

    Attributes:
        rule_name: Name of the graph constraint rule that was violated.
        field_name: Document field that triggered the violation.
        extracted_value: The value extracted from the document.
        expected_constraint: Description of the expected constraint.
        violation_message: Human-readable explanation of the violation.
        source_tokens: Token IDs that contributed to the extracted value.
        severity: Normalised severity score in the range [0.0, 1.0].
            Defaults to ``0.8``.
    """

    rule_name: str
    field_name: str
    extracted_value: str
    expected_constraint: str
    violation_message: str
    source_tokens: list[str] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0, default=0.8)


class Challenge(BaseModel):
    """An adversarial challenge raised by the Challenger agent.

    Attributes:
        challenge_id: Unique identifier (UUID v7) for this challenge.
        target_field: The document field being challenged.
        hypothesis: What the challenger believes could be wrong.
        evidence: Supporting evidence for the challenge.
        proposed_alternative: An alternative value suggested by the
            challenger, if any.
        confidence: Challenger's confidence in the hypothesis, in [0.0, 1.0].
        affected_tokens: Token IDs relevant to the challenged value.
    """

    challenge_id: str = Field(default_factory=lambda: str(uuid7()))
    target_field: str
    hypothesis: str
    evidence: str
    proposed_alternative: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    affected_tokens: list[str] = Field(default_factory=list)


class FailureReport(BaseModel):
    """Post-mortem report for a resolved failure.

    Attributes:
        report_id: Unique identifier (UUID v7) for this report.
        document_id: ID of the document where the failure occurred.
        claim_id: ID of the insurance claim associated with the document.
        root_cause: Identified root cause. One of ``"ocr_misread"``,
            ``"prompt_failure"``, ``"rule_gap"``, or ``"layout_confusion"``.
        failure_type: Enumerated category of the original failure.
        original_value: The incorrect value that was initially extracted.
        corrected_value: The corrected value after resolution.
        affected_field: Name of the document field that was affected.
        affected_tokens: Token IDs involved in the failure.
        resolution_method: How the failure was resolved. One of
            ``"vlm_rescan"``, ``"agent_correction"``, or
            ``"graph_override"``.
        severity: Normalised severity score in the range [0.0, 1.0].
        add_to_dspy_training: Whether this report should be fed back into
            DSPy prompt optimisation as a training example.
        created_at: Timestamp when the report was created.
    """

    report_id: str = Field(default_factory=lambda: str(uuid7()))
    document_id: str
    claim_id: str
    root_cause: str
    failure_type: FailureType
    original_value: str
    corrected_value: str
    affected_field: str
    affected_tokens: list[str] = Field(default_factory=list)
    resolution_method: str
    severity: float = 0.0
    add_to_dspy_training: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
