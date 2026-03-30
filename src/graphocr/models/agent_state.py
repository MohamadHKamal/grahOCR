"""LangGraph agent state models."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import add_messages

from graphocr.models.extraction import ExtractionResult
from graphocr.models.failure import Challenge, FailureClassification, GraphViolation
from graphocr.models.token import BoundingBox, SpatialToken


class RedTeamState(TypedDict):
    """Shared state for the multi-agent red team pipeline.

    All agents read/write to this state. This enables each agent to see
    every previous agent's full output, allowing deeper adversarial challenges.
    """

    # Input
    document_id: str
    spatial_tokens: list[SpatialToken]
    page_images: dict[int, str]  # page_number -> image path

    # RAG — policy context (injected before agent pipeline starts)
    policy_context: str  # Formatted policy text for extractor
    policy_context_validator: str  # Formatted for validator
    policy_context_challenger: str  # Formatted for challenger
    retrieval_method: str  # How the policy was retrieved
    retrieval_warnings: list[str]  # Warnings from the retriever

    # Extractor output
    extraction: ExtractionResult | None

    # Validator output (accumulates across rounds)
    validation_issues: Annotated[list[str], operator.add]

    # Graph validation
    graph_violations: Annotated[list[GraphViolation], operator.add]

    # Challenger output (accumulates across rounds)
    challenges: Annotated[list[Challenge], operator.add]

    # Self-healing
    rescan_regions: list[BoundingBox]
    rescan_results: list[SpatialToken]
    failure_classifications: list[FailureClassification]

    # Self-healing tracking
    self_healing_applied: bool

    # Consensus tracking
    round_number: int
    max_rounds: int
    consensus_reached: bool
    final_result: ExtractionResult | None

    # Audit trail (LangGraph message accumulation)
    messages: Annotated[list, add_messages]
