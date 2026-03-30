"""Context injector — wires RAG-retrieved policy into the agent pipeline.

This module bridges the RAG retriever and the multi-agent red team.
It injects the correct policy context into:
1. The Extractor agent (so it knows which plan rules to apply)
2. The Validator agent (so it validates against the right policy version)
3. The Challenger agent (so it can challenge wrong-policy-version errors)

It also adds a "policy verification" step to the knowledge graph that
checks the claim's policy reference against what was actually retrieved.
"""

from __future__ import annotations

from datetime import date

from graphocr.core.logging import get_logger
from graphocr.models.failure import GraphViolation
from graphocr.models.policy import PolicyChunk, RetrievalContext
from graphocr.models.token import SpatialToken
from graphocr.rag.retriever import TemporalPolicyRetriever
from graphocr.rag.vector_store import PolicyVectorStore

logger = get_logger(__name__)


class PolicyContextInjector:
    """Injects policy context into the claim processing pipeline.

    Sits between Layer 1 (OCR) and Layer 2 (Agents) to ensure every
    agent has the correct policy context for its task.
    """

    def __init__(self, vector_store: PolicyVectorStore):
        self._retriever = TemporalPolicyRetriever(vector_store)

    def get_context_for_claim(
        self,
        tokens: list[SpatialToken],
        claim_date: date | None = None,
        jurisdiction: str = "",
    ) -> RetrievalContext:
        """Retrieve and validate policy context for a claim.

        Returns the policy context plus any warnings about the retrieval.
        """
        context = self._retriever.retrieve(
            tokens=tokens,
            claim_date=claim_date,
            jurisdiction=jurisdiction,
        )

        # Validate retrieval quality
        self._validate_context(context)

        return context

    def format_for_extractor(self, context: RetrievalContext) -> str:
        """Format policy context for the Extractor agent prompt.

        Provides coverage rules, limits, and applicable codes.
        """
        if not context.policy_chunks:
            return "No policy context available. Extract all visible fields."

        lines = [
            f"POLICY CONTEXT (retrieved via {context.retrieval_method}):",
            f"Policy: {context.matched_policy_number} v{context.matched_policy_version}",
        ]

        if context.query_date:
            lines.append(f"Valid on: {context.query_date}")

        if context.warnings:
            lines.append(f"WARNINGS: {'; '.join(context.warnings)}")

        lines.append("\nRELEVANT POLICY SECTIONS:")
        for chunk in context.policy_chunks[:5]:
            lines.append(f"\n--- [{chunk.section_type}] {chunk.section_title} ---")
            lines.append(chunk.text[:500])

        return "\n".join(lines)

    def format_for_validator(self, context: RetrievalContext) -> str:
        """Format policy context for the Validator agent.

        Emphasizes limits, exclusions, and preauth requirements.
        """
        if not context.policy_chunks:
            return "No policy context. Skip policy-specific validation."

        lines = [
            "POLICY RULES FOR VALIDATION:",
            f"Policy: {context.matched_policy_number} v{context.matched_policy_version}",
        ]

        # Prioritize coverage, exclusion, and limit chunks
        priority_types = ["exclusion", "benefit_limit", "preauth", "coverage"]
        sorted_chunks = sorted(
            context.policy_chunks,
            key=lambda c: (
                priority_types.index(c.section_type)
                if c.section_type in priority_types
                else len(priority_types)
            ),
        )

        for chunk in sorted_chunks[:5]:
            lines.append(f"\n[{chunk.section_type.upper()}] {chunk.section_title}")
            lines.append(chunk.text[:400])

        return "\n".join(lines)

    def format_for_challenger(self, context: RetrievalContext) -> str:
        """Format policy context for the Challenger agent.

        Highlights potential version mismatches and ambiguities.
        """
        if not context.policy_chunks:
            return "No policy context. Challenge based on general medical rules only."

        lines = [
            "POLICY CONTEXT FOR ADVERSARIAL REVIEW:",
            f"Policy: {context.matched_policy_number} v{context.matched_policy_version}",
            f"Retrieval method: {context.retrieval_method}",
            f"Retrieval confidence: {context.confidence:.2f}",
        ]

        if context.warnings:
            lines.append(f"\nRETRIEVAL WARNINGS (check these):")
            for w in context.warnings:
                lines.append(f"  - {w}")

        lines.append("\nCHALLENGE ANGLES:")
        lines.append("- Is this the correct policy version for the claim date?")
        lines.append("- Does the claim reference a rider/amendment not retrieved?")
        lines.append("- Are there exclusions that should block this claim?")

        for chunk in context.policy_chunks[:3]:
            lines.append(f"\n[{chunk.section_type}] {chunk.text[:300]}")

        return "\n".join(lines)

    def validate_policy_match(self, context: RetrievalContext) -> list[GraphViolation]:
        """Check for policy retrieval inconsistencies.

        Generates graph violations when:
        1. No policy was found for the reference
        2. Policy version doesn't match the claim date
        3. Retrieval confidence is low
        """
        violations: list[GraphViolation] = []

        if not context.policy_chunks:
            violations.append(GraphViolation(
                rule_name="policy_not_found",
                field_name="policy_reference",
                extracted_value=context.policy_reference_from_claim or "none",
                expected_constraint="A valid policy must be found",
                violation_message="No policy found matching the claim reference",
                severity=0.8,
            ))

        if context.retrieval_method == "semantic_only":
            violations.append(GraphViolation(
                rule_name="policy_retrieval_quality",
                field_name="policy_reference",
                extracted_value=f"method={context.retrieval_method}",
                expected_constraint="Temporal-filtered retrieval preferred",
                violation_message=(
                    "Policy retrieved via pure semantic match (no temporal filtering). "
                    "Risk of Failure Type B — wrong policy version."
                ),
                severity=0.6,
            ))

        if context.confidence < 0.5 and context.policy_chunks:
            violations.append(GraphViolation(
                rule_name="policy_match_confidence",
                field_name="policy_reference",
                extracted_value=f"confidence={context.confidence:.2f}",
                expected_constraint="Confidence >= 0.5",
                violation_message=f"Low policy match confidence ({context.confidence:.2f})",
                severity=0.5,
            ))

        return violations

    def _validate_context(self, context: RetrievalContext) -> None:
        """Internal validation and logging."""
        if context.retrieval_method == "semantic_only":
            logger.warning(
                "type_b_risk",
                reason="Falling back to context-blind retrieval",
                claim_id=context.claim_id,
            )
