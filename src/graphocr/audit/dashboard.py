"""Audit Dashboard — REST-consumable diagnostic reports.

Provides the "Single Source of Truth" audit tool described in Task 1:
- Per-document failure classification (Type A vs Type B)
- Metadata schema dump (SpatialToken provenance mapping)
- Jurisdiction compliance status
- Learned rules from Post-Mortem back-propagation
"""

from __future__ import annotations

from graphocr.audit.failure_analyzer import FailureAnalyzer, FailureBreakdown
from graphocr.core.logging import get_logger

logger = get_logger(__name__)

# Singleton analyzer
_analyzer = FailureAnalyzer()


async def get_failure_stats(
    window_hours: int = 24,
    jurisdiction: str | None = None,
) -> dict:
    """Get aggregated failure statistics — the main audit endpoint.

    Returns Type A/B breakdown, root cause distribution, affected fields,
    resolution methods, and training data eligibility.
    """
    breakdown = await _analyzer.analyze(
        window_hours=window_hours,
        jurisdiction=jurisdiction,
    )
    return breakdown.to_dict()


async def get_failure_detail(report_id: str) -> dict | None:
    """Get detailed failure report by ID."""
    report = await _analyzer.get_report_by_id(report_id)
    if report is None:
        return None
    return report.model_dump(mode="json")


def get_metadata_schema() -> dict:
    """Return the SpatialToken metadata schema — the "Semantic Spatial" mapping.

    This is the enforced schema that ensures provenance consistency across
    the entire team (senior and junior engineers).
    """
    return {
        "description": (
            "SpatialToken is the atomic unit of the Semantic Spatial mapping. "
            "Every extracted field traces back to specific tokens, each anchored "
            "to pixel coordinates on a specific page."
        ),
        "spatial_token": {
            "token_id": "UUID v7 — unique, time-sortable identifier",
            "text": "Raw OCR text content",
            "bbox": {
                "x_min": "Left edge (pixels from page left)",
                "y_min": "Top edge (pixels from page top)",
                "x_max": "Right edge",
                "y_max": "Bottom edge",
                "page_number": "1-indexed page in the document",
            },
            "reading_order": "Integer — spatial reading order assigned by column-aware algorithm",
            "confidence": "Float 0.0-1.0 — OCR engine confidence (boosted when engines agree)",
            "ocr_engine": "Which engine produced this token: paddleocr | surya | vlm_rescan",
            "language": "Detected language: ar | en | mixed | unknown",
            "is_handwritten": "Boolean — handwriting detection flag",
            "zone_label": "Layout zone: header | body | table | stamp | signature | footer | margin",
            "normalized_text": "Post-processed text (Arabic normalization, digit correction)",
        },
        "provenance_chain": {
            "description": (
                "Every FieldExtraction.source_tokens links back to SpatialToken.token_id, "
                "which links to BoundingBox coordinates on a specific page image. "
                "This creates an unbroken provenance chain: "
                "AI output → field → source_tokens → pixel coordinates → page image."
            ),
            "example": (
                "claim.patient_name='Ahmed' → source_tokens=['abc12345'] → "
                "token.bbox=(120,340)-(280,365) page=1 → page_image='/tmp/graphocr/doc_id/page_1.png'"
            ),
        },
        "failure_classification": {
            "type_a_spatial_blind": (
                "OCR reading order error — tokens from different columns/regions were merged. "
                "Root causes: ocr_misread, layout_confusion."
            ),
            "type_b_context_blind": (
                "RAG retrieved wrong policy version or missed temporal context. "
                "Root causes: prompt_failure, rule_gap."
            ),
        },
        "federated_constraints": {
            "description": (
                "Data residency is enforced per jurisdiction. Documents are stored in "
                "jurisdiction-specific MinIO buckets. Only learned patterns (not PII) "
                "are shared across jurisdictions."
            ),
            "jurisdictions": {
                "SA": "Saudi Arabia — local processing required",
                "AE": "UAE — local processing required",
                "EG": "Egypt — local processing required",
                "JO": "Jordan — regional processing allowed",
            },
        },
    }


async def get_learned_rules() -> list[dict]:
    """Fetch all active learned rules from Neo4j.

    These are rules created by the Post-Mortem agent when it detects
    a 'rule_gap' — patterns the knowledge graph now catches automatically.
    """
    try:
        from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient

        client = Neo4jClient()
        await client.connect()
        try:
            records = await client.execute_read(
                """
                MATCH (r:LearnedRule {active: true})
                RETURN r.report_id AS report_id,
                       r.affected_field AS field,
                       r.original_value AS bad_value,
                       r.corrected_value AS good_value,
                       r.root_cause AS root_cause,
                       r.created_at AS created_at
                ORDER BY r.created_at DESC
                LIMIT 100
                """
            )
            return [dict(r) for r in records]
        finally:
            await client.close()
    except Exception as e:
        logger.warning("learned_rules_fetch_failed", error=str(e))
        return []
