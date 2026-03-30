"""Data residency enforcement for federated processing.

Ensures documents are stored and processed within their jurisdiction.
Models run locally; only learned patterns (not data) are shared globally.
"""

from __future__ import annotations

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.models.document import RawDocument

logger = get_logger(__name__)


def get_storage_bucket(document: RawDocument) -> str:
    """Determine the storage bucket based on document jurisdiction.

    Each jurisdiction gets its own MinIO bucket to enforce data separation.
    """
    settings = get_settings()
    base_bucket = settings.minio_bucket
    jurisdiction = document.jurisdiction or "default"
    return f"{base_bucket}-{jurisdiction.lower()}"


def validate_document_routing(document: RawDocument, target_region: str) -> bool:
    """Check if a document can be routed to a target processing region."""
    from graphocr.compliance.jurisdiction import resolve_jurisdiction

    if not document.jurisdiction:
        return True  # No jurisdiction restriction

    rules = resolve_jurisdiction(document.jurisdiction)
    allowed = rules.get("allowed_regions", [])

    if not allowed:
        return True

    if target_region in allowed:
        return True

    logger.warning(
        "routing_blocked",
        document_id=document.document_id,
        jurisdiction=document.jurisdiction,
        target_region=target_region,
        allowed_regions=allowed,
    )
    return False


def filter_shareable_patterns(patterns: dict) -> dict:
    """Filter learned patterns to only include shareable (non-PII) data.

    In federated mode, models run locally but learned patterns can be
    shared globally — as long as they don't contain PII.
    """
    shareable = {}

    # Only share aggregate statistics and structural patterns
    safe_keys = {
        "ocr_confidence_distribution",
        "layout_patterns",
        "failure_type_distribution",
        "language_distribution",
        "zone_distribution",
        "reading_order_patterns",
        "prompt_performance_metrics",
    }

    for key, value in patterns.items():
        if key in safe_keys:
            shareable[key] = value

    return shareable
