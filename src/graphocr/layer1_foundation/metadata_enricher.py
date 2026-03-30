"""Metadata enricher — adds zone labels, handwriting detection."""

from __future__ import annotations

from graphocr.core.logging import get_logger
from graphocr.core.types import ZoneLabel
from graphocr.models.document import PageImage
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


def enrich_tokens_with_zones(
    tokens: list[SpatialToken],
    layout_zones: list[dict],
) -> list[SpatialToken]:
    """Assign zone labels to tokens based on Surya layout detection.

    Args:
        tokens: Tokens to enrich.
        layout_zones: Zone dicts from SuryaLayoutEngine.detect_layout().

    Returns:
        Tokens with zone_label populated.
    """
    for token in tokens:
        best_zone = _find_containing_zone(token.bbox, layout_zones)
        if best_zone:
            token.zone_label = best_zone["zone_label"]

    assigned = sum(1 for t in tokens if t.zone_label is not None)
    logger.info("zone_enrichment_complete", total=len(tokens), assigned=assigned)
    return tokens


def _find_containing_zone(bbox: BoundingBox, zones: list[dict]) -> dict | None:
    """Find the zone that best contains this bounding box."""
    best_zone = None
    best_overlap = 0.0

    for zone in zones:
        zone_bbox: BoundingBox = zone["bbox"]
        overlap = bbox.iou(zone_bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best_zone = zone

    # Also check simple containment
    if best_zone is None:
        for zone in zones:
            zone_bbox = zone["bbox"]
            cx, cy = bbox.center
            if (zone_bbox.x_min <= cx <= zone_bbox.x_max and
                    zone_bbox.y_min <= cy <= zone_bbox.y_max):
                return zone

    return best_zone


def detect_handwriting(tokens: list[SpatialToken], page: PageImage) -> list[SpatialToken]:
    """Flag tokens that appear to be handwritten.

    Uses a heuristic based on stroke width variation and bbox aspect ratio.
    Handwritten text tends to have more irregular bounding boxes.
    """
    if not tokens:
        return tokens

    # Compute statistics for the page
    heights = [t.bbox.height for t in tokens if t.bbox.height > 0]
    if not heights:
        return tokens

    median_height = sorted(heights)[len(heights) // 2]
    height_std = (sum((h - median_height) ** 2 for h in heights) / len(heights)) ** 0.5

    for token in tokens:
        # Heuristic: handwritten tokens tend to have
        # 1. Irregular height (high variance from median)
        # 2. Lower OCR confidence
        # 3. Wider aspect ratios
        height_deviation = abs(token.bbox.height - median_height) / max(median_height, 1)
        aspect_ratio = token.bbox.width / max(token.bbox.height, 1)

        is_hw = (
            height_deviation > 0.5
            and token.confidence < 0.8
            and aspect_ratio > 3.0
        )
        token.is_handwritten = is_hw

    hw_count = sum(1 for t in tokens if t.is_handwritten)
    logger.info("handwriting_detection", total=len(tokens), handwritten=hw_count)
    return tokens
