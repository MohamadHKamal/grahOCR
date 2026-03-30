"""Conflict detector — identifies specific regions where agents disagree.

Compares extractor vs validator vs challenger outputs to pinpoint
the exact bounding box regions that need VLM re-scanning.
"""

from __future__ import annotations

from graphocr.core.logging import get_logger
from graphocr.models.extraction import ExtractionResult
from graphocr.models.failure import Challenge, GraphViolation
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


def detect_conflicting_regions(
    extraction: ExtractionResult,
    challenges: list[Challenge],
    graph_violations: list[GraphViolation],
    spatial_tokens: list[SpatialToken],
) -> list[BoundingBox]:
    """Identify specific page regions that need re-scanning.

    A region needs re-scanning when:
    1. A high-confidence challenge targets tokens in that region
    2. A graph violation references tokens in that region
    3. Multiple issues converge on the same spatial area

    Returns bounding boxes (expanded with padding) for VLM re-scan.
    """
    # Build token lookup
    token_map: dict[str, SpatialToken] = {
        t.token_id: t for t in spatial_tokens
    }
    # Also index by short ID (first 8 chars)
    for t in spatial_tokens:
        token_map[t.token_id[:8]] = t

    conflicting_tokens: list[SpatialToken] = []

    # Collect tokens from high-confidence challenges
    for challenge in challenges:
        if challenge.confidence < 0.6:
            continue
        for tid in challenge.affected_tokens:
            token = token_map.get(tid)
            if token:
                conflicting_tokens.append(token)

    # Collect tokens from graph violations
    for violation in graph_violations:
        if violation.severity < 0.7:
            continue
        for tid in violation.source_tokens:
            token = token_map.get(tid)
            if token:
                conflicting_tokens.append(token)

    # Collect tokens from low-confidence extracted fields
    for field in extraction.fields.values():
        if field.confidence < 0.5 and field.value:
            for tid in field.source_tokens:
                token = token_map.get(tid)
                if token:
                    conflicting_tokens.append(token)

    if not conflicting_tokens:
        return []

    # Merge nearby conflicting tokens into regions
    regions = _merge_into_regions(conflicting_tokens, padding=50)

    logger.info(
        "conflicts_detected",
        conflicting_tokens=len(conflicting_tokens),
        merged_regions=len(regions),
    )
    return regions


def _merge_into_regions(
    tokens: list[SpatialToken],
    padding: float = 50,
) -> list[BoundingBox]:
    """Merge nearby conflicting tokens into larger scan regions.

    Adds padding around each region for VLM context.
    """
    if not tokens:
        return []

    # Group by page
    pages: dict[int, list[SpatialToken]] = {}
    for t in tokens:
        pages.setdefault(t.bbox.page_number, []).append(t)

    regions: list[BoundingBox] = []

    for page_num, page_tokens in pages.items():
        # Sort by position
        sorted_tokens = sorted(page_tokens, key=lambda t: (t.bbox.y_min, t.bbox.x_min))

        # Greedy merge: expand region to include nearby tokens
        current_region: BoundingBox | None = None

        for token in sorted_tokens:
            if current_region is None:
                current_region = BoundingBox(
                    x_min=token.bbox.x_min - padding,
                    y_min=token.bbox.y_min - padding,
                    x_max=token.bbox.x_max + padding,
                    y_max=token.bbox.y_max + padding,
                    page_number=page_num,
                )
            else:
                # Check if token is close enough to merge
                expanded = BoundingBox(
                    x_min=token.bbox.x_min - padding,
                    y_min=token.bbox.y_min - padding,
                    x_max=token.bbox.x_max + padding,
                    y_max=token.bbox.y_max + padding,
                    page_number=page_num,
                )

                if _regions_overlap(current_region, expanded):
                    # Merge
                    current_region = BoundingBox(
                        x_min=min(current_region.x_min, expanded.x_min),
                        y_min=min(current_region.y_min, expanded.y_min),
                        x_max=max(current_region.x_max, expanded.x_max),
                        y_max=max(current_region.y_max, expanded.y_max),
                        page_number=page_num,
                    )
                else:
                    regions.append(current_region)
                    current_region = expanded

        if current_region:
            regions.append(current_region)

    # Clamp coordinates to non-negative
    for region in regions:
        region.x_min = max(0, region.x_min)
        region.y_min = max(0, region.y_min)

    return regions


def _regions_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    """Check if two bounding boxes overlap."""
    return not (
        a.x_max < b.x_min
        or b.x_max < a.x_min
        or a.y_max < b.y_min
        or b.y_max < a.y_min
    )
