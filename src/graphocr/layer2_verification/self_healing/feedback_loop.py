"""Feedback loop (Back-Propagation) — patches conflicting tokens with targeted VLM rescan.

When a Logical Impossibility is detected and VLM re-scan produces new tokens,
this module back-propagates the correction, replacing the original conflicting tokens 
and triggering re-extraction of only the affected fields.
"""

from __future__ import annotations

from graphocr.core.logging import get_logger
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


def patch_tokens(
    original_tokens: list[SpatialToken],
    conflicting_regions: list[BoundingBox],
    rescan_tokens: list[SpatialToken],
) -> list[SpatialToken]:
    """Replace tokens in conflicting regions with VLM rescan results.

    Args:
        original_tokens: Full token stream from Layer 1.
        conflicting_regions: Regions that were re-scanned.
        rescan_tokens: New tokens from VLM re-scan.

    Returns:
        Patched token stream with conflicting tokens replaced.
    """
    if not conflicting_regions or not rescan_tokens:
        return original_tokens

    # Find tokens that fall within conflicting regions
    tokens_to_remove: set[str] = set()
    for token in original_tokens:
        for region in conflicting_regions:
            if _token_in_region(token, region):
                tokens_to_remove.add(token.token_id)
                break

    # Build patched list: keep non-conflicting + add rescan results
    patched = [t for t in original_tokens if t.token_id not in tokens_to_remove]
    patched.extend(rescan_tokens)

    # Re-sort by page and position
    patched.sort(key=lambda t: (t.bbox.page_number, t.bbox.center[1], t.bbox.center[0]))

    # Re-assign reading order
    for idx, token in enumerate(patched):
        token.reading_order = idx

    logger.info(
        "back_propagation_patched",
        removed=len(tokens_to_remove),
        added=len(rescan_tokens),
        total=len(patched),
    )
    return patched


def identify_affected_fields(
    conflicting_regions: list[BoundingBox],
    field_token_map: dict[str, list[str]],
    token_map: dict[str, SpatialToken],
) -> list[str]:
    """Identify which extracted fields are affected by the conflicting regions.

    Args:
        conflicting_regions: Regions that were re-scanned.
        field_token_map: Map of field_name -> list of token_ids.
        token_map: Map of token_id -> SpatialToken.

    Returns:
        List of field names that need re-extraction.
    """
    affected_fields: list[str] = []

    for field_name, token_ids in field_token_map.items():
        for tid in token_ids:
            token = token_map.get(tid)
            if not token:
                continue
            for region in conflicting_regions:
                if _token_in_region(token, region):
                    affected_fields.append(field_name)
                    break
            else:
                continue
            break

    logger.info("affected_fields_identified", fields=affected_fields)
    return affected_fields


def _token_in_region(token: SpatialToken, region: BoundingBox) -> bool:
    """Check if a token's center falls within a region."""
    if token.bbox.page_number != region.page_number:
        return False
    cx, cy = token.bbox.center
    return (
        region.x_min <= cx <= region.x_max
        and region.y_min <= cy <= region.y_max
    )
