"""Spatial assembler — merges outputs from multiple OCR engines.

When two engines detect the same region (IoU > threshold), instead of
discarding one, we MERGE them:
  - Text: from the higher-confidence engine
  - Bounding box: union of both (tighter coverage)
  - Confidence: boosted when both engines agree
  - Zone label: inherited from Surya (layout-aware)
  - Language/handwriting: inherited from whichever detected it
"""

from __future__ import annotations

from graphocr.core.logging import get_logger
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


def assemble_tokens(
    token_streams: list[list[SpatialToken]],
    iou_threshold: float = 0.3,
) -> list[SpatialToken]:
    """Merge token streams from multiple OCR engines.

    When overlapping (IoU > threshold): MERGE both tokens into one,
    combining the best attributes from each engine.
    When no overlap: ADD as a new detection.

    Args:
        token_streams: Lists of tokens from different engines.
        iou_threshold: IoU threshold for merge. 0.3 is more permissive
            than 0.5 — catches partial overlaps from different engines
            that detected the same text region slightly differently.

    Returns:
        Merged list of SpatialTokens with combined metadata.
    """
    if not token_streams:
        return []

    if len(token_streams) == 1:
        return token_streams[0]

    # Start with the primary engine's tokens
    merged = list(token_streams[0])

    # Merge tokens from secondary engines
    for secondary_tokens in token_streams[1:]:
        for sec_token in secondary_tokens:
            best_match, best_iou = _find_best_overlap(sec_token, merged, iou_threshold)

            if best_match is None:
                # No overlap — new detection, add it
                if sec_token.text:  # Only add if it has text
                    merged.append(sec_token)
            else:
                # Overlapping region — merge the two tokens
                _merge_tokens(best_match, sec_token, best_iou)

    logger.info(
        "spatial_assembly_complete",
        input_streams=len(token_streams),
        input_total=sum(len(s) for s in token_streams),
        output_total=len(merged),
    )
    return merged


def _find_best_overlap(
    token: SpatialToken,
    candidates: list[SpatialToken],
    iou_threshold: float,
) -> tuple[SpatialToken | None, float]:
    """Find the candidate with the highest IoU overlap."""
    best_iou = 0.0
    best_match = None

    for candidate in candidates:
        iou = token.bbox.iou(candidate.bbox)
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_match = candidate

    return best_match, best_iou


def _merge_tokens(primary: SpatialToken, secondary: SpatialToken, iou: float) -> None:
    """Merge secondary token INTO primary, combining the best of both.

    Strategy:
      - Text: from whichever engine has higher confidence
      - Bbox: union (encompassing box) for better spatial coverage,
              or intersection (tighter box) when IoU is very high
      - Confidence: boosted by agreement — if two engines see the same
              region, we're more confident it's real
      - Zone label: prefer non-None (Surya provides this, PaddleOCR doesn't)
      - Language: prefer non-UNKNOWN
      - Handwriting: OR (if either detects it, flag it)
      - Engine: record both engines for provenance
    """
    # Text: keep the higher-confidence reading
    if secondary.text and secondary.confidence > primary.confidence:
        primary.text = secondary.text
    elif secondary.text and not primary.text:
        primary.text = secondary.text

    # Bounding box: merge ONLY if secondary has text 
    # (Layout-only boxes like Surya can deform precise text-line boxes)
    if secondary.text:
        if iou > 0.7:
            # High overlap — use intersection (tighter, more precise)
            primary.bbox = BoundingBox(
                x_min=max(primary.bbox.x_min, secondary.bbox.x_min),
                y_min=max(primary.bbox.y_min, secondary.bbox.y_min),
                x_max=min(primary.bbox.x_max, secondary.bbox.x_max),
                y_max=min(primary.bbox.y_max, secondary.bbox.y_max),
                page_number=primary.bbox.page_number,
            )
        else:
            # Moderate overlap — use union (encompassing, don't lose edges)
            primary.bbox = BoundingBox(
                x_min=min(primary.bbox.x_min, secondary.bbox.x_min),
                y_min=min(primary.bbox.y_min, secondary.bbox.y_min),
                x_max=max(primary.bbox.x_max, secondary.bbox.x_max),
                y_max=max(primary.bbox.y_max, secondary.bbox.y_max),
                page_number=primary.bbox.page_number,
            )

    # Confidence: boost when both engines agree on the region
    # Two independent detectors agreeing = higher confidence
    # Formula: 1 - (1 - conf_a) * (1 - conf_b)  (probability of at least one being right)
    combined = 1.0 - (1.0 - primary.confidence) * (1.0 - secondary.confidence)
    primary.confidence = min(combined, 0.99)  # Cap at 0.99

    # Zone label: prefer non-None (Surya detects zones, PaddleOCR usually doesn't)
    if secondary.zone_label and not primary.zone_label:
        primary.zone_label = secondary.zone_label

    # Language: prefer specific over UNKNOWN
    from graphocr.core.types import Language
    if primary.language == Language.UNKNOWN and secondary.language != Language.UNKNOWN:
        primary.language = secondary.language

    # Handwriting: if either engine detects it, flag it
    if secondary.is_handwritten:
        primary.is_handwritten = True

    # Normalized text: prefer non-None
    if secondary.normalized_text and not primary.normalized_text:
        primary.normalized_text = secondary.normalized_text

    # Provenance: record both engines
    if secondary.ocr_engine not in primary.ocr_engine:
        primary.ocr_engine = f"{primary.ocr_engine}+{secondary.ocr_engine}"


def group_into_lines(
    tokens: list[SpatialToken],
    y_tolerance: float = 10.0,
) -> list[list[SpatialToken]]:
    """Group tokens into logical lines based on vertical proximity.

    Args:
        tokens: Ordered tokens.
        y_tolerance: Max vertical distance to consider same line.

    Returns:
        List of lines, each a list of tokens.
    """
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.center[1], t.bbox.center[0]))
    lines: list[list[SpatialToken]] = [[sorted_tokens[0]]]
    line_id = 0

    sorted_tokens[0].line_group_id = f"line_{line_id}"

    for token in sorted_tokens[1:]:
        current_line_y = sum(t.bbox.center[1] for t in lines[-1]) / len(lines[-1])

        if abs(token.bbox.center[1] - current_line_y) <= y_tolerance:
            lines[-1].append(token)
        else:
            line_id += 1
            lines.append([token])

        token.line_group_id = f"line_{line_id}"

    return lines
