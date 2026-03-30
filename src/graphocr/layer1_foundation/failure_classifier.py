"""Failure classifier — distinguishes Type A (spatial-blind OCR) from Type B (context-blind RAG).

Type A: OCR read text but lost the spatial layout (wrong reading order, merged regions).
Type B: OCR extracted correctly but RAG retrieved wrong context (wrong policy, wrong dates).
"""

from __future__ import annotations

import re

from graphocr.core.logging import get_logger
from graphocr.core.types import FailureType
from graphocr.models.failure import FailureClassification
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)


def classify_failures(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Analyze a token stream and classify any detected failures.

    Returns a list of failure classifications. An empty list means
    no failures were detected.
    """
    failures: list[FailureClassification] = []

    # Check for Type A failures
    type_a = _detect_spatial_blind_failures(tokens)
    failures.extend(type_a)

    # Type B failures are detected later by the knowledge graph (Layer 2)
    # and by comparing extracted data against policy context.

    if failures:
        logger.warning(
            "failures_detected",
            count=len(failures),
            types=[f.failure_type.value for f in failures],
        )

    return failures


def _detect_spatial_blind_failures(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Detect Type A (spatial-blind) failures in the token stream.

    Indicators:
    - Tokens with high OCR confidence but nonsensical spatial arrangement
    - Reading order jumps (tokens far apart spatially but adjacent in reading order)
    - Interleaved languages within what should be a single field
    - Merged data from different regions (e.g., dates mixed with amounts)
    - Stamp/seal overlapping body text (obscured fields)
    - Cross-column merge errors (two columns incorrectly serialized as one)
    """
    failures: list[FailureClassification] = []

    # Check 1: Reading order spatial jumps
    jump_failures = _check_reading_order_jumps(tokens)
    failures.extend(jump_failures)

    # Check 2: Nonsensical token sequences (numbers mixed with text in wrong order)
    sequence_failures = _check_nonsensical_sequences(tokens)
    failures.extend(sequence_failures)

    # Check 3: Stamp/seal overlapping body text
    stamp_failures = _check_stamp_overlap(tokens)
    failures.extend(stamp_failures)

    # Check 4: Cross-column merge errors
    column_failures = _check_cross_column_merge(tokens)
    failures.extend(column_failures)

    return failures


def _check_reading_order_jumps(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Detect large spatial jumps between consecutively-ordered tokens."""
    failures: list[FailureClassification] = []
    sorted_by_order = sorted(tokens, key=lambda t: t.reading_order)

    for i in range(1, len(sorted_by_order)):
        prev = sorted_by_order[i - 1]
        curr = sorted_by_order[i]

        if prev.bbox.page_number != curr.bbox.page_number:
            continue

        # Spatial distance between centers
        dx = abs(curr.bbox.center[0] - prev.bbox.center[0])
        dy = abs(curr.bbox.center[1] - prev.bbox.center[1])
        distance = (dx ** 2 + dy ** 2) ** 0.5

        # A jump of >800px between consecutive reading-order tokens is suspicious
        # if both have high confidence (the OCR read them fine, just in wrong order).
        # 800px threshold tuned for 2500px-wide images — handwriting can wander ~500px.
        if distance > 800 and min(prev.confidence, curr.confidence) > 0.7:
            failures.append(
                FailureClassification(
                    failure_type=FailureType.TYPE_A_SPATIAL_BLIND,
                    affected_tokens=[prev.token_id, curr.token_id],
                    severity=min(1.0, distance / 1000),
                    suggested_remedy="vlm_rescan",
                    evidence=(
                        f"Reading order jump: tokens {prev.reading_order}->{curr.reading_order} "
                        f"are {distance:.0f}px apart spatially"
                    ),
                )
            )

    return failures


def _check_nonsensical_sequences(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Detect tokens that appear interleaved from different document regions."""
    failures: list[FailureClassification] = []
    sorted_by_order = sorted(tokens, key=lambda t: t.reading_order)

    # Sliding window: check for rapid alternation between number-heavy and text-heavy tokens
    window_size = 5
    for i in range(len(sorted_by_order) - window_size):
        window = sorted_by_order[i : i + window_size]
        types = [_token_type(t) for t in window]

        # Rapid alternation (e.g., [num, text, num, text, num]) is suspicious
        alternations = sum(1 for j in range(1, len(types)) if types[j] != types[j - 1])
        if alternations >= 4:  # Every token is different from its neighbor
            failures.append(
                FailureClassification(
                    failure_type=FailureType.TYPE_A_SPATIAL_BLIND,
                    affected_tokens=[t.token_id for t in window],
                    severity=0.6,
                    suggested_remedy="vlm_rescan",
                    evidence=f"Rapidly alternating token types in reading order: {types}",
                )
            )

    return failures


def _check_stamp_overlap(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Detect stamp/seal zones overlapping body text.

    When a pharmacy stamp or official seal overlaps a data field (policy number,
    amount, date), the OCR may read the stamp text mixed with the field text.
    This detects bounding box overlap between tokens in different zones.
    """
    from graphocr.core.types import ZoneLabel

    failures: list[FailureClassification] = []

    # Separate tokens by zone type
    stamp_tokens = [t for t in tokens if t.zone_label in (ZoneLabel.STAMP, ZoneLabel.LOGO, ZoneLabel.SIGNATURE)]
    body_tokens = [t for t in tokens if t.zone_label in (ZoneLabel.BODY, ZoneLabel.HEADER, ZoneLabel.TABLE_CELL) or t.zone_label is None]

    if not stamp_tokens or not body_tokens:
        return failures

    for stamp in stamp_tokens:
        overlapping_body: list[SpatialToken] = []
        for body in body_tokens:
            iou = stamp.bbox.iou(body.bbox)
            if iou > 0.05:  # Even slight overlap is concerning
                overlapping_body.append(body)

        if overlapping_body:
            all_affected = [stamp.token_id] + [b.token_id for b in overlapping_body]
            # Severity scales with number of overlapping body tokens and overlap degree
            avg_confidence = sum(b.confidence for b in overlapping_body) / len(overlapping_body)
            failures.append(
                FailureClassification(
                    failure_type=FailureType.TYPE_A_SPATIAL_BLIND,
                    affected_tokens=all_affected,
                    severity=min(1.0, 0.5 + 0.1 * len(overlapping_body)),
                    suggested_remedy="vlm_rescan",
                    evidence=(
                        f"Stamp/seal zone overlaps {len(overlapping_body)} body token(s). "
                        f"Stamp: '{stamp.text[:30]}' at page {stamp.bbox.page_number} "
                        f"({stamp.bbox.x_min:.0f},{stamp.bbox.y_min:.0f})-"
                        f"({stamp.bbox.x_max:.0f},{stamp.bbox.y_max:.0f}). "
                        f"Overlapping body avg confidence: {avg_confidence:.2f}"
                    ),
                )
            )

    return failures


def _check_cross_column_merge(tokens: list[SpatialToken]) -> list[FailureClassification]:
    """Detect cross-column merge errors where the XY-Cut algorithm
    incorrectly serialized two columns as a single reading flow.

    Indicators:
    - Consecutive tokens in reading order have similar Y but large X gap
      (they're on the same line but from different columns)
    - Tokens alternate between two distinct X-ranges (column A, column B)
    """
    failures: list[FailureClassification] = []
    sorted_by_order = sorted(tokens, key=lambda t: t.reading_order)

    # Group tokens by page
    pages: dict[int, list[SpatialToken]] = {}
    for t in sorted_by_order:
        pages.setdefault(t.bbox.page_number, []).append(t)

    for page_num, page_tokens in pages.items():
        if len(page_tokens) < 10:  # Need enough tokens to detect columns
            continue

        # Detect two-column layout: cluster X-centers into 2 groups
        x_centers = [t.bbox.center[0] for t in page_tokens]
        x_min, x_max = min(x_centers), max(x_centers)
        x_range = x_max - x_min

        # Page must be wide enough that the X-spread isn't just handwriting wander.
        # Use 40% of page width as minimum gap — true columns have a clear divide.
        page_widths = [t.bbox.x_max for t in page_tokens]
        page_width = max(page_widths) if page_widths else 1000
        min_column_gap = page_width * 0.4

        if x_range < min_column_gap:
            continue

        x_mid = (x_min + x_max) / 2
        # Use a wider dead zone (15% of page) to avoid classifying
        # centered handwriting as "switching columns"
        dead_zone = page_width * 0.15
        left_tokens = [t for t in page_tokens if t.bbox.center[0] < x_mid - dead_zone]
        right_tokens = [t for t in page_tokens if t.bbox.center[0] > x_mid + dead_zone]

        # Both columns must have substantial content (at least 20% of tokens each)
        min_column_tokens = max(4, len(page_tokens) // 5)
        if len(left_tokens) < min_column_tokens or len(right_tokens) < min_column_tokens:
            continue

        # Check for interleaving: in the reading order, tokens should NOT
        # alternate rapidly between left and right columns
        column_sequence = []
        for t in page_tokens:
            if t.bbox.center[0] < x_mid - dead_zone:
                column_sequence.append("L")
            elif t.bbox.center[0] > x_mid + dead_zone:
                column_sequence.append("R")
            else:
                column_sequence.append("M")  # Middle / spanning

        # Count column switches in reading order (ignoring M transitions)
        switches = 0
        for i in range(1, len(column_sequence)):
            if column_sequence[i] != column_sequence[i - 1] and "M" not in (column_sequence[i], column_sequence[i - 1]):
                switches += 1

        # Threshold scales with token count — more tokens = more expected switches
        # For true two-column, switches should be <= 2 (all left then all right).
        # Allow up to 1 switch per 10 tokens as tolerance for noisy layouts.
        expected_max_switches = max(3, len(page_tokens) // 10)
        if switches > expected_max_switches:
            # Find the specific tokens where column switches happen
            switch_tokens = []
            for i in range(1, len(column_sequence)):
                if column_sequence[i] != column_sequence[i - 1] and "M" not in (column_sequence[i], column_sequence[i - 1]):
                    switch_tokens.append(page_tokens[i - 1].token_id)
                    switch_tokens.append(page_tokens[i].token_id)

            failures.append(
                FailureClassification(
                    failure_type=FailureType.TYPE_A_SPATIAL_BLIND,
                    affected_tokens=list(set(switch_tokens)),
                    severity=min(1.0, 0.4 + 0.05 * switches),
                    suggested_remedy="vlm_rescan",
                    evidence=(
                        f"Cross-column merge detected on page {page_num}: "
                        f"{switches} column switches in reading order "
                        f"(expected <= {expected_max_switches}). "
                        f"Left tokens: {len(left_tokens)}, Right tokens: {len(right_tokens)}"
                    ),
                )
            )

    return failures


def _token_type(token: SpatialToken) -> str:
    """Classify a token as 'num', 'text', or 'mixed'."""
    text = token.text.strip()
    if not text:
        return "empty"
    if re.match(r"^[\d.,/%$€£¥]+$", text):
        return "num"
    if re.match(r"^[a-zA-Z\u0600-\u06FF\s]+$", text):
        return "text"
    return "mixed"
