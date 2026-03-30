"""Reading order algorithm — XY-Cut with RTL awareness for Arabic.

The XY-Cut algorithm recursively splits the page into columns and rows,
then assigns reading order that respects both LTR English and RTL Arabic.
"""

from __future__ import annotations

import numpy as np

from graphocr.core.types import Language
from graphocr.models.token import SpatialToken


def assign_reading_order(tokens: list[SpatialToken], rtl_detection: bool = True) -> list[SpatialToken]:
    """Assign reading order to tokens using XY-Cut algorithm.

    Args:
        tokens: Tokens with bounding boxes but unordered.
        rtl_detection: Whether to detect and handle RTL (Arabic) zones.

    Returns:
        Same tokens with reading_order field updated.
    """
    if not tokens:
        return tokens

    # Group tokens by page
    pages: dict[int, list[SpatialToken]] = {}
    for t in tokens:
        pages.setdefault(t.bbox.page_number, []).append(t)

    ordered: list[SpatialToken] = []
    global_order = 0

    for page_num in sorted(pages):
        page_tokens = pages[page_num]
        sorted_tokens = _xy_cut_order(page_tokens, rtl_detection=rtl_detection)
        for token in sorted_tokens:
            token.reading_order = global_order
            global_order += 1
            ordered.append(token)

    return ordered


def _xy_cut_order(tokens: list[SpatialToken], rtl_detection: bool = True) -> list[SpatialToken]:
    """Apply XY-Cut recursive decomposition to order tokens on a single page."""
    if len(tokens) <= 1:
        return tokens

    # Try vertical split first (separate columns)
    v_split = _find_vertical_split(tokens)
    if v_split is not None:
        left = [t for t in tokens if t.bbox.center[0] < v_split]
        right = [t for t in tokens if t.bbox.center[0] >= v_split]

        # If RTL detection is on and majority of tokens in a group are Arabic, reverse order
        if rtl_detection and _is_majority_rtl(tokens):
            # RTL: read right column first, then left
            return _xy_cut_order(right, rtl_detection) + _xy_cut_order(left, rtl_detection)
        else:
            # LTR: read left column first, then right
            return _xy_cut_order(left, rtl_detection) + _xy_cut_order(right, rtl_detection)

    # Try horizontal split (separate rows/lines)
    h_split = _find_horizontal_split(tokens)
    if h_split is not None:
        top = [t for t in tokens if t.bbox.center[1] < h_split]
        bottom = [t for t in tokens if t.bbox.center[1] >= h_split]
        return _xy_cut_order(top, rtl_detection) + _xy_cut_order(bottom, rtl_detection)

    # No split found — sort within line
    if rtl_detection and _is_majority_rtl(tokens):
        return sorted(tokens, key=lambda t: (-t.bbox.center[1], -t.bbox.center[0]))
    return sorted(tokens, key=lambda t: (t.bbox.center[1], t.bbox.center[0]))


def _find_vertical_split(tokens: list[SpatialToken], min_gap: float = 30.0) -> float | None:
    """Find a vertical gap (column separator) in the token layout."""
    if len(tokens) < 2:
        return None

    # Project tokens onto X-axis, find largest gap
    x_ranges = sorted([(t.bbox.x_min, t.bbox.x_max) for t in tokens], key=lambda r: r[0])

    best_gap = 0.0
    best_split = None

    for i in range(len(x_ranges) - 1):
        gap = x_ranges[i + 1][0] - x_ranges[i][1]
        if gap > best_gap:
            best_gap = gap
            best_split = (x_ranges[i][1] + x_ranges[i + 1][0]) / 2

    return best_split if best_gap >= min_gap else None


def _find_horizontal_split(tokens: list[SpatialToken], min_gap: float = 10.0) -> float | None:
    """Find a horizontal gap (line separator) in the token layout."""
    if len(tokens) < 2:
        return None

    y_ranges = sorted([(t.bbox.y_min, t.bbox.y_max) for t in tokens], key=lambda r: r[0])

    best_gap = 0.0
    best_split = None

    for i in range(len(y_ranges) - 1):
        gap = y_ranges[i + 1][0] - y_ranges[i][1]
        if gap > best_gap:
            best_gap = gap
            best_split = (y_ranges[i][1] + y_ranges[i + 1][0]) / 2

    return best_split if best_gap >= min_gap else None


def _is_majority_rtl(tokens: list[SpatialToken]) -> bool:
    """Check if the majority of tokens in a group are Arabic (RTL)."""
    if not tokens:
        return False
    arabic_count = sum(1 for t in tokens if t.language == Language.ARABIC)
    # Also check by Unicode script if language not yet assigned
    if arabic_count == 0:
        arabic_count = sum(1 for t in tokens if _has_arabic_script(t.text))
    return arabic_count > len(tokens) / 2


def _has_arabic_script(text: str) -> bool:
    """Quick check if text contains Arabic script characters."""
    for char in text:
        if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" or "\uFB50" <= char <= "\uFDFF" or "\uFE70" <= char <= "\uFEFF":
            return True
    return False
