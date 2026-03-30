"""Per-token language classification: Arabic vs English.

Uses Unicode script detection (fast path) with optional fastText fallback.
"""

from __future__ import annotations

import re

from graphocr.core.logging import get_logger
from graphocr.core.types import Language
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)

# Arabic Unicode ranges
_ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_PATTERN = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> Language:
    """Classify a text string as Arabic, English, Mixed, or Unknown."""
    if not text.strip():
        return Language.UNKNOWN

    arabic_chars = len(_ARABIC_PATTERN.findall(text))
    latin_chars = len(_LATIN_PATTERN.findall(text))
    total = arabic_chars + latin_chars

    if total == 0:
        return Language.UNKNOWN

    arabic_ratio = arabic_chars / total

    if arabic_ratio > 0.7:
        return Language.ARABIC
    elif arabic_ratio < 0.3:
        return Language.ENGLISH
    else:
        return Language.MIXED


def assign_languages(tokens: list[SpatialToken]) -> list[SpatialToken]:
    """Assign language to each token based on its text content.

    Args:
        tokens: Tokens with text but no language assigned.

    Returns:
        Same tokens with language field populated.
    """
    for token in tokens:
        token.language = detect_language(token.text)

    stats = _language_stats(tokens)
    logger.info("language_detection_complete", **stats)
    return tokens


def _language_stats(tokens: list[SpatialToken]) -> dict[str, int]:
    """Count tokens per language."""
    counts: dict[str, int] = {}
    for t in tokens:
        key = t.language.value
        counts[key] = counts.get(key, 0) + 1
    return counts
