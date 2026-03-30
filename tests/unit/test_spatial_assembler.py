"""Tests for the spatial assembler — multi-engine token merging."""

import pytest

from graphocr.core.types import Language, ZoneLabel
from graphocr.layer1_foundation.spatial_assembler import assemble_tokens, group_into_lines
from graphocr.models.token import BoundingBox, SpatialToken

pytestmark = [pytest.mark.unit, pytest.mark.layer1]


def _make_token(x, y, text, confidence=0.9, engine="paddleocr", zone=None, lang=Language.UNKNOWN) -> SpatialToken:
    return SpatialToken(
        text=text,
        bbox=BoundingBox(x_min=x, y_min=y, x_max=x + 50, y_max=y + 20, page_number=1),
        reading_order=0,
        confidence=confidence,
        ocr_engine=engine,
        zone_label=zone,
        language=lang,
    )


class TestAssembleTokens:
    def test_single_stream(self):
        tokens = [_make_token(0, 0, "hello")]
        result = assemble_tokens([tokens])
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_no_overlap_adds_both(self):
        stream1 = [_make_token(0, 0, "hello")]
        stream2 = [_make_token(200, 0, "world")]
        result = assemble_tokens([stream1, stream2])
        assert len(result) == 2

    def test_overlap_merges_takes_higher_confidence_text(self):
        """When overlapping, text comes from the higher-confidence engine."""
        stream1 = [_make_token(0, 0, "hello", confidence=0.7, engine="engine1")]
        stream2 = [_make_token(0, 0, "world", confidence=0.95, engine="engine2")]
        result = assemble_tokens([stream1, stream2])
        assert len(result) == 1
        assert result[0].text == "world"  # Higher confidence text

    def test_overlap_boosts_confidence(self):
        """When both engines detect same region, confidence is boosted."""
        stream1 = [_make_token(0, 0, "hello", confidence=0.7, engine="engine1")]
        stream2 = [_make_token(0, 0, "hello", confidence=0.8, engine="engine2")]
        result = assemble_tokens([stream1, stream2])
        # Combined: 1 - (1-0.7)*(1-0.8) = 1 - 0.06 = 0.94
        assert result[0].confidence > 0.9

    def test_overlap_inherits_zone_label(self):
        """Zone label from Surya is inherited when PaddleOCR has none."""
        stream1 = [_make_token(0, 0, "Patient", confidence=0.9, engine="paddleocr")]
        stream2 = [_make_token(0, 0, "", confidence=0.8, engine="surya", zone=ZoneLabel.HEADER)]
        result = assemble_tokens([stream1, stream2])
        assert result[0].zone_label == ZoneLabel.HEADER
        assert result[0].text == "Patient"  # Text from PaddleOCR

    def test_overlap_inherits_language(self):
        """Language from secondary is inherited when primary is UNKNOWN."""
        stream1 = [_make_token(0, 0, "محمد", confidence=0.9, engine="engine1")]
        stream2 = [_make_token(0, 0, "محمد", confidence=0.8, engine="engine2", lang=Language.ARABIC)]
        result = assemble_tokens([stream1, stream2])
        assert result[0].language == Language.ARABIC

    def test_overlap_records_both_engines(self):
        """Provenance records both engines."""
        stream1 = [_make_token(0, 0, "text", confidence=0.8, engine="paddleocr")]
        stream2 = [_make_token(0, 0, "text", confidence=0.7, engine="surya")]
        result = assemble_tokens([stream1, stream2])
        assert "paddleocr" in result[0].ocr_engine
        assert "surya" in result[0].ocr_engine

    def test_empty_streams(self):
        assert assemble_tokens([]) == []
        assert assemble_tokens([[]]) == []

    def test_empty_text_secondary_not_added(self):
        """Secondary tokens with no text are not added as new detections."""
        stream1 = [_make_token(0, 0, "hello")]
        stream2 = [_make_token(200, 0, "")]  # Empty text, no overlap
        result = assemble_tokens([stream1, stream2])
        assert len(result) == 1  # Only the primary token


class TestGroupIntoLines:
    def test_single_line(self):
        tokens = [
            _make_token(0, 100, "a"),
            _make_token(60, 100, "b"),
            _make_token(120, 100, "c"),
        ]
        lines = group_into_lines(tokens, y_tolerance=15)
        assert len(lines) == 1
        assert len(lines[0]) == 3

    def test_two_lines(self):
        tokens = [
            _make_token(0, 100, "line1"),
            _make_token(0, 200, "line2"),
        ]
        lines = group_into_lines(tokens, y_tolerance=15)
        assert len(lines) == 2

    def test_assigns_line_group_id(self):
        tokens = [
            _make_token(0, 100, "a"),
            _make_token(0, 200, "b"),
        ]
        lines = group_into_lines(tokens, y_tolerance=15)
        assert tokens[0].line_group_id is not None
        assert tokens[0].line_group_id != tokens[1].line_group_id
