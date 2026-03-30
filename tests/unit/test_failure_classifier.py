"""Tests for the failure classifier."""

import pytest

from graphocr.core.types import FailureType, Language, ZoneLabel
from graphocr.layer1_foundation.failure_classifier import classify_failures
from graphocr.models.token import BoundingBox, SpatialToken

pytestmark = [pytest.mark.unit, pytest.mark.layer1]


def _make_token(x, y, text, order, confidence=0.9, zone=None, page=1) -> SpatialToken:
    return SpatialToken(
        text=text,
        bbox=BoundingBox(x_min=x, y_min=y, x_max=x + 50, y_max=y + 20, page_number=page),
        reading_order=order,
        confidence=confidence,
        ocr_engine="test",
        zone_label=zone,
    )


class TestFailureClassifier:
    def test_no_failures_for_clean_tokens(self):
        tokens = [
            _make_token(0, 0, "a", 0),
            _make_token(60, 0, "b", 1),
            _make_token(120, 0, "c", 2),
        ]
        failures = classify_failures(tokens)
        assert len(failures) == 0

    def test_detects_spatial_jump(self):
        """Tokens that are far apart spatially but consecutive in reading order."""
        tokens = [
            _make_token(0, 0, "a", 0, confidence=0.9),
            _make_token(800, 800, "b", 1, confidence=0.9),  # Huge jump
        ]
        failures = classify_failures(tokens)
        spatial_failures = [f for f in failures if f.failure_type == FailureType.TYPE_A_SPATIAL_BLIND]
        assert len(spatial_failures) > 0

    def test_empty_tokens(self):
        failures = classify_failures([])
        assert len(failures) == 0


class TestStampOverlapDetection:
    def test_detects_stamp_over_body(self):
        """A stamp zone overlapping body text should be flagged."""
        tokens = [
            # Body token
            _make_token(100, 100, "Policy: SA-2018-001", 0, zone=ZoneLabel.BODY),
            # Stamp overlapping the body token
            SpatialToken(
                text="APPROVED",
                bbox=BoundingBox(x_min=90, y_min=95, x_max=160, y_max=125, page_number=1),
                reading_order=1,
                confidence=0.7,
                ocr_engine="test",
                zone_label=ZoneLabel.STAMP,
            ),
        ]
        failures = classify_failures(tokens)
        stamp_failures = [f for f in failures if "tamp" in f.evidence.lower() or "seal" in f.evidence.lower()]
        assert len(stamp_failures) > 0

    def test_no_stamp_overlap_when_separate(self):
        """Non-overlapping stamp and body should not be flagged."""
        tokens = [
            _make_token(100, 100, "Policy: SA-2018-001", 0, zone=ZoneLabel.BODY),
            _make_token(500, 500, "APPROVED", 1, zone=ZoneLabel.STAMP),
        ]
        failures = classify_failures(tokens)
        stamp_failures = [f for f in failures if "tamp" in f.evidence.lower()]
        assert len(stamp_failures) == 0

    def test_no_false_positive_without_zones(self):
        """Tokens without zone labels should not trigger stamp detection."""
        tokens = [
            _make_token(100, 100, "text1", 0),
            _make_token(110, 100, "text2", 1),
        ]
        failures = classify_failures(tokens)
        stamp_failures = [f for f in failures if "tamp" in f.evidence.lower()]
        assert len(stamp_failures) == 0


class TestCrossColumnMergeDetection:
    def test_detects_interleaved_columns(self):
        """Tokens alternating between left and right columns in reading order.
        Needs 10+ tokens, wide X-spread (>40% of page), and 20%+ tokens per side."""
        tokens = [
            # Simulates a 1000px-wide page with two clear columns
            # Interleaved: L, R, L, R, L, R, L, R, L, R, L, R
            _make_token(50,  100, "left1", 0),
            _make_token(800, 100, "right1", 1),
            _make_token(50,  150, "left2", 2),
            _make_token(800, 150, "right2", 3),
            _make_token(50,  200, "left3", 4),
            _make_token(800, 200, "right3", 5),
            _make_token(50,  250, "left4", 6),
            _make_token(800, 250, "right4", 7),
            _make_token(50,  300, "left5", 8),
            _make_token(800, 300, "right5", 9),
            _make_token(50,  350, "left6", 10),
            _make_token(800, 350, "right6", 11),
        ]
        failures = classify_failures(tokens)
        column_failures = [f for f in failures if "column" in f.evidence.lower()]
        assert len(column_failures) > 0

    def test_no_column_error_for_single_column(self):
        """All tokens in one column should not trigger column detection."""
        tokens = [
            _make_token(50, 100, "a", 0),
            _make_token(50, 150, "b", 1),
            _make_token(50, 200, "c", 2),
            _make_token(50, 250, "d", 3),
        ]
        failures = classify_failures(tokens)
        column_failures = [f for f in failures if "column" in f.evidence.lower()]
        assert len(column_failures) == 0

    def test_correct_two_column_reading(self):
        """All left then all right is correct reading order — no failure."""
        tokens = [
            # All left first, then all right
            _make_token(50, 100, "left1", 0),
            _make_token(50, 150, "left2", 1),
            _make_token(50, 200, "left3", 2),
            _make_token(500, 100, "right1", 3),
            _make_token(500, 150, "right2", 4),
            _make_token(500, 200, "right3", 5),
        ]
        failures = classify_failures(tokens)
        column_failures = [f for f in failures if "column" in f.evidence.lower()]
        assert len(column_failures) == 0
