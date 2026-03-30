"""Tests for the XY-Cut reading order algorithm."""

import pytest

from graphocr.core.types import Language
from graphocr.layer1_foundation.reading_order import (
    assign_reading_order,
    _has_arabic_script,
)
from graphocr.models.token import BoundingBox, SpatialToken

pytestmark = [pytest.mark.unit, pytest.mark.layer1]


def _make_token(x, y, text="test", lang=Language.ENGLISH, page=1) -> SpatialToken:
    return SpatialToken(
        text=text,
        bbox=BoundingBox(x_min=x, y_min=y, x_max=x + 50, y_max=y + 20, page_number=page),
        reading_order=0,
        language=lang,
        confidence=0.9,
        ocr_engine="test",
    )


class TestReadingOrder:
    def test_single_token(self):
        tokens = [_make_token(0, 0)]
        result = assign_reading_order(tokens)
        assert result[0].reading_order == 0

    def test_top_to_bottom(self):
        tokens = [_make_token(0, 100, "bottom"), _make_token(0, 0, "top")]
        result = assign_reading_order(tokens)
        assert result[0].text == "top"
        assert result[1].text == "bottom"

    def test_left_to_right_same_line(self):
        tokens = [_make_token(200, 0, "right"), _make_token(0, 0, "left")]
        result = assign_reading_order(tokens)
        # Should be left then right (LTR)
        texts = [t.text for t in result]
        assert texts[0] == "left"
        assert texts[1] == "right"

    def test_multi_page_ordering(self):
        tokens = [
            _make_token(0, 0, "page2_top", page=2),
            _make_token(0, 0, "page1_top", page=1),
        ]
        result = assign_reading_order(tokens)
        assert result[0].text == "page1_top"
        assert result[1].text == "page2_top"
        assert result[0].reading_order == 0
        assert result[1].reading_order == 1

    def test_empty_list(self):
        assert assign_reading_order([]) == []


class TestArabicDetection:
    def test_arabic_text(self):
        assert _has_arabic_script("محمد أحمد")

    def test_english_text(self):
        assert not _has_arabic_script("John Smith")

    def test_mixed_text(self):
        assert _has_arabic_script("Patient محمد")

    def test_empty(self):
        assert not _has_arabic_script("")
