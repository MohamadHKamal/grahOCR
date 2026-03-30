"""Tests for the SpatialToken model — the pipeline's provenance backbone."""

import pytest

from graphocr.models.token import BoundingBox, SpatialToken
from graphocr.core.types import Language

pytestmark = [pytest.mark.unit, pytest.mark.layer1, pytest.mark.smoke]


class TestBoundingBox:
    def test_center(self, sample_bbox):
        assert sample_bbox.center == (55.0, 35.0)

    def test_dimensions(self, sample_bbox):
        assert sample_bbox.width == 90
        assert sample_bbox.height == 30

    def test_area(self, sample_bbox):
        assert sample_bbox.area == 2700

    def test_iou_identical(self, sample_bbox):
        assert sample_bbox.iou(sample_bbox) == 1.0

    def test_iou_no_overlap(self):
        a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, page_number=1)
        b = BoundingBox(x_min=20, y_min=20, x_max=30, y_max=30, page_number=1)
        assert a.iou(b) == 0.0

    def test_iou_partial_overlap(self):
        a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, page_number=1)
        b = BoundingBox(x_min=5, y_min=5, x_max=15, y_max=15, page_number=1)
        iou = a.iou(b)
        assert 0.1 < iou < 0.3  # 25 / (100+100-25) ≈ 0.143

    def test_iou_different_pages(self):
        a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, page_number=1)
        b = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, page_number=2)
        assert a.iou(b) == 0.0


class TestSpatialToken:
    def test_creation(self, sample_tokens):
        assert len(sample_tokens) == 6
        assert sample_tokens[0].text == "Patient Name:"
        assert sample_tokens[1].language == Language.ARABIC

    def test_provenance_string(self, sample_tokens):
        prov = sample_tokens[0].to_provenance_str()
        assert "Patient Name:" in prov
        assert "page=1" in prov
        assert "paddleocr" in prov

    def test_auto_id(self):
        t = SpatialToken(
            text="test",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, page_number=1),
            reading_order=0,
            confidence=0.9,
            ocr_engine="test",
        )
        assert t.token_id  # Auto-generated UUID
        assert len(t.token_id) > 10
