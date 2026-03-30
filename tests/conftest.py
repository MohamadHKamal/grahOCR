"""Shared test fixtures."""

import pytest

from graphocr.core.types import Language, ZoneLabel
from graphocr.models.token import BoundingBox, SpatialToken


@pytest.fixture
def sample_tokens() -> list[SpatialToken]:
    """Sample spatial tokens for testing."""
    return [
        SpatialToken(
            token_id="tok_001",
            text="Patient Name:",
            bbox=BoundingBox(x_min=50, y_min=100, x_max=200, y_max=130, page_number=1),
            reading_order=0,
            language=Language.ENGLISH,
            confidence=0.95,
            ocr_engine="paddleocr",
            zone_label=ZoneLabel.HEADER,
        ),
        SpatialToken(
            token_id="tok_002",
            text="محمد أحمد",
            bbox=BoundingBox(x_min=210, y_min=100, x_max=350, y_max=130, page_number=1),
            reading_order=1,
            language=Language.ARABIC,
            confidence=0.88,
            ocr_engine="paddleocr",
            zone_label=ZoneLabel.BODY,
        ),
        SpatialToken(
            token_id="tok_003",
            text="Date:",
            bbox=BoundingBox(x_min=50, y_min=150, x_max=120, y_max=180, page_number=1),
            reading_order=2,
            language=Language.ENGLISH,
            confidence=0.97,
            ocr_engine="paddleocr",
        ),
        SpatialToken(
            token_id="tok_004",
            text="2026-03-15",
            bbox=BoundingBox(x_min=130, y_min=150, x_max=280, y_max=180, page_number=1),
            reading_order=3,
            language=Language.ENGLISH,
            confidence=0.92,
            ocr_engine="paddleocr",
        ),
        SpatialToken(
            token_id="tok_005",
            text="Total: 1,500.00 SAR",
            bbox=BoundingBox(x_min=50, y_min=500, x_max=300, y_max=530, page_number=1),
            reading_order=4,
            language=Language.ENGLISH,
            confidence=0.90,
            ocr_engine="paddleocr",
        ),
        SpatialToken(
            token_id="tok_006",
            text="E11.9",
            bbox=BoundingBox(x_min=50, y_min=250, x_max=130, y_max=280, page_number=1),
            reading_order=5,
            language=Language.ENGLISH,
            confidence=0.93,
            ocr_engine="paddleocr",
        ),
    ]


@pytest.fixture
def sample_bbox() -> BoundingBox:
    return BoundingBox(x_min=10, y_min=20, x_max=100, y_max=50, page_number=1)
