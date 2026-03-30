"""Tests for the uncertainty-based traffic controller."""

import pytest

from graphocr.core.types import Language, ProcessingPath
from graphocr.layer3_inference.traffic_controller import route_document
from graphocr.models.token import BoundingBox, SpatialToken

pytestmark = [pytest.mark.unit, pytest.mark.layer3, pytest.mark.smoke]


def _make_token(confidence=0.95, lang=Language.ENGLISH, handwritten=False) -> SpatialToken:
    return SpatialToken(
        text="test",
        bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20, page_number=1),
        reading_order=0,
        language=lang,
        confidence=confidence,
        ocr_engine="test",
        is_handwritten=handwritten,
    )


class TestTrafficController:
    def test_high_confidence_goes_cheap(self):
        tokens = [_make_token(0.95) for _ in range(10)]
        decision = route_document(tokens)
        assert decision.path == ProcessingPath.CHEAP_RAIL

    def test_low_confidence_goes_vlm(self):
        tokens = [_make_token(0.3) for _ in range(10)]
        decision = route_document(tokens)
        assert decision.path == ProcessingPath.VLM_CONSENSUS

    def test_handwriting_increases_uncertainty(self):
        clean = [_make_token(0.85) for _ in range(10)]
        handwritten = [_make_token(0.85, handwritten=True) for _ in range(10)]

        clean_decision = route_document(clean)
        hw_decision = route_document(handwritten)

        assert hw_decision.uncertainty_score > clean_decision.uncertainty_score

    def test_empty_tokens_goes_vlm(self):
        decision = route_document([])
        assert decision.path == ProcessingPath.VLM_CONSENSUS

    def test_routing_decision_has_components(self):
        tokens = [_make_token(0.9) for _ in range(10)]
        decision = route_document(tokens)

        assert 0.0 <= decision.uncertainty_score <= 1.0
        assert 0.0 <= decision.confidence_mean <= 1.0
        assert decision.reason
