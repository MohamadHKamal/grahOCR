"""Tests for the self-healing pipeline: conflict detection + token patching."""

import pytest

from graphocr.core.types import Language

pytestmark = [pytest.mark.unit, pytest.mark.layer2]
from graphocr.layer2_verification.self_healing.conflict_detector import (
    detect_conflicting_regions,
    _regions_overlap,
)
from graphocr.layer2_verification.self_healing.feedback_loop import (
    patch_tokens,
    identify_affected_fields,
    _token_in_region,
)
from graphocr.models.extraction import ExtractionResult, FieldExtraction
from graphocr.models.failure import Challenge, GraphViolation
from graphocr.models.token import BoundingBox, SpatialToken


def _make_token(tid, x, y, text="test", confidence=0.9) -> SpatialToken:
    return SpatialToken(
        token_id=tid,
        text=text,
        bbox=BoundingBox(x_min=x, y_min=y, x_max=x + 50, y_max=y + 20, page_number=1),
        reading_order=0,
        confidence=confidence,
        ocr_engine="test",
    )


class TestConflictDetector:
    def test_detects_high_confidence_challenge_region(self):
        tokens = [_make_token("tok1", 100, 100, "value")]
        extraction = ExtractionResult(
            claim_id="c1", document_id="d1",
            fields={"f1": FieldExtraction(field_name="f1", value="v", confidence=0.9)},
        )
        challenges = [
            Challenge(
                target_field="f1",
                hypothesis="wrong",
                evidence="test",
                confidence=0.8,
                affected_tokens=["tok1"],
            )
        ]
        regions = detect_conflicting_regions(extraction, challenges, [], tokens)
        assert len(regions) >= 1

    def test_no_regions_for_low_confidence_challenges(self):
        tokens = [_make_token("tok1", 100, 100)]
        extraction = ExtractionResult(claim_id="c1", document_id="d1")
        challenges = [
            Challenge(
                target_field="f1", hypothesis="maybe", evidence="weak",
                confidence=0.3, affected_tokens=["tok1"],
            )
        ]
        regions = detect_conflicting_regions(extraction, challenges, [], tokens)
        assert len(regions) == 0

    def test_detects_graph_violation_region(self):
        tokens = [_make_token("tok2", 200, 200, "3000mg")]
        extraction = ExtractionResult(claim_id="c1", document_id="d1")
        violations = [
            GraphViolation(
                rule_name="dosage", field_name="meds",
                extracted_value="3000mg", expected_constraint="max 2550",
                violation_message="overdose", severity=0.9,
                source_tokens=["tok2"],
            )
        ]
        regions = detect_conflicting_regions(extraction, [], violations, tokens)
        assert len(regions) >= 1


class TestRegionsOverlap:
    def test_overlapping(self):
        a = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100, page_number=1)
        b = BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150, page_number=1)
        assert _regions_overlap(a, b)

    def test_non_overlapping(self):
        a = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100, page_number=1)
        b = BoundingBox(x_min=200, y_min=200, x_max=300, y_max=300, page_number=1)
        assert not _regions_overlap(a, b)


class TestFeedbackLoop:
    def test_patch_replaces_tokens_in_region(self):
        original = [
            _make_token("tok1", 100, 100, "wrong"),
            _make_token("tok2", 200, 100, "keep"),
        ]
        region = BoundingBox(x_min=80, y_min=80, x_max=160, y_max=140, page_number=1)
        rescan = [_make_token("tok3", 100, 100, "corrected")]

        patched = patch_tokens(original, [region], rescan)

        texts = [t.text for t in patched]
        assert "wrong" not in texts
        assert "corrected" in texts
        assert "keep" in texts

    def test_patch_preserves_reading_order(self):
        original = [
            _make_token("tok1", 100, 100, "a"),
            _make_token("tok2", 200, 200, "b"),
        ]
        region = BoundingBox(x_min=80, y_min=80, x_max=160, y_max=140, page_number=1)
        rescan = [_make_token("tok3", 100, 100, "fixed")]

        patched = patch_tokens(original, [region], rescan)
        orders = [t.reading_order for t in patched]
        assert orders == sorted(orders)

    def test_token_in_region(self):
        tok = _make_token("t1", 100, 100)
        region = BoundingBox(x_min=90, y_min=90, x_max=200, y_max=200, page_number=1)
        assert _token_in_region(tok, region)

    def test_token_not_in_region(self):
        tok = _make_token("t1", 100, 100)
        region = BoundingBox(x_min=500, y_min=500, x_max=600, y_max=600, page_number=1)
        assert not _token_in_region(tok, region)

    def test_identify_affected_fields(self):
        tok = _make_token("tok1", 100, 100)
        region = BoundingBox(x_min=80, y_min=80, x_max=160, y_max=140, page_number=1)
        field_map = {"patient_name": ["tok1"], "date": ["tok2"]}
        token_map = {"tok1": tok, "tok2": _make_token("tok2", 500, 500)}

        affected = identify_affected_fields([region], field_map, token_map)
        assert "patient_name" in affected
        assert "date" not in affected
