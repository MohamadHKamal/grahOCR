"""Tests for DSPy custom metrics."""

import pytest

from graphocr.dspy_layer.metrics import (
    field_level_f1,
    exact_match,
    code_accuracy,
    _normalize_arabic,
    _field_similarity,
)

pytestmark = [pytest.mark.unit, pytest.mark.dspy]


class TestFieldLevelF1:
    def test_perfect_match(self):
        example = {"claim_fields_json": '{"patient_name": "Mohammed"}'}
        pred = {"claim_fields_json": '{"patient_name": "Mohammed"}'}
        assert field_level_f1(example, pred) == 1.0

    def test_complete_mismatch(self):
        example = {"claim_fields_json": '{"patient_name": "Mohammed"}'}
        pred = {"claim_fields_json": '{"patient_name": ""}'}
        assert field_level_f1(example, pred) == 0.0

    def test_partial_fields(self):
        example = {"claim_fields_json": '{"a": "1", "b": "2"}'}
        pred = {"claim_fields_json": '{"a": "1", "b": "wrong"}'}
        score = field_level_f1(example, pred)
        assert 0.3 < score < 0.8  # One right, one wrong

    def test_invalid_json(self):
        assert field_level_f1({"claim_fields_json": "not json"}, {"claim_fields_json": "{}"}) == 0.0


class TestExactMatch:
    def test_match(self):
        assert exact_match({"normalized_text": "foo"}, {"normalized_text": "foo"}) == 1.0

    def test_no_match(self):
        assert exact_match({"normalized_text": "foo"}, {"normalized_text": "bar"}) == 0.0


class TestCodeAccuracy:
    def test_exact_code(self):
        assert code_accuracy({"icd10_code": "E11.9"}, {"icd10_code": "E11.9"}) == 1.0

    def test_same_category(self):
        assert code_accuracy({"icd10_code": "E11.9"}, {"icd10_code": "E11.0"}) == 0.5

    def test_different_category(self):
        assert code_accuracy({"icd10_code": "E11.9"}, {"icd10_code": "I10"}) == 0.0


class TestArabicNormalization:
    def test_removes_diacritics(self):
        # Fatha + damma + kasra should be removed
        assert _normalize_arabic("\u0641\u064E\u0639\u064F\u0644\u0650") == "\u0641\u0639\u0644"

    def test_normalizes_alef_variants(self):
        assert _normalize_arabic("\u0623\u062D\u0645\u062F") == "\u0627\u062D\u0645\u062F"

    def test_normalizes_taa_marbuta(self):
        # Taa marbuta -> Haa
        assert _normalize_arabic("\u0645\u062F\u0631\u0633\u0629") == "\u0645\u062F\u0631\u0633\u0647"


class TestFieldSimilarity:
    def test_numeric_exact(self):
        assert _field_similarity("total_amount", "1500.00", "1500.00") == 1.0

    def test_numeric_close(self):
        score = _field_similarity("total_amount", "1500", "1490")
        assert score > 0.9

    def test_code_f1(self):
        assert _field_similarity("diagnosis_codes", "E11.9,I10", "E11.9,I10") == 1.0

    def test_code_partial(self):
        score = _field_similarity("diagnosis_codes", "E11.9,I10", "E11.9")
        assert 0.5 < score < 0.9  # One match, one miss
