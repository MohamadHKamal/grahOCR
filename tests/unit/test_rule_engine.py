"""Tests for the Neo4j rule engine — date sanity and amount validation."""

from datetime import date
from decimal import Decimal

import pytest

from graphocr.layer2_verification.knowledge_graph.validators import validate_date_sanity
from graphocr.layer2_verification.knowledge_graph.rule_engine import _validate_amounts
from graphocr.models.claim import InsuranceClaim

pytestmark = [pytest.mark.unit, pytest.mark.layer2]


class TestDateSanity:
    def test_valid_dates(self):
        violations = validate_date_sanity(
            date_of_service=date(2026, 3, 15),
            patient_dob=date(1980, 5, 12),
        )
        assert len(violations) == 0

    def test_future_date_of_service(self):
        violations = validate_date_sanity(
            date_of_service=date(2030, 1, 1),
            patient_dob=None,
        )
        assert len(violations) == 1
        assert "future" in violations[0].violation_message.lower()

    def test_ancient_date(self):
        violations = validate_date_sanity(
            date_of_service=date(2010, 1, 1),
            patient_dob=None,
        )
        assert len(violations) == 1
        assert "before" in violations[0].violation_message.lower()

    def test_implausible_age(self):
        violations = validate_date_sanity(
            date_of_service=None,
            patient_dob=date(1800, 1, 1),
        )
        assert any("age" in v.violation_message.lower() for v in violations)

    def test_dob_after_service(self):
        violations = validate_date_sanity(
            date_of_service=date(2020, 1, 1),
            patient_dob=date(2025, 1, 1),
        )
        assert any("after" in v.violation_message.lower() for v in violations)


class TestAmountValidation:
    def test_valid_amounts(self):
        claim = InsuranceClaim(total_amount=Decimal("1500"))
        violations = _validate_amounts(claim)
        assert len(violations) == 0

    def test_negative_amount(self):
        claim = InsuranceClaim(total_amount=Decimal("-100"))
        violations = _validate_amounts(claim)
        assert any("negative" in v.violation_message.lower() for v in violations)

    def test_excessive_amount(self):
        claim = InsuranceClaim(total_amount=Decimal("5000000"))
        violations = _validate_amounts(claim)
        assert any("exceeds" in v.violation_message.lower() for v in violations)
