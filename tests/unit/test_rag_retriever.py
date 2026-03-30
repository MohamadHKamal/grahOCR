"""Tests for the temporal-aware policy retriever.

These tests verify that the retriever correctly:
1. Extracts policy references from claim text
2. Filters by temporal validity (preventing Failure Type B)
3. Falls back gracefully with appropriate warnings
"""

from datetime import date

import pytest

from graphocr.rag.retriever import TemporalPolicyRetriever

pytestmark = [pytest.mark.unit, pytest.mark.rag]


class TestPolicyReferenceExtraction:
    """Test extraction of policy references from claim text."""

    def test_extract_policy_number(self):
        ref = TemporalPolicyRetriever._extract_policy_reference(
            "Policy No: SA-2018-STD-001 Patient: Mohammed"
        )
        assert ref == "SA-2018-STD-001"

    def test_extract_rider(self):
        ref = TemporalPolicyRetriever._extract_policy_reference(
            "Claims under Rider 2018-R3 are subject to..."
        )
        assert ref is not None
        assert "2018" in ref

    def test_extract_plan(self):
        ref = TemporalPolicyRetriever._extract_policy_reference(
            "Plan: Standard 2025"
        )
        assert ref is not None

    def test_extract_arabic_reference(self):
        ref = TemporalPolicyRetriever._extract_policy_reference(
            "بوليصة رقم: SA-2018-001 المريض: محمد"
        )
        assert ref == "SA-2018-001"

    def test_no_reference_returns_none(self):
        ref = TemporalPolicyRetriever._extract_policy_reference(
            "Patient name: Mohammed Ahmed, Date: 2026-03-15"
        )
        assert ref is None


class TestDateExtraction:
    """Test date extraction from claim text."""

    def test_iso_date(self):
        d = TemporalPolicyRetriever._extract_date("Date of Service: 2018-06-15")
        assert d == date(2018, 6, 15)

    def test_slash_date(self):
        d = TemporalPolicyRetriever._extract_date("Service: 2018/06/15")
        assert d == date(2018, 6, 15)

    def test_no_date(self):
        d = TemporalPolicyRetriever._extract_date("Patient: Mohammed Ahmed")
        assert d is None


class TestTemporalFilter:
    """Test the temporal filtering that prevents Failure Type B."""

    def setup_method(self):
        # We don't need a real vector store for filter tests
        self.retriever = TemporalPolicyRetriever.__new__(TemporalPolicyRetriever)

    def test_filters_expired_policy(self):
        hits = [
            {
                "chunk_id": "1",
                "text": "Coverage under 2018 plan",
                "metadata": {
                    "effective_date": "2018-01-01",
                    "expiry_date": "2019-12-31",
                },
                "distance": 0.1,
            },
            {
                "chunk_id": "2",
                "text": "Coverage under 2025 plan",
                "metadata": {
                    "effective_date": "2025-01-01",
                    "expiry_date": "2026-12-31",
                },
                "distance": 0.2,
            },
        ]

        # A claim from 2018 should only match the 2018 policy
        filtered = self.retriever._temporal_filter(hits, date(2018, 6, 15))
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "1"

    def test_filters_not_yet_effective(self):
        hits = [
            {
                "chunk_id": "1",
                "text": "2025 plan",
                "metadata": {
                    "effective_date": "2025-01-01",
                    "expiry_date": "",
                },
                "distance": 0.1,
            },
        ]

        # A claim from 2024 should NOT match a 2025 policy
        filtered = self.retriever._temporal_filter(hits, date(2024, 6, 15))
        assert len(filtered) == 0

    def test_keeps_currently_valid(self):
        hits = [
            {
                "chunk_id": "1",
                "text": "Active plan",
                "metadata": {
                    "effective_date": "2025-01-01",
                    "expiry_date": "2026-12-31",
                },
                "distance": 0.1,
            },
        ]

        filtered = self.retriever._temporal_filter(hits, date(2026, 3, 15))
        assert len(filtered) == 1

    def test_handles_no_expiry(self):
        """Policies without expiry are considered perpetually valid."""
        hits = [
            {
                "chunk_id": "1",
                "text": "Perpetual plan",
                "metadata": {
                    "effective_date": "2020-01-01",
                    "expiry_date": "",
                },
                "distance": 0.1,
            },
        ]

        filtered = self.retriever._temporal_filter(hits, date(2026, 3, 15))
        assert len(filtered) == 1


class TestSemanticQueryBuilder:
    """Test the focused semantic query builder."""

    def test_extracts_icd_codes(self):
        query = TemporalPolicyRetriever._build_semantic_query(
            "Patient diagnosed with E11.9 type 2 diabetes and I10 hypertension"
        )
        assert "E11.9" in query
        assert "I10" in query

    def test_extracts_cpt_codes(self):
        query = TemporalPolicyRetriever._build_semantic_query(
            "Procedure 99213 office visit performed"
        )
        assert "99213" in query

    def test_fallback_to_truncated_text(self):
        query = TemporalPolicyRetriever._build_semantic_query(
            "Patient Mohammed Ahmed visited the clinic for a routine checkup"
        )
        assert len(query) <= 200
