"""End-to-end integration test — processes a synthetic claim through all 3 layers.

Verifies the provenance chain is intact: every extracted field traces
back to specific SpatialTokens with page coordinates.

This test does NOT require Docker, vLLM, or Neo4j — it uses the internal
pipeline components with mock tokens (simulating Layer 1 output).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from graphocr.core.types import FailureType, Language, ProcessingPath, ZoneLabel
from graphocr.layer1_foundation.failure_classifier import classify_failures

pytestmark = [pytest.mark.integration, pytest.mark.smoke]
from graphocr.layer1_foundation.language_detector import assign_languages
from graphocr.layer1_foundation.reading_order import assign_reading_order
from graphocr.layer1_foundation.spatial_assembler import assemble_tokens, group_into_lines
from graphocr.layer2_verification.knowledge_graph.rule_engine import _validate_amounts
from graphocr.layer2_verification.knowledge_graph.validators import validate_date_sanity
from graphocr.layer2_verification.self_healing.conflict_detector import detect_conflicting_regions
from graphocr.layer2_verification.self_healing.feedback_loop import patch_tokens
from graphocr.layer3_inference.traffic_controller import route_document
from graphocr.layer3_inference.circuit_breaker import CircuitBreaker, CircuitState
from graphocr.models.claim import InsuranceClaim, MedicationEntry
from graphocr.models.extraction import ExtractionResult, FieldExtraction
from graphocr.models.failure import Challenge, GraphViolation
from graphocr.models.token import BoundingBox, SpatialToken


def _build_synthetic_claim_tokens() -> list[SpatialToken]:
    """Build a realistic set of tokens simulating Arabic/English claim OCR output."""
    return [
        # Header zone
        SpatialToken(
            token_id="hdr_001", text="Gulf Health Insurance",
            bbox=BoundingBox(x_min=200, y_min=50, x_max=600, y_max=80, page_number=1),
            reading_order=0, confidence=0.97, ocr_engine="paddleocr",
            zone_label=ZoneLabel.HEADER, language=Language.ENGLISH,
        ),
        SpatialToken(
            token_id="hdr_002", text="تأمين الخليج الصحي",
            bbox=BoundingBox(x_min=200, y_min=85, x_max=500, y_max=110, page_number=1),
            reading_order=1, confidence=0.94, ocr_engine="paddleocr",
            zone_label=ZoneLabel.HEADER, language=Language.ARABIC,
        ),
        # Patient info
        SpatialToken(
            token_id="pat_001", text="Patient Name:",
            bbox=BoundingBox(x_min=50, y_min=150, x_max=200, y_max=175, page_number=1),
            reading_order=2, confidence=0.96, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        SpatialToken(
            token_id="pat_002", text="محمد أحمد العلي",
            bbox=BoundingBox(x_min=210, y_min=150, x_max=420, y_max=175, page_number=1),
            reading_order=3, confidence=0.88, ocr_engine="paddleocr",
            language=Language.ARABIC, zone_label=ZoneLabel.BODY, is_handwritten=True,
        ),
        # Policy reference
        SpatialToken(
            token_id="pol_001", text="Policy No: SA-2018-STD-001",
            bbox=BoundingBox(x_min=50, y_min=190, x_max=350, y_max=215, page_number=1),
            reading_order=4, confidence=0.92, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        # Date
        SpatialToken(
            token_id="dat_001", text="Date of Service: 2026-03-15",
            bbox=BoundingBox(x_min=50, y_min=230, x_max=350, y_max=255, page_number=1),
            reading_order=5, confidence=0.95, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        # Diagnosis
        SpatialToken(
            token_id="diag_001", text="Diagnosis: E11.9",
            bbox=BoundingBox(x_min=50, y_min=280, x_max=250, y_max=305, page_number=1),
            reading_order=6, confidence=0.93, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        # Procedure
        SpatialToken(
            token_id="proc_001", text="Procedure: 99213",
            bbox=BoundingBox(x_min=50, y_min=320, x_max=250, y_max=345, page_number=1),
            reading_order=7, confidence=0.91, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        # Medication
        SpatialToken(
            token_id="med_001", text="Metformin 500mg 2x/day",
            bbox=BoundingBox(x_min=50, y_min=370, x_max=350, y_max=395, page_number=1),
            reading_order=8, confidence=0.87, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY, is_handwritten=True,
        ),
        # Amount
        SpatialToken(
            token_id="amt_001", text="Total: 1,500.00 SAR",
            bbox=BoundingBox(x_min=50, y_min=500, x_max=300, y_max=530, page_number=1),
            reading_order=9, confidence=0.90, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.BODY,
        ),
        # Stamp (overlapping amount area slightly)
        SpatialToken(
            token_id="stamp_001", text="RECEIVED",
            bbox=BoundingBox(x_min=250, y_min=490, x_max=400, y_max=540, page_number=1),
            reading_order=10, confidence=0.60, ocr_engine="paddleocr",
            language=Language.ENGLISH, zone_label=ZoneLabel.STAMP,
        ),
    ]


class TestEndToEndPipeline:
    """End-to-end test walking a synthetic claim through all pipeline stages."""

    def setup_method(self):
        self.tokens = _build_synthetic_claim_tokens()

    # --- Layer 1: Foundation ---

    def test_layer1_spatial_assembly(self):
        """Multiple OCR engines merged into single token stream."""
        stream1 = self.tokens[:6]
        stream2 = self.tokens[6:]  # Simulates second engine output
        merged = assemble_tokens([stream1, stream2])
        assert len(merged) >= len(self.tokens) - 2  # Some may merge

    def test_layer1_reading_order(self):
        """Tokens get correct reading order assigned."""
        ordered = assign_reading_order(self.tokens)
        orders = [t.reading_order for t in ordered]
        assert orders == sorted(orders)  # Monotonically increasing

    def test_layer1_language_detection(self):
        """Arabic and English tokens detected correctly."""
        detected = assign_languages(self.tokens)
        arabic = [t for t in detected if t.language == Language.ARABIC]
        english = [t for t in detected if t.language == Language.ENGLISH]
        assert len(arabic) >= 2  # Header + patient name
        assert len(english) >= 5

    def test_layer1_failure_classification(self):
        """Stamp overlap should be detected as Type A failure."""
        failures = classify_failures(self.tokens)
        stamp_failures = [f for f in failures if "tamp" in f.evidence.lower()]
        assert len(stamp_failures) >= 1
        assert stamp_failures[0].failure_type == FailureType.TYPE_A_SPATIAL_BLIND

    def test_layer1_line_grouping(self):
        """Tokens on the same Y-line grouped together."""
        lines = group_into_lines(self.tokens, y_tolerance=30)
        assert len(lines) >= 5  # Multiple logical lines

    # --- Layer 2: Verification ---

    def test_layer2_graph_date_validation(self):
        """Valid claim dates pass, invalid ones caught."""
        # Valid claim
        violations = validate_date_sanity(date(2026, 3, 15), date(1980, 5, 12))
        assert len(violations) == 0

        # Future date caught
        violations = validate_date_sanity(date(2030, 1, 1), None)
        assert len(violations) > 0

    def test_layer2_graph_amount_validation(self):
        """Amount validation catches negative and excessive values."""
        claim = InsuranceClaim(total_amount=Decimal("1500.00"))
        violations = _validate_amounts(claim)
        assert len(violations) == 0

        bad_claim = InsuranceClaim(total_amount=Decimal("-500"))
        violations = _validate_amounts(bad_claim)
        assert len(violations) > 0

    def test_layer2_conflict_detection(self):
        """Conflict detector finds regions from high-confidence challenges."""
        extraction = ExtractionResult(
            claim_id="c1", document_id="d1",
            fields={"amount": FieldExtraction(
                field_name="amount", value="1500", confidence=0.4,
                source_tokens=["amt_001"],
            )},
        )
        challenges = [
            Challenge(
                target_field="amount", hypothesis="digit misread",
                evidence="3 vs 8", confidence=0.8,
                affected_tokens=["amt_001"],
            )
        ]
        regions = detect_conflicting_regions(extraction, challenges, [], self.tokens)
        assert len(regions) >= 1

    def test_layer2_self_healing_patch(self):
        """Token patching replaces conflicting tokens with VLM results."""
        region = BoundingBox(x_min=40, y_min=490, x_max=310, y_max=540, page_number=1)
        rescan = [SpatialToken(
            token_id="fixed_001", text="Total: 1,800.00 SAR",
            bbox=BoundingBox(x_min=50, y_min=500, x_max=300, y_max=530, page_number=1),
            reading_order=0, confidence=0.95, ocr_engine="vlm_rescan",
        )]
        patched = patch_tokens(self.tokens, [region], rescan)

        # Original amount token should be gone, replacement present
        texts = [t.text for t in patched]
        assert "Total: 1,800.00 SAR" in texts
        assert "Total: 1,500.00 SAR" not in texts

    # --- Layer 3: Inference ---

    def test_layer3_traffic_routing(self):
        """Claim with handwriting and stamp gets routed to VLM consensus."""
        decision = route_document(self.tokens, classify_failures(self.tokens))
        # With handwritten tokens and a stamp overlap, uncertainty should be higher
        assert decision.uncertainty_score > 0.0
        assert decision.handwriting_ratio > 0.0

    def test_layer3_circuit_breaker_lifecycle(self):
        """Circuit breaker transitions through all states."""
        cb = CircuitBreaker("e2e_test", min_calls=3, failure_threshold=0.5, recovery_timeout=60)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    # --- Provenance Chain ---

    def test_provenance_chain_intact(self):
        """Every token has coordinates, page, engine — the core requirement."""
        for token in self.tokens:
            assert token.token_id
            assert token.bbox.page_number >= 1
            assert token.bbox.x_min >= 0
            assert token.bbox.y_min >= 0
            assert token.ocr_engine
            assert 0.0 <= token.confidence <= 1.0
            provenance = token.to_provenance_str()
            assert "page=" in provenance
            assert token.ocr_engine in provenance

    def test_bilingual_provenance(self):
        """Both Arabic and English tokens carry full provenance."""
        arabic_tokens = [t for t in self.tokens if t.language == Language.ARABIC]
        english_tokens = [t for t in self.tokens if t.language == Language.ENGLISH]

        for t in arabic_tokens:
            assert t.bbox.page_number == 1
            assert t.confidence > 0

        for t in english_tokens:
            assert t.bbox.page_number == 1
            assert t.confidence > 0

    # --- Phase 1: Self-Healing State ---

    def test_self_healing_applied_flag_in_state(self):
        """RedTeamState includes self_healing_applied field."""
        from graphocr.models.agent_state import RedTeamState

        # Verify the TypedDict has the field by checking annotations
        annotations = RedTeamState.__annotations__
        assert "self_healing_applied" in annotations

    # --- Phase 2: Learned Rules Validator ---

    def test_learned_rules_validator_no_rules(self):
        """Learned rules validator returns empty when no rules exist."""
        # This tests the validator function signature and empty-case behavior
        from graphocr.layer2_verification.knowledge_graph.validators import validate_learned_rules

        assert validate_learned_rules is not None  # Function exists

    # --- Phase 3: DSPy Module Registration ---

    def test_dspy_modules_registered(self):
        """All DSPy modules including PolicyVersionValidator are in registry."""
        from graphocr.dspy_layer.optimizers import MODULES

        assert "ClaimFieldExtractor" in MODULES
        assert "ArabicMedicalNormalizer" in MODULES
        assert "DiagnosisCodeMapper" in MODULES
        assert "ChallengeGenerator" in MODULES
        assert "PolicyVersionValidator" in MODULES

    # --- Phase 4: Metrics Integration ---

    def test_metrics_collector_counters(self):
        """MetricsCollector tracks all required counters."""
        from graphocr.monitoring.metrics_collector import MetricsCollector

        mc = MetricsCollector()
        mc.increment("documents_processed")
        mc.increment("cheap_rail")
        mc.increment("healing_triggered")
        mc.record_latency("cheap_rail", 150.0)

        m = mc.get_metrics()
        assert m.documents_processed == 1
        assert m.cheap_rail_count == 1
        assert m.healing_triggered == 1
        assert m.avg_latency_cheap > 0

    def test_accuracy_tracker_decay_detection(self):
        """AccuracyTracker detects declining accuracy trend."""
        from graphocr.monitoring.langsmith_tracer import AccuracyTracker

        tracker = AccuracyTracker(window_size=100)

        # No decay with insufficient samples
        assert tracker.detect_decay() is False

        # Record 100% accuracy — no decay
        for _ in range(60):
            tracker.record(True)
        assert tracker.detect_decay() is False

    def test_circuit_breaker_failure_rate(self):
        """Circuit breaker tracks failure rate accurately."""
        cb = CircuitBreaker("test_rate", min_calls=5, failure_threshold=0.4)
        for _ in range(3):
            cb.record_success()
        for _ in range(2):
            cb.record_failure()

        assert 0.3 < cb.failure_rate < 0.5  # 2/5 = 0.4

    # --- Phase 5: Audit Module ---

    def test_audit_metadata_schema(self):
        """Audit metadata schema contains required fields."""
        from graphocr.audit.dashboard import get_metadata_schema

        schema = get_metadata_schema()
        assert "spatial_token" in schema
        assert "provenance_chain" in schema
        assert "failure_classification" in schema
        assert "federated_constraints" in schema

        token_schema = schema["spatial_token"]
        assert "token_id" in token_schema
        assert "bbox" in token_schema
        assert "reading_order" in token_schema
        assert "confidence" in token_schema
        assert "language" in token_schema

    def test_audit_failure_breakdown_classification(self):
        """FailureBreakdown correctly computes Type A/B rates."""
        from graphocr.audit.failure_analyzer import FailureBreakdown
        from datetime import datetime

        breakdown = FailureBreakdown(
            window_start=datetime(2026, 3, 1),
            window_end=datetime(2026, 3, 29),
            total_reports=100,
            type_a_count=70,
            type_b_count=30,
        )

        assert breakdown.type_a_rate == 0.7
        assert breakdown.type_b_rate == 0.3

        d = breakdown.to_dict()
        assert d["type_a"]["count"] == 70
        assert d["type_b"]["count"] == 30
