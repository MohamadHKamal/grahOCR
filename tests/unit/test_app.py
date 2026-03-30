"""Tests for the FastAPI application endpoints.

Markers: unit, smoke
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.smoke]

import io
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from graphocr.monitoring.metrics_collector import MetricsCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset global singletons before each test so state doesn't leak."""
    import graphocr.app as app_module
    from graphocr.monitoring import metrics_collector
    from graphocr.monitoring import langsmith_tracer
    from graphocr.monitoring.langsmith_tracer import AccuracyTracker

    fresh_metrics = MetricsCollector()
    fresh_tracker = AccuracyTracker()

    # Reset in both the source module AND in app.py's imported reference
    metrics_collector.metrics = fresh_metrics
    langsmith_tracer.accuracy_tracker = fresh_tracker
    app_module.metrics = fresh_metrics
    app_module.accuracy_tracker = fresh_tracker


@pytest.fixture()
def client():
    """TestClient that skips the lifespan (no Neo4j / DSPy needed)."""
    from graphocr.app import app

    # Disable lifespan for unit tests
    app.router.lifespan_context = _noop_lifespan
    return TestClient(app)


from contextlib import asynccontextmanager

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ===== /health =====

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_is_get_only(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


# ===== /metrics =====

class TestMetrics:
    def test_metrics_default_values(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()

        assert data["documents_processed"] == 0
        assert "routing" in data
        assert data["routing"]["cheap_rail"] == 0
        assert data["routing"]["vlm_consensus"] == 0
        assert data["routing"]["escalated"] == 0
        assert "latency" in data
        assert "self_healing" in data
        assert "accuracy" in data
        assert "circuit_breakers" in data

    def test_metrics_after_increments(self, client):
        import graphocr.app as app_module

        app_module.metrics.increment("documents_processed", 5)
        app_module.metrics.increment("cheap_rail", 4)
        app_module.metrics.increment("vlm_consensus", 1)

        resp = client.get("/metrics")
        data = resp.json()

        assert data["documents_processed"] == 5
        assert data["routing"]["cheap_rail"] == 4
        assert data["routing"]["vlm_consensus"] == 1

    def test_metrics_latency_recording(self, client):
        import graphocr.app as app_module

        app_module.metrics.record_latency("cheap_rail", 100.0)
        app_module.metrics.record_latency("cheap_rail", 200.0)

        resp = client.get("/metrics")
        data = resp.json()

        assert data["latency"]["avg_cheap_ms"] == 150.0

    def test_metrics_accuracy_tracking(self, client):
        import graphocr.app as app_module

        for _ in range(8):
            app_module.accuracy_tracker.record(True)
        for _ in range(2):
            app_module.accuracy_tracker.record(False)

        resp = client.get("/metrics")
        data = resp.json()

        assert data["accuracy"] == 0.8


# ===== /supervisor/status =====

class TestSupervisorStatus:
    def test_supervisor_not_running(self, client):
        import graphocr.app as app_module
        app_module._supervisor = None

        resp = client.get("/supervisor/status")
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_running"}

    def test_supervisor_running(self, client):
        import graphocr.app as app_module

        mock_supervisor = MagicMock()
        mock_supervisor.get_status.return_value = {
            "status": "running",
            "optimizations": 3,
            "gradient_alerts": 1,
        }
        app_module._supervisor = mock_supervisor

        resp = client.get("/supervisor/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "running"
        assert data["optimizations"] == 3
        mock_supervisor.get_status.assert_called_once()

        # Cleanup
        app_module._supervisor = None


# ===== /process =====

class TestProcess:
    def test_process_rejects_unsupported_format(self, client):
        fake_file = io.BytesIO(b"not a real document")
        resp = client.post(
            "/process",
            files={"file": ("test.docx", fake_file, "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_process_requires_file(self, client):
        resp = client.post("/process")
        assert resp.status_code == 422  # FastAPI validation error

    @patch("graphocr.app.process_cheap_rail")
    @patch("graphocr.app.route_document")
    @patch("graphocr.app.classify_failures")
    @patch("graphocr.app.assign_languages")
    @patch("graphocr.app.assign_reading_order")
    @patch("graphocr.app.assemble_tokens")
    @patch("graphocr.app.PaddleOCREngine")
    @patch("graphocr.app.load_document")
    def test_process_happy_path_cheap_rail(
        self,
        mock_load_doc,
        mock_ocr_cls,
        mock_assemble,
        mock_reading,
        mock_lang,
        mock_classify,
        mock_route,
        mock_cheap_rail,
        client,
    ):
        from graphocr.core.types import ProcessingPath
        from graphocr.models.token import BoundingBox, SpatialToken
        from graphocr.models.extraction import ExtractionResult, FieldExtraction

        # Set up mocks
        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.image_path = "/tmp/test/page_1.png"
        mock_load_doc.return_value = [mock_page]

        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = []
        mock_ocr_cls.return_value = mock_ocr

        tokens = [
            SpatialToken(
                token_id="t1",
                text="Test",
                bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30, page_number=1),
                reading_order=0,
                confidence=0.95,
                ocr_engine="paddleocr",
            )
        ]
        mock_assemble.return_value = tokens
        mock_reading.return_value = tokens
        mock_lang.return_value = tokens
        mock_classify.return_value = []

        mock_routing = MagicMock()
        mock_routing.path = ProcessingPath.CHEAP_RAIL
        mock_route.return_value = mock_routing

        mock_extraction = ExtractionResult(
            claim_id="claim-001",
            document_id="test-doc",
            fields={
                "patient_name": FieldExtraction(
                    field_name="patient_name",
                    value="Ahmed",
                    confidence=0.92,
                    source_tokens=["t1"],
                ),
            },
            overall_confidence=0.92,
            rounds_taken=1,
            escalated=False,
        )
        mock_cheap_rail.return_value = mock_extraction

        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = client.post(
            "/process",
            files={"file": ("claim.pdf", fake_pdf, "application/pdf")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["processing_path"] == "cheap_rail"
        assert data["overall_confidence"] == 0.92
        assert data["rounds"] == 1
        assert data["escalated"] is False
        assert "patient_name" in data["fields"]
        assert data["fields"]["patient_name"]["value"] == "Ahmed"
        assert data["latency_ms"] > 0

    @patch("graphocr.app.process_vlm_consensus")
    @patch("graphocr.app.route_document")
    @patch("graphocr.app.classify_failures")
    @patch("graphocr.app.assign_languages")
    @patch("graphocr.app.assign_reading_order")
    @patch("graphocr.app.assemble_tokens")
    @patch("graphocr.app.PaddleOCREngine")
    @patch("graphocr.app.load_document")
    def test_process_vlm_consensus_path(
        self,
        mock_load_doc,
        mock_ocr_cls,
        mock_assemble,
        mock_reading,
        mock_lang,
        mock_classify,
        mock_route,
        mock_vlm,
        client,
    ):
        from graphocr.core.types import ProcessingPath
        from graphocr.models.token import BoundingBox, SpatialToken
        from graphocr.models.extraction import ExtractionResult, FieldExtraction

        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.image_path = "/tmp/test/page_1.png"
        mock_load_doc.return_value = [mock_page]

        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = []
        mock_ocr_cls.return_value = mock_ocr

        tokens = [
            SpatialToken(
                token_id="t1",
                text="Test",
                bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30, page_number=1),
                reading_order=0,
                confidence=0.95,
                ocr_engine="paddleocr",
            )
        ]
        mock_assemble.return_value = tokens
        mock_reading.return_value = tokens
        mock_lang.return_value = tokens
        mock_classify.return_value = []

        mock_routing = MagicMock()
        mock_routing.path = ProcessingPath.VLM_CONSENSUS
        mock_route.return_value = mock_routing

        mock_extraction = ExtractionResult(
            claim_id="claim-002",
            document_id="test-doc",
            fields={},
            overall_confidence=0.85,
            rounds_taken=2,
            escalated=True,
        )
        mock_vlm.return_value = mock_extraction

        fake_png = io.BytesIO(b"\x89PNG fake")
        resp = client.post(
            "/process",
            files={"file": ("scan.png", fake_png, "image/png")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["processing_path"] == "vlm_consensus"
        assert data["escalated"] is True
        assert data["rounds"] == 2


# ===== /audit/* =====

class TestAuditStats:
    @patch("graphocr.audit.dashboard.get_failure_stats")
    def test_audit_stats_default(self, mock_stats, client):
        mock_stats.return_value = {
            "window": {"start": "2026-03-29T00:00:00", "end": "2026-03-30T00:00:00"},
            "total_reports": 10,
            "type_a": {"count": 6, "rate": 0.6, "root_causes": {"ocr_misread": 4, "layout_confusion": 2}},
            "type_b": {"count": 4, "rate": 0.4, "root_causes": {"prompt_failure": 3, "rule_gap": 1}},
            "affected_fields": {"patient_name": 3, "date_of_service": 2},
            "resolution_methods": {"vlm_rescan": 5, "self_healing": 3},
            "training_eligible": 7,
        }

        resp = client.get("/audit/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_reports"] == 10
        assert data["type_a"]["count"] == 6
        assert data["type_b"]["count"] == 4
        mock_stats.assert_called_once_with(window_hours=24, jurisdiction=None)

    @patch("graphocr.audit.dashboard.get_failure_stats")
    def test_audit_stats_with_params(self, mock_stats, client):
        mock_stats.return_value = {"total_reports": 0}

        resp = client.get("/audit/stats?window_hours=48&jurisdiction=SA")
        assert resp.status_code == 200
        mock_stats.assert_called_once_with(window_hours=48, jurisdiction="SA")


class TestAuditFailureDetail:
    @patch("graphocr.audit.dashboard.get_failure_detail")
    def test_failure_detail_found(self, mock_detail, client):
        mock_detail.return_value = {
            "report_id": "rpt-001",
            "failure_type": "type_a_spatial_blind",
            "root_cause": "ocr_misread",
            "affected_field": "patient_name",
        }

        resp = client.get("/audit/failure/rpt-001")
        assert resp.status_code == 200
        assert resp.json()["report_id"] == "rpt-001"

    @patch("graphocr.audit.dashboard.get_failure_detail")
    def test_failure_detail_not_found(self, mock_detail, client):
        mock_detail.return_value = None

        resp = client.get("/audit/failure/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestAuditMetadataSchema:
    def test_metadata_schema_structure(self, client):
        resp = client.get("/audit/metadata-schema")
        assert resp.status_code == 200
        data = resp.json()

        assert "description" in data
        assert "spatial_token" in data
        assert "provenance_chain" in data
        assert "failure_classification" in data
        assert "federated_constraints" in data

        # Check spatial token fields
        st = data["spatial_token"]
        assert "token_id" in st
        assert "text" in st
        assert "bbox" in st
        assert "confidence" in st
        assert "language" in st
        assert "zone_label" in st

    def test_metadata_schema_bbox_fields(self, client):
        resp = client.get("/audit/metadata-schema")
        bbox = resp.json()["spatial_token"]["bbox"]

        assert "x_min" in bbox
        assert "y_min" in bbox
        assert "x_max" in bbox
        assert "y_max" in bbox
        assert "page_number" in bbox

    def test_metadata_schema_jurisdictions(self, client):
        resp = client.get("/audit/metadata-schema")
        jurisdictions = resp.json()["federated_constraints"]["jurisdictions"]

        assert "SA" in jurisdictions
        assert "AE" in jurisdictions


class TestAuditLearnedRules:
    @patch("graphocr.audit.dashboard.get_learned_rules")
    def test_learned_rules_returns_list(self, mock_rules, client):
        mock_rules.return_value = [
            {
                "report_id": "rpt-001",
                "field": "drug_name",
                "bad_value": "Amoxicillin 5000mg",
                "good_value": "Amoxicillin 500mg",
                "root_cause": "ocr_misread",
                "created_at": "2026-03-29T10:00:00",
            }
        ]

        resp = client.get("/audit/learned-rules")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data) == 1
        assert data[0]["field"] == "drug_name"
        assert data[0]["root_cause"] == "ocr_misread"

    @patch("graphocr.audit.dashboard.get_learned_rules")
    def test_learned_rules_empty(self, mock_rules, client):
        mock_rules.return_value = []

        resp = client.get("/audit/learned-rules")
        assert resp.status_code == 200
        assert resp.json() == []


# ===== Response model validation =====

class TestProcessResponse:
    def test_response_model_fields(self):
        from graphocr.app import ProcessResponse

        resp = ProcessResponse(
            document_id="doc-123",
            processing_path="cheap_rail",
            fields={"name": {"value": "Test", "confidence": 0.9}},
            overall_confidence=0.9,
            violations=0,
            challenges=0,
            rounds=1,
            escalated=False,
            latency_ms=150.5,
        )
        assert resp.document_id == "doc-123"
        assert resp.latency_ms == 150.5
        assert resp.escalated is False
