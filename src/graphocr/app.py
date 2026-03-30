"""FastAPI application — REST API for the Hybrid Graph-OCR Pipeline."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger, setup_logging
from graphocr.dspy_layer.supervisor import DSPySupervisor
from graphocr.layer1_foundation.failure_classifier import classify_failures
from graphocr.layer1_foundation.ingestion import load_document
from graphocr.layer1_foundation.language_detector import assign_languages
from graphocr.layer1_foundation.ocr_paddleocr import PaddleOCREngine
from graphocr.layer1_foundation.reading_order import assign_reading_order
from graphocr.layer1_foundation.spatial_assembler import assemble_tokens
from graphocr.layer3_inference.cheap_rail import process_cheap_rail
from graphocr.layer3_inference.circuit_breaker import circuit_breakers
from graphocr.layer3_inference.traffic_controller import route_document
from graphocr.layer3_inference.vlm_consensus import process_vlm_consensus
from graphocr.monitoring.langsmith_tracer import accuracy_tracker, configure_langsmith
from graphocr.monitoring.metrics_collector import metrics
from graphocr.models.document import RawDocument

logger = get_logger(__name__)

# DSPy supervisor instance
_supervisor: DSPySupervisor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application startup and shutdown."""
    global _supervisor

    settings = get_settings()
    setup_logging(settings.log_level)
    configure_langsmith()

    # Seed Neo4j knowledge graph on startup
    try:
        from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
        from graphocr.layer2_verification.knowledge_graph.schema_loader import load_schema

        neo4j_client = Neo4jClient()
        await neo4j_client.connect()
        await load_schema(neo4j_client)
        await neo4j_client.close()
        logger.info("neo4j_schema_seeded_on_startup")
    except Exception as e:
        logger.warning("neo4j_schema_seed_skipped", error=str(e))

    # Start DSPy supervisor in background
    _supervisor = DSPySupervisor()
    supervisor_task = asyncio.create_task(_supervisor.start())

    logger.info("pipeline_started")
    yield

    # Shutdown
    if _supervisor:
        _supervisor.stop()
    supervisor_task.cancel()
    logger.info("pipeline_stopped")


app = FastAPI(
    title="GraphOCR Pipeline",
    description="Hybrid Graph-OCR Pipeline: Deterministic Trust Layer for Insurance Claims",
    version="0.1.0",
    lifespan=lifespan,
)


class ProcessResponse(BaseModel):
    document_id: str
    processing_path: str
    fields: dict
    overall_confidence: float
    violations: int
    challenges: int
    rounds: int
    escalated: bool
    latency_ms: float


@app.post("/process", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """Process a single insurance claim document through the pipeline."""
    start = time.time()

    # Save uploaded file
    settings = get_settings()
    upload_dir = Path("/tmp/graphocr_uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / (file.filename or "upload.pdf")
    content = await file.read()
    file_path.write_bytes(content)

    file_format = file_path.suffix.lstrip(".").lower()
    if file_format not in ("pdf", "tiff", "jpeg", "jpg", "png"):
        raise HTTPException(400, f"Unsupported format: {file_format}")

    # Create document
    doc = RawDocument(
        source_path=str(file_path),
        file_format=file_format if file_format != "jpg" else "jpeg",
        file_size_bytes=len(content),
    )

    # Layer 1: Ingest and OCR
    pages = load_document(doc, str(upload_dir / doc.document_id))
    ocr = PaddleOCREngine()
    token_streams = [ocr.extract(page) for page in pages]
    tokens = assemble_tokens(token_streams)
    tokens = assign_reading_order(tokens)
    tokens = assign_languages(tokens)
    failures = classify_failures(tokens)

    # Layer 3: Route
    routing = route_document(tokens, failures)

    # Process
    if routing.path.value == "cheap_rail":
        cb = circuit_breakers.get_or_create("cheap_rail")
        try:
            cb.check()
            extraction = await process_cheap_rail(doc.document_id, tokens)
            cb.record_success()
            metrics.increment("cheap_rail")
        except Exception as e:
            cb.record_failure()
            # Fallback to VLM consensus
            extraction = await process_vlm_consensus(
                doc.document_id,
                tokens,
                {p.page_number: p.image_path for p in pages},
            )
            metrics.increment("vlm_consensus")
    else:
        extraction = await process_vlm_consensus(
            doc.document_id,
            tokens,
            {p.page_number: p.image_path for p in pages},
        )
        metrics.increment("vlm_consensus")

    metrics.increment("documents_processed")
    latency_ms = (time.time() - start) * 1000
    metrics.record_latency(routing.path.value, latency_ms)

    return ProcessResponse(
        document_id=doc.document_id,
        processing_path=routing.path.value,
        fields={k: {"value": v.value, "confidence": v.confidence} for k, v in extraction.fields.items()},
        overall_confidence=extraction.overall_confidence,
        violations=0,  # TODO: from extraction
        challenges=0,
        rounds=extraction.rounds_taken,
        escalated=extraction.escalated,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics():
    """Pipeline metrics for monitoring dashboards."""
    m = metrics.get_metrics()
    return {
        "documents_processed": m.documents_processed,
        "documents_per_minute": round(m.documents_per_minute, 2),
        "routing": {
            "cheap_rail": m.cheap_rail_count,
            "vlm_consensus": m.vlm_consensus_count,
            "escalated": m.escalated_count,
        },
        "latency": {
            "avg_cheap_ms": round(m.avg_latency_cheap, 1),
            "avg_vlm_ms": round(m.avg_latency_vlm, 1),
            "p95_ms": round(m.p95_latency, 1),
        },
        "self_healing": {
            "triggered": m.healing_triggered,
            "successful": m.healing_successful,
        },
        "accuracy": round(accuracy_tracker.accuracy, 4),
        "circuit_breakers": circuit_breakers.all_metrics(),
    }


@app.get("/supervisor/status")
async def supervisor_status():
    """DSPy supervisor status and gradient analysis."""
    if _supervisor:
        return _supervisor.get_status()
    return {"status": "not_running"}


# === Audit Endpoints (Task 1: Diagnostic Tool) ===


@app.get("/audit/stats")
async def audit_stats(window_hours: int = 24, jurisdiction: str | None = None):
    """Aggregated failure statistics — Type A/B breakdown.

    This is the diagnostic tool that distinguishes Input Failure (OCR)
    from Intelligence Failure (RAG). Supports federated queries.
    """
    from graphocr.audit.dashboard import get_failure_stats

    return await get_failure_stats(window_hours=window_hours, jurisdiction=jurisdiction)


@app.get("/audit/failure/{report_id}")
async def audit_failure_detail(report_id: str):
    """Detailed failure report by ID."""
    from graphocr.audit.dashboard import get_failure_detail

    result = await get_failure_detail(report_id)
    if result is None:
        raise HTTPException(404, f"Failure report not found: {report_id}")
    return result


@app.get("/audit/metadata-schema")
async def audit_metadata_schema():
    """SpatialToken metadata schema — the Semantic Spatial mapping.

    This is the enforced schema that serves as the Single Source of Truth
    for both senior and junior engineers.
    """
    from graphocr.audit.dashboard import get_metadata_schema

    return get_metadata_schema()


@app.get("/audit/learned-rules")
async def audit_learned_rules():
    """Active learned rules from Post-Mortem back-propagation."""
    from graphocr.audit.dashboard import get_learned_rules

    return await get_learned_rules()
