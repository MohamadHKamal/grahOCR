"""Full inference pipeline — single entry point for end-to-end document processing.

Orchestrates all 3 layers:
  Layer 1: Ingestion → OCR → Spatial Assembly → Reading Order → Language → Failures
  RAG:     Policy Retrieval (if vector store available)
  Layer 3: Traffic Routing → Cheap Rail / VLM Consensus
  Output:  InsuranceClaim with provenance + audit trail

Usage:
    from graphocr.pipeline import Pipeline

    pipeline = Pipeline()
    result = await pipeline.process("/path/to/claim.pdf")
    print(result.claim)          # InsuranceClaim
    print(result.tokens)         # SpatialToken[]
    print(result.failures)       # FailureClassification[]
    print(result.routing)        # RoutingDecision
    print(result.extraction)     # ExtractionResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import ProcessingPath
from graphocr.layer3_inference.circuit_breaker import circuit_breakers
from graphocr.monitoring.langsmith_tracer import accuracy_tracker
from graphocr.monitoring.metrics_collector import metrics
from graphocr.layer1_foundation.failure_classifier import classify_failures
from graphocr.layer1_foundation.ingestion import load_document
from graphocr.layer1_foundation.language_detector import assign_languages
from graphocr.layer1_foundation.metadata_enricher import enrich_tokens_with_zones
from graphocr.layer1_foundation.ocr_paddleocr import PaddleOCREngine
from graphocr.layer1_foundation.reading_order import assign_reading_order
from graphocr.layer1_foundation.spatial_assembler import assemble_tokens, group_into_lines
from graphocr.layer3_inference.traffic_controller import RoutingDecision, route_document
from graphocr.models.claim import InsuranceClaim
from graphocr.models.document import PageImage, RawDocument
from graphocr.models.extraction import ExtractionResult
from graphocr.models.failure import FailureClassification
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Complete output from a single document processing run."""

    # Identity
    document_id: str
    source_path: str
    processed_at: datetime = field(default_factory=datetime.utcnow)

    # Layer 1 outputs
    pages: list[PageImage] = field(default_factory=list)
    tokens: list[SpatialToken] = field(default_factory=list)
    lines: list[list[SpatialToken]] = field(default_factory=list)
    failures: list[FailureClassification] = field(default_factory=list)

    # Layer 1 stats
    total_tokens: int = 0
    arabic_tokens: int = 0
    english_tokens: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # Routing
    routing: RoutingDecision | None = None

    # Layer 2+3 outputs (populated when LLM is available)
    extraction: ExtractionResult | None = None
    claim: InsuranceClaim | None = None

    # Timing
    latency_ms: float = 0.0
    layer1_ms: float = 0.0
    layer23_ms: float = 0.0

    # Errors
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.total_tokens > 0

    def text_dump(self) -> str:
        """Return all extracted text in reading order."""
        return "\n".join(
            f"[{t.language.value}] ({t.confidence:.0%}) {t.text}"
            for t in sorted(self.tokens, key=lambda x: x.reading_order)
        )

    def provenance_dump(self, limit: int = 20) -> str:
        """Return provenance strings for first N tokens."""
        return "\n".join(
            t.to_provenance_str()
            for t in sorted(self.tokens, key=lambda x: x.reading_order)[:limit]
        )

    def full_text_ordered(self) -> str:
        """Plain text concatenation of all tokens in reading order."""
        return " ".join(
            t.text for t in sorted(self.tokens, key=lambda x: x.reading_order)
        )

    def ocr_content(self) -> list[dict]:
        """Per-token OCR content with metadata for reports."""
        return [
            {
                "token_id": t.token_id[:8],
                "text": t.text,
                "language": t.language.value,
                "confidence": round(t.confidence, 4),
                "page": t.bbox.page_number,
                "bbox": {
                    "x_min": round(t.bbox.x_min, 1),
                    "y_min": round(t.bbox.y_min, 1),
                    "x_max": round(t.bbox.x_max, 1),
                    "y_max": round(t.bbox.y_max, 1),
                },
                "reading_order": t.reading_order,
                "is_handwritten": t.is_handwritten,
                "zone": t.zone_label.value if t.zone_label else None,
                "ocr_engine": t.ocr_engine,
            }
            for t in sorted(self.tokens, key=lambda x: x.reading_order)
        ]

    def failure_details(self) -> list[dict]:
        """Structured failure details for reports."""
        return [
            {
                "type": f.failure_type.value,
                "severity": round(f.severity, 3),
                "remedy": f.suggested_remedy,
                "evidence": f.evidence,
                "affected_tokens": f.affected_tokens[:5],
            }
            for f in self.failures
        ]

    def summary(self) -> dict:
        """Machine-readable summary for reports — includes OCR content."""
        return {
            "document_id": self.document_id,
            "source": self.source_path,
            "success": self.success,
            "pages": len(self.pages),
            "tokens": self.total_tokens,
            "arabic": self.arabic_tokens,
            "english": self.english_tokens,
            "avg_confidence": round(self.avg_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "failures": len(self.failures),
            "failure_types": [f.failure_type.value for f in self.failures],
            "failure_details": self.failure_details(),
            "route": self.routing.path.value if self.routing else "unknown",
            "uncertainty": round(self.routing.uncertainty_score, 4) if self.routing else 0,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            # OCR content
            "full_text": self.full_text_ordered(),
            "ocr_tokens": self.ocr_content(),
        }


class Pipeline:
    """End-to-end document processing pipeline.

    Handles the complete flow from raw document to structured output.
    Works in two modes:
      - Layer 1 only (no LLM required): OCR + analysis + routing decision
      - Full pipeline (requires vLLM): adds LLM extraction + Neo4j validation

    Surya layout detection is optional — when enabled, the assembler merges
    PaddleOCR text tokens with Surya region bboxes + zone labels, boosting
    confidence and enabling zone-based failure detection (stamp overlap).
    """

    def __init__(self, ocr_lang: str = "ar", use_surya: bool = False, use_paddle: bool = True, use_rag: bool = True):
        self._use_paddle = use_paddle
        self._ocr = PaddleOCREngine(lang=ocr_lang) if use_paddle else None
        self._surya = None
        if use_surya:
            try:
                from graphocr.layer1_foundation.ocr_surya import SuryaLayoutEngine
                self._surya = SuryaLayoutEngine()
                logger.info("pipeline_initialized", ocr="paddleocr" if use_paddle else "none", layout="surya")
            except Exception as e:
                logger.warning("surya_init_failed", error=str(e))
                logger.info("pipeline_initialized", ocr="paddleocr" if use_paddle else "none", layout="none")
        else:
            logger.info("pipeline_initialized", ocr="paddleocr" if use_paddle else "none", layout="none")
            
        self._rag_injector = None
        if use_rag:
            try:
                from graphocr.rag.vector_store import PolicyVectorStore
                from graphocr.rag.context_injector import PolicyContextInjector
                store = PolicyVectorStore()
                self._rag_injector = PolicyContextInjector(store)
                logger.info("rag_initialized")
            except Exception as e:
                logger.warning("rag_init_failed", error=str(e))

    async def process(
        self,
        file_path: str,
        run_llm: bool = False,
        jurisdiction: str = "",
    ) -> PipelineResult:
        """Process a single document end-to-end.

        Args:
            file_path: Path to document (PDF, PNG, JPEG, TIFF).
            run_llm: If True, run Layer 2+3 (requires vLLM). If False, Layer 1 only.
            jurisdiction: Jurisdiction code for data residency.

        Returns:
            PipelineResult with all outputs and provenance.
        """
        start = time.time()
        path = Path(file_path)

        if not path.exists():
            return PipelineResult(
                document_id="", source_path=file_path,
                error=f"File not found: {file_path}",
            )

        fmt = path.suffix.lstrip(".").lower()
        if fmt == "jpg":
            fmt = "jpeg"

        doc = RawDocument(
            source_path=str(path),
            file_format=fmt,
            file_size_bytes=path.stat().st_size,
            jurisdiction=jurisdiction,
        )

        result = PipelineResult(document_id=doc.document_id, source_path=str(path))

        try:
            # === LAYER 1 ===
            l1_start = time.time()

            # Ingest (auto-rotate + normalize + resize)
            result.pages = load_document(doc, f"/tmp/graphocr/{doc.document_id}")

            # OCR — PaddleOCR (text + bboxes)
            paddle_tokens: list[SpatialToken] = []
            if self._use_paddle:
                for page in result.pages:
                    page_tokens = self._ocr.extract(page)
                    paddle_tokens.extend(page_tokens)

            # Layout detection — Surya (region bboxes + zone labels)
            surya_tokens: list[SpatialToken] = []
            layout_zones: list[dict] = []
            if self._surya:
                for page in result.pages:
                    try:
                        surya_page_tokens = self._surya.extract(page)
                        surya_tokens.extend(surya_page_tokens)
                        page_zones = self._surya.detect_layout(page)
                        layout_zones.extend(page_zones)
                    except Exception as e:
                        logger.warning("surya_page_failed", page=page.page_number, error=str(e))

            # Spatial assembly — merge both engines
            # When Surya is active: merges PaddleOCR text with Surya region bboxes,
            # inherits zone labels, boosts confidence where both agree
            streams_to_assemble = []
            if paddle_tokens:
                streams_to_assemble.append(paddle_tokens)
            if surya_tokens:
                streams_to_assemble.append(surya_tokens)
                
            if streams_to_assemble:
                result.tokens = assemble_tokens(streams_to_assemble)
            else:
                result.tokens = []

            # Enrich with zone labels from Surya layout detection
            if layout_zones:
                result.tokens = enrich_tokens_with_zones(result.tokens, layout_zones)

            result.lines = group_into_lines(result.tokens)

            # Reading order + language
            result.tokens = assign_reading_order(result.tokens)
            result.tokens = assign_languages(result.tokens)

            # Failure detection
            result.failures = classify_failures(result.tokens)

            # Stats
            result.total_tokens = len(result.tokens)
            if result.tokens:
                confs = [t.confidence for t in result.tokens]
                result.avg_confidence = sum(confs) / len(confs)
                result.min_confidence = min(confs)
                result.max_confidence = max(confs)
                result.arabic_tokens = sum(1 for t in result.tokens if t.language.value == "ar")
                result.english_tokens = sum(1 for t in result.tokens if t.language.value == "en")

            # Routing decision
            result.routing = route_document(result.tokens, result.failures)
            result.layer1_ms = (time.time() - l1_start) * 1000

            # === LAYER 2 + 3 (if LLM available) ===
            if run_llm and result.tokens:
                l23_start = time.time()
                result.extraction = await self._run_extraction(
                    doc, result.tokens, result.pages, result.routing, result.failures,
                )
                if result.extraction:
                    from graphocr.layer3_inference.output_assembler import assemble_claim
                    result.claim = assemble_claim(result.extraction, result.tokens)

                    # Record metrics
                    if getattr(result.extraction, "rounds_taken", 0) > 1:
                        metrics.increment("healing_triggered")
                    if getattr(result.extraction, "escalated", False):
                        metrics.increment("escalated")

                route_name = result.routing.path.value if result.routing else "unknown"
                metrics.increment("documents_processed")
                metrics.increment(route_name)
                result.layer23_ms = (time.time() - l23_start) * 1000
                metrics.record_latency(route_name, result.layer23_ms)

        except Exception as e:
            result.error = str(e)
            logger.error("pipeline_error", document_id=doc.document_id, error=str(e))

        result.latency_ms = (time.time() - start) * 1000

        logger.info(
            "pipeline_complete",
            document_id=doc.document_id,
            tokens=result.total_tokens,
            failures=len(result.failures),
            route=result.routing.path.value if result.routing else "N/A",
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    async def process_batch(
        self,
        folder_path: str,
        run_llm: bool = False,
        extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"),
    ) -> list[PipelineResult]:
        """Process all documents in a folder.

        Args:
            folder_path: Path to folder containing documents.
            run_llm: Whether to run LLM extraction.
            extensions: File extensions to include.

        Returns:
            List of PipelineResult, one per document.
        """
        folder = Path(folder_path)
        files = sorted(
            f for f in folder.rglob('*')
            if f.is_file() and f.suffix.lower() in extensions
        )

        logger.info("batch_started", folder=str(folder), files=len(files))
        results: list[PipelineResult] = []

        for i, file in enumerate(files):
            logger.info("batch_progress", file=file.name, index=i + 1, total=len(files))
            result = await self.process(str(file), run_llm=run_llm)
            results.append(result)

        logger.info("batch_complete", files=len(files), success=sum(1 for r in results if r.success))
        return results

    async def _run_extraction(
        self,
        doc: RawDocument,
        tokens: list[SpatialToken],
        pages: list[PageImage],
        routing: RoutingDecision,
        failures: list[FailureClassification],
    ) -> ExtractionResult | None:
        """Run Layer 2+3 extraction based on routing decision."""
        from graphocr.core.exceptions import CircuitBreakerOpenError

        # Retrieve RAG context if enabled
        policy_context_extractor = ""
        policy_context_validator = ""
        policy_context_challenger = ""
        retrieval_method = ""
        retrieval_warnings = []

        if self._rag_injector:
            try:
                context = self._rag_injector.get_context_for_claim(
                    tokens=tokens,
                    jurisdiction=doc.jurisdiction,
                )
                policy_context_extractor = self._rag_injector.format_for_extractor(context)
                policy_context_validator = self._rag_injector.format_for_validator(context)
                policy_context_challenger = self._rag_injector.format_for_challenger(context)
                retrieval_method = context.retrieval_method
                retrieval_warnings = context.warnings

                # Validate policy match — detect Type B (wrong policy version) risk
                policy_violations = self._rag_injector.validate_policy_match(context)
                if policy_violations:
                    for v in policy_violations:
                        retrieval_warnings.append(v.violation_message)
                    logger.warning(
                        "policy_match_violations",
                        document_id=doc.document_id,
                        violations=len(policy_violations),
                    )
            except Exception as e:
                logger.warning("rag_retrieval_failed", document_id=doc.document_id, error=str(e))

        # Circuit breaker check
        route_name = routing.path.value
        cb = circuit_breakers.get_or_create(route_name)

        try:
            cb.check()

            if routing.path == ProcessingPath.CHEAP_RAIL:
                from graphocr.layer3_inference.cheap_rail import process_cheap_rail
                result = await process_cheap_rail(
                    doc.document_id,
                    tokens,
                    policy_context=policy_context_extractor,
                )
            else:
                from graphocr.layer3_inference.vlm_consensus import process_vlm_consensus
                result = await process_vlm_consensus(
                    doc.document_id,
                    tokens,
                    {p.page_number: p.image_path for p in pages},
                    policy_context_extractor=policy_context_extractor,
                    policy_context_validator=policy_context_validator,
                    policy_context_challenger=policy_context_challenger,
                    retrieval_method=retrieval_method,
                    retrieval_warnings=retrieval_warnings,
                )

            cb.record_success()
            accuracy_tracker.record(result.overall_confidence > 0.7)

            # Check for accuracy decay — trigger circuit breaker if detected
            if accuracy_tracker.detect_decay():
                logger.warning(
                    "accuracy_decay_detected",
                    document_id=doc.document_id,
                    accuracy=accuracy_tracker.accuracy,
                )
                metrics.increment("accuracy_decay_events")

            return result

        except CircuitBreakerOpenError:
            logger.warning(
                "circuit_breaker_open_fallback",
                document_id=doc.document_id,
                breaker=route_name,
            )
            # Fallback: cheap_rail -> vlm_consensus, vlm -> escalate
            if routing.path == ProcessingPath.CHEAP_RAIL:
                from graphocr.layer3_inference.vlm_consensus import process_vlm_consensus
                metrics.increment("circuit_breaker_fallback")
                return await process_vlm_consensus(
                    doc.document_id, tokens,
                    {p.page_number: p.image_path for p in pages},
                    policy_context_extractor=policy_context_extractor,
                    policy_context_validator=policy_context_validator,
                    policy_context_challenger=policy_context_challenger,
                    retrieval_method=retrieval_method,
                    retrieval_warnings=retrieval_warnings,
                )
            return None

        except Exception as e:
            cb.record_failure()
            accuracy_tracker.record(False)
            logger.error("extraction_failed", document_id=doc.document_id, error=str(e))
            return None
