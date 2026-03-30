# Infrastructure & Configuration

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Docker services, API endpoints, CLI commands, YAML configuration, observability
**Purpose**: Complete reference for deployment, configuration, and operational management

---

## 1. Infrastructure Services

| Service | Image | Ports | Purpose |
|---|---|---|---|
| **Neo4j 5.x** | `neo4j:5-community` | 7474 (HTTP), 7687 (Bolt) | Knowledge graph with APOC plugin — domain rules, learned rules |
| **Redis 7** | `redis:7-alpine` | 6379 | Cache, task queue, FailureStore (postmortem reports, DSPy training data) |
| **MinIO** | `minio/minio` | 9000 (API), 9001 (Console) | Object storage for page images (jurisdiction-partitioned buckets) |
| **vLLM** | `vllm/vllm-openai` | 8000 | Serves Qwen2.5-7B, Llama-3.1-70B, Qwen2-VL-7B via OpenAI-compatible API |

### Service Dependencies

```
Pipeline
  ├── PaddleOCR (local, CPU/GPU)
  ├── Surya (local, optional, MPS/GPU)
  ├── vLLM ──── Qwen2.5-7B (cheap rail + validator + postmortem)
  |          ├── Llama-3.1-70B (VLM consensus extractor + challenger)
  |          └── Qwen2-VL-7B (VLM re-scanner)
  ├── Neo4j ──── Domain rules, learned rules, validation queries
  ├── Redis ──── FailureStore, DSPy training data, stats counters
  ├── MinIO ──── Page image storage, intermediate results
  └── Langfuse ── Observability tracing (external SaaS or self-hosted)
```

---

## 2. API Endpoints

**File**: `app.py` (FastAPI)
**Launch**: `graphocr serve --host 0.0.0.0 --port 8080`

### 2.1 Core Processing

**`POST /process`** — Process a single document

| Parameter | Type | Description |
|---|---|---|
| `file` | UploadFile | PDF, TIFF, JPEG, or PNG |
| `jurisdiction` | str (optional) | Jurisdiction code (SA, AE, EG, JO) |

**Response** (`ProcessResponse`):
```json
{
    "document_id": "uuid",
    "processing_path": "CHEAP_RAIL|VLM_CONSENSUS",
    "fields": {"patient_name": "...", "diagnosis_codes": ["E11.2"]},
    "overall_confidence": 0.89,
    "violations": [{"rule": "...", "severity": 0.8}],
    "challenges": [{"field": "...", "hypothesis": "..."}],
    "rounds": 1,
    "escalated": false,
    "latency_ms": 2340
}
```

### 2.2 Health & Monitoring

**`GET /health`** → `{"status": "ok"}`

**`GET /metrics`** → `PipelineMetrics`:
```json
{
    "documents_processed": 45230,
    "documents_per_minute": 72.5,
    "cheap_rail": 40707,
    "vlm_consensus": 4523,
    "escalated": 142,
    "healing_triggered": 891,
    "healing_successful": 834,
    "challenges_raised": 12500,
    "graph_violations": 3200,
    "accuracy": 0.983,
    "avg_latency_cheap": 2100,
    "avg_latency_vlm": 18500,
    "p95_latency": 25000,
    "circuit_breakers": {"cheap_rail": "CLOSED", "vlm_consensus": "CLOSED"}
}
```

### 2.3 DSPy Supervisor

**`GET /supervisor/status`** → `SupervisorState`:
```json
{
    "running": true,
    "last_run": "2026-03-29T14:30:00Z",
    "total_optimizations": 7,
    "modules": {
        "ClaimFieldExtractor": {
            "baseline": 0.92,
            "current": 0.94,
            "degradation": -0.02,
            "samples": 1250,
            "optimizations_today": 1,
            "last_optimized": "2026-03-29T10:00:00Z"
        }
    },
    "gradient_stability": {
        "ClaimFieldExtractor": {
            "stability_score": 0.85,
            "is_diverging": false,
            "trend": "stable"
        }
    },
    "recent_alerts": []
}
```

### 2.4 Audit Endpoints

**`GET /audit/stats?window_hours=24&jurisdiction=SA`** — Failure statistics
- Type A/B breakdown (Input vs Intelligence failure)
- Federated query support (filter by jurisdiction)

**`GET /audit/failure/{report_id}`** — Detailed failure report with root cause, corrected values, provenance

**`GET /audit/metadata-schema`** — SpatialToken metadata schema (the Single Source of Truth definition)

**`GET /audit/learned-rules`** — Active learned rules from postmortem back-propagation

---

## 3. CLI Commands

**File**: `cli.py`

| Command | Description |
|---|---|
| `graphocr process FILE [--output path] [--verbose]` | Process single document, output JSON result |
| `graphocr serve [--host 0.0.0.0] [--port 8080] [--reload]` | Start FastAPI server with Uvicorn |
| `graphocr seed-graph` | Seed Neo4j with domain rules from `neo4j_rules.yaml` |
| `graphocr supervisor-status` | Print DSPy supervisor status as JSON |

---

## 4. Configuration Files

All YAML files in `config/`:

### 4.1 `pipeline.yaml` — Core Pipeline Settings

```yaml
pipeline:
  batch_size: 32
  max_concurrent_claims: 64
  max_agent_rounds: 2

ocr:
  primary_engine: "paddleocr"
  layout_engine: "surya"
  merge_iou_threshold: 0.3
  min_confidence: 0.3
  paddleocr:
    lang: "ar"
  surya:
    use_layout: true
    use_recognition: true

reading_order:
  algorithm: "xy_cut"
  rtl_detection: true
  column_gap_threshold: 30    # px, vertical split min gap
  line_merge_threshold: 10    # px, horizontal split min gap
```

### 4.2 `agents.yaml` — Agent Model Configuration

```yaml
extractor:
  llm:
    cheap_rail: "qwen2.5-7b-instruct"
    vlm_consensus: "llama-3.1-70b-instruct"
  temperature: 0.1
  max_tokens: 4096
  dspy_module: "ClaimFieldExtractor"

validator:
  llm: "qwen2.5-7b-instruct"
  temperature: 0.0              # Fully deterministic
  max_tokens: 2048
  neo4j_checks:
    - procedure_diagnosis_compatibility
    - drug_dosage_limits
    - contraindicated_drugs
    - provider_specialty
    - date_sanity
    - amount_reasonableness
    - age_plausibility

challenger:
  llm: "llama-3.1-70b-instruct"  # Different model for diversity
  temperature: 0.3               # Creative adversarial
  max_tokens: 3072
  max_challenges_per_round: 5
  strategies:
    - arabic_character_confusion
    - digit_ocr_errors
    - stamp_obscuration
    - merged_line_items
    - date_format_ambiguity
    - currency_symbol_misread
    - handwriting_ambiguity

postmortem:
  llm: "qwen2.5-7b-instruct"
  min_severity_to_log: 0.3
  dspy_training_threshold: 0.5

consensus:
  min_agreement_ratio: 0.8
  max_rounds: 2
  escalation_threshold: 0.5
```

### 4.3 `neo4j_rules.yaml` — Domain Knowledge

```yaml
constraints:
  date_ranges:
    min_date_of_service: "2015-01-01"
    max_patient_age: 130
    min_patient_age: 0
  amount_limits:
    max_single_line_item_usd: 500000
    max_claim_total_usd: 2000000
    min_line_item_usd: 0.01
  dosage_limits:
    metformin: 2550.0
    atorvastatin: 80.0
    warfarin: 15.0
    paracetamol: 4000.0
    ibuprofen: 3200.0
    # ... more drugs

relationships:
  contraindicated_drugs:
    - ["Warfarin", "Aspirin"]
    - ["Metformin", "Iodinated_Contrast"]
    - ["Lisinopril", "Potassium_Supplements"]
    - ["Sildenafil", "Nitrates"]
    - ["Fluoxetine", "Tramadol"]
  procedure_diagnosis_compatibility:
    "99213": ["E11", "I10", "J06", "M54", "Z00"]
    "85025": ["D50", "D64", "R79", "Z01"]
  specialty_requirements:
    "27447": "Orthopedic Surgery"
    "33533": "Cardiothoracic Surgery"
    "47562": "General Surgery"
    "66984": "Ophthalmology"

temporal_rules:
  policy_effective_ranges:
    "RIDER_2018_A": ["2018-01-01", "2020-12-31"]
    "RIDER_2022_ENH": ["2022-01-01", "2025-12-31"]
    "POLICY_STANDARD_2025": ["2025-01-01", "2028-12-31"]

indexes:
  - "DiagnosisCode.code"
  - "ProcedureCode.code"
  - "Medication.name"
  - "Provider.id"
  - "Patient.id"
```

### 4.4 `dspy_config.yaml` — DSPy Optimization

```yaml
lm:
  provider: "vllm"
  endpoint: "http://localhost:8000/v1"
  model: "qwen2.5-7b-instruct"

modules:
  ClaimFieldExtractor:
    optimizer: "mipro"
    metric: "field_level_f1"
    baseline_score: 0.92
    degradation_threshold: 0.05
    max_bootstrapped_demos: 4
    max_labeled_demos: 8
  ArabicMedicalNormalizer:
    optimizer: "bootstrap_fewshot"
    metric: "exact_match"
    baseline_score: 0.88
    max_bootstrapped_demos: 8
  DiagnosisCodeMapper:
    optimizer: "mipro"
    metric: "code_accuracy"
    baseline_score: 0.90
    degradation_threshold: 0.05

supervisor:
  check_interval_minutes: 30
  performance_window_minutes: 60
  min_samples_for_reoptimize: 100
  max_reoptimize_per_day: 3

gradient_monitor:
  stability_threshold: 0.7
  window_size: 5
  divergence_alert_threshold: 0.3
```

### 4.5 `monitoring.yaml` — Observability & Traffic Control

```yaml
circuit_breaker:
  window_seconds: 300
  failure_rate_threshold: 0.15
  min_calls_in_window: 50
  recovery_timeout_seconds: 60
  half_open_max_calls: 10

traffic_controller:
  cheap_rail_confidence_threshold: 0.65
  handwriting_penalty: 0.15
  mixed_language_penalty: 0.02
  failure_classification_penalty: 0.10
  max_cheap_rail_ratio: 0.95
  min_cheap_rail_ratio: 0.70

accuracy_monitoring:
  target_accuracy: 0.98
  alert_threshold: 0.955
  decay_detection:
    window_minutes: 60
    min_samples: 200
    slope_threshold: -0.001

langsmith:
  project: "graphocr"
  tracing_enabled: true
  trace_sample_rate: 1.0

langfuse:
  enabled: false
  public_key: ""
  secret_key: ""
  host: "https://cloud.langfuse.com"
```

### 4.6 `rag.yaml` — RAG Configuration

```yaml
vector_store:
  backend: "chromadb"
  persist_dir: "./data/vectorstore"
  collection_name: "policy_chunks"

embeddings:
  model: "intfloat/multilingual-e5-large"
  dimension: 1024
  batch_size: 32

retriever:
  n_results: 5
  allow_semantic_fallback: true
  min_confidence_threshold: 0.3

chunking:
  max_chunk_size: 1000
  overlap: 100
  priority_sections:
    - coverage
    - exclusion
    - benefit_limit
    - preauth
```

---

## 5. Observability Stack

### 5.1 Langfuse Integration (`monitoring/langfuse_tracer.py`)

| Feature | Implementation |
|---|---|
| Trace creation | `create_trace(document_id, processing_path, metadata)` |
| Agent spans | `trace_agent_span(trace, agent_name, document_id, round_number)` — context manager recording latency |
| Score recording | `record_score(trace, name, value, comment)` — extraction confidence, accuracy |
| Flush | `flush()` — send pending events |

### 5.2 LangSmith Integration (`monitoring/langsmith_tracer.py`)

| Feature | Implementation |
|---|---|
| Configuration | Sets `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` env vars |
| Metadata tagging | `get_trace_metadata(document_id, processing_path, agent_role, round_number)` |
| Accuracy tracking | `AccuracyTracker` with rolling window and linear regression decay detection |

### 5.3 Metrics Collection (`monitoring/metrics_collector.py`)

Global `MetricsCollector` singleton tracking:
- Document counters (processed, cheap_rail, vlm_consensus, escalated)
- Self-healing counters (triggered, successful)
- Agent counters (challenges, violations)
- DSPy counters (optimizations, gradient alerts, accuracy decay events)
- Latency histograms (cheap_rail, vlm_consensus — avg, p95)

---

## 6. Pipeline Orchestration

**File**: `pipeline.py`
**Class**: `Pipeline`

### 6.1 Initialization

```python
Pipeline(
    ocr_lang="ar",          # PaddleOCR language
    use_surya=False,        # Enable Surya layout engine
    use_paddle=True,        # Enable PaddleOCR
    use_rag=True,           # Enable temporal RAG
)
```

### 6.2 Processing Flow

```python
async def process(file_path, run_llm=False, jurisdiction="") -> PipelineResult:
```

1. **Input Validation** — Load `RawDocument` (path, format, size, jurisdiction)
2. **Layer 1** — OCR + Spatial Foundation
   - `load_document()` → `list[PageImage]`
   - PaddleOCR → `list[SpatialToken]`
   - Surya layout (optional) → zone labels
   - `assemble_tokens()` → merged token stream
   - `enrich_tokens_with_zones()` → zone labels on tokens
   - `group_into_lines()` → line grouping
   - `assign_reading_order()` → global ordering
   - `assign_languages()` → per-token language
   - `classify_failures()` → Type A detection
   - `route_document()` → `RoutingDecision`
3. **Layer 2+3** (Routing + Verification, if `run_llm=True`)
   - RAG context retrieval (if enabled)
   - Circuit breaker check
   - Cheap rail OR VLM consensus
   - Claim assembly
   - Metrics recording + accuracy decay check
4. **Output** — `PipelineResult`

### 6.3 PipelineResult

```python
class PipelineResult:
    document_id: str
    source_path: str
    processed_at: datetime
    pages: list[PageImage]
    tokens: list[SpatialToken]
    lines: list[list[SpatialToken]]
    failures: list[FailureClassification]
    statistics: dict                    # total_tokens, ar/en counts, confidence range
    routing: RoutingDecision
    extraction: ExtractionResult | None
    claim: InsuranceClaim | None
    latency_ms: float
    layer1_ms: float
    layer23_ms: float | None
    error: str | None
```

**Methods**:
- `success` — True if no error and tokens > 0
- `text_dump()` — all text in reading order
- `provenance_dump(limit)` — first N token provenance strings
- `full_text_ordered()` — plain text concatenation
- `ocr_content()` — per-token metadata dicts
- `failure_details()` — structured failure info
- `summary()` — machine-readable dict for reports

### 6.4 Batch Processing

```python
async def process_batch(folder_path, run_llm=False, extensions=("pdf","tiff","jpeg","png"))
    -> list[PipelineResult]
```

---

## 7. Startup Sequence

On application startup (`app.py` lifespan):

1. Configure LangSmith tracing
2. Seed Neo4j with domain rules (`seed_neo4j.py`)
3. Start DSPy Supervisor background loop

On shutdown:
1. Stop DSPy Supervisor
2. Close connections

---

## 8. File-to-Architecture Mapping

| Component | Source Files |
|---|---|
| Document Ingestion | `layer1_foundation/ingestion.py` |
| Dual-Engine OCR | `layer1_foundation/ocr_engine.py`, `ocr_paddleocr.py`, `ocr_surya.py` |
| Spatial Assembly + Merge | `layer1_foundation/spatial_assembler.py`, `metadata_enricher.py` |
| Reading Order | `layer1_foundation/reading_order.py` |
| Language Detection | `layer1_foundation/language_detector.py` |
| Failure Classification | `layer1_foundation/failure_classifier.py` |
| SpatialToken Schema | `models/token.py` |
| Extractor Agent | `layer2_verification/agents/extractor.py` |
| Validator Agent | `layer2_verification/agents/validator.py` |
| Challenger Agent | `layer2_verification/agents/challenger.py` |
| Postmortem Agent | `layer2_verification/agents/postmortem.py` |
| Neo4j Rules | `layer2_verification/knowledge_graph/client.py`, `rule_engine.py`, `validators.py`, `schema_loader.py` |
| Self-Healing | `layer2_verification/self_healing/conflict_detector.py`, `vlm_rescanner.py`, `feedback_loop.py` |
| Temporal RAG | `rag/retriever.py`, `vector_store.py`, `policy_chunker.py`, `embeddings.py`, `context_injector.py` |
| Traffic Controller | `layer3_inference/traffic_controller.py` |
| Cheap Rail | `layer3_inference/cheap_rail.py` |
| VLM Consensus | `layer3_inference/vlm_consensus.py` |
| Circuit Breaker | `layer3_inference/circuit_breaker.py` |
| Output Assembly | `layer3_inference/output_assembler.py` |
| Monitoring | `monitoring/langfuse_tracer.py`, `langsmith_tracer.py`, `metrics_collector.py` |
| Failure Store | `layer2_verification/agents/failure_store.py` |
| DSPy Layer | `dspy_layer/modules.py`, `optimizers.py`, `metrics.py`, `supervisor.py`, `gradient_monitor.py` |
| Compliance | `compliance/jurisdiction.py`, `compliance/data_residency.py` |
| Pipeline Orchestrator | `pipeline.py` |
| API | `app.py` |
| CLI | `cli.py` |
| Configuration | `config/pipeline.yaml`, `agents.yaml`, `neo4j_rules.yaml`, `dspy_config.yaml`, `monitoring.yaml`, `rag.yaml` |
| Scripts | `scripts/batch_test.py`, `test_document.py`, `calibrate_threshold.py`, `seed_neo4j.py`, `ingest_policies.py` |
