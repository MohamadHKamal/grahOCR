# Hybrid Graph-OCR Pipeline

Deterministic Trust Layer for Insurance Claims — processes 100K multilingual (Arabic/English) scanned documents per day with full provenance tracking, knowledge graph validation, and self-healing.

**Documentation:**
- [Handwritten Draft (PDF)](HandWrittingArchOverview_draft%20planing.pdf) — Original handwritten architecture planning notes and flowcharts
- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) — Simplified architecture overview with diagrams (no implementation details)
- [ARCHITECTURE.md](ARCHITECTURE.md) — Full technical architecture with all thresholds, formulas, and code paths
- [TASK_RESPONSE.md](TASK_RESPONSE.md) — Direct answers to every question in the technical task brief
- [EVALUATION.md](EVALUATION.md) — Codebase evaluation against the problem brief
- [docs/](docs/README.md) — Deep-dive architecture docs for each subsystem (Layer 1, Layer 2, Layer 3, RAG, DSPy, Compliance, Infrastructure)

## Architecture

```
Layer 1: OCR + Metadata Foundation
  PaddleOCR (text) + Surya (text + layout) -> SpatialTokens -> Reading Order -> Language Detection

Layer 2: Tiered Inference + Routing
  Traffic Controller -> Cheap Rail (90%) / VLM Consensus (10%)
  Circuit Breaker + Langfuse Tracing + DSPy Supervisor

Layer 3: Adversarial Verification + Self-Healing
  Multi-Agent Red Team (LangGraph) + Neo4j Knowledge Graph + VLM Re-scan
  Agents: Extractor (Qwen2.5) -> Validator (+ Neo4j) -> Challenger (Llama-3.1)

RAG: Temporal-Aware Policy Retrieval
  ChromaDB + multilingual-e5-large -> Temporal Filter -> Context Injection
```

## Tech Stack

| Component | Tool |
|-----------|------|
| OCR | PaddleOCR (text), Surya (text + layout) |
| LLMs | Qwen2.5-7B, Llama-3.1-70B (via vLLM) |
| VLM | Qwen2-VL-7B |
| Agents | LangGraph |
| Knowledge Graph | Neo4j Community 5.x |
| RAG | ChromaDB + sentence-transformers |
| Prompt Optimization | DSPy (MIPRO) |
| Monitoring | Langfuse |
| API | FastAPI |
| Queue | Redis |
| Storage | MinIO (infra only) |

## Prerequisites

- Conda (Anaconda or Miniconda)
- Docker and Docker Compose
- GPU recommended (for OCR and LLM inference)

## Installation

### 1. Create the conda environment

```bash
git clone <repo-url> grahpOCR
cd grahpOCR

# Create and activate the environment
conda create -n graphocr python=3.11 -y
conda activate graphocr
```

### 2. Install all dependencies

```bash
# Install the project + all dependencies
pip install -e ".[dev]"

# Install poppler for PDF support (required for pdf2image)
conda install -c conda-forge poppler -y
```

This installs: PaddleOCR, Surya, LangGraph, LangChain, DSPy, Neo4j, ChromaDB, sentence-transformers, FastAPI, pytest, poppler, and more.

### 3. Verify the installation

```bash
# Run the test suite to confirm everything works
python -m pytest tests/ -v
```

You should see **147 passed**.

### 4. Configure your IDE

Set your IDE's Python interpreter to the conda environment:

```
/Users/<your-username>/anaconda3/envs/graphocr/bin/python
```

**VS Code**: `Cmd+Shift+P` -> "Python: Select Interpreter" -> select `graphocr`.

**PyCharm**: Settings -> Project -> Python Interpreter -> Add -> Conda Environment -> Existing -> select `graphocr`.

This fixes the "Import could not be resolved" errors in Pylance/Pyright.

### 5. Environment configuration

```bash
cp .env.example .env
# Edit .env with your settings (Neo4j password, Langfuse API key, etc.)
```

### 6. Start infrastructure

```bash
docker compose up -d
```

This starts:
- **Neo4j** on `localhost:7474` (browser) / `localhost:7687` (bolt)
- **Redis** on `localhost:6379`
- **MinIO** on `localhost:9000` (API) / `localhost:9001` (console)

### 7. Seed both databases (one command)

```bash
# Seeds Neo4j + ChromaDB if not already seeded — safe to run multiple times
make setup-dbs

# Or run the script directly
python scripts/setup_dbs.py
```

This checks both databases and only seeds them if they're empty:
- **Neo4j**: domain rules (drug dosage limits, procedure-diagnosis compatibility, contraindicated drug pairs, provider specialty requirements)
- **ChromaDB**: sample insurance policies from `tests/fixtures/sample_policies/`

You can also seed them individually:

```bash
# Neo4j only
python scripts/setup_dbs.py --neo4j
# or: python scripts/seed_neo4j.py

# ChromaDB only
python scripts/setup_dbs.py --chroma
# or: python scripts/ingest_policies.py --policy-dir tests/fixtures/sample_policies/
```

To ingest your own policies:
```bash
python scripts/ingest_policies.py --policy-dir /path/to/your/policies/
```

Policy files are JSON. See `tests/fixtures/sample_policies/` for the format.

### 8. Setup + test in one step

```bash
# Ensures DBs are seeded, then runs the full test suite
make setup-and-test
```

### 9. Start vLLM (requires GPU)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Or uncomment the `vllm` service in `docker-compose.yml` and run `docker compose up -d`.

## Usage

### Quick test — single document

```bash
conda activate graphocr

# Test an image (JPG, PNG, TIFF)
python scripts/test_document.py "sample data/images/1000093095.jpg"

# Test a PDF
python scripts/test_document.py "sample data/pdfs/test_column_issue.pdf"
```

Prints: extracted tokens, language tags, detected failures, routing decision, provenance trail, confidence stats.

### Batch test — folder of images

```bash
# Process all images and generate HTML + JSON reports
python scripts/batch_test.py "sample data/images" --format both

# Open the HTML report
open "sample data/images/graphocr_report.html"
```

### Batch test — folder of PDFs

```bash
# Process all PDFs and generate reports
python scripts/batch_test.py "sample data/pdfs" --format both

# Open the HTML report
open "sample data/pdfs/graphocr_report.html"
```

### Batch test options

```bash
# HTML report only (default)
python scripts/batch_test.py /path/to/folder

# JSON report only
python scripts/batch_test.py /path/to/folder -f json

# Both HTML and JSON
python scripts/batch_test.py /path/to/folder --format both

# Custom output path
python scripts/batch_test.py /path/to/folder -o my_report.html
```

Supported formats: PDF, PNG, JPEG, TIFF. The pipeline auto-detects the format.

### HTML report contents

Each report includes:
- **Summary cards**: document count, tokens, avg confidence, failures, routing split, avg latency
- **Summary table**: per-document stats at a glance
- **Per-document detail**: full OCR text, annotated token list, token detail table (text + language + confidence + bounding box + zone), failure descriptions with severity, and provenance trail

### Python API (Pipeline class)

```python
from graphocr.pipeline import Pipeline

pipeline = Pipeline()

# Single document (Layer 1 only — no LLM needed)
result = await pipeline.process("claim.jpg")
print(result.tokens)          # list[SpatialToken]
print(result.failures)        # list[FailureClassification]
print(result.routing)         # RoutingDecision
print(result.text_dump())     # All text in reading order
print(result.full_text_ordered())  # Plain text concatenation
print(result.summary())       # Full dict with OCR content for reports

# Batch processing
results = await pipeline.process_batch("sample data/images")
for r in results:
    print(f"{r.source_path}: {r.total_tokens} tokens, {len(r.failures)} failures")

# Full pipeline with LLM extraction (requires vLLM running)
result = await pipeline.process("claim.jpg", run_llm=True)
print(result.claim)           # InsuranceClaim
print(result.extraction)      # ExtractionResult
```

### Image Preprocessing Pipeline

The pipeline applies 6 preprocessing steps before OCR, optimized for Arabic medical documents:

| Step | What it does | Why it helps Arabic OCR |
|------|-------------|------------------------|
| **1. EXIF rotation** | Fixes phone camera metadata rotation | Pixels match what a human sees |
| **2. Orientation detection** | Detects and corrects 90/180/270 rotation via text line contour analysis | Handwritten prescriptions are often photographed sideways |
| **3. Auto-resize** | Scales to max 2500px | Prevents OOM; PaddleOCR optimal at 1500-2500px |
| **4. Stamp suppression** | Detects high-saturation colored pixels (HSV) and fades them | Pharmacy stamps (red/blue/green) overlap prescription text |
| **5. Lighting correction** | Morphological background estimation + division normalization | Phone camera shadows hide Arabic thin strokes |
| **6. Denoising (FNLM h=3)** | Fast Non-Local Means with mild strength | Removes paper texture; h=3 preserves Arabic dots and connections |
| **7. CLAHE (2.5)** | Adaptive contrast enhancement | Recovers faded ink and diacritical dots |
| **8. Deskew** | Projection profile analysis (not Hough — more robust for Arabic) | Even 1-2 degrees of skew misaligns Arabic detection boxes |

No binarization is applied — PaddleOCR uses gradient information internally. Pre-binarizing destroys this.

No manual pre-processing needed.

### CLI

```bash
# Process a single document
graphocr process /path/to/claim.pdf

# Process with verbose output
graphocr process /path/to/claim.pdf -v

# Output to file
graphocr process /path/to/claim.pdf --output result.json

# Seed Neo4j
graphocr seed-graph

# Check DSPy supervisor status
graphocr supervisor-status
```

### API Server

```bash
# Start the server
graphocr serve --host 0.0.0.0 --port 8080

# Or with auto-reload for development
graphocr serve --reload
```

#### Endpoints

**Process a document:**
```bash
curl -X POST http://localhost:8080/process \
  -F "file=@/path/to/claim.pdf"
```

Response:
```json
{
  "document_id": "...",
  "processing_path": "cheap_rail",
  "fields": {
    "patient_name": {"value": "محمد أحمد", "confidence": 0.92},
    "date_of_service": {"value": "2026-03-15", "confidence": 0.95},
    "total_amount": {"value": "1500.00", "confidence": 0.90},
    "diagnosis_codes": {"value": "E11.9", "confidence": 0.88}
  },
  "overall_confidence": 0.91,
  "violations": 0,
  "challenges": 0,
  "rounds": 1,
  "escalated": false,
  "latency_ms": 2340.5
}
```

**Health check:**
```bash
curl http://localhost:8080/health
```

**Pipeline metrics:**
```bash
curl http://localhost:8080/metrics
```

**DSPy supervisor status:**
```bash
curl http://localhost:8080/supervisor/status
```

**Audit — failure statistics (Type A/B breakdown):**
```bash
curl "http://localhost:8080/audit/stats?window_hours=24"
```

**Audit — failure detail by report ID:**
```bash
curl http://localhost:8080/audit/failure/{report_id}
```

**Audit — SpatialToken metadata schema:**
```bash
curl http://localhost:8080/audit/metadata-schema
```

**Audit — active learned rules from Post-Mortem:**
```bash
curl http://localhost:8080/audit/learned-rules
```

## Testing

### CLI test command

```bash
# Run all tests
graphocr test

# Run by category
graphocr test --unit              # Unit tests only (no external services)
graphocr test --integration       # Integration tests only
graphocr test --smoke             # Fast critical-path subset

# Run by layer
graphocr test --layer1            # Layer 1 — OCR Foundation
graphocr test --layer2            # Layer 2 — Verification
graphocr test --layer3            # Layer 3 — Inference

# Run by component
graphocr test --rag               # RAG retriever tests
graphocr test --dspy              # DSPy metrics tests

# Combine flags (OR logic)
graphocr test --layer1 --layer2   # Layer 1 + Layer 2
graphocr test --smoke --cov       # Smoke tests with coverage
graphocr test --unit --cov -v     # All unit tests, coverage, verbose
```

You should see **147 passed**.

### Test pipeline with OCR engines

```bash
# PaddleOCR only (default)
graphocr test-pipeline claim.pdf

# PaddleOCR + Surya (full accuracy — zone labels, confidence boost)
graphocr test-pipeline claim.pdf --surya

# Surya only (no PaddleOCR)
graphocr test-pipeline claim.pdf --no-paddle --surya

# Compare all 3 engine combos side-by-side
graphocr test-pipeline claim.pdf --all-engines

# Verbose output (token details + failure details)
graphocr test-pipeline claim.pdf --surya -v
```

### Make targets

```bash
make test              # All tests with coverage
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-smoke        # Fast smoke tests
make test-layer1       # Layer 1 tests
make test-layer2       # Layer 2 tests
make test-layer3       # Layer 3 tests
make test-rag          # RAG tests
make test-dspy         # DSPy tests
make test-cov          # Full coverage with HTML report
```

### pytest directly

```bash
# Run by marker
pytest tests/ -v -m "layer1 or layer2"
pytest tests/ -v -m smoke

# Run a specific test file
pytest tests/unit/test_token_model.py -v
pytest tests/unit/test_rag_retriever.py -v
```

The end-to-end test (`test_end_to_end.py`) walks a synthetic Arabic/English claim through all 3 layers and verifies the provenance chain is intact.

### Test real documents

```bash
# Test images (handwritten Arabic prescriptions)
python scripts/batch_test.py "sample data/images" --format both
open "sample data/images/graphocr_report.html"

# Test PDFs (typed English documents)
python scripts/batch_test.py "sample data/pdfs" --format both
open "sample data/pdfs/graphocr_report.html"
```

### Test files

| File | Markers | What it covers |
|------|---------|---------------|
| `test_token_model.py` | unit, layer1, smoke | BoundingBox IoU, SpatialToken creation, provenance strings |
| `test_spatial_assembler.py` | unit, layer1 | Multi-engine token merging, line grouping |
| `test_reading_order.py` | unit, layer1 | XY-Cut algorithm, RTL Arabic ordering |
| `test_failure_classifier.py` | unit, layer1 | Spatial jumps, stamp overlap, cross-column merge detection |
| `test_rule_engine.py` | unit, layer2 | Date sanity, amount validation, age plausibility |
| `test_self_healing.py` | unit, layer2 | Conflict detection, token patching, affected field identification |
| `test_traffic_controller.py` | unit, layer3, smoke | Uncertainty routing, handwriting/language penalties |
| `test_circuit_breaker.py` | unit, layer3 | Circuit breaker state transitions, failure rate tracking |
| `test_rag_retriever.py` | unit, rag | Policy reference extraction, temporal filtering, query building |
| `test_dspy_metrics.py` | unit, dspy | Field-level F1, code accuracy, Arabic normalization |
| `test_app.py` | unit, smoke | FastAPI endpoints: /health, /process, /metrics, /supervisor/status, /audit/* |
| `test_end_to_end.py` | integration, smoke | Full 3-layer pipeline with provenance chain verification |

### Linting

```bash
ruff check src/ tests/
ruff format --check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Calibrate traffic controller threshold

```bash
# Run ROC analysis with synthetic data
python scripts/calibrate_threshold.py --synthetic 1000

# Or with your own labeled dataset
python scripts/calibrate_threshold.py --dataset labeled_claims.json
```

## Project Structure

```
grahpOCR/
├── src/graphocr/
│   ├── core/                    # Config, types, exceptions, logging
│   ├── models/                  # Pydantic models (SpatialToken, Claim, Policy)
│   ├── layer1_foundation/       # OCR engines, spatial assembly, reading order
│   │                            # Auto-rotation, failure classification
│   ├── layer2_verification/
│   │   ├── agents/              # LangGraph red team (extractor, validator, challenger)
│   │   │                        # Post-mortem agent, failure store (Redis)
│   │   ├── knowledge_graph/     # Neo4j client, validators, rule engine
│   │   └── self_healing/        # Conflict detection, VLM rescan, feedback loop
│   ├── layer3_inference/        # Traffic controller, cheap rail, circuit breaker
│   ├── rag/                     # Policy retrieval (embeddings, vector store, retriever)
│   ├── dspy_layer/              # DSPy modules, optimizers, supervisor, gradient monitor
│   ├── monitoring/              # Langfuse/LangSmith tracing, metrics collector
│   ├── audit/                   # Failure analytics dashboard, failure analyzer
│   ├── compliance/              # Jurisdiction resolver, data residency
│   └── pipeline.py              # End-to-end orchestrator (single entry point)
├── config/                      # YAML configs (pipeline, agents, neo4j, dspy, rag)
├── tests/
│   ├── unit/                    # 10 unit test files (no external services)
│   ├── integration/             # End-to-end test (no external services)
│   └── fixtures/                # Sample claims, policies, Neo4j seeds
├── scripts/
│   ├── test_document.py         # Process a single document (Layer 1)
│   ├── batch_test.py            # Batch test + HTML/JSON report generation
│   ├── calibrate_threshold.py   # ROC analysis for traffic controller
│   ├── seed_neo4j.py            # Populate Neo4j with domain rules
│   └── ingest_policies.py       # Index policies into vector store
├── sample data/
│   ├── images/                  # Handwritten Arabic medical prescriptions (JPG)
│   └── pdfs/                    # Typed English test PDFs
├── docker-compose.yml           # Neo4j, Redis, MinIO
├── pyproject.toml
└── Makefile
```

## Make Commands

```bash
make install          # pip install -e .
make dev              # pip install -e ".[dev]"
make up               # docker compose up -d
make down             # docker compose down
make test             # pytest with coverage
make lint             # ruff check + format check
make format           # ruff auto-fix + format
make seed-neo4j       # populate Neo4j with domain rules
make seed-chroma      # populate ChromaDB with sample policies
make setup-dbs        # seed both DBs if not already seeded
make setup-and-test   # seed DBs + run full test suite
make serve            # start FastAPI server
```
