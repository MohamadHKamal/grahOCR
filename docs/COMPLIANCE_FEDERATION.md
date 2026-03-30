# Compliance — Federated Architecture & Sovereign Data

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Jurisdictional routing, data residency enforcement, federated learning, PII protection
**Problem Solved**: **Federated systems that respect sovereign data constraints** — health data must never leave jurisdictional boundaries

---

## 1. Purpose & Problem Context

The problem statement requires: *"Assume you are using Federated systems that respect sovereign data constraints."*

Health data in the Global South is subject to strict jurisdictional sovereignty laws:
- **Saudi Arabia**: Data must be processed and stored within Saudi borders
- **UAE**: Similar data residency requirements
- **HIPAA** (US-originated claims): Protected health information constraints
- **GDPR** (EU policyholders): Right to data locality

The compliance layer ensures that every document is processed exclusively within infrastructure zones that satisfy its data residency requirements. A claim originating from Saudi Arabia is processed on Saudi-resident infrastructure — it never leaves the jurisdiction boundary, even for VLM re-scan.

---

## 2. Architecture Overview

```
Incoming Document
    |
    v
[Jurisdiction Resolver] ── jurisdiction_code from policy/claim metadata
    |
    ├── validate_processing_region() ── is target region allowed?
    |
    ├── get_storage_bucket() ── route to jurisdiction-local MinIO
    |
    └── Pipeline processes on jurisdiction-local infrastructure
            |
            ├── vLLM endpoint: jurisdiction-local
            ├── MinIO storage: jurisdiction-local bucket
            ├── Neo4j: jurisdiction-scoped database
            └── Redis: jurisdiction-local instance
    |
    v
[Postmortem Learning]
    |
    ├── LearnedRule: scoped to originating jurisdiction
    └── filter_shareable_patterns() ── only non-PII aggregate stats shared globally
```

**Source Files:**
| Component | File |
|---|---|
| Jurisdiction Rules | `compliance/jurisdiction.py` |
| Data Residency Enforcement | `compliance/data_residency.py` |

---

## 3. Jurisdiction Rules

**File**: `compliance/jurisdiction.py`
**Function**: `resolve_jurisdiction(jurisdiction_code) -> dict`

### 3.1 Rules Map

```python
_JURISDICTION_RULES = {
    "SA": {
        "name": "Saudi Arabia",
        "allowed_regions": ["sa-riyadh", "sa-jeddah"],
        "requires_local_processing": True,
        "data_classification": "sensitive"
    },
    "AE": {
        "name": "United Arab Emirates",
        "allowed_regions": ["ae-dubai", "ae-abudhabi"],
        "requires_local_processing": True,
        "data_classification": "sensitive"
    },
    "EG": {
        "name": "Egypt",
        "allowed_regions": ["eg-cairo"],
        "requires_local_processing": True,
        "data_classification": "standard"
    },
    "JO": {
        "name": "Jordan",
        "allowed_regions": ["jo-amman"],
        "requires_local_processing": False,
        "data_classification": "standard"
    },
}
```

### 3.2 Validation

```python
def validate_processing_region(jurisdiction_code: str, processing_region: str) -> None:
    rules = resolve_jurisdiction(jurisdiction_code)
    if processing_region not in rules["allowed_regions"]:
        raise DataResidencyError(
            f"Region {processing_region} not allowed for jurisdiction {jurisdiction_code}"
        )
```

**Default behavior**: Unknown jurisdictions default to **strict** (`requires_local_processing=True`) — fail closed, not open.

---

## 4. Data Residency Enforcement

**File**: `compliance/data_residency.py`

### 4.1 Storage Locality

```python
def get_storage_bucket(document: RawDocument) -> str:
    return f"{base_bucket}-{document.jurisdiction.lower()}"
    # e.g., "claims-sa", "claims-ae", "claims-eg"
```

MinIO object storage is **partitioned by jurisdiction**. Page images and intermediate SpatialToken streams are written only to the jurisdiction-local MinIO instance.

### 4.2 Processing Locality

```python
def validate_document_routing(document: RawDocument, target_region: str) -> bool:
    rules = resolve_jurisdiction(document.jurisdiction)
    if target_region not in rules["allowed_regions"]:
        logger.warning("routing_blocked", jurisdiction=document.jurisdiction,
                       target_region=target_region)
        return False
    return True
```

vLLM inference endpoints are deployed per-region. The pipeline configuration maps each jurisdiction to its local vLLM endpoint, ensuring **no raw document data crosses sovereign boundaries** during LLM inference.

### 4.3 Federated Pattern Sharing

```python
def filter_shareable_patterns(patterns: dict) -> dict:
    SHAREABLE_KEYS = [
        "ocr_confidence_distribution",
        "layout_patterns",
        "failure_type_distribution",
        "language_distribution",
        "zone_distribution",
        "reading_order_patterns",
        "prompt_performance_metrics",
    ]
    return {k: v for k, v in patterns.items() if k in SHAREABLE_KEYS}
```

**Federated model**: Raw data stays local. Only **non-PII aggregate statistics** are shared globally:
- OCR confidence distributions (not actual text)
- Layout patterns (column counts, zone ratios)
- Failure type distributions (Type A vs Type B rates)
- Language distributions (ar/en ratios)
- Prompt performance metrics (F1 scores, not examples)

This enables global model improvement without exposing patient data.

---

## 5. Three Residency Constraints

### Constraint 1: Storage Locality

| What | Where | Enforcement |
|---|---|---|
| Page images (PNG) | Jurisdiction-local MinIO bucket | `get_storage_bucket()` |
| SpatialToken streams | Jurisdiction-local MinIO bucket | Same |
| FailureReports | Jurisdiction-local Redis instance | Redis deployment per region |
| Extraction results | Jurisdiction-local storage | Pipeline routing |

### Constraint 2: Processing Locality

| What | Where | Enforcement |
|---|---|---|
| vLLM inference (Qwen, Llama) | Jurisdiction-local GPU cluster | vLLM endpoint per region in config |
| VLM re-scan (Qwen2-VL) | Jurisdiction-local GPU cluster | Same endpoint |
| OCR processing (PaddleOCR, Surya) | Jurisdiction-local compute | Pipeline deployment per region |
| Neo4j validation queries | Jurisdiction-local Neo4j instance | `neo4j_uri` per region |

### Constraint 3: Graph Isolation

| What | Scope | Cross-Jurisdiction? |
|---|---|---|
| Policy rules | Per-jurisdiction Neo4j database | No |
| Provider registry | Per-jurisdiction | Read-only projections for multi-country providers (non-PII metadata only) |
| LearnedRule nodes | Scoped to originating jurisdiction by default | Promotion via compliance-approved review queue |
| Drug dosage limits | Universal (not jurisdiction-specific) | Shared globally (medical facts, not PII) |

---

## 6. Federated Self-Healing

### Default: Jurisdiction-Scoped

When the Postmortem Agent creates a `LearnedRule` node (e.g., "patient name 'Mohaned' is commonly misread for 'Mohamed'"), the rule is created **only in the originating jurisdiction's Neo4j instance**.

### Promotion: Global Rules

A compliance review queue allows approved rules to be promoted across jurisdictions when they represent **universal medical/logical constraints**:

| Promotable (Universal) | Not Promotable (Jurisdiction-Specific) |
|---|---|
| Drug dosage limits (clinical stoichiometric) | Policy-specific coverage rules |
| Contraindicated drug combinations | Jurisdiction-specific amount thresholds |
| ICD-10/CPT code mappings | Provider credentialing data |
| Arabic OCR character confusion patterns | Patient data patterns |
| Date format conventions | Jurisdiction-specific document layouts |

---

## 7. Audit Trail

Every `SpatialToken` and `ExtractionResult` carries a **`jurisdiction_code`** field. The Langfuse tracer logs jurisdiction routing decisions, enabling compliance auditors to verify:

1. No data crossed a sovereignty boundary during processing
2. Storage was in the correct jurisdiction-local bucket
3. LLM inference used the jurisdiction-local vLLM endpoint
4. Learned rules are properly scoped

---

## 8. Pipeline Integration

The compliance layer integrates at the **entry point** of the pipeline (`pipeline.py`):

```python
async def process(file_path, run_llm=False, jurisdiction=""):
    # 1. Load document with jurisdiction tag
    raw_doc = RawDocument(path=file_path, jurisdiction=jurisdiction)

    # 2. Validate processing region
    validate_document_routing(raw_doc, current_region)

    # 3. Route storage to jurisdiction-local bucket
    bucket = get_storage_bucket(raw_doc)

    # 4. Process on jurisdiction-local infrastructure
    # ... (all LLM calls, Neo4j queries, MinIO storage use local endpoints)
```

### API Integration

- `POST /process` — accepts `jurisdiction` parameter
- `GET /audit/stats?jurisdiction=SA` — filtered audit statistics
- `GET /audit/failure/{id}` — includes jurisdiction in report

---

## 9. How Compliance Connects to the Problem Statement

| Problem Statement Requirement | Compliance Solution |
|---|---|
| "Federated systems that respect sovereign data constraints" | Per-jurisdiction routing, storage, processing, and graph isolation |
| "Single Source of Truth for both Senior and Junior engineers" | SpatialToken carries `jurisdiction_code` — same schema, jurisdiction-aware |
| "Metadata schema consistent across the whole team's work" | Jurisdiction is a first-class field in the token/extraction schema |
| "Self-Healing pipeline" | Federated self-healing: rules scoped to jurisdiction, promotable via review |
| "100k docs/day" | Each jurisdiction processes its own volume independently — horizontal scaling by region |
