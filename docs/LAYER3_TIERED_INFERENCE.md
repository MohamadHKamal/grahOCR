# Layer 3 — Tiered Inference & Routing

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Traffic controller, uncertainty scoring, cheap rail, VLM consensus, circuit breaker, output assembly
**Problem Solved**: **Maintain 98% accuracy at scale** — a "Traffic Controller" that routes documents by uncertainty, keeping costs manageable while ensuring accuracy

---

## 1. Purpose & Problem Context

Processing 100k insurance claims per day with full multi-agent adversarial verification on every document is computationally prohibitive. The problem statement demands:
- A **"Traffic Controller" agent** that assigns an **"Uncertainty Score"** to every document
- A **mathematically defined threshold** that determines which ~10% get expensive VLM-Consensus vs ~90% standard rails
- An automated **"Circuit Breaker"** strategy if accuracy dips
- 98% accuracy target maintained at scale

---

## 2. Architecture Overview

```
SpatialTokens + FailureClassifications
    |
    v
[Traffic Controller]
    |
    ├── U <= 0.35 (~90% of documents)
    |       |
    |       v
    |   [Cheap Rail]
    |       ├── Regex extraction (dates, amounts, codes)
    |       ├── Single LLM call (Qwen2.5-7B, temp=0.1)
    |       └── Quick Neo4j validation
    |       |
    |       v
    |   ExtractionResult
    |
    └── U > 0.35 (~10% of documents)
            |
            v
        [VLM Consensus]
            ├── Full Red Team pipeline (Layer 3)
            ├── Extractor → Validator → Challenger
            ├── Self-healing loop (up to 2 rounds)
            └── 15-30 sec latency
            |
            v
        ExtractionResult (or Escalation)

    [Circuit Breaker] monitors both paths
    [Accuracy Tracker] detects gradual decay
```

**Source Files:**
| Component | File |
|---|---|
| Traffic Controller | `layer3_inference/traffic_controller.py` |
| Cheap Rail | `layer3_inference/cheap_rail.py` |
| VLM Consensus | `layer3_inference/vlm_consensus.py` |
| Circuit Breaker | `layer3_inference/circuit_breaker.py` |
| Output Assembler | `layer3_inference/output_assembler.py` |

---

## 3. Traffic Controller — Uncertainty Scoring

**Function**: `route_document(tokens, failures) -> RoutingDecision`

### 3.1 Uncertainty Score Formula

```
U = (1 - C_mean) + 0.15 * R_hw + 0.02 * R_mix + 0.10 * S_fail + 0.10 * H_norm
```

| Component | Symbol | Weight | Description |
|---|---|---|---|
| OCR Confidence Gap | `1 - C_mean` | 1.0 (base) | Mean OCR confidence inverted. Low confidence = high uncertainty |
| Handwriting Ratio | `R_hw` | **0.15** | Fraction of tokens flagged as handwritten |
| Language Mixing Entropy | `R_mix` | **0.02** | Shannon entropy of language distribution. Tuned down from 0.05 — mixed ar/en is normal for medical docs |
| Failure Severity | `S_fail` | **0.10** | Max severity from FailureClassifier. Tuned down from 0.20 — handwritten docs naturally have spatial irregularity |
| Confidence Entropy | `H_norm` | **0.10** | Normalized Shannon entropy of confidence distribution (discretized into 10 bins) |

### 3.2 Routing Threshold

```
T = 1.0 - cheap_rail_confidence_threshold
T = 1.0 - 0.65 = 0.35
```

| Condition | Path | Expected Volume | Latency |
|---|---|---|---|
| U <= 0.35 | **Cheap Rail** | ~90% | 2-3 sec |
| U > 0.35 | **VLM Consensus** | ~10% | 15-30 sec |

### 3.3 RoutingDecision Output

```python
class RoutingDecision:
    path: ProcessingPath          # CHEAP_RAIL or VLM_CONSENSUS
    uncertainty_score: float       # 0.0-1.0
    confidence_mean: float
    handwriting_ratio: float
    language_mixing_ratio: float
    failure_severity: float
    confidence_entropy: float
    reason: str                    # Human-readable explanation
```

### 3.4 Helper Functions

**Language Mixing** (`_language_mixing_ratio`): Shannon entropy over language distribution (ar/en/mixed/unknown counts)

**Confidence Entropy** (`_confidence_entropy`): Discretizes confidence values into 10 bins (0.0-0.1, 0.1-0.2, ...), computes Shannon entropy, normalizes by `log2(10)`

### 3.5 Configuration (`monitoring.yaml`)

```yaml
traffic_controller:
  cheap_rail_confidence_threshold: 0.65
  handwriting_penalty: 0.15
  mixed_language_penalty: 0.02
  failure_classification_penalty: 0.10
  max_cheap_rail_ratio: 0.95
  min_cheap_rail_ratio: 0.70
```

---

## 4. Cheap Rail — Fast Processing Path

**File**: `layer3_inference/cheap_rail.py`
**Function**: `process_cheap_rail(document_id, tokens, policy_context) -> ExtractionResult`

### 4.1 Three-Stage Pipeline

**Stage 1: Regex Extraction** (`_regex_extract`)

| Field | Pattern | Confidence |
|---|---|---|
| date_of_service | `\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}\|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b` | 0.9 |
| total_amount | `\b(\d{1,3}(?:[,. ]\d{3})*(?:\.\d{1,2})?)\b` | 0.9 |
| diagnosis_codes | `\b([A-Z]\d{2}(?:\.\d{1,4})?)\b` (ICD-10) | 0.9 |
| procedure_codes | `\b(\d{5})\b` (CPT) | 0.9 |

Regex fields take precedence — structured data is best captured deterministically.

**Stage 2: Single LLM Call**

- Model: `qwen2.5-7b-instruct`
- Temperature: 0.1
- Max tokens: 2048
- Extracts remaining fields: patient_name, patient_id, provider_name, provider_id, medications, currency, policy_reference

LLM-extracted fields get confidence 0.7 (lower than regex's 0.9).

**Stage 3: Quick Neo4j Validation**

Runs `run_all_validations()` — same graph checks as full pipeline but on cheap-rail extraction. Returns any graph violations for the output.

### 4.2 Performance

~2-3 seconds per document. Handles ~90% of the 100k daily volume.

---

## 5. VLM Consensus — Full Verification Path

**File**: `layer3_inference/vlm_consensus.py`
**Function**: `process_vlm_consensus(document_id, tokens, page_images, policy_contexts, ...) -> ExtractionResult`

Calls the full Layer 3 red-team pipeline (`run_red_team()` from `agents/graph_builder.py`):
- Extractor → Validator → Challenger → Consensus Check
- Self-healing loop (up to `max_rounds=2`)
- VLM visual verification of challenges
- Full postmortem analysis

### 5.1 Inputs

- `page_images: dict[int, str]` — page_number → image path (for VLM visual verification)
- `policy_context_extractor/validator/challenger` — role-specific RAG context formatting
- `retrieval_method` + `retrieval_warnings` — RAG audit metadata

### 5.2 Output

`ExtractionResult` with:
- `rounds_taken: int` (1-3)
- `escalated: bool` (True if max rounds exceeded)
- `overall_confidence: float`
- Full extracted fields with per-field confidence
- Agent consensus metadata

### 5.3 Performance

~15-30 seconds per document. Handles ~10% of daily volume (the uncertain/complex ones).

---

## 6. Circuit Breaker — Automated Failure Response

**File**: `layer3_inference/circuit_breaker.py`

### 6.1 Three-State Machine

```
CLOSED ──(failure_rate >= 15% over 50+ calls)──> OPEN
  ^                                                  |
  |                                              (60s timeout)
  |                                                  |
  └──(10 consecutive successes)── HALF_OPEN <────────┘
                                      |
                                  (any failure)
                                      |
                                      └──> OPEN
```

### 6.2 Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `window_seconds` | 300 (5 min) | Sliding window for failure rate calculation |
| `failure_rate_threshold` | 0.15 (15%) | Triggers OPEN state |
| `min_calls_in_window` | 50 | Minimum calls before evaluating rate |
| `recovery_timeout` | 60 seconds | Before transitioning OPEN → HALF_OPEN |
| `half_open_max_calls` | 10 | Test calls allowed in HALF_OPEN |

### 6.3 Fallback Strategy

| Path | When OPEN | Action |
|---|---|---|
| Cheap Rail | Circuit opens | Fallback to VLM Consensus |
| VLM Consensus | Circuit opens | Escalate (return None, queue for human review) |

### 6.4 Implementation

- **Thread-safe** with `threading.Lock`
- **Global registry**: `CircuitBreakerRegistry` with lazy instantiation per path name
- Records success/failure after each extraction
- `check()` raises `CircuitBreakerOpenError` if OPEN
- `metrics` property exposes current state, failure rate, call counts

---

## 7. Output Assembler

**File**: `layer3_inference/output_assembler.py`
**Function**: `assemble_claim(extraction, tokens) -> InsuranceClaim`

Converts raw `ExtractionResult` into a typed `InsuranceClaim`:

| Field | Parsing |
|---|---|
| Dates | `date.fromisoformat()` |
| Codes | Split comma-separated lists |
| Amounts | `Decimal` (handles comma/space separators) |
| Medications | JSON array → `list[MedicationEntry]` (name, dosage, frequency) |
| Currency | Default: "SAR" (Saudi Riyal) |

---

## 8. Accuracy Monitoring & Decay Detection

**File**: `monitoring/langsmith_tracer.py`
**Class**: `AccuracyTracker`

### 8.1 Rolling Window

Maintains a rolling window of `(timestamp, correct)` tuples (window_size=1000).

```python
accuracy = correct_count / total_count  # in recent window
```

### 8.2 Decay Detection

```python
detect_decay(window_minutes=60, slope_threshold=-0.001)
```

Uses **linear regression** over the 60-minute window:
- If slope < -0.001 → signals systemic degradation
- Combined with the circuit breaker, this automates the **"2:00 PM accuracy dip"** response from the problem statement

### 8.3 Configuration (`monitoring.yaml`)

```yaml
accuracy_monitoring:
  target_accuracy: 0.98
  alert_threshold: 0.955
  decay_detection:
    window_minutes: 60
    min_samples: 200
    slope_threshold: -0.001
```

---

## 9. Metrics Collection

**File**: `monitoring/metrics_collector.py`
**Class**: `MetricsCollector`

### Tracked Counters

| Counter | Description |
|---|---|
| `documents_processed` | Total documents through pipeline |
| `cheap_rail` | Documents routed to cheap rail |
| `vlm_consensus` | Documents routed to VLM consensus |
| `escalated` | Documents escalated to human review |
| `healing_triggered` | Self-healing loops initiated |
| `healing_successful` | Self-healing loops that resolved |
| `challenges_raised` | Adversarial challenges generated |
| `graph_violations` | Neo4j rule violations detected |
| `dspy_optimizations` | DSPy reoptimization runs |
| `gradient_alerts` | Gradient stability alerts |
| `accuracy_decay_events` | Accuracy decay detections |

### Latency Tracking

- `cheap_rail` and `vlm_consensus` latency histories (ms)
- Auto-prunes to last 5000 measurements (when >10000)

### Output: `PipelineMetrics`

```python
class PipelineMetrics:
    documents_per_minute: float
    # All counters above
    accuracy: float
    avg_latency_cheap: float
    avg_latency_vlm: float
    p95_latency: float
```

Exposed via `GET /metrics` API endpoint.

---

## 10. How Layer 2 Connects to the Problem Statement

| Problem Statement Requirement | Layer 2 Solution |
|---|---|
| "Traffic Controller agent using low-cost model to assign Uncertainty Score" | `route_document()` computes U from 5 weighted components, routes to cheap/VLM path |
| "Mathematically define the Threshold for 10% expensive vs 90% standard" | `U = (1-C̄) + 0.15·R_hw + 0.02·R_mix + 0.10·S_fail + 0.10·H_norm`, T=0.35 |
| "Monitor Accuracy Decay in real-time" | AccuracyTracker with linear regression over 60-min window, slope threshold -0.001 |
| "If accuracy dips to 95.5% at 2:00 PM, automated Circuit Breaker" | 3-state circuit breaker: CLOSED→OPEN at 15% failure rate, auto-recovers via HALF_OPEN |
| "Maintain 98% accuracy at scale" | Target accuracy 0.98 in config, alert at 0.955, automated decay response |
| "Deterministic, not better prompting" | Mathematical uncertainty formula, configurable thresholds, automated circuit breaker — no manual intervention |
