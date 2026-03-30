# Technical Task Response — Hybrid Graph-OCR Deterministic Trust Layer

**Author**: Mohamed Hussein
**Date**: 2026-03-29
**Repository**: [grahpOCR](https://github.com/YOUR_USERNAME/grahpOCR)
**Architecture Document**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Task 1: Audit — Diagnostic Tool & Metadata Schema

### Q: How do you architect the diagnostic/evaluation tool between Input Failure (OCR messed up the reading order) and Intelligence Failure (RAG grabbed the 2025 policy instead of the 2018 rider)?

The diagnostic tool is split across two layers because the two failure types are fundamentally different and detectable at different stages:

**Input Failure (Type A — Spatial-Blind OCR)** is detected at Layer 1 by `failure_classifier.py` before any LLM or RAG is involved. Four concrete detectors run on the raw SpatialToken stream:

| Detector | What it catches | How it works |
|----------|----------------|-------------|
| Spatial jump | OCR read across columns instead of down | Consecutive tokens in reading order >800px apart spatially, both with high OCR confidence |
| Nonsensical sequences | "Serialization gore" — dates merged wigth prices | Sliding window of 5 tokens checking for rapid alternation between numeric and text types |
| Stamp overlap | Pharmacy stamp overlaps policy number | Bounding box IoU > 0.05 between STAMP zone tokens and BODY zone tokens |
| Cross-column merge | Two-column layout incorrectly serialized as one | Clusters tokens into left/right by X-center, counts column switches in reading order |

Each detector produces a `FailureClassification` with: `failure_type`, `affected_tokens` (specific token IDs), `severity` (0.0-1.0), `suggested_remedy` ("vlm_rescan" or "escalate"), and `evidence` (human-readable explanation).

**Intelligence Failure (Type B — Context-Blind RAG)** is detected at Layer 2 by the temporal RAG retriever (`rag/retriever.py`). The core insight: a naive retriever matches by semantic similarity alone, so a 2025 Standard Plan always outranks the correct 2018 Rider because their text is similar. Our retriever implements a 3-stage strategy:

1. **EXTRACT**: Regex-extract the policy reference and date from OCR tokens (supports both English "Policy No: SA-2018-R3" and Arabic "بوليصة رقم: ...")
2. **FILTER**: Narrow the ChromaDB vector store to only policies where `effective_date <= claim_date <= expiry_date`
3. **RANK**: Semantic similarity within the temporally-filtered set

The retriever degrades gracefully through 4 confidence levels:

| Mode | Condition | Confidence | Risk |
|------|-----------|------------|------|
| temporal_filtered | Reference + date found | 0.9 | Low |
| temporal_semantic_hybrid | Date only, no reference | 0.7 | Medium |
| reference_only | Reference only, no date | 0.5 | Medium-High |
| semantic_only | Neither found | 0.3 | **Type B risk — explicit warning** |

When the retriever falls back to `semantic_only`, it generates a `GraphViolation` with `rule_name="policy_retrieval_quality"` so the Validator and Challenger agents know to question the policy match.

**Code**: `src/graphocr/layer1_foundation/failure_classifier.py`, `src/graphocr/rag/retriever.py`, `src/graphocr/rag/context_injector.py`

---

### Q: How do you ensure this tool works as a "Single Source of Truth" for both Senior and Junior engineers?

The **SpatialToken** (`models/token.py`) is the Single Source of Truth. Every token extracted by OCR — regardless of which engine produced it — is normalized into the same schema:

```
SpatialToken:
  token_id:        UUID7 (time-ordered, globally unique)
  text:            Raw OCR text
  bbox:            BoundingBox (x_min, y_min, x_max, y_max, page_number)
  reading_order:   Global sequence index
  language:        ARABIC | ENGLISH | MIXED | UNKNOWN
  confidence:      0.0-1.0 from OCR engine
  ocr_engine:      "paddleocr" | "surya" | "vlm_rescan"
  zone_label:      HEADER | BODY | STAMP | SIGNATURE | TABLE_CELL
  is_handwritten:  bool
  line_group_id:   Groups tokens into logical lines
  normalized_text: Post-normalization (Arabic diacritics removed)
```

Every downstream component consumes this schema. Every extracted claim field carries `source_tokens` (list of token IDs). The provenance chain is unbroken: `pixel coordinates → SpatialToken → FieldExtraction → InsuranceClaim`.

Both senior and junior engineers work with the same `SpatialToken` objects — there is no separate "senior" or "junior" data format. The `to_provenance_str()` method generates audit-ready strings:

```
[a1b2c3d4] 'metformin' page=1 (234,567)-(345,589) conf=0.92 lang=en engine=paddleocr
```

**Code**: `src/graphocr/models/token.py`, `src/graphocr/models/extraction.py`

---

### Q: What is the metadata schema you'd enforce to ensure the "Semantic Spatial" mapping is consistent across the whole team's work?

The schema is the `SpatialToken` plus `BoundingBox` Pydantic models above. Consistency is enforced by:

1. **Pydantic validation**: All fields are type-checked at runtime. `confidence` must be in [0.0, 1.0], `language` must be a valid enum, etc.
2. **Single entry point**: All OCR engines (PaddleOCR, Surya) must output `list[SpatialToken]` through the `OCREngine` abstract base class. No raw engine-specific formats leak into the pipeline.
3. **Spatial assembler**: Merges PaddleOCR text tokens with Surya region bboxes via IoU merge (threshold 0.3). Overlapping tokens are combined — text from PaddleOCR, zone labels from Surya, confidence boosted by agreement (`1 - (1-a)(1-b)`), bounding box via union or intersection.
4. **Reading order assignment**: The XY-Cut algorithm assigns a global `reading_order` index. This is the canonical serialization — all downstream processing uses this ordering.

---

### Q: How do you handle Federated systems that respect sovereign data constraints?

Two modules enforce data sovereignty:

**Jurisdiction resolver** (`compliance/jurisdiction.py`): Maps country codes (SA, AE, EG, JO) to rules — which processing regions are allowed, whether local-only processing is required, and data classification level.

**Data residency enforcer** (`compliance/data_residency.py`):
- `get_storage_bucket()`: Routes documents to jurisdiction-specific MinIO buckets (e.g., `claims-sa`, `claims-ae`)
- `validate_document_routing()`: Blocks routing to unauthorized regions
- `filter_shareable_patterns()`: In federated mode, only aggregate statistics (OCR confidence distributions, layout patterns, failure type distributions) are shared globally — never PII or raw text

**Code**: `src/graphocr/compliance/jurisdiction.py`, `src/graphocr/compliance/data_residency.py`

---

## Task 2: Adversarial — Multi-Agent Red Team & Self-Healing

### Q: Architect a layered verification pipeline. Not passive "LLM as a judge" but a multi-agent red team.

The pipeline uses **4 agents with distinct roles, different LLM models, and different temperatures** wired into a LangGraph cyclic state machine:

| Agent | LLM | Temperature | Role |
|-------|-----|-------------|------|
| Extractor | Qwen2.5-7B (cheap) / Llama-3.1-70B (VLM) | 0.1 | Extract structured fields with provenance |
| Validator | Qwen2.5-7B | 0.0 (deterministic) | Run Neo4j rules + consistency checks |
| Challenger | **Llama-3.1-70B** (different model) | **0.3** (creative) | Adversarially attack the extraction |
| PostMortem | Qwen2.5-7B | 0.0 | Classify root cause, update system logic |

This is NOT passive scoring. The Challenger agent uses 7 adversarial strategies specific to Arabic medical documents:
1. Arabic character confusion (ع vs غ, ي vs ى)
2. OCR digit errors (3 vs 8, 1 vs 7)
3. Stamp obscuration
4. Merged line items
5. Date format ambiguity (DD/MM vs MM/DD)
6. Currency symbol misread
7. Handwriting ambiguity

The Challenger uses a **different LLM** (Llama-3.1-70B) from the Extractor (Qwen2.5-7B) to ensure genuine model diversity — they don't share the same blind spots.

**State machine flow**:
```
Extractor → Validator (+ Neo4j) → Challenger → Consensus Check
  ├─ Agreed → Output Assembly → PostMortem → END
  └─ Disagreed → Self-Healing (VLM rescan) → Extractor (retry, max 2 rounds)
                                             → Escalate (if exhausted) → PostMortem → END
```

**Code**: `src/graphocr/layer2_verification/agents/graph_builder.py`

---

### Q: If the "Adjuster" agent gets tricked, how do you architect an automated "Post-Mortem Agent" that updates the system's core logic to prevent that specific failure pattern globally?

The PostMortem Agent (`agents/postmortem.py`) activates after every processing run (both successful consensus and escalations). It:

1. **Classifies root cause** into 4 categories: `ocr_misread`, `prompt_failure`, `rule_gap`, `layout_confusion`
2. **Persists FailureReport** to Redis FailureStore with 30-day TTL. Each report contains: original_value, corrected_value, affected_field, root_cause, resolution_method
3. **Updates Neo4j** when root_cause = "rule_gap": Creates a `LearnedRule` node directly in the knowledge graph so future claims are caught by the validator
4. **Tags for DSPy training**: Sets `add_to_dspy_training=True` so the DSPy Supervisor pulls the case as a training example during the next re-optimization cycle

The DSPy Supervisor runs every 30 minutes. It:
- Fetches training data from the FailureStore
- Converts FailureReport objects to `dspy.Example` objects (module-specific mapping)
- Runs MIPRO optimization (max 3/day)
- Validates the new prompt against a held-out set
- Atomically swaps the optimized prompt

This is fully automated — no human intervention. The pipeline gets stronger with every failure it encounters.

**Code**: `src/graphocr/layer2_verification/agents/postmortem.py`, `src/graphocr/layer2_verification/agents/failure_store.py`, `src/graphocr/dspy_layer/supervisor.py`

---

### Q: How do you use the Neo4j Knowledge Graph to detect "Logical Impossibilities" in OCR output?

The Validator Agent executes **6 categories of constraint checks** against Neo4j:

| Rule | Example Catch | Cypher Pattern | Severity |
|------|--------------|----------------|----------|
| Procedure-Diagnosis compatibility | Knee MRI billed with diabetes diagnosis | Check `valid_diagnosis_prefixes` on ProcedureCode node | 0.8 |
| Drug dosage limits | Metformin 5000mg/day (max is 2550) | Compare `daily_dosage_mg` vs `max_daily_dosage_mg` | 0.9 |
| Contraindicated drug combinations | Warfarin + Aspirin prescribed together | Traverse `CONTRAINDICATED_WITH` edges | 0.95 |
| Provider specialty | General practitioner performing cardiac surgery | Check `REQUIRES_SPECIALTY` vs `HAS_SPECIALTY` | 0.7 |
| Date sanity | 1940 birthdate on 2026 policy | Age 0-130, DOB < service date, no future dates | 0.9 |
| Amount reasonableness | Negative total, line items don't sum | Computed total vs stated total, threshold checks | 0.6-0.9 |

The graph stores **domain rules as relationships**, not patient data:
- `(:ProcedureCode)-[:COMPATIBLE_WITH]->(:DiagnosisCode)`
- `(:Medication)-[:CONTRAINDICATED_WITH]->(:Medication)`
- `(:ProcedureCode)-[:REQUIRES_SPECIALTY]->(:Specialty)`
- `(:LearnedRule)` — auto-created by PostMortem when rule gaps are found

**Code**: `src/graphocr/layer2_verification/knowledge_graph/validators.py`, `src/graphocr/layer2_verification/knowledge_graph/rule_engine.py`

---

### Q: Describe the "Back-Propagation" mechanism: When the Graph identifies a logical conflict, how does the system automatically trigger a targeted VLM Re-Scan of the specific source coordinates without restarting the entire 100k-doc batch?

The self-healing loop operates at the **individual document level**, not the batch level:

**Phase 1 — Conflict Detection** (`conflict_detector.py`):
- Collects `token_id`s from: high-confidence challenges (>0.6), severe graph violations (>0.7), low-confidence extractions (<0.5)
- Looks up each token's `BoundingBox` coordinates
- Groups by page, merges nearby tokens with 50px padding into minimal re-scan regions

**Phase 2 — Targeted VLM Re-Scan** (`vlm_rescanner.py`):
- Crops the **exact conflicting region** from the original page image using PIL
- Encodes the cropped region to base64
- Sends to Qwen2-VL-7B with a specialized prompt targeting Arabic character confusion, digit misreads, stamp obscuration, and handwriting
- Returns new `SpatialToken` objects with `ocr_engine="vlm_rescan"`

**Phase 3 — Token Patch** (`feedback_loop.py`):
- Removes old tokens in the conflicting region from the token stream
- Inserts new VLM tokens
- Re-sorts by spatial position, re-assigns reading order
- Identifies **only the affected fields** via `field_token_map` and re-extracts only those fields

**The batch is never restarted.** Only the specific document, specific page region, and specific fields are re-processed. The loop runs max 2 rounds before escalation.

**Code**: `src/graphocr/layer2_verification/self_healing/conflict_detector.py`, `src/graphocr/layer2_verification/self_healing/vlm_rescanner.py`, `src/graphocr/layer2_verification/self_healing/feedback_loop.py`

---

## Task 3: Tiered Inference Pipeline & Monitoring

### Q: How do you mathematically define the "Threshold" that determines which 10% of docs get the expensive VLM-Consensus treatment versus the 90% that pass through standard rails?

The Traffic Controller (`traffic_controller.py`) computes an **Uncertainty Score U** for every document using a weighted linear combination of 5 measurable signals from Layer 1:

```
U = (1 - C̄) + 0.15 · R_hw + 0.02 · R_mix + 0.10 · S_fail + 0.10 · H_norm
```

| Component | Symbol | What it measures | Weight |
|-----------|--------|-----------------|--------|
| Mean OCR confidence | C̄ | Overall extraction quality | 1.0 (inverted) |
| Handwriting ratio | R_hw | Fraction of handwritten tokens | 0.15 |
| Language mixing entropy | R_mix | Arabic/English mixing level | 0.02 |
| Max failure severity | S_fail | Worst spatial failure detected | 0.10 |
| Confidence entropy | H_norm | Spread of confidence distribution | 0.10 |

**Threshold selection**: T is set via ROC analysis on a labeled calibration dataset (`scripts/calibrate_threshold.py`). The optimal T maximizes the Youden index J = sensitivity + specificity - 1. Default T = 0.35.

- U <= T → **Cheap Rail** (~90% of traffic): Regex + single Qwen2.5-7B call + quick Neo4j check
- U > T → **VLM Consensus** (~10% of traffic): Full 4-agent red team pipeline with adversarial challenges

The threshold is **not a magic number** — it is empirically derived and recalibrable:
```bash
python scripts/calibrate_threshold.py --synthetic 1000
```

**Code**: `src/graphocr/layer3_inference/traffic_controller.py`, `scripts/calibrate_threshold.py`

---

### Q: How do you use Langfuse at this scale to monitor "Accuracy Decay" in real-time? If accuracy dips to 95.5% at 2:00 PM, what is your automated "Circuit Breaker" strategy?

We use **LangSmith** (functionally equivalent to Langfuse, native LangGraph integration). Two monitoring systems work in parallel:

**1. AccuracyTracker** (`monitoring/langsmith_tracer.py`):
- Rolling window of 1000 most recent results
- `detect_decay()` uses **linear regression** over a 60-minute window
- If the slope of accuracy over time is negative beyond -0.001, it signals systemic degradation — catching the "2:00 PM accuracy dip" scenario

**2. Circuit Breaker** (`layer3_inference/circuit_breaker.py`):
Three-state sliding window pattern:

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation. Tracks success/failure in a 300-second window |
| **OPEN** | Path disabled. Triggered when failure rate >= 15% over 50+ calls. All traffic rerouted to alternate path |
| **HALF_OPEN** | Testing recovery. After 60-second timeout, allows 10 test calls. Any failure → back to OPEN. 10 successes → CLOSED |

**Automated response when accuracy dips to 95.5%:**
1. AccuracyTracker detects negative slope → alerts via structured logging
2. If cheap_rail failure rate hits 15% → circuit breaker OPENS → all traffic reroutes to VLM consensus
3. DSPy Supervisor detects module F1 drop > 5% → triggers automatic MIPRO re-optimization
4. Gradient Monitor checks if prompts are oscillating → alerts if unstable

**Code**: `src/graphocr/monitoring/langsmith_tracer.py`, `src/graphocr/layer3_inference/circuit_breaker.py`

---

### Q: How do you design an "Agentic Supervisor" that watches the Junior use DSPy? How would this supervisor explain the "math of textual gradients" when the Junior's prompt optimization starts to hallucinate or drift?

The DSPy Supervisor (`dspy_layer/supervisor.py`) and Gradient Monitor (`dspy_layer/gradient_monitor.py`) form a **deterministic, automated monitoring system** — not "better prompting."

**Supervisor loop** (runs every 30 minutes):
1. Fetches module performance metrics from LangSmith API
2. Compares against baseline F1 (default 0.92)
3. If degradation > 5%: triggers re-optimization via MIPRO (max 3/day)
4. Records a gradient snapshot after each optimization

**Gradient Monitor** — the "math of textual gradients":

After every DSPy optimization run, the monitor records a snapshot: `(prompt_hash, prompt_text, metric_score, step)`. Over a window of the last 5 snapshots, it computes:

| Metric | Formula | What it detects |
|--------|---------|-----------------|
| **Prompt similarity chain** | `SequenceMatcher(prompt[i-1], prompt[i])` ratio | How much the prompt text changes between runs |
| **Direction consistency** | Count sign changes in metric deltas | Oscillation: metric improving then worsening alternately |
| **Stability score** | `0.4 * avg_similarity + 0.6 * direction_consistency` | Overall prompt health (0.0 = chaotic, 1.0 = stable) |
| **Magnitude trend** | Classify as increasing/decreasing/oscillating/stable | Whether optimization is actually improving things |

**Actionable recommendations** (generated automatically, not by human):

| Situation | Recommendation |
|-----------|---------------|
| Oscillating prompts | "Increase training data diversity or widen the MIPRO search space. Consider freezing the current best prompt." |
| Declining metrics | "Check for data distribution shift. Roll back to last known-good prompt. Investigate training data quality." |
| Low stability | "Monitor closely. May benefit from more training examples in weak areas." |
| Healthy + increasing | "Optimization is improving. Continue current approach." |

This is a **deterministic system**: thresholds trigger actions, not human intuition. The "twist" constraint ("do not suggest better prompting") is satisfied because the supervisor monitors, detects, and intervenes automatically based on numerical analysis.

**Code**: `src/graphocr/dspy_layer/supervisor.py`, `src/graphocr/dspy_layer/gradient_monitor.py`, `src/graphocr/dspy_layer/optimizers.py`

---

## Evidence: Working System

### Test results on real Arabic medical prescriptions

```
Documents:       3
Successful:      3
Total tokens:    134
Arabic tokens:   81
English tokens:  39
Avg confidence:  59.93%
Failures caught: 6 (spatial-blind Type A)
```

| Document | Type | Tokens | Arabic | Confidence | Failures | Auto-Rotation | Route |
|----------|------|--------|--------|------------|----------|---------------|-------|
| 1000093095.jpg | Pediatric Rx | 39 | 34 | 45% | 0 | 270° corrected | VLM consensus |
| 1000093096.jpg | Psychiatric Rx | 77 | 31 | 52% | 6 | None needed | VLM consensus |
| 1000093097.jpg | Internal Med Rx | 18 | 16 | 83% | 0 | 90° corrected | Cheap rail |

### Test results on typed PDFs

```
Documents:       3
Successful:      3
Total tokens:    19
Avg confidence:  98.94%
Cheap rail:      3/3 (100%)
```

### Unit test suite

```
112 passed in 0.55s
```

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| **Architecture document** | [ARCHITECTURE.md](ARCHITECTURE.md) — full system design with Mermaid diagrams |
| **Source code** | `src/graphocr/` — 80+ Python files across 3 layers + RAG + DSPy + monitoring |
| **Tests** | `tests/` — 112 unit + integration tests |
| **Sample data** | `sample data/images/` (Arabic prescriptions), `sample data/pdfs/` (typed documents) |
| **HTML/JSON reports** | Generated via `scripts/batch_test.py` |
| **Configuration** | `config/` — 6 YAML files (pipeline, agents, neo4j_rules, dspy, monitoring, rag) |
| **Infrastructure** | `docker-compose.yml` (Neo4j, Redis, MinIO), conda environment |
| **README** | [README.md](README.md) — installation, testing, usage |

---

## How to Run

```bash
# Setup
conda create -n graphocr python=3.11 -y
conda activate graphocr
pip install -e ".[dev]"
conda install -c conda-forge poppler -y

# Run unit tests (112 tests, no external services needed)
python -m pytest tests/ -v

# Test on real images
python scripts/batch_test.py "sample data/images" --format both
open "sample data/images/graphocr_report.html"

# Test on PDFs
python scripts/batch_test.py "sample data/pdfs" --format both
open "sample data/pdfs/graphocr_report.html"

# Start full pipeline (requires Docker + vLLM)
docker compose up -d
python scripts/seed_neo4j.py
graphocr serve
```
