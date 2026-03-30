# Layer 2 — Adversarial Verification & Self-Healing

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Multi-agent red team, Neo4j knowledge graph validation, self-healing back-propagation, postmortem learning
**Problem Solved**: **Adversarial verification at 100k docs/day** — manual review is impossible. A "Self-Healing" pipeline with multi-agent red team, not passive "LLM as a judge"

---

## 1. Purpose & Problem Context

At 100k documents per day, manual review is impossible. The problem statement demands:
- A **layered verification pipeline** that is not a passive "LLM as a judge"
- A **multi-agent red team** where agents adversarially challenge each other
- **Governance**: if the "Adjuster" agent gets tricked, an automated "Post-Mortem Agent" updates the system's core logic to prevent that specific failure pattern globally
- A **Hybrid Graph-OCR Validation** using Neo4j to detect "Logical Impossibilities" (1940 birthdate on a 2026 policy, drug dosage exceeding clinical limits)
- **Back-Propagation**: when the graph identifies a conflict, the system triggers a targeted VLM re-scan of specific source coordinates **without restarting the entire 100k-doc batch**

---

## 2. Agent Pipeline Architecture

```
SpatialTokens + Policy Context
    |
    v
[Extractor Agent] ─── Qwen2.5-7B (cheap) or Llama-3.1-70B (VLM)
    |                  DSPy-optimized prompts, temp=0.1
    |                  Output: ExtractionResult with field→token_id mapping
    v
[Validator Agent] ─── Qwen2.5-7B, temp=0 (fully deterministic)
    |                  Neo4j 6-rule check + LLM consistency analysis
    |                  Output: list[GraphViolation] with severity scores
    v
[Challenger Agent] ── Llama-3.1-70B, temp=0.3 (intentionally creative)
    |                  7 adversarial strategies + VLM visual verification
    |                  Output: list[Challenge] with affected_token_ids
    v
[Consensus Check]
    |
    ├── Consensus reached (no high-conf challenges, no critical violations)
    |       |
    |       v
    |   [Output Assembly] → [Postmortem Agent] → END
    |
    └── Conflicts detected
            |
            v
        [Self-Healing Loop]
            |
            ├── round < 2: → back to [Extractor Agent]
            └── round >= 2: → [Escalate to Human] → [Postmortem Agent] → END
```

**Source Files:**
| Component | File |
|---|---|
| Graph Builder (LangGraph state machine) | `layer2_verification/agents/graph_builder.py` |
| Extractor Agent | `layer2_verification/agents/extractor.py` |
| Validator Agent | `layer2_verification/agents/validator.py` |
| Challenger Agent | `layer2_verification/agents/challenger.py` |
| Postmortem Agent | `layer2_verification/agents/postmortem.py` |
| Failure Store | `layer2_verification/agents/failure_store.py` |
| Neo4j Client | `layer2_verification/knowledge_graph/client.py` |
| Rule Engine | `layer2_verification/knowledge_graph/rule_engine.py` |
| Validators | `layer2_verification/knowledge_graph/validators.py` |
| Schema Loader | `layer2_verification/knowledge_graph/schema_loader.py` |
| Conflict Detector | `layer2_verification/self_healing/conflict_detector.py` |
| VLM Rescanner | `layer2_verification/self_healing/vlm_rescanner.py` |
| Feedback Loop | `layer2_verification/self_healing/feedback_loop.py` |

---

## 3. Agent Roles & Model Selection

Each agent uses a **deliberately different model and temperature** to prevent shared blind spots:

### 3.1 Extractor Agent (`agents/extractor.py`)

| Parameter | Value |
|---|---|
| Model (cheap rail) | `qwen2.5-7b-instruct` |
| Model (VLM consensus) | `llama-3.1-70b-instruct` |
| Temperature | 0.1 |
| Max tokens | 4096 |

**Behavior**:
1. Attempts **DSPy-optimized extraction** first (loads from `optimized_modules/ClaimFieldExtractor_latest`)
2. Falls back to LangChain LLM call if DSPy unavailable
3. Detects document language (Arabic/English/mixed) by counting language tokens
4. Selects model based on `round_number`: Round 0 = cheap model, retries = heavy model

**Extracted Fields**: patient_name, patient_id, patient_dob, provider_name, provider_id, diagnosis_codes (ICD-10), procedure_codes (CPT), line_items (description, code, amount, quantity), date_of_service, total_amount, currency, policy_reference

Every field carries **`source_tokens`** — the token_ids that sourced the value.

### 3.2 Validator Agent (`agents/validator.py`)

| Parameter | Value |
|---|---|
| Model | `qwen2.5-7b-instruct` |
| Temperature | **0.0** (fully deterministic) |
| Max tokens | 2048 |

**Behavior** (2-stage validation):
1. **Graph Validation**: Builds `InsuranceClaim` from extraction, runs `Neo4jClient.run_all_validations()` → returns `list[GraphViolation]`
2. **LLM Consistency Check**: Sends extraction + graph violations to LLM for semantic analysis of internal consistency, medical plausibility, data quality, completeness, cross-field validation

This agent is the **"deterministic anchor"** — it does not generate or interpret text creatively. It executes structured graph queries and returns pass/fail verdicts.

### 3.3 Challenger Agent (`agents/challenger.py`)

| Parameter | Value |
|---|---|
| Model | `llama-3.1-70b-instruct` (different from extractor!) |
| Temperature | **0.3** (intentionally creative) |
| Max tokens | 3072 |
| Max challenges/round | 5 |

**7 Adversarial Attack Strategies**:

| # | Strategy | Example |
|---|---|---|
| 1 | Arabic Character Confusion | ع vs غ, ي vs ى (dots change meaning) |
| 2 | OCR Digit Errors | 3 vs 8, 1 vs 7, 0 vs O |
| 3 | Stamp Obscuration | Seals covering text underneath |
| 4 | Merged Line Items | Two items read as one due to column merge |
| 5 | Date Format Ambiguity | DD/MM/YYYY vs MM/DD/YYYY |
| 6 | Currency Symbol Misread | SAR vs USD, ر.س confusion |
| 7 | Handwriting Ambiguity | Multiple plausible readings |

**VLM Visual Verification**: For the top 3 high-confidence (>0.7) challenges:
1. Finds affected tokens, computes bounding box + 20px padding
2. Crops region from original page image
3. Base64-encodes and sends to `qwen2-vl-7b-instruct`
4. VLM confirms or rejects the challenge
5. Adjusts confidence: supports = +0.1, rejects = -0.3

Each challenge carries: `target_field`, `hypothesis`, `evidence`, `proposed_alternative`, `confidence`, `affected_tokens`

### 3.4 Postmortem Agent (`agents/postmortem.py`)

| Parameter | Value |
|---|---|
| Model | `qwen2.5-7b-instruct` |
| Temperature | 0.0 |
| Max tokens | 2048 |

**Root Cause Classification** (4 types):

| Root Cause | Failure Type | Description |
|---|---|---|
| `ocr_misread` | TYPE_A_SPATIAL_BLIND | Character confusion, digit errors |
| `prompt_failure` | TYPE_B_CONTEXT_BLIND | LLM extraction prompt failed |
| `rule_gap` | TYPE_B_CONTEXT_BLIND | Knowledge graph missing a constraint |
| `layout_confusion` | TYPE_A_SPATIAL_BLIND | Reading order or spatial assembly wrong |

**Actions**:
1. Classifies each corrected field's root cause
2. Persists `FailureReport` to Redis via `FailureStore`
3. Tags cases with `add_to_dspy_training=True` for self-healing loop
4. If `root_cause == "rule_gap"`: **creates `LearnedRule` node in Neo4j** via `_update_neo4j_rules()`

---

## 4. Consensus Mechanism

**Function**: `consensus_check_node(state)`

Consensus is reached when:
- **ZERO** high-confidence challenges (`confidence > 0.7`)
- **ZERO** critical graph violations (`severity > 0.8`)

If consensus fails → route to self-healing loop.

---

## 5. Neo4j Knowledge Graph — Hybrid Graph-OCR Validation

### 5.1 Purpose

The Neo4j Knowledge Graph catches **Logical Impossibilities** that OCR serialization gore creates. Instead of passively accepting OCR output, the Validator Agent ensures the data obeys strict medical and logical realities.

### 5.2 Data Model

```
[ProcedureCode] ──COMPATIBLE_WITH──> [DiagnosisCode]
[Medication] ──CONTRAINDICATED_WITH──> [Medication]
[ProcedureCode] ──REQUIRES_SPECIALTY──> [Specialty]
[Provider] ──HAS_SPECIALTY──> [Specialty]
[DiagnosisCode] ──TREATED_BY──> [Medication]
[Medication] ──HAS_DOSAGE_LIMIT──> [LearnedRule]
```

**Node Types**:
- `ProcedureCode` (code UNIQUE, description, category, valid_diagnosis_prefixes, required_specialty)
- `DiagnosisCode` (code UNIQUE, description, category)
- `Medication` (name UNIQUE, max_daily_dosage_mg, unit)
- `Specialty` (name UNIQUE, department)
- `Provider` (id UNIQUE, name, facility)
- `Patient` (id UNIQUE, dob)
- `LearnedRule` (report_id, affected_field, original_value, corrected_value, root_cause, active, created_at)

### 5.3 Six Validation Rule Categories

**Rule 1: Procedure-Diagnosis Compatibility** (`validate_procedure_diagnosis`)

```cypher
MATCH (p:ProcedureCode {code: $code})
RETURN p.valid_diagnosis_prefixes AS valid_prefixes
```
Checks if diagnosis codes start with valid ICD-10 prefixes for the given procedure. **Severity: 0.8**

**Rule 2: Drug Dosage Limits** (`validate_drug_dosage`)

```cypher
MATCH (m:Medication {name: $name})
RETURN m.max_daily_dosage_mg AS max_dosage
```
Example: If OCR misreads "15 mg" as "1500 mg" for Warfarin (max 15mg), the graph catches it. **Severity: 0.9**

**Rule 3: Contraindicated Drug Combinations** (`validate_contraindicated_drugs`)

```cypher
UNWIND $names AS name1
UNWIND $names AS name2
WITH name1, name2 WHERE name1 < name2
MATCH (a:Medication {name: name1})-[:CONTRAINDICATED_WITH]->(b:Medication {name: name2})
RETURN a.name AS drug1, b.name AS drug2
```
Example: Warfarin + Aspirin prescribed due to column merging. **Severity: 0.95**

**Rule 4: Provider Specialty Requirements** (`validate_provider_specialty`)

```cypher
MATCH (p:ProcedureCode {code: $proc_code})-[:REQUIRES_SPECIALTY]->(s:Specialty)
OPTIONAL MATCH (prov:Provider {id: $provider_id})-[:HAS_SPECIALTY]->(s)
RETURN s.name AS required_specialty, prov IS NOT NULL AS has_specialty
```
Example: Cardiac surgery requires cardiology specialty. **Severity: 0.7**

**Rule 5: Date Sanity** (`validate_date_sanity`) — local, no Neo4j

| Check | Severity |
|---|---|
| Date of service in future | 0.9 |
| Date of service before 2015-01-01 | 0.7 |
| Patient age outside 0–130 years | **0.95** (logical impossibility — catches "1940 birthdate on 2026 policy") |
| Patient DOB after date of service | **0.95** |

**Rule 6: Amount Reasonableness** (`_validate_amounts`) — local

| Check | Severity |
|---|---|
| Negative total amount | 0.9 |
| Total > $2,000,000 | 0.7 |
| Single line item > $500,000 | 0.7 |
| Computed total vs stated total diff > 1% | 0.6 |

**Rule 7: Learned Rules** (`validate_learned_rules`) — from Postmortem feedback

```cypher
MATCH (r:LearnedRule {active: true})
WHERE r.affected_field IN $fields
RETURN r.affected_field, r.original_value, r.corrected_value, r.root_cause
```
Matches extracted values against known-bad patterns from previous postmortems. **Severity: 0.85**

### 5.4 Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| CMS.gov | ICD-10 codes, CPT codes, valid pairs | Annually (October) |
| FDA / Lexicomp / First Databank | Drug names, max dosages, contraindications | Quarterly |
| Internal Policy (Actuarial) | Amount limits, date ranges, age limits | As needed |
| Provider Registry | Provider-specialty credentialing | Monthly |
| Postmortem Feedback | **LearnedRule nodes from rule_gap root causes** | **Runtime** |

### 5.5 Schema Construction (`schema_loader.py`)

Idempotent seeding sequence:
1. Create uniqueness constraints (DiagnosisCode, ProcedureCode, Medication, Provider, Patient, Claim)
2. Create indexes
3. Seed dosage limits (`MERGE (m:Medication) SET m.max_daily_dosage_mg`)
4. Seed contraindications (`MERGE (a)-[:CONTRAINDICATED_WITH]->(b)`)
5. Seed procedure-diagnosis compatibility
6. Seed specialty requirements
7. Seed temporal rules (policy effective ranges)
8. Create LearnedRule indexes

Example from `neo4j_rules.yaml`:
```yaml
constraints:
  dosage_limits:
    metformin: 2550.0
    warfarin: 15.0
    paracetamol: 4000.0
relationships:
  contraindicated_drugs:
    - ["Warfarin", "Aspirin"]
    - ["Metformin", "Iodinated_Contrast"]
  procedure_diagnosis_compatibility:
    "99213": ["E11", "I10", "J06", "M54", "Z00"]
  specialty_requirements:
    "27447": "Orthopedic Surgery"
    "33533": "Cardiothoracic Surgery"
temporal_rules:
  policy_effective_ranges:
    "RIDER_2018_A": ["2018-01-01", "2020-12-31"]
    "POLICY_STANDARD_2025": ["2025-01-01", "2028-12-31"]
```

---

## 6. Self-Healing Back-Propagation Loop

This is the core mechanism that answers: "When the Graph identifies a logical conflict, how does the system automatically trigger a targeted VLM Re-Scan of the specific source coordinates **without restarting the entire 100k-doc batch**?"

### Phase 1 — Conflict Detection (`conflict_detector.py`)

**Function**: `detect_conflicting_regions(extraction, challenges, graph_violations, spatial_tokens) -> list[BoundingBox]`

Collects conflicting token_ids from three sources:
| Source | Threshold | Rationale |
|---|---|---|
| Challenger challenges | confidence >= **0.6** | High-confidence adversarial finding |
| Graph violations | severity >= **0.7** | Severe logical impossibility |
| Extracted fields | confidence < **0.5** | Low-confidence extraction |

Process:
1. Build token map: `token_id -> SpatialToken` (supports full and short IDs)
2. Collect all conflicting tokens from above sources
3. Group by `page_number`
4. **Greedy merge with 50px padding** — nearby tokens become one re-scan region
5. Clamp coordinates to non-negative
6. Return minimal set of `BoundingBox` regions

**Region overlap check** (AABB collision):
```python
not (a.x_max < b.x_min or b.x_max < a.x_min or
     a.y_max < b.y_min or b.y_max < a.y_min)
```

### Phase 2 — Targeted VLM Re-Scan (`vlm_rescanner.py`)

**Function**: `rescan_region(page_image_path, region, vlm_model="qwen2-vl-7b-instruct") -> list[SpatialToken]`

1. Load original page PNG from MinIO storage
2. **Crop** to exact conflicting region coordinates (clamped to image bounds)
3. Base64-encode the cropped region
4. Send to **Qwen2-VL-7B** via vLLM OpenAI-compatible API (temp=0.1)
5. Specialized prompt targeting:
   - Arabic character confusion (ع vs غ, ي vs ى)
   - Digit misreads (3 vs 8, 1 vs 7, 0 vs O)
   - Stamp obscuration
   - Handwriting ambiguity
6. Parse VLM JSON response into new `SpatialToken` objects with `ocr_engine="vlm_rescan"`

**This avoids restarting the entire 100k-doc batch** — only the specific conflicting pixel region is re-examined.

### Phase 3 — Token Stream Patch (`feedback_loop.py`)

**Function**: `patch_tokens(original_tokens, conflicting_regions, rescan_tokens) -> list[SpatialToken]`

1. Identify which original tokens fall within conflicting regions (center-point check)
2. **Remove** old conflicting tokens
3. **Insert** VLM rescan tokens
4. Re-sort by page + position (Y, then X)
5. Re-assign reading order (0, 1, 2, ...)

**Function**: `identify_affected_fields(conflicting_regions, field_token_map, token_map) -> list[str]`

Only fields whose `source_tokens` fall within conflicting regions are re-extracted — not the entire document.

### Phase 4 — Postmortem Learning

1. Classify root cause (`ocr_misread` | `prompt_failure` | `rule_gap` | `layout_confusion`)
2. Generate `FailureReport` with `corrected_value`
3. Tag: `add_to_dspy_training=True` (if severity > 0.5)
4. Store in **Redis FailureStore** (TTL: 30 days)
5. DSPy Supervisor pulls at 30-minute intervals for MIPRO reoptimization

### Loop Constraints

- Maximum **2 rounds** of self-healing
- If consensus not reached after 2 rounds → **escalate to human review** with full provenance
- Each round: Detect → Re-scan → Patch → Re-extract → Re-validate → Re-challenge

---

## 7. Failure Store — Redis Infrastructure

**File**: `layer2_verification/agents/failure_store.py`
**Class**: `FailureStore`

| Key | Type | Purpose |
|---|---|---|
| `graphocr:failure_report:{id}` | String (JSON) | Individual failure report, TTL=30 days |
| `graphocr:dspy_training_set` | List | Reports tagged for DSPy training (max 9999) |
| `graphocr:failure_stats` | Hash | Counters: root_cause:{type}, failure_type:{type}, total |

**Methods**:
- `save_report(report)` — stores report, pushes to training set if flagged, increments stats
- `get_training_data(limit=500, root_cause_filter=None)` — consumed by DSPy Supervisor
- `get_rule_gap_reports(limit=100)` — consumed by Postmortem for Neo4j updates
- `get_stats()` — returns root cause distribution

---

## 8. How Layer 3 Connects to the Problem Statement

| Problem Statement Requirement | Layer 3 Solution |
|---|---|
| "Not passive LLM as a judge, but multi-agent red team" | 4 agents with different models/temperatures adversarially challenging each other |
| "If Adjuster agent gets tricked, Post-Mortem Agent updates core logic" | Postmortem classifies root cause, creates `LearnedRule` in Neo4j, tags for DSPy retraining |
| "Self-Healing pipeline at 100k docs/day" | Automated: Detect → VLM Re-scan → Patch → Re-extract. No human in the loop until escalation |
| "Hybrid Graph-OCR Validation for Logical Impossibilities" | Neo4j 6+1 rule categories: dosage limits, contraindications, date sanity, procedure-diagnosis, specialty, amounts, learned rules |
| "1940 birthdate on a 2026 policy" | Date sanity check: patient age 0–130 years, severity 0.95 |
| "Drug dosage exceeding clinical stoichiometric limits" | `validate_drug_dosage`: Cypher query checks `max_daily_dosage_mg`, severity 0.9 |
| "Back-Propagation without restarting the entire batch" | Targeted VLM re-scan crops exact bounding box region, patches only affected tokens, re-extracts only affected fields |
| "Deterministic and automated system" | Validator at temp=0, Neo4j graph rules are deterministic, DSPy MIPRO replaces manual prompting |
