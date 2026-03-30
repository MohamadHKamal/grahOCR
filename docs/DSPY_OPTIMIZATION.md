# DSPy Layer — Automated Prompt Optimization & Mentorship

**System**: GraphOCR Deterministic Trust Layer
**Scope**: DSPy modules, MIPRO optimizer, metrics, agentic supervisor, gradient monitor, mentorship mode
**Problem Solved**: **Deterministic automated system** — "clearly do not suggest better prompting." This is MIPRO-based machine learning over prompts, not manual prompt engineering. Plus an **Agentic Supervisor** that explains textual gradients to junior engineers.

---

## 1. Purpose & Problem Context

The problem statement explicitly states: *"Do not suggest better prompting. We are after a deterministic and automated system."*

The DSPy layer replaces manual prompt engineering with **MIPRO (Multi-prompt Instruction PRoposal Optimizer)** — a Bayesian optimizer that treats prompt optimization as a machine learning problem. It jointly optimizes instructions and few-shot examples via data-driven trials, using real failure data from the Postmortem Agent.

Additionally, the problem asks: *"How do you design an Agentic Supervisor that watches the Junior use DSPy? Describe how this supervisor agent would step in to explain the math of textual gradients when the Junior's prompt optimization starts to hallucinate or drift."*

---

## 2. Architecture Overview

```
[Postmortem Agent]
    |
    | FailureReports (Redis FailureStore)
    v
[DSPy Supervisor] ──── 30-min check intervals
    |
    ├── Fetch metrics from Langfuse API
    ├── Check degradation > 5% threshold
    |
    ├── If degraded:
    |   ├── Fetch training data from FailureStore
    |   ├── Run MIPRO reoptimization (max 3/day)
    |   ├── Save new prompts (atomic symlink swap)
    |   └── Record gradient snapshot
    |
    ├── [Gradient Monitor]
    |   ├── Track prompt similarity chain
    |   ├── Detect oscillation/divergence
    |   └── Alert if unstable
    |
    └── [Mentor Mode]
        ├── Explain gradient direction
        ├── Report convergence status
        ├── Diagnose drift/oscillation
        └── Detect hallucinated gradients
```

**Source Files:**
| Component | File |
|---|---|
| DSPy Modules | `dspy_layer/modules.py` |
| Optimizers | `dspy_layer/optimizers.py` |
| Metrics | `dspy_layer/metrics.py` |
| Supervisor | `dspy_layer/supervisor.py` |
| Gradient Monitor | `dspy_layer/gradient_monitor.py` |

---

## 3. DSPy Modules — Specialized Extraction Components

**File**: `dspy_layer/modules.py`

Five specialized modules, each a DSPy `Signature` + `Module` pair using `ChainOfThought` for multi-step reasoning:

### 3.1 ClaimFieldExtractor

| | |
|---|---|
| **Inputs** | `spatial_tokens_text`, `document_language` (ar/en/mixed), `context_hints` |
| **Output** | `claim_fields_json` (patient_name, patient_id, diagnosis_codes, procedure_codes, etc.) |
| **Used By** | Extractor Agent (Layer 3) |
| **Metric** | `field_level_f1` |
| **Baseline** | 0.92 |

### 3.2 ArabicMedicalNormalizer

| | |
|---|---|
| **Inputs** | `arabic_text`, `context` |
| **Outputs** | `normalized_text`, `confidence` (0.0-1.0) |
| **Purpose** | Normalize Arabic diacritics, abbreviations, OCR variants |
| **Metric** | `exact_match` |
| **Baseline** | 0.88 |

### 3.3 DiagnosisCodeMapper

| | |
|---|---|
| **Inputs** | `diagnosis_text`, `language` |
| **Outputs** | `icd10_code`, `code_description`, `mapping_confidence` |
| **Purpose** | Map free-text diagnosis descriptions → ICD-10 codes |
| **Metric** | `code_accuracy` |
| **Baseline** | 0.90 |

### 3.4 PolicyVersionValidator

| | |
|---|---|
| **Inputs** | `claim_text`, `policy_context` |
| **Outputs** | `is_correct_version` (bool), `confidence`, `mismatch_explanation` |
| **Purpose** | Detect Type B failures (wrong policy version from RAG) |

### 3.5 ChallengeGenerator

| | |
|---|---|
| **Inputs** | `extracted_fields`, `token_sample`, `known_issues` |
| **Output** | `challenges_json` (adversarial challenges with hypothesis, evidence, confidence) |
| **Used By** | Challenger Agent (Layer 3) |

---

## 4. Optimizers

**File**: `dspy_layer/optimizers.py`

### 4.1 Module Registry

```python
MODULES = {
    "ClaimFieldExtractor": ClaimFieldExtractor,
    "ArabicMedicalNormalizer": ArabicMedicalNormalizer,
    "DiagnosisCodeMapper": DiagnosisCodeMapper,
    "ChallengeGenerator": ChallengeGenerator,
    "PolicyVersionValidator": PolicyVersionValidator,
}
```

### 4.2 MIPRO (Primary Optimizer)

```python
optimizer = MIPROv2(auto="medium", num_threads=4)
optimized = optimizer.compile(
    module,
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)
```

**How MIPRO works**: Unlike manual prompting, MIPRO uses a Bayesian search over the space of possible instructions and few-shot examples. It:
1. Generates candidate instruction variations
2. Evaluates each on the training set using the module's metric
3. Uses Bayesian optimization to explore the most promising regions
4. Jointly optimizes instructions AND few-shot example selection

### 4.3 BootstrapFewShot (Fallback)

```python
optimizer = BootstrapFewShot(max_bootstrapped_demos=8)
```

Used for simpler modules (e.g., ArabicMedicalNormalizer).

### 4.4 Persistence

- `save_optimized_module(module, path)` — saves to disk
- `load_optimized_module(module_name, path)` — loads from disk
- **Atomic symlink swap**: `{module}_latest` always points to newest optimization

---

## 5. Custom Evaluation Metrics

**File**: `dspy_layer/metrics.py`

### 5.1 field_level_f1

Primary metric for `ClaimFieldExtractor`. Compares expected vs predicted JSON claim fields using per-field similarity:

| Field Type | Similarity Function |
|---|---|
| Code fields (ICD-10, CPT) | Set-based F1 on extracted codes |
| Numeric (amounts) | `1 - |expected - predicted| / expected` |
| Date | Exact match on digits only |
| Text (names, etc.) | `SequenceMatcher.ratio()` fuzzy match |

Returns average F1 across all fields (0.0-1.0).

### 5.2 exact_match

For normalization tasks — binary match after stripping whitespace.

### 5.3 code_accuracy

For ICD-10/CPT code mapping:
- Full code match: **1.0**
- Category prefix match (e.g., "E11" matches "E11.2"): **0.5**
- No match: **0.0**

### 5.4 arabic_fuzzy_match

For Arabic medical text:
1. Strip diacritics (tashkeel)
2. Normalize alef variants (أ/إ/آ → ا)
3. Normalize taa marbuta (ة → ه)
4. `SequenceMatcher.ratio()`

### 5.5 Metric Registry

```python
METRICS = {
    "field_level_f1": field_level_f1,
    "exact_match": exact_match,
    "code_accuracy": code_accuracy,
    "arabic_fuzzy_match": arabic_fuzzy_match,
}
```

---

## 6. DSPy Supervisor — Automated Performance Monitoring

**File**: `dspy_layer/supervisor.py`
**Class**: `DSPySupervisor`

### 6.1 Core Loop

Background async loop running every **30 minutes**:

```
1. Fetch module performance metrics from Langfuse API
2. Check for metric degradation against baseline
3. If degradation > 5% AND samples >= 100 AND daily_count < 3:
   → Trigger MIPRO reoptimization
4. Check gradient stability via GradientMonitor
5. Return actions per module
```

### 6.2 Reoptimization Pipeline (`_reoptimize_module`)

1. **Fetch training data** from Redis FailureStore (postmortem-curated failures)
2. Convert `FailureReport` → `dspy.Example` objects
3. Run `optimize_module()` (MIPRO or BootstrapFewShot)
4. Save optimized module to disk with timestamp
5. **Atomic symlink swap**: `{module}_latest` → newest version
6. Record gradient snapshot for stability analysis
7. Increment daily counter

### 6.3 Module Configuration (`dspy_config.yaml`)

```yaml
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
```

### 6.4 Supervisor State

```python
class SupervisorState:
    modules: dict[str, ModulePerformance]
    total_optimizations: int
    last_run: datetime | None
    alerts: list[dict]
```

Exposed via `GET /supervisor/status` API and `graphocr supervisor-status` CLI.

---

## 7. Gradient Monitor — Prompt Stability Analysis

**File**: `dspy_layer/gradient_monitor.py`
**Class**: `GradientMonitor`

### 7.1 What Are Textual Gradients?

In DSPy's MIPRO optimizer, "textual gradients" are **natural-language feedback signals** that describe *how* a prompt should change to improve a metric. Unlike numerical gradients in neural networks, these are LLM-generated critiques.

Example: *"The prompt failed to extract the drug dosage because it didn't instruct the model to check for Arabic numeral formats."*

MIPRO uses these critiques to propose new prompt candidates in the next optimization trial.

### 7.2 Gradient Snapshots

```python
class GradientSnapshot:
    timestamp: datetime
    module_name: str
    prompt_hash: str        # MD5 of prompt text (first 12 chars)
    prompt_text: str
    metric_score: float
    optimization_step: int
```

### 7.3 Stability Analysis

```python
class GradientAnalysis:
    module_name: str
    stability_score: float          # 0=chaotic, 1=stable
    is_diverging: bool
    direction_consistency: float    # How consistent are metric changes
    magnitude_trend: str            # "increasing"|"decreasing"|"stable"|"oscillating"
    recommendation: str             # Actionable advice
    window_size: int
    metric_trend: list[float]
```

**Stability Score Formula**:
```
stability = 0.4 * avg_prompt_similarity + 0.6 * direction_consistency
```

Where:
- `avg_prompt_similarity`: `SequenceMatcher.ratio()` between consecutive prompt versions
- `direction_consistency`: fraction of metric deltas that share the same sign

**Magnitude Trend Classification**:
| Condition | Classification |
|---|---|
| >70% positive deltas | "increasing" |
| >70% negative deltas | "decreasing" |
| Mixed with reversals | "oscillating" |
| Small changes | "stable" |

**Divergence Detection**:
- `stability_score < divergence_alert_threshold (0.3)` → diverging
- `decreasing` trend with >= 3 snapshots → diverging

### 7.4 Configuration (`dspy_config.yaml`)

```yaml
gradient_monitor:
  stability_threshold: 0.7
  window_size: 5
  divergence_alert_threshold: 0.3
```

---

## 8. Mentor Mode — Gradient Explanation Interface

The DSPy Supervisor includes a **Mentor Mode** (`MentorMode` class in `supervisor.py`) that generates human-readable explanations for junior engineers working with DSPy.

### 8.1 What It Explains in Real-Time

**Gradient Direction**:
> "The optimizer is pushing the extraction prompt toward more explicit Arabic character handling — 3 of the last 5 gradient signals mention Arabic digit confusion (٥ vs 5)."

**Convergence Status**:
> "Prompt quality score has plateaued at F1=0.94 for the last 4 trials. The gradients are becoming contradictory (one says 'be more specific about dates', another says 'be more general about formats') — this indicates convergence. No further optimization needed."

**Drift/Oscillation Alerts with Root Cause**:
> "WARNING: The prompt is oscillating between two states — Trial 7 added a rule about stamp regions, Trial 8 removed it, Trial 9 re-added it. Root cause: the training examples contain conflicting stamp-handling cases. Action: review the 3 flagged training examples in the Langfuse dashboard."

**Hallucination Detection**:
> "The optimizer proposed a prompt that references 'ICD-11 codes' — but our Neo4j graph only contains ICD-10. This gradient is hallucinated. The Supervisor has rejected this trial and logged it."

### 8.2 Recommendations by Scenario

| Scenario | Recommendation |
|---|---|
| Oscillating prompts | Increase training data diversity or widen MIPRO search space |
| Decreasing performance | Check for data distribution shift, rollback to previous prompt version |
| Low stability score | Review training data for label noise or contradictory examples |
| Diverging gradients | Reduce optimization frequency, increase `min_samples_for_reoptimize` |
| Stable & increasing | Continue — optimization is working as intended |

### 8.3 Access Points

Each explanation includes:
- Raw gradient text
- Plain-language interpretation
- Metric delta (before/after)
- Recommended action (continue, stop, review training data)

Exposed via:
- `GET /supervisor/status` API endpoint
- `graphocr supervisor-status` CLI command

---

## 9. Self-Healing Data Flow

```
[Postmortem Agent]
    │
    │ FailureReport {root_cause, corrected_value, add_to_dspy_training}
    ▼
[Redis FailureStore]
    │ graphocr:dspy_training_set (bounded list, max 9999)
    │ graphocr:failure_stats (counters by root_cause, failure_type)
    ▼
[DSPy Supervisor] (every 30 min)
    │
    │ get_training_data(limit=500)
    │ Convert FailureReport → dspy.Example
    ▼
[MIPRO Optimizer]
    │
    │ optimize_module(trainset, valset)
    │ max_bootstrapped_demos=4, max_labeled_demos=8
    ▼
[Optimized Module]
    │
    │ save_optimized_module() → disk
    │ Atomic symlink: ClaimFieldExtractor_latest → new version
    ▼
[Extractor Agent] loads _latest on next invocation
    │
    │ Improved extraction accuracy
    │ Fewer failures → fewer FailureReports
    ▼
[Self-Healing Loop Closes]
```

---

## 10. How DSPy Connects to the Problem Statement

| Problem Statement Requirement | DSPy Solution |
|---|---|
| "Do not suggest better prompting" | MIPRO treats prompts as ML parameters optimized via Bayesian search — no manual prompt hacking |
| "Deterministic and automated system" | Supervisor runs on 30-min intervals, auto-triggers reoptimization based on F1 degradation thresholds |
| "Agentic Supervisor watches Junior use DSPy" | MentorMode generates real-time explanations of gradient direction, convergence, drift |
| "Explain math of textual gradients when optimization hallucinates or drifts" | GradientMonitor detects oscillation/divergence, MentorMode explains in plain language with root cause and recommended action |
| "Self-Healing pipeline" | Postmortem → FailureStore → Supervisor → MIPRO → Improved prompts → Fewer failures (closed loop) |
| "Handle large volumes at scale" | Max 3 reoptimizations/day, min 100 samples before trigger, atomic symlink swap for zero-downtime prompt updates |
