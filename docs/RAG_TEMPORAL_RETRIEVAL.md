# RAG — Temporal Policy Retrieval

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Temporal-aware policy retrieval, vector store, policy chunking, multilingual embeddings, context injection
**Problem Solved**: **Contextual Hallucination (Type B / Intelligence Failure)** — standard RAG retrieves the "2025 Standard Plan" when the claim explicitly refers to an "Obsolete 2018 Rider"

---

## 1. Purpose & Problem Context

Standard RAG systems commit **Intelligence Failure** because they retrieve policies based purely on text similarity. When a handwritten claim from 2018 mentions "Rider 2018-A", a naive RAG system ignores the date entirely and retrieves the most semantically similar document — often a 2025 standard plan that has completely different coverage rules.

The AI then "hallucinates" a denial because it's evaluating the claim against the **wrong policy version**. This is not an OCR problem — it's a retrieval problem.

The Temporal RAG system solves this by enforcing **hard temporal filtering before semantic similarity**, ensuring the LLM is never looking at the wrong roadmap.

### Claims vs Policies — Two Different Document Types

The pipeline works with two fundamentally different types of documents. Understanding the distinction is essential:

**Insurance Claims (scanned documents — what OCR processes):**
- Paper documents submitted by patients after receiving medical treatment
- Messy, real-world artifacts: handwritten prescriptions, receipts with pharmacy stamps, doctor notes, lab results
- Contain: patient name, diagnosis, procedures performed, medications prescribed, amounts charged, dates of service
- These are what Layer 1 processes — PaddleOCR extracts text, the pipeline builds SpatialTokens
- **Claims are NEVER stored in the vector database**

**Insurance Policies (rulebooks — what the vector database stores):**
- Formal contracts written by the insurance company *before* any claim is filed
- Define the rules: what's covered, what's excluded, benefit limits, deductibles, copays, preauthorization requirements
- Example: *"Rider 2018-A covers prescription medications up to 50,000 SAR/year. Effective 2018-01-01 to 2020-12-31. Excludes pre-existing conditions diagnosed before enrollment."*
- Multiple versions exist over time (2018 rider, 2022 amendment, 2025 standard plan)
- These are chunked, embedded with `multilingual-e5-large`, and stored in ChromaDB with temporal metadata
- Ingestion is done via `scripts/ingest_policies.py`

**How they interact:**

```
Patient submits:  CLAIM (scanned paper)        -->  OCR extracts data (SpatialTokens)
                     |
Insurance asks:   "Is this claim valid         -->  RAG retrieves the correct POLICY
                   under the patient's              version to check the rules
                   policy?"
                     |
System checks:    Does the CLAIM comply        -->  Agents + Neo4j validate claim
                  with the POLICY rules?             data against policy constraints
```

**Concrete example:**
- **Claim** (OCR output): *"Dr. Ahmed prescribed Warfarin 15mg for patient Mohamed, date 2018-06-15, Policy Rider 2018-A"*
- **Policy** (retrieved from ChromaDB): *"Rider 2018-A covers prescription medications up to 50,000 SAR/year. Effective 2018-01-01 to 2020-12-31"*
- **Neo4j**: Validates Warfarin max dosage is 15mg (passes), checks no contraindicated drugs

Without the correct **policy version**, the system cannot determine if the **claim** should be approved or denied. That's why temporal filtering is critical — a 2018 claim must be evaluated against the 2018 policy, not the 2025 version.

In short: **policies go in, claim data queries against them.** The vector store is a policy library, not a document archive.

---

## 2. Architecture Overview

```
SpatialTokens + claim_date + jurisdiction
    |
    v
[Temporal Policy Retriever]
    |
    ├── Stage 1: EXTRACT
    |   ├── Regex: policy reference (Policy No, Plan, Rider — English + Arabic)
    |   └── Regex: service date (YYYY-MM-DD, DD/MM/YYYY)
    |
    ├── Stage 2: FILTER (hard temporal constraint)
    |   └── effective_date <= claim_date <= expiry_date
    |
    └── Stage 3: RANK (semantic similarity within filtered set)
        └── multilingual-e5-large embeddings (Arabic + English)
    |
    v
RetrievalContext (policy chunks + method + confidence + warnings)
    |
    v
[Policy Context Injector]
    ├── format_for_extractor()  — coverage rules, limits, codes
    ├── format_for_validator()  — exclusions, limits, preauth
    └── format_for_challenger() — version info, warnings, challenge angles
```

**Source Files:**
| Component | File |
|---|---|
| Temporal Retriever | `rag/retriever.py` |
| Vector Store (ChromaDB) | `rag/vector_store.py` |
| Policy Chunker | `rag/policy_chunker.py` |
| Embeddings | `rag/embeddings.py` |
| Context Injector | `rag/context_injector.py` |

---

## 3. Temporal Policy Retriever

**File**: `rag/retriever.py`
**Class**: `TemporalPolicyRetriever`

### 3.1 Three-Stage Retrieval Strategy

#### Stage 1: EXTRACT

Extract structured references directly from the SpatialToken text stream:

**Policy Reference Patterns** (`_extract_policy_reference`):
```python
# English
r"(?:Policy|Plan|Rider|Contract)\s*(?:No\.?:?|Number:?|#|:)\s*([\w\-/]+)"
r"(Rider\s+\d{4}[\-\w]*)"
r"(Plan\s+\d{4}[\-\w]*)"
r"(?:Policy\s+)([\w]{2,3}-\d{4,}-\d+)"

# Arabic
r"(?:بوليصة|خطة|ملحق|عقد)\s*(?:رقم:?|#|:)\s*([\w\-/]+)"
```

**Service Date Patterns** (`_extract_date`):
```python
r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})"   # YYYY-MM-DD
r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})"   # DD/MM/YYYY
```

#### Stage 2+3: FILTER + RANK

The retrieval strategy adapts based on what was extracted:

| Extracted | Method | Confidence | Warning |
|---|---|---|---|
| Policy ref + service date | `temporal_filtered` | **0.9** | None |
| Service date only | `temporal_semantic_hybrid` | **0.7** | "No explicit policy reference found" |
| Policy ref only | `reference_only` | **0.5** | "No date for temporal filtering" |
| Neither | `semantic_only` | **0.3** | **"Failure Type B risk!"** |

### 3.2 Temporal Filtering — The Core Defense

```python
def _temporal_filter(hits, service_date):
    for hit in hits:
        effective = parse_date(hit["metadata"]["effective_date"])
        expiry = parse_date(hit["metadata"]["expiry_date"])

        if service_date < effective: continue   # Policy not yet active
        if service_date > expiry: continue      # Policy expired

        filtered.append(hit)
```

This is the **hard constraint** that prevents Intelligence Failure. A 2018 claim will never be evaluated against a 2025 policy because the temporal filter eliminates it before semantic similarity is considered.

### 3.3 Semantic Query Construction

Instead of embedding the full claim text (which contains noise), the retriever extracts **medical keywords**:

```python
def _build_semantic_query(full_text):
    # ICD codes: A12.3, B45
    icd_codes = re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b", text)
    # CPT codes: 99213
    cpt_codes = re.findall(r"\b\d{5}\b", text)
    # Insurance terms (English + Arabic)
    terms = ["coverage", "benefit", "exclusion", "deductible",
             "تغطية", "فوائد", "استثناء", "خصم"]
    # Return first 10 keywords, fallback to first 200 chars
```

### 3.4 RetrievalContext Output

```python
class RetrievalContext:
    policy_chunks: list[PolicyChunk]
    retrieval_method: str           # "temporal_filtered", "semantic_only", etc.
    retrieval_confidence: float     # 0.3-0.9
    retrieval_warnings: list[str]   # Risk flags
    policy_reference: str | None
    service_date: date | None
```

---

## 4. Policy Context Injector

**File**: `rag/context_injector.py`
**Class**: `PolicyContextInjector`

Formats retrieved policy chunks differently for each agent role:

### 4.1 For Extractor

Focus: coverage rules, benefit limits, code lists
```
Policy: {number} v{version}
Claim Date: {date}
Retrieved via: {method}
Warnings: {warnings}
--- Policy Sections ---
1. {section_title}: {text}  (coverage rules, limits, codes)
```

### 4.2 For Validator

Focus: exclusions, limits, preauthorization requirements
```
--- Validation-Relevant Policy Sections ---
Exclusions: ...
Benefit Limits: ...
Preauth Requirements: ...
Coverage Rules: ...
```

### 4.3 For Challenger

Focus: version info, retrieval confidence, challenge angles
```
Policy Version: {version}
Retrieved via: {method} (confidence: {conf})
⚠ Warnings: {warnings}

Challenge angles to consider:
- Is this the correct policy version for the claim date?
- Does the claim reference a rider/amendment not retrieved?
- Are there exclusions that should block this claim?
```

### 4.4 Policy Match Validation

`validate_policy_match(context) -> list[GraphViolation]`

| Condition | Severity | Description |
|---|---|---|
| No policy found | 0.8 | Cannot validate claim without policy |
| `semantic_only` method | 0.6 | Type B risk — no temporal/reference filtering |
| Low confidence (< 0.5) | 0.5 | Uncertain policy match |

---

## 5. Vector Store — ChromaDB Backend

**File**: `rag/vector_store.py`
**Class**: `PolicyVectorStore`

### 5.1 Configuration

```yaml
vector_store:
  backend: "chromadb"
  persist_dir: "./data/vectorstore"
  collection_name: "policy_chunks"
  distance_metric: "cosine"
```

### 5.2 Metadata Schema

Every chunk stored with temporal + hierarchical metadata:

```python
{
    "policy_id": "uuid",
    "policy_number": "POL-2018-001",
    "policy_type": "rider",
    "policy_version": "2018-A",
    "effective_date": "2018-01-01",    # ISO string for filtering
    "expiry_date": "2020-12-31",       # ISO string for filtering
    "section_title": "Coverage Rules",
    "section_type": "coverage",
    "parent_policy_id": "uuid",
    "jurisdiction": "SA"
}
```

**Critical**: Temporal metadata stored as ISO strings — enables ChromaDB `where` filtering.

### 5.3 Methods

- `add_chunks(chunks)` — embed + store with metadata
- `search(query, n_results=5, where=None)` — semantic search with optional metadata filters
- `delete_policy(policy_id)` — cleanup when policy superseded

---

## 6. Policy Chunker

**File**: `rag/policy_chunker.py`

### 6.1 Chunking Strategy

```python
chunk_policy(policy, max_chunk_size=1000, overlap=100) -> list[PolicyChunk]
```

1. **Split into sections** by detecting boundaries:
   - Markdown headings (`# Section`)
   - Formal sections (`Section 1:`, `Article 2:`)
   - Keyword sections (`Coverage:`, `Exclusions:`, `Benefits:`)
   - Arabic sections (`القسم:`, `المادة:`, `الباب:`)

2. **Classify section type** by keyword matching:

| Type | Keywords (English) | Keywords (Arabic) |
|---|---|---|
| coverage | coverage, benefit, covered | تغطية, فوائد |
| exclusion | exclusion, excluded | استثناء |
| benefit_limit | limit, maximum, annual | حد, أقصى |
| deductible | deductible, copay | خصم |
| definition | definition, means | تعريف |
| preauth | preauthorization, pre-auth | تفويض مسبق |
| eligibility | eligibility, eligible | أهلية |

3. **Split large sections** respecting sentence boundaries (`(?<=[.!?])\s+`), with 100-token overlap for context

4. **Propagate metadata**: temporal dates, jurisdiction, parent policy ID inherited by every chunk

### 6.2 Configuration (`rag.yaml`)

```yaml
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

## 7. Multilingual Embeddings

**File**: `rag/embeddings.py`

### 7.1 Model

**`intfloat/multilingual-e5-large`**
- Dimensionality: 1024
- Supports: Arabic, English, and 50+ languages
- Runs **locally** (no API calls — data sovereignty)
- Lazy-loaded on first use

### 7.2 E5 Prefix Convention

E5 models require specific prefixes:
- **Documents**: `"passage: " + text`
- **Queries**: `"query: " + text`

```python
def embed_document(text: str) -> list[float]:
    return embed_texts([text], prefix="passage: ")[0]

def embed_query(query: str) -> list[float]:
    return embed_texts([query], prefix="query: ")[0]
```

### 7.3 Why This Model

Balanced Arabic/English performance is critical for a pipeline processing multilingual insurance claims. `multilingual-e5-large` provides strong cross-lingual retrieval — an Arabic claim description should match English policy text about the same condition.

---

## 8. How RAG Connects to the Problem Statement

| Problem Statement Requirement | RAG Solution |
|---|---|
| "RAG retrieves 2025 Standard Plan when claim refers to Obsolete 2018 Rider" | Hard temporal filter: `effective_date <= claim_date <= expiry_date` eliminates wrong-version policies before semantic search |
| "AI hallucinates a denial because it's looking at the wrong map" | Temporal RAG ensures the LLM always sees the correct policy version for the claim date |
| "Inconsistent policy references" | Regex extraction of policy refs from spatial tokens (English + Arabic patterns), then reference-based lookup |
| "Multilingual Arabic/English" | multilingual-e5-large embeddings, Arabic section detection in chunker, Arabic regex patterns in retriever |
| "OCR extracts text in the wrong order" | RAG consumes the already-corrected SpatialToken stream from Layer 1 (XY-Cut ordered) |
| "Distinguish Input Failure vs Intelligence Failure" | `retrieval_method` metadata: "temporal_filtered" (0.9) vs "semantic_only" (0.3 + Type B warning) — enables root cause separation |
