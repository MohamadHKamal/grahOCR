# GraphOCR — Documentation Index

**Deterministic Trust Layer for Insurance Claim Processing**
**100k multilingual handwritten claims/day | 98% accuracy target**

---

## Top-Level Documents

| Document | Description |
|---|---|
| [Handwritten Draft (PDF)](../HandWrittingArchOverview_draft%20planing.pdf) | Original handwritten architecture planning notes and flowcharts |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Full technical architecture with all thresholds, formulas, code paths, and implementation details |
| [ARCHITECTURE_OVERVIEW.md](../ARCHITECTURE_OVERVIEW.md) | Simplified architecture overview — descriptions, diagrams, no implementation details |

---

## Detailed Documentation (docs/)

Each document below is a deep-dive into one subsystem, derived from the source code and aligned with the [Problem Statement](../src/graphocr/problemStatment.txt).

| Document | Scope | Problem Solved |
|---|---|---|
| [Layer 1 — OCR + Spatial Foundation](LAYER1_OCR_SPATIAL_FOUNDATION.md) | Ingestion, dual-engine OCR, spatial assembly, reading order, failure classification | **Serialization Gore** — OCR reads across columns, mixing data |
| [Layer 2 — Adversarial Verification](LAYER2_ADVERSARIAL_VERIFICATION.md) | Multi-agent red team, Neo4j graph validation, self-healing back-propagation | **Self-Healing Pipeline** — multi-agent red team, not passive LLM-as-judge |
| [Layer 3 — Tiered Inference & Routing](LAYER3_TIERED_INFERENCE.md) | Traffic controller, uncertainty scoring, circuit breaker, accuracy monitoring | **98% Accuracy at Scale** — mathematical routing, automated failure response |
| [RAG — Temporal Retrieval](RAG_TEMPORAL_RETRIEVAL.md) | Temporal policy retriever, vector store, chunking, embeddings | **Contextual Hallucination** — wrong policy version retrieved |
| [DSPy — Optimization & Mentorship](DSPY_OPTIMIZATION.md) | MIPRO optimizer, gradient monitor, supervisor, mentor mode | **Deterministic Automation** — not "better prompting", plus junior mentorship |
| [Compliance — Federation](COMPLIANCE_FEDERATION.md) | Jurisdiction routing, data residency, federated learning | **Sovereign Data Constraints** — health data stays in-jurisdiction |
| [Infrastructure & Config](INFRASTRUCTURE_CONFIG.md) | Docker services, API, CLI, YAML configuration, observability | Complete deployment and operations reference |
| [Sample Data](../sample%20data/README.md) | 3 handwritten Arabic prescriptions + 3 synthetic PDFs | Each document targets a specific pipeline failure mode |

---

## Problem Statement Coverage Matrix

| Problem Requirement | Doc(s) |
|---|---|
| Serialization Gore (horizontal reading across columns) | Layer 1 |
| Contextual Hallucination (wrong policy version) | RAG |
| 100k claims/day, multilingual, handwritten | Layer 1, Layer 3 |
| Single Source of Truth (SpatialToken schema) | Layer 1 |
| Distinguish Input Failure vs Intelligence Failure | Layer 1 (Type A), RAG (Type B) |
| Federated / sovereign data constraints | Compliance |
| Traffic Controller with Uncertainty Score | Layer 3 |
| Mathematical threshold (10% expensive / 90% cheap) | Layer 3 |
| Accuracy Decay monitoring + Circuit Breaker | Layer 3 |
| Multi-agent red team (not passive LLM-as-judge) | Layer 2 |
| Post-Mortem Agent updates core logic globally | Layer 2 |
| Hybrid Graph-OCR Validation (logical impossibilities) | Layer 2 |
| Back-propagation without restarting batch | Layer 2 |
| Agentic Supervisor for DSPy mentorship | DSPy |
| Deterministic system, not "better prompting" | DSPy |
