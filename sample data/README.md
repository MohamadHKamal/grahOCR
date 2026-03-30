# Sample Data — Test Documents

Each sample document is designed to trigger specific pipeline behaviors. Use these to verify that each layer is working correctly.

---

## Images (Handwritten Arabic Prescriptions)

Real-world scanned medical prescriptions from Egyptian clinics. These are the hardest documents for OCR — handwritten, bilingual, with stamps and overlapping elements.

### `1000093095.jpg` — Pediatric Prescription with Photos

| Property | Value |
|----------|-------|
| **Type** | Handwritten Arabic pediatric prescription |
| **Doctor** | Dr. Ahmed Fathy Gaber (Pediatrics, Ain Shams University) |
| **Language** | Arabic + English (drug names) |
| **Orientation** | Sideways (270 degree rotation needed) |
| **Key Challenges** | Photos of children pasted on the prescription (OCR noise), heavy handwriting, colored stamps at bottom, multiple Rx entries across the page, phone numbers and clinic details in margins |
| **Tests** | Auto-orientation correction, stamp suppression (422K pixels detected), mixed language detection, low confidence handling (avg ~0.45), VLM consensus routing |
| **Expected Route** | **VLM Consensus** (uncertainty ~0.62 — handwritten + low confidence) |

### `1000093096.jpg` — Handwritten Prescription with QR Code

| Property | Value |
|----------|-------|
| **Type** | Handwritten Arabic medical prescription |
| **Doctor** | Dr. Mahmoud Khalifa (clinic details in margin) |
| **Language** | Arabic + English (drug names: Differin, Fusidic, Imuran) |
| **Key Challenges** | QR code in corner (OCR noise), blue ink handwriting on white paper, circled numbers, Rx symbols, margin text with phone numbers and clinic stamps |
| **Tests** | Handwriting recognition, mixed script handling, noise filtering from QR code, blue ink detection |
| **Expected Route** | **VLM Consensus** (handwritten, moderate confidence) |

### `1000093097.jpg` — Handwritten Prescription with Printed Header

| Property | Value |
|----------|-------|
| **Type** | Handwritten Arabic prescription on printed clinic letterhead |
| **Doctor** | Dr. Khaled Youssef |
| **Language** | Arabic + English (drug names: Declac, Pariet ELN, Ranihs) |
| **Date** | 15/12/2024 |
| **Key Challenges** | Mix of printed header (clinic logo, name fields, vitals table) and handwritten prescription body. Blue pen on light blue paper. Printed form fields partially filled. Sideways orientation. |
| **Tests** | Mixed printed/handwritten detection, form field extraction, orientation correction, zone detection (header vs body) |
| **Expected Route** | **VLM Consensus** (handwritten body, mixed layout) |

---

## PDFs (Synthetic Typed Documents)

Purpose-built test documents designed to trigger specific pipeline failure modes. Each one targets a different aspect of the system.

### `test_column_issue.pdf` — Two-Column Layout (Serialization Gore Test)

| Property | Value |
|----------|-------|
| **Type** | Hospital Discharge Summary |
| **Layout** | **Two columns** — Patient Details (left) and Clinical Diagnosis (right) |
| **Content** | Left: Name: Ahmed Ali, DOB: 15-05-1985, ID: PAT-99201. Right: Code: ICD-10-J45 (Asthma), Admission: 10-02-2026, Ward: Respiratory Unit |
| **Purpose** | Tests the **Serialization Gore** problem — standard OCR reads horizontally across both columns, mixing "Name: Ahmed Ali Code: ICD-10-J45" into nonsense. The XY-Cut algorithm should detect the column gap and read each column separately. |
| **Expected Failures** | Spatial jump detected (~1974px between columns) |
| **Expected Route** | **Cheap Rail** (typed text, high confidence ~0.99) |
| **What To Verify** | Failure classifier flags the spatial jump. Reading order processes left column fully before right column. |

### `test_logical_conflict.pdf` — Date Impossibility (Neo4j Validation Test)

| Property | Value |
|----------|-------|
| **Type** | Insurance Policy Enrollment |
| **Content** | Client: Mohamed Hussein, DOB: **12-10-2020**, Policy Effective Date: **01-01-2015** |
| **Purpose** | Tests the **Logical Impossibility** detection — the policy effective date (2015) is BEFORE the client's birth date (2020). This is impossible. The document even says "Status: Approved (System Error Expected)". |
| **Expected Behavior** | When Layer 2 is active, the Neo4j Validator should flag this with **date sanity violation** (severity 0.95) — DOB is after policy date. Without LLM (Layer 1 only), the OCR reads the text correctly but cannot detect the logical conflict. |
| **What To Verify** | OCR extracts both dates correctly. With `run_llm=True`, the validator catches the impossible date relationship. |

### `test_low_confidence.pdf` — Pharmacy Invoice (Amount Extraction Test)

| Property | Value |
|----------|-------|
| **Type** | Pharmacy Invoice |
| **Content** | Total Amount: $1,500.00, Drug: Amoxicillin 500mg, handwritten note "(Urgent Delivery)" |
| **Purpose** | Tests **amount extraction** (regex pattern matching for currency), **drug name recognition**, and **mixed typed/handwritten content**. The amount has a gray highlight box around "500.00" to simulate OCR ambiguity. |
| **Expected Behavior** | Regex extracts $1,500.00 as total_amount. Drug name "Amoxicillin 500mg" extracted. The italic handwritten note may have lower confidence. |
| **Expected Route** | **Cheap Rail** (mostly typed, high confidence) |
| **What To Verify** | Amount parsing handles comma separator. Drug name extracted with dosage. |

---

## How To Run

```bash
# Single document
graphocr test-pipeline "sample data/images/1000093095.jpg" -v

# Single document with report
graphocr test-pipeline "sample data/pdfs/test_column_issue.pdf" -f both

# Compare engines on handwritten doc
graphocr test-pipeline "sample data/images/1000093095.jpg" --all-engines

# Batch test all images
python scripts/batch_test.py "sample data/images" --format both

# Batch test all PDFs
python scripts/batch_test.py "sample data/pdfs" --format both
```

---

## Expected Results Summary

| Document | Type | Tokens | Confidence | Route | Key Finding |
|----------|------|--------|------------|-------|-------------|
| `1000093095.jpg` | Handwritten Arabic | ~39 | ~45% | VLM Consensus | Auto-rotated 270, stamp suppressed, bilingual |
| `1000093096.jpg` | Handwritten Arabic | ~30-40 | ~40-50% | VLM Consensus | QR code noise, blue ink |
| `1000093097.jpg` | Handwritten + Printed | ~20-30 | ~50-60% | VLM Consensus | Mixed layout, form fields |
| `test_column_issue.pdf` | Typed English | ~10 | ~99% | Cheap Rail | Spatial jump detected (2-column) |
| `test_logical_conflict.pdf` | Typed English | ~8 | ~99% | Cheap Rail | DOB after policy date (Neo4j catches) |
| `test_low_confidence.pdf` | Typed + Handwritten | ~6 | ~95% | Cheap Rail | Amount + drug extraction |
