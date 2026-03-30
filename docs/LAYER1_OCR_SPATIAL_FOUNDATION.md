# Layer 1 — OCR + Spatial Foundation

**System**: GraphOCR Deterministic Trust Layer
**Scope**: Document ingestion, dual-engine OCR, spatial assembly, reading order, language detection, failure classification
**Problem Solved**: **Serialization Gore** — standard OCR reads horizontally across pages, mixing patients' data, merging prices with dates, creating "meaningless soup"

---

## 1. Purpose & Problem Context

When a doctor writes a diagnosis across two columns, or a pharmacy stamp overlaps a policy number, standard OCR pipelines commit **Input Failure (Type A)**. They read left-to-right across the entire page width, producing nonsensical token sequences that downstream LLMs cannot recover from.

Layer 1 solves this by producing a spatially-aware, ordered token stream where every token carries its exact pixel coordinates, reading order, language, zone label, and OCR provenance. This is the **atomic unit** that flows through the entire pipeline.

---

## 2. Architecture Overview

```
Scanned Document (PDF / TIFF / JPEG / PNG)
    |
    v
[Document Ingestion] ──── 8-step image normalization
    |
    v
300 DPI Normalized PNG (grayscale-in-BGR, NOT binarized)
    |
    ├──> [PaddleOCR] ──── text + bboxes + confidence
    |
    └──> [Surya] ──── region bboxes + zone labels (optional)
              |
              v
        [Spatial Assembler] ──── IoU merge (threshold=0.3)
              |
              v
        [Reading Order] ──── XY-Cut with RTL detection
              |
              v
        [Language Detector] ──── per-token Unicode analysis
              |
              v
        [Metadata Enricher] ──── zone labels + handwriting flags
              |
              v
        [Failure Classifier] ──── 4 Type-A checks
              |
              v
        SpatialToken Stream (the Single Source of Truth)
```

**Source Files:**
| Component | File |
|---|---|
| Document Ingestion | `layer1_foundation/ingestion.py` |
| OCR Engine ABC | `layer1_foundation/ocr_engine.py` |
| PaddleOCR Adapter | `layer1_foundation/ocr_paddleocr.py` |
| Surya Layout Engine | `layer1_foundation/ocr_surya.py` |
| Spatial Assembler | `layer1_foundation/spatial_assembler.py` |
| Reading Order | `layer1_foundation/reading_order.py` |
| Language Detector | `layer1_foundation/language_detector.py` |
| Metadata Enricher | `layer1_foundation/metadata_enricher.py` |
| Failure Classifier | `layer1_foundation/failure_classifier.py` |
| SpatialToken Schema | `models/token.py` |
| Core Types | `core/types.py` |

---

## 3. The SpatialToken — Single Source of Truth

Every token in the system is a `SpatialToken` (`models/token.py`). This is the **metadata schema** that enforces "Semantic Spatial" mapping consistency across the entire team — both senior and junior engineers consume the same structure.

```python
class SpatialToken(BaseModel):
    token_id: str          # UUID v7 (time-ordered, unique)
    text: str              # Raw OCR text
    bbox: BoundingBox      # x_min, y_min, x_max, y_max, page_number
    reading_order: int     # Global index (0, 1, 2, ...)
    language: Language      # ARABIC | ENGLISH | MIXED | UNKNOWN
    confidence: float       # 0.0–1.0
    ocr_engine: str        # "paddleocr", "surya", or "paddleocr+surya"
    zone_label: ZoneLabel   # HEADER | BODY | STAMP | TABLE_CELL | FOOTER | ...
    is_handwritten: bool    # True if handwriting detected
    line_group_id: str      # "line_0", "line_1", ...
    normalized_text: str    # Post-processed (diacritics removed, etc.)
```

### BoundingBox

```python
class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page_number: int
```

**Properties**: `center` (cx, cy), `width`, `height`, `area`, `iou(other)` (returns 0.0 for cross-page comparisons).

### Provenance Tracking

```python
token.to_provenance_str()
# → "[a1b2c3d4] 'metformin' page=1 (234,567)-(345,589) conf=0.92 lang=en engine=paddleocr"
```

Every downstream extraction field carries `source_tokens` (list of token_ids), enabling **coordinate-level traceability** from final output back to pixel positions on the original scan. This is the "Deterministic Trust Layer" — every word the AI says can be traced to a specific set of coordinates on a specific page.

### Enums (`core/types.py`)

```python
class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class ZoneLabel(str, Enum):
    HEADER, BODY, STAMP, SIGNATURE, TABLE_CELL, FOOTER, MARGIN_NOTE, LOGO
```

---

## 4. Document Ingestion Pipeline

**File**: `layer1_foundation/ingestion.py`
**Function**: `load_document(doc, output_dir, max_side=2500)`
**Output**: `list[PageImage]` — normalized page images at 300 DPI

The ingestion module normalizes raw scans through an **8-step image processing pipeline**, optimized for Arabic medical documents based on Invizo (2025), Arabic OCR surveys, and PaddleOCR best practices.

### 4.1 Pre-OCR Steps

| Step | Function | What It Does | Why |
|---|---|---|---|
| 1 | `_fix_exif_rotation` | Reads EXIF tag 274, applies rotation (90/180/270) | Phone cameras store orientation in metadata rather than rotating pixels |
| 2 | `_correct_orientation` | Contour analysis: if >60% regions taller than wide → rotate. Upside-down: compares ink density in top vs bottom 15% (threshold: ratio > 3.0) | Catches documents scanned upside-down or sideways |
| 3 | Auto-resize | Lanczos interpolation if max side > 2500px | PaddleOCR optimal at 1500–2500px |

### 4.2 Image Enhancement Pipeline (`_normalize_image`)

**Order is critical** — each step depends on the output of the previous:

| Step | Function | Parameters | Why This Matters for Arabic Medical Docs |
|---|---|---|---|
| 4 | `_suppress_stamps` | HSV: saturation>80, value>50. Red (hue 0-10/170-180), Blue (100-130), Green (35-85). **Conservative fade: +80 brightness** (not erasure) | Pharmacy stamps overlap policy numbers. Fading preserves underlying dark ink strokes |
| 5 | `_correct_lighting` | Morphological closing, kernel = max(51, 6% of min dimension). Division: `gray / background * 255` | **Highest-impact step** for phone-captured documents. Flattens gradients from camera shadows |
| 6 | FNLM Denoising | `h=3`, templateWindow=7, searchWindow=21 | Standard aggressive `h>5` **destroys Arabic diacritical dots** (ب vs ت are different characters distinguished only by dots) |
| 7 | CLAHE | clipLimit=2.5, tileGrid=8x8 | Recovers faded prescription ink and diacritical dots |
| 8 | `_deskew_projection` | Projection profile, -5° to +5° in 0.2° steps, maximizes horizontal projection variance | More robust than Hough lines for Arabic — Arabic has dense vertical strokes that confuse Hough |

### 4.3 Critical Design Decision: No Binarization

PaddleOCR's internal DBNet text detector relies on **grayscale spatial gradients** to locate text boundaries. Pre-binarizing destroys this sub-pixel gradient information and severely hurts detection accuracy. The output is **grayscale-in-BGR** (3-channel) for PaddleOCR compatibility.

### 4.4 PageImage Output

```python
class PageImage(BaseModel):
    page_id: str           # UUID v7
    document_id: str
    page_number: int
    image_path: str        # Path to 300 DPI PNG
    width_px: int
    height_px: int
    dpi: int = 300
    is_deskewed: bool = True
    is_contrast_enhanced: bool = True
```

---

## 5. Dual-Engine OCR Strategy

### 5.1 OCR Engine Abstraction (`ocr_engine.py`)

```python
class OCREngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def extract(self, page: PageImage) -> list[SpatialToken]: ...

    def extract_batch(self, pages: list[PageImage]) -> dict[int, list[SpatialToken]]:
        # Default: sequential. Override for GPU batching.
```

**Contract**: Reading order is NOT assigned here — that's the spatial assembler's job. Each token is tagged with the engine name for provenance.

### 5.2 PaddleOCR — Primary Text Extraction (`ocr_paddleocr.py`)

```python
class PaddleOCREngine(OCREngine):
    def __init__(self, lang="ar"):
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,  # We handle rotation
            use_doc_unwarping=False,              # We handle deskew
            use_textline_orientation=False,
        )
```

- **Best Arabic script support** with built-in angle correction
- Handles mixed Arabic/English text natively
- Returns: text content + bounding box polygons + per-token confidence scores
- **Always active** — this is the primary engine for all text extraction
- Bounding box conversion: receives polygon (4 corners) → extracts min/max x,y for axis-aligned box

### 5.3 Surya — Layout Region Detection (`ocr_surya.py`, Optional)

```python
class SuryaLayoutEngine(OCREngine):
    def __init__(self, use_recognition=True):
        self._foundation_predictor = FoundationPredictor()
        self._det_predictor = DetectionPredictor()
        self._layout_predictor = LayoutPredictor(self._foundation_predictor)
        if use_recognition:
            self._rec_predictor = RecognitionPredictor(self._foundation_predictor)
```

**Two levels of output**:
1. `RecognitionPredictor` → text extraction with bounding boxes, line-level confidence, sorted reading order
2. `LayoutPredictor` → zone-labeled bounding boxes (what TYPE of region each area is)

**Zone Mapping** (`_SURYA_ZONE_MAP`):
| Surya Label | GraphOCR ZoneLabel |
|---|---|
| Text | BODY |
| Title | HEADER |
| Table | TABLE_CELL |
| Figure | LOGO |
| Caption | FOOTER |
| Header / Page-header | HEADER |
| Footer / Page-footer | FOOTER |

### 5.4 Pipeline Modes

| Mode | Engines | Zone Labels | Stamp Detection | Latency |
|---|---|---|---|---|
| `Pipeline()` | PaddleOCR only | None | Not available | Fast |
| `Pipeline(use_surya=True)` | PaddleOCR + Surya | Full | Active | +2-4 min/page |
| `Pipeline(use_surya=True, use_paddle=False)` | Surya only | Full | Active | +2-4 min/page |

---

## 6. Spatial Assembler — Token Merge Strategy

**File**: `layer1_foundation/spatial_assembler.py`
**Function**: `assemble_tokens(token_streams, iou_threshold=0.3)`

Instead of discarding overlapping detections from multiple engines, the assembler **combines the best attributes of both** into a single enriched token.

### 6.1 Merge Algorithm

1. Start with primary engine's tokens (PaddleOCR)
2. For each secondary engine token (Surya):
   - Find best IoU overlap in primary set
   - If IoU >= 0.3: **MERGE** both tokens
   - If no overlap: **ADD** as new detection (if has text)

### 6.2 Merge Strategy (`_merge_tokens`)

| Attribute | Source | Strategy |
|---|---|---|
| Text | Higher confidence engine | PaddleOCR always provides content (Surya may not) |
| Bounding box | Both | **Union** (IoU 0.3–0.7) for coverage; **Intersection** (IoU > 0.7) for precision |
| Confidence | Both | **Bayesian boost**: `1 - (1 - conf_a) * (1 - conf_b)`, capped at 0.99 |
| Zone label | Surya | Inherited (HEADER, BODY, STAMP, TABLE) — PaddleOCR doesn't detect zones |
| Language | Either | Prefer non-UNKNOWN |
| Handwriting | Either | OR — if either detects it, flag it |
| Engine | Both | Concatenated: `"paddleocr+surya"` |

**Why IoU threshold is 0.3** (not 0.5): Different engines produce slightly different bounding boxes for the same text region. 0.3 catches these partial overlaps while avoiding false merges of genuinely separate regions.

**Confidence Boosting Example**: PaddleOCR=0.7, Surya=0.8 → Merged=`1 - (0.3)(0.2) = 0.94`. Two independent detectors agreeing is stronger evidence than either alone.

### 6.3 Line Grouping

```python
group_into_lines(tokens, y_tolerance=10.0)
```
- Sorts tokens by `(center_y, center_x)`
- Groups consecutive tokens with `|delta_y| <= 10px`
- Assigns `line_group_id` (e.g., `"line_0"`, `"line_1"`)

---

## 7. Reading Order — XY-Cut with RTL Awareness

**File**: `layer1_foundation/reading_order.py`
**Function**: `assign_reading_order(tokens, rtl_detection=True)`

The **XY-Cut algorithm** is a well-established technique for Document Image Analysis. It recursively decomposes the page into columns and rows by finding the widest "valleys" (white spaces) between token clusters.

### 7.1 Algorithm

1. **Vertical Split** (`_find_vertical_split`): Find largest horizontal gap >= 30px between x-ranges → split into left/right columns, recurse
2. **Horizontal Split** (`_find_horizontal_split`): Find largest vertical gap >= 10px between y-ranges → split into top/bottom rows, recurse
3. **Within-Line Sort**: If no splits possible, sort by `(center_y, center_x)` for LTR or reversed for RTL

### 7.2 RTL Detection (`_is_majority_rtl`)

Counts Arabic script characters in Unicode ranges:
- `\u0600-\u06FF` (Arabic Base)
- `\u0750-\u077F` (Arabic Supplement)
- `\uFB50-\uFDFF` (Arabic Presentation Forms)
- `\uFE70-\uFEFF` (Arabic Ligatures)

If **>50%** of characters in the token group are Arabic → apply RTL ordering (right column first).

This context-aware spatial serialization **prevents the "Serialization Gore"** described in the problem brief — reading across two distinct medical columns haphazardly is eliminated.

---

## 8. Language Detection

**File**: `layer1_foundation/language_detector.py`

### Per-Token Classification

```python
def detect_language(text: str) -> Language:
    arabic_ratio = arabic_chars / (arabic_chars + latin_chars)
    if arabic_ratio > 0.7:  return ARABIC
    if arabic_ratio < 0.3:  return ENGLISH
    if no characters:       return UNKNOWN
    else:                   return MIXED
```

Arabic character regex: `[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]`

---

## 9. Metadata Enrichment

**File**: `layer1_foundation/metadata_enricher.py`

### Zone Label Assignment (`enrich_tokens_with_zones`)

For tokens not already labeled by the spatial assembler merge:
1. Find best-overlapping Surya layout zone via IoU
2. Fallback: check if token center is inside any zone
3. Assign `zone_label` from best match

### Handwriting Detection (`detect_handwriting`)

Three conditions must ALL be true:
| Condition | Threshold | Rationale |
|---|---|---|
| Height deviation | > 0.5 (50% from page median) | Handwriting is irregular in size |
| OCR confidence | < 0.8 | OCR engines are less confident on handwriting |
| Aspect ratio | > 3.0 (width/height) | Handwritten words tend to be wider |

---

## 10. Failure Classifier — Type A Detection

**File**: `layer1_foundation/failure_classifier.py`
**Function**: `classify_failures(tokens) -> list[FailureClassification]`

Detects four patterns of **Type A (spatial-blind) failure** that indicate the OCR reading order is corrupted:

### Check 1: Spatial Jumps (`_check_reading_order_jumps`)

Consecutive tokens in reading order >**800px** apart spatially (both with confidence >0.7).

- Severity: `min(1.0, distance / 1000)`
- Threshold tuned for 2500px images — handwriting can wander ~500px without being a real column merge

### Check 2: Nonsensical Sequences (`_check_nonsensical_sequences`)

Sliding window of 5 tokens checking for rapid type alternation (num-text-num-text-num) — classic horizontal-scan corruption.

- Token types: `"num"` (digits/currency), `"text"` (letters), `"mixed"`, `"empty"`
- Threshold: >= 4 type changes in 5-token window
- Severity: 0.6

### Check 3: Stamp/Seal Overlap (`_check_stamp_overlap`)

Detects bounding box overlap (IoU > **0.05**) between STAMP/LOGO/SIGNATURE zone tokens and BODY tokens.

- Catches the "pharmacy stamp overlaps a policy number" scenario
- Severity: `min(1.0, 0.5 + 0.1 * num_overlapping)`

### Check 4: Cross-Column Merge (`_check_cross_column_merge`)

Clusters tokens into left/right columns by X-center:
- Column gap threshold: **40%** of page width
- Dead zone (ignore middle): **15%** of page width
- Min tokens per column: max(4, 20% of total)
- Max allowed column switches: `max(3, token_count / 10)`

If switches exceed threshold → Type A failure. Prevents false positives on single-column handwritten prescriptions where text wanders.

---

## 11. Key Thresholds Summary

| Component | Parameter | Value | Tuning Note |
|---|---|---|---|
| Ingestion | MAX_SIDE | 2500 px | Reduce to 1500 for CPU |
| Ingestion | Stamp saturation | > 80 HSV | Conservative fade, not erasure |
| Ingestion | CLAHE clipLimit | 2.5 | Tuned for faded prescriptions |
| Ingestion | Deskew range | +/- 5 degrees, 0.2 steps | Projection profile (not Hough) |
| Ingestion | Denoising h | 3 | Low to preserve Arabic dots |
| Spatial Assembler | IoU merge threshold | 0.3 | Catches partial engine overlaps |
| Spatial Assembler | Line y-tolerance | 10 px | For `group_into_lines()` |
| Reading Order | Vertical split gap | >= 30 px | Column separator detection |
| Reading Order | Horizontal split gap | >= 10 px | Line separator detection |
| Reading Order | Arabic RTL threshold | > 50% Arabic chars | Triggers right-to-left ordering |
| Language Detector | Arabic classification | > 0.7 ratio | Per-token language assignment |
| Language Detector | English classification | < 0.3 ratio | Per-token language assignment |
| Failure Classifier | Spatial jump | > 800 px | Tuned for 2500px images |
| Failure Classifier | Stamp overlap IoU | > 0.05 | Low threshold catches partial overlaps |
| Failure Classifier | Column gap | 40% page width | Minimum column separation |
| Failure Classifier | Column dead zone | 15% page width | Ignore middle region |
| Failure Classifier | Max column switches | max(3, N/10) | Scales with document size |
| Handwriting | Height deviation | > 0.5 | 50% from page median |
| Handwriting | OCR confidence | < 0.8 | Lower confidence = likely handwritten |
| Handwriting | Aspect ratio | > 3.0 | Width/height ratio |

---

## 12. How Layer 1 Connects to the Problem Statement

| Problem Statement Requirement | Layer 1 Solution |
|---|---|
| "OCR reads horizontally across the page, mixing two different patients' data" | XY-Cut algorithm splits columns before assigning reading order. RTL-aware for Arabic. |
| "Merging a price with a date, creating meaningless soup" | Failure Classifier Check 2 detects nonsensical type alternation (num-text-num-text) |
| "Pharmacy stamp overlaps a policy number" | Stamp suppression in preprocessing (step 4) + Failure Classifier Check 3 (stamp/body overlap) |
| "Handwritten, multi-lingual" | Per-token language detection (Arabic/English/Mixed), handwriting flags, confidence tracking |
| "Scans of handwritten/illegible doctor notes" | FNLM denoising preserves Arabic dots, CLAHE recovers faded ink, deskew fixes tilted scans |
| "Single Source of Truth for Senior and Junior engineers" | SpatialToken schema — same structure consumed by all downstream components |
| "Metadata schema for Semantic Spatial mapping" | SpatialToken carries: token_id, text, bbox, reading_order, language, engine, zone_label, handwriting flag, line_group, normalized_text |
| "Every word traceable to coordinates on a specific page" | `to_provenance_str()` → `[id] 'text' page=N (x1,y1)-(x2,y2) conf=C lang=L engine=E` |
