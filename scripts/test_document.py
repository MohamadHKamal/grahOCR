"""Quick test script — process a document through Layer 1 (OCR + analysis).

Works without vLLM, Neo4j, or Redis. Just needs the conda env.

Usage:
    conda activate graphocr
    python scripts/test_document.py /path/to/claim.pdf
    python scripts/test_document.py /path/to/claim.jpg
    python scripts/test_document.py /path/to/claim.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Suppress noisy OCR logs
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


def main(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    fmt = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"

    print(f"\n{'='*60}")
    print(f"  GraphOCR Layer 1 Test — {path.name}")
    print(f"{'='*60}\n")

    # --- Step 1: Ingest ---
    print("[1/6] Ingesting document...")
    from graphocr.models.document import RawDocument
    from graphocr.layer1_foundation.ingestion import load_document

    doc = RawDocument(
        source_path=str(path),
        file_format=fmt,
        file_size_bytes=path.stat().st_size,
    )
    pages = load_document(doc, f"/tmp/graphocr_test/{doc.document_id}")
    print(f"       Pages: {len(pages)}")
    for p in pages:
        print(f"       Page {p.page_number}: {p.width_px}x{p.height_px}px @ {p.dpi}dpi")

    # --- Step 2: OCR ---
    print("\n[2/6] Running PaddleOCR...")
    from graphocr.layer1_foundation.ocr_paddleocr import PaddleOCREngine

    ocr = PaddleOCREngine()
    all_tokens = []
    for page in pages:
        tokens = ocr.extract(page)
        all_tokens.extend(tokens)
        print(f"       Page {page.page_number}: {len(tokens)} tokens extracted")

    if not all_tokens:
        print("\n  No tokens extracted! The document may be empty or unreadable.")
        return

    # --- Step 3: Spatial Assembly ---
    print("\n[3/6] Assembling spatial tokens...")
    from graphocr.layer1_foundation.spatial_assembler import assemble_tokens, group_into_lines

    assembled = assemble_tokens([all_tokens])
    lines = group_into_lines(assembled)
    print(f"       Tokens: {len(assembled)}")
    print(f"       Lines: {len(lines)}")

    # --- Step 4: Reading Order + Language ---
    print("\n[4/6] Assigning reading order + language...")
    from graphocr.layer1_foundation.reading_order import assign_reading_order
    from graphocr.layer1_foundation.language_detector import assign_languages

    ordered = assign_reading_order(assembled)
    ordered = assign_languages(ordered)

    ar_count = sum(1 for t in ordered if t.language.value == "ar")
    en_count = sum(1 for t in ordered if t.language.value == "en")
    mixed_count = sum(1 for t in ordered if t.language.value == "mixed")
    print(f"       Arabic tokens:  {ar_count}")
    print(f"       English tokens: {en_count}")
    print(f"       Mixed tokens:   {mixed_count}")

    # --- Step 5: Failure Classification ---
    print("\n[5/6] Checking for failures...")
    from graphocr.layer1_foundation.failure_classifier import classify_failures

    failures = classify_failures(ordered)
    if failures:
        print(f"       DETECTED {len(failures)} failure(s):")
        for f in failures:
            print(f"         - [{f.failure_type.value}] {f.evidence[:80]}")
            print(f"           Severity: {f.severity:.2f} | Remedy: {f.suggested_remedy}")
    else:
        print("       No failures detected.")

    # --- Step 6: Traffic Routing ---
    print("\n[6/6] Traffic routing decision...")
    from graphocr.layer3_inference.traffic_controller import route_document

    decision = route_document(ordered, failures)
    print(f"       Route: {decision.path.value}")
    print(f"       Uncertainty score: {decision.uncertainty_score:.4f}")
    print(f"       Confidence mean:   {decision.confidence_mean:.4f}")
    print(f"       Handwriting ratio: {decision.handwriting_ratio:.4f}")
    print(f"       Language mixing:   {decision.language_mixing_ratio:.4f}")
    print(f"       Reason: {decision.reason}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  EXTRACTED TEXT (reading order)")
    print(f"{'='*60}\n")

    for token in sorted(ordered, key=lambda t: t.reading_order):
        lang_tag = f"[{token.language.value:2s}]"
        conf_tag = f"({token.confidence:.0%})"
        hw_tag = " [HW]" if token.is_handwritten else ""
        page_tag = f"p{token.bbox.page_number}"
        print(f"  {page_tag} {lang_tag} {conf_tag}{hw_tag} {token.text}")

    # --- Provenance dump ---
    print(f"\n{'='*60}")
    print(f"  PROVENANCE TRAIL (first 10 tokens)")
    print(f"{'='*60}\n")
    for token in sorted(ordered, key=lambda t: t.reading_order)[:10]:
        print(f"  {token.to_provenance_str()}")

    # --- Stats ---
    confidences = [t.confidence for t in ordered]
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)

    print(f"\n{'='*60}")
    print(f"  STATISTICS")
    print(f"{'='*60}")
    print(f"  Total tokens:     {len(ordered)}")
    print(f"  Total lines:      {len(lines)}")
    print(f"  Total pages:      {len(pages)}")
    print(f"  Avg confidence:   {avg_conf:.4f}")
    print(f"  Min confidence:   {min_conf:.4f}")
    print(f"  Max confidence:   {max_conf:.4f}")
    print(f"  Failures:         {len(failures)}")
    print(f"  Routing:          {decision.path.value}")
    print(f"  Document ID:      {doc.document_id}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_document.py <path-to-document>")
        print("Supported: PDF, PNG, JPEG, TIFF")
        sys.exit(1)

    main(sys.argv[1])
