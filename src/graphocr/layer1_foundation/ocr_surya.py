"""Surya adapter — layout-aware OCR with text recognition."""

from __future__ import annotations

from PIL import Image
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor

from graphocr.core.logging import get_logger
from graphocr.core.types import ZoneLabel
from graphocr.layer1_foundation.ocr_engine import OCREngine
from graphocr.models.document import PageImage
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)

# Map Surya layout labels to our zone labels
_SURYA_ZONE_MAP: dict[str, ZoneLabel] = {
    "Text": ZoneLabel.BODY,
    "Title": ZoneLabel.HEADER,
    "Table": ZoneLabel.TABLE_CELL,
    "Figure": ZoneLabel.LOGO,
    "Caption": ZoneLabel.FOOTER,
    "Header": ZoneLabel.HEADER,
    "Footer": ZoneLabel.FOOTER,
    "Page-footer": ZoneLabel.FOOTER,
    "Page-header": ZoneLabel.HEADER,
}


class SuryaLayoutEngine(OCREngine):
    """Surya adapter for OCR text recognition + layout detection.

    Provides both text extraction (via RecognitionPredictor) and
    structural analysis (via LayoutPredictor) for columns, tables, zones.
    """

    def __init__(self, use_recognition: bool = True):
        self._use_recognition = use_recognition
        self._foundation_predictor = FoundationPredictor()
        self._det_predictor = DetectionPredictor()
        self._layout_predictor = LayoutPredictor(self._foundation_predictor)
        if use_recognition:
            self._rec_predictor = RecognitionPredictor(self._foundation_predictor)
        else:
            self._rec_predictor = None

    @property
    def name(self) -> str:
        return "surya"

    def extract(self, page: PageImage) -> list[SpatialToken]:
        """Extract text + bounding boxes from a page using Surya OCR."""
        image = Image.open(page.image_path)
        return self._extract_from_image(image, page.page_number)

    def _extract_from_image(self, image: Image.Image, page_number: int) -> list[SpatialToken]:
        """Run Surya OCR on a PIL image — detection + recognition."""
        if not self._rec_predictor:
            # Layout-only mode: return region bboxes without text
            return self._extract_layout_only(image, page_number)

        # Full OCR: detection + recognition
        ocr_results = self._rec_predictor(
            [image],
            det_predictor=self._det_predictor,
            sort_lines=True,
        )

        tokens: list[SpatialToken] = []
        if not ocr_results:
            return tokens

        for idx, text_line in enumerate(ocr_results[0].text_lines):
            text = text_line.text.strip()
            if not text:
                continue

            bbox = text_line.bbox  # [x_min, y_min, x_max, y_max]
            token = SpatialToken(
                text=text,
                bbox=BoundingBox(
                    x_min=bbox[0],
                    y_min=bbox[1],
                    x_max=bbox[2],
                    y_max=bbox[3],
                    page_number=page_number,
                ),
                reading_order=idx,
                confidence=text_line.confidence or 0.0,
                ocr_engine=self.name,
            )
            tokens.append(token)

        logger.info("surya_ocr_complete", page_number=page_number, tokens=len(tokens))
        return tokens

    def _extract_layout_only(self, image: Image.Image, page_number: int) -> list[SpatialToken]:
        """Fallback: layout detection without text recognition."""
        layout_results = self._layout_predictor([image])

        tokens: list[SpatialToken] = []
        if not layout_results:
            return tokens

        for idx, layout_box in enumerate(layout_results[0].bboxes):
            bbox = layout_box.bbox
            token = SpatialToken(
                text="",
                bbox=BoundingBox(
                    x_min=bbox[0],
                    y_min=bbox[1],
                    x_max=bbox[2],
                    y_max=bbox[3],
                    page_number=page_number,
                ),
                reading_order=idx,
                confidence=layout_box.confidence or 0.9,
                ocr_engine=self.name,
            )
            tokens.append(token)

        logger.info("surya_detected_regions", page_number=page_number, regions=len(tokens))
        return tokens

    def detect_layout(self, page: PageImage) -> list[dict]:
        """Detect layout zones (header, body, table, etc.) on a page.

        Returns a list of dicts with 'bbox', 'zone_label', and 'confidence'.
        """
        image = Image.open(page.image_path)
        layout_results = self._layout_predictor([image])

        zones: list[dict] = []
        if not layout_results:
            return zones

        for layout_box in layout_results[0].bboxes:
            bbox = layout_box.bbox
            label = layout_box.label
            zone_label = _SURYA_ZONE_MAP.get(label, ZoneLabel.BODY)
            zones.append({
                "bbox": BoundingBox(
                    x_min=bbox[0], y_min=bbox[1],
                    x_max=bbox[2], y_max=bbox[3],
                    page_number=page.page_number,
                ),
                "zone_label": zone_label,
                "confidence": layout_box.confidence or 0.9,
                "raw_label": label,
            })

        logger.info("surya_layout_detected", page_id=page.page_id, zones=len(zones))
        return zones
