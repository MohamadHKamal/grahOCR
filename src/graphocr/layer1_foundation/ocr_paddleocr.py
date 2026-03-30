"""PaddleOCR adapter — primary OCR engine with strong Arabic support."""

from __future__ import annotations

from paddleocr import PaddleOCR

from graphocr.core.logging import get_logger
from graphocr.layer1_foundation.ocr_engine import OCREngine
from graphocr.models.document import PageImage
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


class PaddleOCREngine(OCREngine):
    """PaddleOCR adapter.

    Configured for multilingual Arabic/English extraction.
    Returns bounding boxes natively — no post-processing needed.
    """

    def __init__(self, lang: str = "ar"):
        # Disable PaddleOCR's built-in orientation/unwarping — we handle
        # rotation in ingestion.py already. This saves ~2s per init.
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @property
    def name(self) -> str:
        return "paddleocr"

    def extract(self, page: PageImage) -> list[SpatialToken]:
        """Run PaddleOCR on a page image and return SpatialTokens."""
        result = self._ocr.ocr(page.image_path)

        tokens: list[SpatialToken] = []
        if not result:
            logger.warning("paddleocr_empty_result", page_id=page.page_id)
            return tokens

        first = result[0]

        # PaddleOCR v3: result[0] is an OCRResult object with dict-like access
        # Keys: rec_texts (list[str]), dt_polys (list[ndarray]), rec_scores (list[float])
        if hasattr(first, "get") and first.get("rec_texts") is not None:
            texts = first.get("rec_texts", [])
            polys = first.get("dt_polys", [])
            scores_raw = first.get("rec_scores", None)
            scores = list(scores_raw) if scores_raw is not None else [0.9] * len(texts)

            for idx in range(len(texts)):
                text = str(texts[idx]).strip()
                if not text:
                    continue

                poly = polys[idx]
                # poly is a numpy array of shape (4,2) or a list of 4 points
                poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                xs = [p[0] for p in poly_list]
                ys = [p[1] for p in poly_list]

                confidence = float(scores[idx]) if idx < len(scores) else 0.9

                tokens.append(SpatialToken(
                    text=text,
                    bbox=BoundingBox(
                        x_min=float(min(xs)),
                        y_min=float(min(ys)),
                        x_max=float(max(xs)),
                        y_max=float(max(ys)),
                        page_number=page.page_number,
                    ),
                    reading_order=idx,
                    confidence=confidence,
                    ocr_engine=self.name,
                ))

        # PaddleOCR v2: result[0] is a list of [bbox_points, (text, confidence)]
        elif isinstance(first, list):
            for idx, line in enumerate(first):
                try:
                    bbox_points, (text, confidence) = line
                    text = str(text).strip()
                    if not text:
                        continue
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    tokens.append(SpatialToken(
                        text=text,
                        bbox=BoundingBox(
                            x_min=float(min(xs)),
                            y_min=float(min(ys)),
                            x_max=float(max(xs)),
                            y_max=float(max(ys)),
                            page_number=page.page_number,
                        ),
                        reading_order=idx,
                        confidence=float(confidence),
                        ocr_engine=self.name,
                    ))
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug("paddleocr_parse_line_failed", idx=idx, error=str(e))
                    continue

        logger.info(
            "paddleocr_extracted",
            page_id=page.page_id,
            tokens=len(tokens),
            avg_confidence=sum(t.confidence for t in tokens) / max(len(tokens), 1),
        )
        return tokens
