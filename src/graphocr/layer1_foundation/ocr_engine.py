"""Abstract OCR engine interface (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from graphocr.models.document import PageImage
from graphocr.models.token import SpatialToken


class OCREngine(ABC):
    """Base class for all OCR engine adapters.

    Each adapter must convert engine-specific output into a list of
    SpatialTokens with bounding boxes and confidence scores.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier used in SpatialToken.ocr_engine."""

    @abstractmethod
    def extract(self, page: PageImage) -> list[SpatialToken]:
        """Extract spatial tokens from a single page image.

        Args:
            page: Normalized page image.

        Returns:
            List of SpatialTokens with bounding boxes, text, and confidence.
            Reading order is NOT set here — that's the spatial assembler's job.
        """

    def extract_batch(self, pages: list[PageImage]) -> dict[int, list[SpatialToken]]:
        """Extract tokens from multiple pages. Override for GPU batching.

        Returns:
            Dict mapping page_number -> list of tokens.
        """
        return {page.page_number: self.extract(page) for page in pages}
