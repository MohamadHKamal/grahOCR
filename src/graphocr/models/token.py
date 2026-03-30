"""SpatialToken — the atomic unit of the pipeline.

Every OCR-extracted token carries its spatial, linguistic, and provenance
metadata through all three layers. This is the provenance backbone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from uuid_extensions import uuid7

from graphocr.core.types import Language, ZoneLabel


class BoundingBox(BaseModel):
    """Pixel coordinates on the source page image."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page_number: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union with another bounding box."""
        if self.page_number != other.page_number:
            return 0.0
        x_overlap = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        y_overlap = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = x_overlap * y_overlap
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class SpatialToken(BaseModel):
    """The atomic unit of the pipeline.

    Every text token carries its spatial, linguistic, and provenance
    metadata end-to-end through all three layers.
    """

    token_id: str = Field(default_factory=lambda: str(uuid7()))
    text: str
    bbox: BoundingBox
    reading_order: int = Field(description="Global reading order index on the page")
    language: Language = Language.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, description="OCR engine confidence")
    ocr_engine: str = Field(description="Which engine produced this token")
    zone_label: ZoneLabel | None = None
    is_handwritten: bool = False
    line_group_id: str | None = Field(
        default=None, description="Groups tokens into logical lines"
    )
    normalized_text: str | None = Field(
        default=None, description="Post-normalization (e.g., Arabic diacritic removal)"
    )

    def to_provenance_str(self) -> str:
        """Human-readable provenance string for audit trails."""
        b = self.bbox
        return (
            f"[{self.token_id[:8]}] '{self.text}' "
            f"page={b.page_number} ({b.x_min:.0f},{b.y_min:.0f})-({b.x_max:.0f},{b.y_max:.0f}) "
            f"conf={self.confidence:.2f} lang={self.language.value} engine={self.ocr_engine}"
        )
