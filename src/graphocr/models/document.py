"""Document models for ingestion and page representation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from uuid_extensions import uuid7


class PageImage(BaseModel):
    """A single normalized page image ready for OCR."""

    page_id: str = Field(default_factory=lambda: str(uuid7()))
    document_id: str
    page_number: int
    image_path: str  # Local path or MinIO object key
    width_px: int
    height_px: int
    dpi: int = 300
    is_deskewed: bool = False
    is_contrast_enhanced: bool = False


class RawDocument(BaseModel):
    """An incoming document before processing."""

    document_id: str = Field(default_factory=lambda: str(uuid7()))
    source_path: str  # Original file path or upload key
    file_format: str  # pdf, tiff, jpeg, png
    file_size_bytes: int
    num_pages: int = 0
    jurisdiction: str = ""  # For data residency
    received_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentBatch(BaseModel):
    """A batch of documents for pipeline processing."""

    batch_id: str = Field(default_factory=lambda: str(uuid7()))
    documents: list[RawDocument]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    jurisdiction: str = ""

    @property
    def size(self) -> int:
        return len(self.documents)
