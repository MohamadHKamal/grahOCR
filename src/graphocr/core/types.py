"""Shared type aliases and enums."""

from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    TYPE_A_SPATIAL_BLIND = "spatial_blind"
    TYPE_B_CONTEXT_BLIND = "context_blind"


class ProcessingPath(str, Enum):
    CHEAP_RAIL = "cheap_rail"
    VLM_CONSENSUS = "vlm_consensus"


class ValidationStatus(str, Enum):
    VALID = "valid"
    FLAGGED = "flagged"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class AgentRole(str, Enum):
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    CHALLENGER = "challenger"
    POSTMORTEM = "postmortem"


class ZoneLabel(str, Enum):
    HEADER = "header"
    BODY = "body"
    STAMP = "stamp"
    SIGNATURE = "signature"
    TABLE_CELL = "table_cell"
    FOOTER = "footer"
    MARGIN_NOTE = "margin_note"
    LOGO = "logo"
