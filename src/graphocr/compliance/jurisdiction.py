"""Jurisdiction resolver — determines data residency rules for documents."""

from __future__ import annotations

from graphocr.core.exceptions import DataResidencyError
from graphocr.core.logging import get_logger

logger = get_logger(__name__)

# Jurisdiction -> allowed processing regions
_JURISDICTION_RULES: dict[str, dict] = {
    "SA": {  # Saudi Arabia
        "name": "Saudi Arabia",
        "allowed_regions": ["sa-riyadh", "sa-jeddah"],
        "requires_local_processing": True,
        "data_classification": "sensitive",
    },
    "AE": {  # UAE
        "name": "United Arab Emirates",
        "allowed_regions": ["ae-dubai", "ae-abudhabi"],
        "requires_local_processing": True,
        "data_classification": "sensitive",
    },
    "EG": {  # Egypt
        "name": "Egypt",
        "allowed_regions": ["eg-cairo"],
        "requires_local_processing": True,
        "data_classification": "standard",
    },
    "JO": {  # Jordan
        "name": "Jordan",
        "allowed_regions": ["jo-amman"],
        "requires_local_processing": False,
        "data_classification": "standard",
    },
}


def resolve_jurisdiction(jurisdiction_code: str) -> dict:
    """Get data residency rules for a jurisdiction."""
    rules = _JURISDICTION_RULES.get(jurisdiction_code.upper())
    if not rules:
        logger.warning("unknown_jurisdiction", code=jurisdiction_code)
        return {
            "name": "Unknown",
            "allowed_regions": [],
            "requires_local_processing": True,  # Default to strict
            "data_classification": "sensitive",
        }
    return rules


def validate_processing_region(
    jurisdiction_code: str,
    processing_region: str,
) -> None:
    """Validate that processing can occur in the given region.

    Raises DataResidencyError if data would leave its jurisdiction.
    """
    rules = resolve_jurisdiction(jurisdiction_code)
    allowed = rules.get("allowed_regions", [])

    if allowed and processing_region not in allowed:
        raise DataResidencyError(
            f"Data from {rules['name']} ({jurisdiction_code}) cannot be processed "
            f"in region '{processing_region}'. Allowed regions: {allowed}"
        )
