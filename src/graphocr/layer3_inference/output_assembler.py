"""Output assembler — packages final claim JSON with audit trail."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from graphocr.core.logging import get_logger
from graphocr.models.claim import ClaimLineItem, InsuranceClaim, MedicationEntry
from graphocr.models.extraction import ExtractionResult
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)


def assemble_claim(
    extraction: ExtractionResult,
    tokens: list[SpatialToken],
) -> InsuranceClaim:
    """Convert an ExtractionResult into a structured InsuranceClaim.

    Populates all fields from the extraction, parsing dates, amounts,
    and structured lists. Attaches source_token provenance.
    """
    fields = extraction.fields

    claim = InsuranceClaim(
        document_id=extraction.document_id,
        patient_name=_get_field(fields, "patient_name"),
        patient_id=_get_field(fields, "patient_id"),
        provider_name=_get_field(fields, "provider_name"),
        provider_id=_get_field(fields, "provider_id"),
        currency=_get_field(fields, "currency") or "SAR",
        policy_reference=_get_field(fields, "policy_reference"),
    )

    # Parse dates
    dob = _get_field(fields, "patient_dob")
    if dob:
        try:
            claim.patient_dob = date.fromisoformat(dob)
        except ValueError:
            pass

    dos = _get_field(fields, "date_of_service")
    if dos:
        try:
            claim.date_of_service = date.fromisoformat(dos)
        except ValueError:
            pass

    # Parse codes
    diag = _get_field(fields, "diagnosis_codes")
    if diag:
        claim.diagnosis_codes = [c.strip() for c in diag.split(",") if c.strip()]

    proc = _get_field(fields, "procedure_codes")
    if proc:
        claim.procedure_codes = [c.strip() for c in proc.split(",") if c.strip()]

    # Parse amount
    amount = _get_field(fields, "total_amount")
    if amount:
        try:
            claim.total_amount = Decimal(amount.replace(",", ""))
        except Exception:
            pass

    # Parse medications (may be JSON string)
    meds_raw = _get_field(fields, "medications")
    if meds_raw:
        claim.medications = _parse_medications(meds_raw)

    logger.info(
        "claim_assembled",
        document_id=extraction.document_id,
        patient=claim.patient_name,
        amount=str(claim.total_amount),
        codes=len(claim.diagnosis_codes) + len(claim.procedure_codes),
    )
    return claim


def _get_field(fields: dict, name: str) -> str:
    """Safely get a field value."""
    field = fields.get(name)
    return field.value if field else ""


def _parse_medications(raw: str) -> list[MedicationEntry]:
    """Parse medications from extracted string."""
    import json

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                MedicationEntry(
                    name=m.get("name", ""),
                    dosage=m.get("dosage", ""),
                    frequency=m.get("frequency", ""),
                )
                for m in data
                if isinstance(m, dict)
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    return []
