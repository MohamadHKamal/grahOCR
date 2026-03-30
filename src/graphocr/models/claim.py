"""Insurance claim models.

This module defines the Pydantic data models for structured insurance
claims extracted from scanned documents via OCR. The models preserve
traceability to raw OCR tokens and support fraud detection through
computed vs. stated total comparison.

Classes:
    MedicationEntry: A prescribed medication extracted from the claim.
    ClaimLineItem: A single billable line item on the claim.
    InsuranceClaim: Top-level structured insurance claim.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field
from uuid_extensions import uuid7

from graphocr.core.types import Language


class MedicationEntry(BaseModel):
    """A prescribed medication extracted from the claim.

    Attributes:
        name: Commercial or generic medication name.
        dosage: Dosage string as written on the prescription (e.g. "500mg").
        frequency: How often the medication is taken (e.g. "twice daily").
        duration_days: Prescribed duration in days. Defaults to 0 if unknown.
        daily_dosage_mg: Estimated daily dosage in milligrams, if computable.
        source_tokens: Raw OCR tokens from which this entry was extracted,
            retained for audit and traceability.
    """

    name: str
    dosage: str
    frequency: str = ""
    duration_days: int = 0
    daily_dosage_mg: float | None = None
    source_tokens: list[str] = Field(default_factory=list)


class ClaimLineItem(BaseModel):
    """A single billable line item on an insurance claim.

    Attributes:
        description: Human-readable description of the service or product.
        code: CPT or local procedure code. Empty string if not available.
        amount: Billed amount per unit as a Decimal for financial precision.
        quantity: Number of units billed. Defaults to 1.
        source_tokens: Raw OCR tokens from which this item was extracted,
            retained for audit and traceability.
    """

    description: str
    code: str = ""  # CPT or local procedure code
    amount: Decimal
    quantity: int = 1
    source_tokens: list[str] = Field(default_factory=list)


class InsuranceClaim(BaseModel):
    """Structured insurance claim extracted from scanned documents.

    Aggregates patient, provider, clinical, financial, and temporal
    data into a single validated object. Supports fraud detection by
    comparing the stated ``total_amount`` against ``computed_total``.

    Attributes:
        claim_id: Unique claim identifier, auto-generated as UUID7.
        document_id: Identifier of the source document this claim was
            extracted from.
        patient_name: Full name of the insured patient.
        patient_id: Patient's national or insurer-assigned ID.
        patient_dob: Patient's date of birth, if available.
        provider_name: Name of the healthcare provider or facility.
        provider_id: Provider's license or registration number.
        diagnosis_codes: List of ICD-10 diagnosis codes.
        procedure_codes: List of CPT procedure codes.
        medications: Prescribed medications extracted from the claim.
        line_items: Individual billable services or products.
        total_amount: Stated total amount on the claim document.
        currency: ISO 4217 currency code. Defaults to "SAR" (Saudi Riyal).
        date_of_service: Date the medical service was provided.
        date_of_submission: Date the claim was submitted to the insurer.
        jurisdiction: Regulatory/geographic region governing the claim
            (e.g. "SA" for Saudi Arabia).
        language_primary: Primary language of the source document.
        policy_reference: Insurance policy or contract reference number.
    """

    claim_id: str = Field(default_factory=lambda: str(uuid7()))
    document_id: str = ""

    # Patient
    patient_name: str = ""
    patient_id: str = ""
    patient_dob: date | None = None

    # Provider
    provider_name: str = ""
    provider_id: str = ""

    # Clinical
    diagnosis_codes: list[str] = Field(default_factory=list)  # ICD-10
    procedure_codes: list[str] = Field(default_factory=list)  # CPT
    medications: list[MedicationEntry] = Field(default_factory=list)

    # Financial
    line_items: list[ClaimLineItem] = Field(default_factory=list)
    total_amount: Decimal = Decimal("0")
    currency: str = "SAR"

    # Temporal
    date_of_service: date | None = None
    date_of_submission: date | None = None

    # Metadata
    jurisdiction: str = ""
    language_primary: Language = Language.UNKNOWN
    policy_reference: str = ""

    @property
    def computed_total(self) -> Decimal:
        """Recalculate the total from line items (amount * quantity).

        Returns:
            Sum of all line item costs. Useful for validating against
            the stated ``total_amount`` to detect billing errors or fraud.
        """
        return sum((item.amount * item.quantity for item in self.line_items), Decimal("0"))
