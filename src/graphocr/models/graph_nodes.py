"""Pydantic mirrors of Neo4j node types for type-safe graph operations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnosisCode(BaseModel):
    code: str  # ICD-10 code
    description: str = ""
    category: str = ""


class ProcedureCode(BaseModel):
    code: str  # CPT code
    description: str = ""
    category: str = ""
    required_specialty: str | None = None


class Medication(BaseModel):
    name: str
    max_daily_dosage_mg: float = 0.0
    controlled: bool = False
    drug_class: str = ""


class Provider(BaseModel):
    id: str
    name: str = ""
    specialties: list[str] = Field(default_factory=list)
    license_active: bool = True
    jurisdiction: str = ""


class Patient(BaseModel):
    id: str
    name: str = ""
    dob: str = ""  # ISO date string
    jurisdiction: str = ""


class DrugContraindication(BaseModel):
    """Represents a contraindication relationship between two drugs."""

    drug_a: str
    drug_b: str
    severity: str = "high"  # "high" | "moderate" | "low"
    description: str = ""


class ProcedureDiagnosisRule(BaseModel):
    """Maps a procedure code to valid diagnosis code prefixes."""

    procedure_code: str
    valid_diagnosis_prefixes: list[str] = Field(default_factory=list)
