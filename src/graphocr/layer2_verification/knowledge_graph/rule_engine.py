"""Rule engine — orchestrates all Neo4j validation checks against an extraction."""

from __future__ import annotations

from graphocr.core.logging import get_logger
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
from graphocr.layer2_verification.knowledge_graph.validators import (
    validate_contraindicated_drugs,
    validate_date_sanity,
    validate_drug_dosage,
    validate_learned_rules,
    validate_procedure_diagnosis,
    validate_provider_specialty,
)
from graphocr.models.claim import InsuranceClaim
from graphocr.models.failure import GraphViolation

logger = get_logger(__name__)


async def run_all_validations(
    client: Neo4jClient,
    claim: InsuranceClaim,
) -> list[GraphViolation]:
    """Run all domain rule validations against an extracted claim.

    Returns all violations found. An empty list means the claim passed
    all graph-based checks.
    """
    violations: list[GraphViolation] = []

    # 1. Procedure-Diagnosis compatibility
    if claim.procedure_codes and claim.diagnosis_codes:
        v = await validate_procedure_diagnosis(
            client, claim.procedure_codes, claim.diagnosis_codes
        )
        violations.extend(v)

    # 2. Drug dosage limits
    if claim.medications:
        med_dicts = [
            {"name": m.name, "daily_dosage_mg": m.daily_dosage_mg}
            for m in claim.medications
            if m.daily_dosage_mg
        ]
        if med_dicts:
            v = await validate_drug_dosage(client, med_dicts)
            violations.extend(v)

    # 3. Contraindicated drug combinations
    if len(claim.medications) >= 2:
        med_names = [m.name for m in claim.medications]
        v = await validate_contraindicated_drugs(client, med_names)
        violations.extend(v)

    # 4. Provider specialty
    if claim.provider_id and claim.procedure_codes:
        v = await validate_provider_specialty(
            client, claim.provider_id, claim.procedure_codes
        )
        violations.extend(v)

    # 5. Date sanity (no Neo4j needed)
    v = validate_date_sanity(claim.date_of_service, claim.patient_dob)
    violations.extend(v)

    # 6. Amount reasonableness
    v = _validate_amounts(claim)
    violations.extend(v)

    # 7. Learned rules from Post-Mortem back-propagation
    extracted_fields = {}
    if hasattr(claim, "raw_fields") and claim.raw_fields:
        extracted_fields = claim.raw_fields
    else:
        # Build field map from claim attributes
        field_map = {
            "patient_name": claim.patient_name,
            "provider_id": claim.provider_id,
            "policy_number": getattr(claim, "policy_number", ""),
            "total_amount": str(claim.total_amount),
        }
        extracted_fields = {k: v for k, v in field_map.items() if v}

    if extracted_fields:
        v = await validate_learned_rules(client, extracted_fields)
        violations.extend(v)

    logger.info(
        "validation_complete",
        claim_id=claim.claim_id,
        violations=len(violations),
        rules_checked=7,
    )
    return violations


def _validate_amounts(claim: InsuranceClaim) -> list[GraphViolation]:
    """Check claim amounts are reasonable."""
    violations: list[GraphViolation] = []

    if claim.total_amount < 0:
        violations.append(
            GraphViolation(
                rule_name="amount_reasonableness",
                field_name="total_amount",
                extracted_value=str(claim.total_amount),
                expected_constraint="Must be >= 0",
                violation_message="Negative claim total",
                severity=0.9,
            )
        )

    if claim.total_amount > 2_000_000:
        violations.append(
            GraphViolation(
                rule_name="amount_reasonableness",
                field_name="total_amount",
                extracted_value=str(claim.total_amount),
                expected_constraint="Must be <= 2,000,000",
                violation_message="Claim total exceeds maximum threshold",
                severity=0.7,
            )
        )

    # Check individual line items
    for item in claim.line_items:
        if item.amount > 500_000:
            violations.append(
                GraphViolation(
                    rule_name="amount_reasonableness",
                    field_name="line_items",
                    extracted_value=f"{item.description}: {item.amount}",
                    expected_constraint="Single line item <= 500,000",
                    violation_message=f"Line item '{item.description}' amount exceeds maximum",
                    severity=0.7,
                )
            )

    # Check computed total vs stated total
    computed = claim.computed_total
    if computed > 0 and claim.total_amount > 0:
        diff_ratio = abs(float(computed - claim.total_amount)) / float(claim.total_amount)
        if diff_ratio > 0.01:  # More than 1% discrepancy
            violations.append(
                GraphViolation(
                    rule_name="amount_reasonableness",
                    field_name="total_amount",
                    extracted_value=f"Stated: {claim.total_amount}, Computed: {computed}",
                    expected_constraint="Stated total should match sum of line items",
                    violation_message=f"Total discrepancy: {diff_ratio:.1%}",
                    severity=0.6,
                )
            )

    return violations
