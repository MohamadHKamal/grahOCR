"""Cypher validation queries for domain rule enforcement.

Each validator returns a list of GraphViolations when the extracted data
violates known domain constraints.
"""

from __future__ import annotations

from datetime import date, datetime

from graphocr.core.logging import get_logger
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
from graphocr.models.failure import GraphViolation

logger = get_logger(__name__)


async def validate_procedure_diagnosis(
    client: Neo4jClient,
    procedure_codes: list[str],
    diagnosis_codes: list[str],
) -> list[GraphViolation]:
    """Check if procedure codes are compatible with diagnosis codes."""
    violations: list[GraphViolation] = []

    for proc_code in procedure_codes:
        records = await client.execute_read(
            """
            MATCH (p:ProcedureCode {code: $code})
            RETURN p.valid_diagnosis_prefixes AS valid_prefixes
            """,
            {"code": proc_code},
        )

        if not records:
            continue

        valid_prefixes = records[0].get("valid_prefixes", [])
        if not valid_prefixes:
            continue

        for diag_code in diagnosis_codes:
            if not any(diag_code.startswith(prefix) for prefix in valid_prefixes):
                violations.append(
                    GraphViolation(
                        rule_name="procedure_diagnosis_compatibility",
                        field_name="procedure_codes",
                        extracted_value=f"{proc_code} with {diag_code}",
                        expected_constraint=f"Procedure {proc_code} valid with prefixes: {valid_prefixes}",
                        violation_message=f"Procedure {proc_code} is not compatible with diagnosis {diag_code}",
                        severity=0.8,
                    )
                )

    return violations


async def validate_drug_dosage(
    client: Neo4jClient,
    medications: list[dict],
) -> list[GraphViolation]:
    """Check if prescribed dosages exceed known clinical limits.

    Args:
        medications: List of dicts with 'name' and 'daily_dosage_mg'.
    """
    violations: list[GraphViolation] = []

    for med in medications:
        name = med.get("name", "").lower()
        dosage = med.get("daily_dosage_mg", 0)
        if not name or not dosage:
            continue

        records = await client.execute_read(
            """
            MATCH (m:Medication {name: $name})
            RETURN m.max_daily_dosage_mg AS max_dosage
            """,
            {"name": name},
        )

        if not records:
            continue

        max_dosage = records[0].get("max_dosage")
        if max_dosage and dosage > max_dosage:
            violations.append(
                GraphViolation(
                    rule_name="drug_dosage_limits",
                    field_name="medications",
                    extracted_value=f"{name}: {dosage}mg/day",
                    expected_constraint=f"Max {max_dosage}mg/day",
                    violation_message=f"Logical Impossibility: {name} dosage {dosage}mg exceeds clinical stoichiometric limit of {max_dosage}mg",
                    severity=0.9,
                )
            )

    return violations


async def validate_contraindicated_drugs(
    client: Neo4jClient,
    medication_names: list[str],
) -> list[GraphViolation]:
    """Check for contraindicated drug combinations."""
    violations: list[GraphViolation] = []

    if len(medication_names) < 2:
        return violations

    records = await client.execute_read(
        """
        UNWIND $names AS name1
        UNWIND $names AS name2
        WITH name1, name2 WHERE name1 < name2
        MATCH (a:Medication {name: name1})-[:CONTRAINDICATED_WITH]->(b:Medication {name: name2})
        RETURN a.name AS drug1, b.name AS drug2
        """,
        {"names": [n.lower() for n in medication_names]},
    )

    for record in records:
        violations.append(
            GraphViolation(
                rule_name="contraindicated_drugs",
                field_name="medications",
                extracted_value=f"{record['drug1']} + {record['drug2']}",
                expected_constraint="These drugs should not be prescribed together",
                violation_message=f"Contraindicated combination: {record['drug1']} and {record['drug2']}",
                severity=0.95,
            )
        )

    return violations


async def validate_provider_specialty(
    client: Neo4jClient,
    provider_id: str,
    procedure_codes: list[str],
) -> list[GraphViolation]:
    """Check if provider has required specialty for procedures."""
    violations: list[GraphViolation] = []

    for proc_code in procedure_codes:
        records = await client.execute_read(
            """
            MATCH (p:ProcedureCode {code: $proc_code})-[:REQUIRES_SPECIALTY]->(s:Specialty)
            OPTIONAL MATCH (prov:Provider {id: $provider_id})-[:HAS_SPECIALTY]->(s)
            RETURN s.name AS required_specialty, prov IS NOT NULL AS has_specialty
            """,
            {"proc_code": proc_code, "provider_id": provider_id},
        )

        for record in records:
            if not record.get("has_specialty", False):
                violations.append(
                    GraphViolation(
                        rule_name="provider_specialty",
                        field_name="provider_id",
                        extracted_value=f"Provider {provider_id} performing {proc_code}",
                        expected_constraint=f"Requires specialty: {record['required_specialty']}",
                        violation_message=f"Provider lacks required specialty: {record['required_specialty']}",
                        severity=0.7,
                    )
                )

    return violations


async def validate_learned_rules(
    client: Neo4jClient,
    extracted_fields: dict[str, str],
) -> list[GraphViolation]:
    """Check extracted values against learned failure patterns from Post-Mortem.

    LearnedRule nodes are created by the Post-Mortem agent when it detects
    a 'rule_gap' — a failure pattern the knowledge graph should have caught.
    This validator queries those learned rules to prevent repeat failures.
    """
    violations: list[GraphViolation] = []

    if not extracted_fields:
        return violations

    field_names = list(extracted_fields.keys())
    records = await client.execute_read(
        """
        MATCH (r:LearnedRule {active: true})
        WHERE r.affected_field IN $fields
        RETURN r.affected_field AS field,
               r.original_value AS bad_value,
               r.corrected_value AS good_value,
               r.root_cause AS root_cause,
               r.report_id AS report_id
        """,
        {"fields": field_names},
    )

    for record in records:
        field = record["field"]
        bad_value = record.get("bad_value", "")
        extracted_value = extracted_fields.get(field, "")

        # Check if the extracted value matches a known-bad pattern
        if bad_value and extracted_value and bad_value.lower() == extracted_value.lower():
            violations.append(
                GraphViolation(
                    rule_name="learned_rule",
                    field_name=field,
                    extracted_value=extracted_value,
                    expected_constraint=f"Known-bad value (corrected to: {record.get('good_value', 'N/A')})",
                    violation_message=(
                        f"Field '{field}' matches a previously corrected failure pattern "
                        f"(root cause: {record.get('root_cause', 'unknown')}, "
                        f"report: {record.get('report_id', 'N/A')})"
                    ),
                    severity=0.85,
                )
            )

    return violations


def validate_date_sanity(
    date_of_service: date | None,
    patient_dob: date | None,
    min_date: str = "2015-01-01",
) -> list[GraphViolation]:
    """Validate dates are within plausible ranges. No Neo4j needed."""
    violations: list[GraphViolation] = []
    today = date.today()
    min_dt = date.fromisoformat(min_date)

    if date_of_service:
        if date_of_service > today:
            violations.append(
                GraphViolation(
                    rule_name="date_sanity",
                    field_name="date_of_service",
                    extracted_value=str(date_of_service),
                    expected_constraint=f"Must be <= {today}",
                    violation_message="Date of service is in the future",
                    severity=0.9,
                )
            )
        if date_of_service < min_dt:
            violations.append(
                GraphViolation(
                    rule_name="date_sanity",
                    field_name="date_of_service",
                    extracted_value=str(date_of_service),
                    expected_constraint=f"Must be >= {min_date}",
                    violation_message=f"Date of service is before {min_date}",
                    severity=0.7,
                )
            )

    if patient_dob:
        age = (today - patient_dob).days / 365.25
        if age < 0 or age > 130:
            violations.append(
                GraphViolation(
                    rule_name="age_plausibility",
                    field_name="patient_dob",
                    extracted_value=str(patient_dob),
                    expected_constraint="Age must be 0-130",
                    violation_message=f"Logical Impossibility: Implausible patient age: {age:.0f} years",
                    severity=0.95,
                )
            )

        if date_of_service and patient_dob > date_of_service:
            violations.append(
                GraphViolation(
                    rule_name="date_sanity",
                    field_name="patient_dob",
                    extracted_value=f"DOB: {patient_dob}, Service: {date_of_service}",
                    expected_constraint="DOB must be before date of service",
                    violation_message="Logical Impossibility: Patient DOB is after date of service",
                    severity=0.95,
                )
            )

    return violations
