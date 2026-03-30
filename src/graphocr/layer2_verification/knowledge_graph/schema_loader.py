"""Load domain rules from YAML into Neo4j. Idempotent."""

from __future__ import annotations

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient

logger = get_logger(__name__)


async def load_schema(client: Neo4jClient) -> None:
    """Create constraints, indexes, and seed reference data into Neo4j."""
    rules = get_settings().neo4j_rules

    await _create_constraints(client)
    await _create_indexes(client, rules.get("indexes", []))
    await _seed_dosage_limits(client, rules.get("constraints", {}).get("dosage_limits", {}))
    await _seed_contraindications(client, rules.get("relationships", {}).get("contraindicated_drugs", []))
    await _seed_procedure_diagnosis(client, rules.get("relationships", {}).get("procedure_diagnosis_compatibility", {}))
    await _seed_specialty_requirements(client, rules.get("relationships", {}).get("specialty_requirements", {}))
    await _seed_temporal_rules(client, rules.get("temporal_rules", {}).get("policy_effective_ranges", {}))
    await _create_learned_rule_index(client)

    logger.info("neo4j_schema_loaded")


async def _create_constraints(client: Neo4jClient) -> None:
    """Create uniqueness constraints."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DiagnosisCode) REQUIRE d.code IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:ProcedureCode) REQUIRE p.code IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Provider) REQUIRE pr.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (pt:Patient) REQUIRE pt.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    ]
    for cypher in constraints:
        await client.execute_write(cypher)
    logger.info("neo4j_constraints_created", count=len(constraints))


async def _create_indexes(client: Neo4jClient, indexes: list[dict]) -> None:
    """Create indexes from config."""
    for idx in indexes:
        label = idx["label"]
        prop = idx["property"]
        await client.execute_write(
            f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
        )
    logger.info("neo4j_indexes_created", count=len(indexes))


async def _seed_dosage_limits(client: Neo4jClient, limits: dict[str, float]) -> None:
    """Seed medication nodes with max dosage limits."""
    for drug_name, max_dosage in limits.items():
        await client.execute_write(
            """
            MERGE (m:Medication {name: $name})
            SET m.max_daily_dosage_mg = $max_dosage
            """,
            {"name": drug_name, "max_dosage": max_dosage},
        )
    logger.info("neo4j_dosage_limits_seeded", count=len(limits))


async def _seed_contraindications(client: Neo4jClient, pairs: list[list[str]]) -> None:
    """Seed drug contraindication relationships."""
    for pair in pairs:
        if len(pair) != 2:
            continue
        await client.execute_write(
            """
            MERGE (a:Medication {name: $drug_a})
            MERGE (b:Medication {name: $drug_b})
            MERGE (a)-[:CONTRAINDICATED_WITH]->(b)
            MERGE (b)-[:CONTRAINDICATED_WITH]->(a)
            """,
            {"drug_a": pair[0], "drug_b": pair[1]},
        )
    logger.info("neo4j_contraindications_seeded", count=len(pairs))


async def _seed_procedure_diagnosis(client: Neo4jClient, rules: dict[str, list[str]]) -> None:
    """Seed procedure-diagnosis compatibility rules."""
    for proc_code, valid_prefixes in rules.items():
        await client.execute_write(
            """
            MERGE (p:ProcedureCode {code: $code})
            SET p.valid_diagnosis_prefixes = $prefixes
            """,
            {"code": proc_code, "prefixes": valid_prefixes},
        )
    logger.info("neo4j_procedure_diagnosis_seeded", count=len(rules))


async def _seed_specialty_requirements(client: Neo4jClient, rules: dict[str, str]) -> None:
    """Seed procedure-specialty requirements."""
    for proc_code, specialty in rules.items():
        await client.execute_write(
            """
            MERGE (p:ProcedureCode {code: $code})
            SET p.required_specialty = $specialty
            MERGE (s:Specialty {name: $specialty})
            MERGE (p)-[:REQUIRES_SPECIALTY]->(s)
            """,
            {"code": proc_code, "specialty": specialty},
        )
    logger.info("neo4j_specialty_requirements_seeded", count=len(rules))


async def _seed_temporal_rules(client: Neo4jClient, ranges: dict[str, list[str]]) -> None:
    """Seed policy effective date ranges to catch Type B (wrong version) failures.

    Each rider/policy version has an effective date range. If a claim references
    a rider outside its effective period, it's a logical impossibility.
    """
    for rider_code, date_range in ranges.items():
        if len(date_range) != 2:
            continue
        await client.execute_write(
            """
            MERGE (p:PolicyRider {code: $code})
            SET p.effective_from = date($from_date),
                p.effective_to = date($to_date)
            """,
            {"code": rider_code, "from_date": date_range[0], "to_date": date_range[1]},
        )
    logger.info("neo4j_temporal_rules_seeded", count=len(ranges))


async def _create_learned_rule_index(client: Neo4jClient) -> None:
    """Create index on LearnedRule nodes for efficient back-propagation queries."""
    await client.execute_write(
        "CREATE INDEX IF NOT EXISTS FOR (r:LearnedRule) ON (r.affected_field)"
    )
    await client.execute_write(
        "CREATE INDEX IF NOT EXISTS FOR (r:LearnedRule) ON (r.active)"
    )
    logger.info("neo4j_learned_rule_indexes_created")
