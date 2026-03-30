"""Validator Agent — checks extraction consistency + Neo4j rule enforcement.

Cross-checks the extractor's output for internal consistency and
validates against the knowledge graph domain rules.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
from graphocr.layer2_verification.knowledge_graph.rule_engine import run_all_validations
from graphocr.models.agent_state import RedTeamState
from graphocr.models.claim import InsuranceClaim, MedicationEntry
from graphocr.models.extraction import ExtractionResult
from graphocr.models.failure import GraphViolation

logger = get_logger(__name__)

VALIDATION_PROMPT = """You are an expert insurance claim validator.
Review the following extracted claim data and identify any inconsistencies, errors, or suspicious values.

EXTRACTED FIELDS:
{fields_json}

NEO4J GRAPH VIOLATIONS (already detected):
{graph_violations}

Check for:
1. Internal consistency — do amounts sum correctly? Do dates make sense together?
2. Medical plausibility — does the diagnosis match the procedures and medications?
3. Data quality — are there obvious OCR errors (e.g., letter O vs digit 0)?
4. Completeness — are critical fields missing?
5. Cross-field validation — does patient age match DOB? Does currency match jurisdiction?

Return a JSON list of issues found. Each issue should have:
- "field": the affected field name
- "issue": description of the problem
- "severity": 0.0-1.0
- "suggested_action": "accept" | "flag" | "reject" | "rescan"

Return ONLY valid JSON. Empty list [] if no issues found."""


async def validator_node(state: RedTeamState) -> dict:
    """LangGraph node: Validate extraction against rules and consistency."""
    settings = get_settings()
    agent_config = settings.agents.get("validator", {})
    extraction = state.get("extraction")

    if not extraction:
        return {
            "validation_issues": ["No extraction to validate"],
            "messages": [AIMessage(content="[Validator] No extraction found — skipping.")],
        }

    # Step 1: Neo4j graph validation
    graph_violations = await _run_graph_validation(extraction)

    # Step 2: LLM-based consistency validation
    llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=agent_config.get("llm", "qwen2.5-7b-instruct"),
        temperature=agent_config.get("temperature", 0.0),
        max_tokens=agent_config.get("max_tokens", 2048),
    )

    fields_json = json.dumps(
        {k: {"value": v.value, "confidence": v.confidence} for k, v in extraction.fields.items()},
        indent=2,
        ensure_ascii=False,
    )
    violations_str = "\n".join(
        f"- [{v.rule_name}] {v.violation_message} (severity: {v.severity})"
        for v in graph_violations
    ) or "None detected."

    prompt = VALIDATION_PROMPT.format(
        fields_json=fields_json,
        graph_violations=violations_str,
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    llm_issues = _parse_validation_response(response.content)

    # Combine issues
    all_issues = [
        f"[GRAPH] {v.violation_message}" for v in graph_violations
    ] + [
        f"[LLM] {issue}" for issue in llm_issues
    ]

    logger.info(
        "validator_complete",
        document_id=state["document_id"],
        graph_violations=len(graph_violations),
        llm_issues=len(llm_issues),
    )

    return {
        "validation_issues": all_issues,
        "graph_violations": graph_violations,
        "messages": [
            AIMessage(
                content=f"[Validator] Found {len(graph_violations)} graph violations and {len(llm_issues)} consistency issues."
            )
        ],
    }


async def _run_graph_validation(extraction: ExtractionResult) -> list[GraphViolation]:
    """Build a temporary InsuranceClaim from extraction and validate against Neo4j."""
    fields = extraction.fields

    # Build claim from extracted fields
    claim = InsuranceClaim(
        document_id=extraction.document_id,
        patient_name=fields.get("patient_name", _empty_field()).value,
        patient_id=fields.get("patient_id", _empty_field()).value,
        provider_name=fields.get("provider_name", _empty_field()).value,
        provider_id=fields.get("provider_id", _empty_field()).value,
    )

    # Parse diagnosis codes
    diag_raw = fields.get("diagnosis_codes", _empty_field()).value
    if diag_raw:
        claim.diagnosis_codes = [c.strip() for c in diag_raw.split(",") if c.strip()]

    # Parse procedure codes
    proc_raw = fields.get("procedure_codes", _empty_field()).value
    if proc_raw:
        claim.procedure_codes = [c.strip() for c in proc_raw.split(",") if c.strip()]

    # Parse dates
    dob_raw = fields.get("patient_dob", _empty_field()).value
    if dob_raw:
        try:
            claim.patient_dob = date.fromisoformat(dob_raw)
        except ValueError:
            pass

    dos_raw = fields.get("date_of_service", _empty_field()).value
    if dos_raw:
        try:
            claim.date_of_service = date.fromisoformat(dos_raw)
        except ValueError:
            pass

    # Parse total amount
    amount_raw = fields.get("total_amount", _empty_field()).value
    if amount_raw:
        try:
            claim.total_amount = Decimal(amount_raw.replace(",", ""))
        except Exception:
            pass

    # Run Neo4j validation
    try:
        client = Neo4jClient()
        await client.connect()
        try:
            violations = await run_all_validations(client, claim)
        finally:
            await client.close()
        return violations
    except Exception as e:
        logger.warning("neo4j_validation_skipped", error=str(e))
        return []


def _parse_validation_response(raw: str) -> list[str]:
    """Parse LLM validation response into issue strings."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        issues = json.loads(json_str)
        if isinstance(issues, list):
            return [
                f"{i.get('field', '?')}: {i.get('issue', '?')} (severity: {i.get('severity', '?')})"
                for i in issues
                if isinstance(i, dict)
            ]
    except json.JSONDecodeError:
        pass

    return []


def _empty_field():
    """Return a stub FieldExtraction for missing fields."""
    from graphocr.models.extraction import FieldExtraction
    return FieldExtraction(field_name="", value="", confidence=0.0)
