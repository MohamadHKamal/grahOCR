"""Post-Mortem Agent — analyzes resolved failures for global learning.

When a failure passes through verification and gets corrected, this agent
classifies the root cause, logs a structured report, and adds it to the
DSPy training set for future optimization.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import FailureType
from graphocr.models.agent_state import RedTeamState
from graphocr.models.failure import FailureReport

logger = get_logger(__name__)

POSTMORTEM_PROMPT = """You are a failure analysis expert for an OCR insurance claim pipeline.
A conflict was detected and resolved during processing. Analyze the root cause.

ORIGINAL EXTRACTION (before healing):
{original_fields}

CORRECTED EXTRACTION (after healing):
{corrected_fields}

CHALLENGES THAT TRIGGERED HEALING:
{challenges}

GRAPH VIOLATIONS:
{violations}

Classify the root cause as ONE of:
- "ocr_misread": The OCR engine misread characters (e.g., Arabic character confusion, digit errors)
- "prompt_failure": The LLM extraction prompt failed to correctly interpret the tokens
- "rule_gap": The knowledge graph was missing a rule that should have caught this earlier
- "layout_confusion": The reading order or spatial assembly was incorrect

For each corrected field, explain:
1. What went wrong (root cause)
2. Why it wasn't caught earlier
3. What rule or prompt change would prevent this in the future
4. Whether this case should be added to DSPy training data (yes/no)

Return ONLY valid JSON:
{{
  "root_cause": "ocr_misread|prompt_failure|rule_gap|layout_confusion",
  "corrections": [
    {{
      "field": "...",
      "original_value": "...",
      "corrected_value": "...",
      "explanation": "...",
      "prevention": "...",
      "add_to_training": true/false
    }}
  ]
}}"""


async def postmortem_node(state: RedTeamState) -> dict:
    """LangGraph node: Analyze resolved conflicts and generate failure reports."""
    settings = get_settings()
    agent_config = settings.agents.get("postmortem", {})

    if not state.get("self_healing_applied", False) and not state.get("challenges"):
        return {
            "messages": [AIMessage(content="[PostMortem] No failures to analyze.")],
        }

    llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=agent_config.get("llm", "qwen2.5-7b-instruct"),
        temperature=agent_config.get("temperature", 0.0),
        max_tokens=agent_config.get("max_tokens", 2048),
    )

    extraction = state.get("extraction")
    final = state.get("final_result")

    original_fields = {}
    corrected_fields = {}
    if extraction:
        original_fields = {k: v.value for k, v in extraction.fields.items()}
    if final:
        corrected_fields = {k: v.value for k, v in final.fields.items()}

    challenges_str = "\n".join(
        f"- [{c.target_field}] {c.hypothesis} (conf: {c.confidence})"
        for c in state.get("challenges", [])
    ) or "None"

    violations_str = "\n".join(
        f"- [{v.rule_name}] {v.violation_message}"
        for v in state.get("graph_violations", [])
    ) or "None"

    prompt = POSTMORTEM_PROMPT.format(
        original_fields=json.dumps(original_fields, ensure_ascii=False, indent=2),
        corrected_fields=json.dumps(corrected_fields, ensure_ascii=False, indent=2),
        challenges=challenges_str,
        violations=violations_str,
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    reports = _parse_postmortem(response.content, state)

    # Persist reports to the failure store (Redis)
    await _persist_reports(reports)

    # If any reports have root_cause="rule_gap", update Neo4j
    rule_gap_reports = [r for r in reports if r.root_cause == "rule_gap"]
    if rule_gap_reports:
        await _update_neo4j_rules(rule_gap_reports)

    for report in reports:
        logger.info(
            "postmortem_report",
            document_id=report.document_id,
            root_cause=report.root_cause,
            field=report.affected_field,
            add_to_training=report.add_to_dspy_training,
        )

    return {
        "messages": [
            AIMessage(
                content=f"[PostMortem] Generated {len(reports)} failure reports. "
                f"Root causes: {set(r.root_cause for r in reports)}. "
                f"Persisted to store. Rule gap updates: {len(rule_gap_reports)}."
            )
        ],
    }


async def _persist_reports(reports: list[FailureReport]) -> None:
    """Persist failure reports to Redis for DSPy training and analytics."""
    if not reports:
        return
    try:
        from graphocr.layer2_verification.agents.failure_store import FailureStore

        store = FailureStore()
        await store.connect()
        try:
            await store.save_reports_batch(reports)
        finally:
            await store.close()
    except Exception as e:
        logger.warning("failure_store_persist_failed", error=str(e))


async def _update_neo4j_rules(reports: list[FailureReport]) -> None:
    """When a 'rule_gap' is detected, add the missing rule to Neo4j.

    This closes the self-healing loop: PostMortem detects that the knowledge
    graph was missing a constraint -> adds the constraint -> future claims
    are caught automatically.
    """
    try:
        from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient

        client = Neo4jClient()
        await client.connect()
        try:
            for report in reports:
                # Create a learned rule node in Neo4j
                await client.execute_write(
                    """
                    CREATE (r:LearnedRule {
                        report_id: $report_id,
                        affected_field: $field,
                        original_value: $original,
                        corrected_value: $corrected,
                        root_cause: $root_cause,
                        created_at: datetime(),
                        active: true
                    })
                    """,
                    {
                        "report_id": report.report_id,
                        "field": report.affected_field,
                        "original": report.original_value,
                        "corrected": report.corrected_value,
                        "root_cause": report.root_cause,
                    },
                )
                logger.info(
                    "neo4j_rule_gap_learned",
                    report_id=report.report_id,
                    field=report.affected_field,
                )
        finally:
            await client.close()
    except Exception as e:
        logger.warning("neo4j_rule_update_failed", error=str(e))


def _parse_postmortem(raw: str, state: RedTeamState) -> list[FailureReport]:
    """Parse postmortem LLM response into FailureReport objects."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error("postmortem_json_parse_failed", raw=raw[:200])
        return []

    root_cause = data.get("root_cause", "unknown")
    corrections = data.get("corrections", [])

    # Map root cause to failure type
    failure_type = FailureType.TYPE_A_SPATIAL_BLIND
    if root_cause in ("prompt_failure", "rule_gap"):
        failure_type = FailureType.TYPE_B_CONTEXT_BLIND

    reports = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue

        reports.append(
            FailureReport(
                document_id=state["document_id"],
                claim_id=state.get("extraction", {}).claim_id if state.get("extraction") else "",
                root_cause=root_cause,
                failure_type=failure_type,
                original_value=correction.get("original_value", ""),
                corrected_value=correction.get("corrected_value", ""),
                affected_field=correction.get("field", ""),
                resolution_method="vlm_rescan" if root_cause == "ocr_misread" else "agent_correction",
                add_to_dspy_training=correction.get("add_to_training", False),
            )
        )

    return reports
