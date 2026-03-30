"""Cheap rail — fast extraction path for clean, high-confidence documents.

Uses regex for structured fields (dates, amounts, codes) and a single
small LLM call for unstructured fields. No multi-agent; single-pass
extraction + Neo4j validation.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import ProcessingPath
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
from graphocr.layer2_verification.knowledge_graph.rule_engine import run_all_validations
from graphocr.models.claim import InsuranceClaim
from graphocr.models.extraction import ExtractionResult, FieldExtraction
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)

# Regex patterns for structured field extraction
_DATE_PATTERN = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b")
_AMOUNT_PATTERN = re.compile(r"\b(\d{1,3}(?:[,. ]\d{3})*(?:\.\d{1,2})?)\b")
_ICD_PATTERN = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b")
_CPT_PATTERN = re.compile(r"\b(\d{5})\b")

CHEAP_EXTRACTION_PROMPT = """Extract insurance claim fields from these OCR tokens.
Return ONLY valid JSON with the fields and their values.

{policy_context}

TOKENS (ordered by reading sequence):
{token_text}

PRE-EXTRACTED (regex):
{regex_fields}

Complete the extraction for fields not yet found. Return JSON:
{{
  "patient_name": "...",
  "patient_id": "...",
  "provider_name": "...",
  "provider_id": "...",
  "diagnosis_codes": ["..."],
  "procedure_codes": ["..."],
  "medications": [{{"name": "...", "dosage": "...", "frequency": "..."}}],
  "date_of_service": "YYYY-MM-DD",
  "total_amount": "...",
  "currency": "...",
  "policy_reference": "..."
}}"""


async def process_cheap_rail(
    document_id: str,
    tokens: list[SpatialToken],
    policy_context: str = "",
) -> ExtractionResult:
    """Process a document through the fast cheap rail.

    Steps:
    1. Regex extraction for structured fields
    2. Single LLM call for remaining fields
    3. Quick Neo4j validation
    """
    settings = get_settings()

    # Step 1: Regex extraction
    full_text = " ".join(t.text for t in tokens)
    regex_fields = _regex_extract(full_text, tokens)

    # Step 2: LLM for remaining fields
    token_text = "\n".join(
        f"[{t.reading_order}] '{t.text}' (conf={t.confidence:.2f})"
        for t in tokens
    )
    regex_str = json.dumps(regex_fields, ensure_ascii=False, indent=2)

    llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.agents.get("extractor", {}).get("llm", {}).get("cheap_rail", "qwen2.5-7b-instruct"),
        temperature=0.1,
        max_tokens=2048,
    )

    response = await llm.ainvoke([
        HumanMessage(content=CHEAP_EXTRACTION_PROMPT.format(
            policy_context=policy_context or "No policy context retrieved.",
            token_text=token_text[:3000],  # Truncate for speed
            regex_fields=regex_str,
        ))
    ])

    # Parse and merge
    extraction = _build_extraction(response.content, regex_fields, document_id, tokens)

    # Step 3: Quick Neo4j validation
    try:
        client = Neo4jClient()
        await client.connect()
        try:
            graph_violations = await run_all_validations(client, _extraction_to_claim(extraction))
        finally:
            await client.close()

        if graph_violations:
            extraction.graph_violations = graph_violations
            logger.warning(
                "cheap_rail_graph_violations",
                document_id=document_id,
                violations=len(graph_violations),
            )
    except Exception as e:
        logger.warning("cheap_rail_neo4j_skipped", error=str(e))

    logger.info(
        "cheap_rail_complete",
        document_id=document_id,
        fields=len(extraction.fields),
        confidence=extraction.overall_confidence,
    )
    return extraction


def _regex_extract(text: str, tokens: list[SpatialToken]) -> dict[str, str]:
    """Extract structured fields using regex patterns."""
    fields: dict[str, str] = {}

    # Dates
    dates = _DATE_PATTERN.findall(text)
    if dates:
        fields["date_of_service"] = dates[0]

    # Amounts
    amounts = _AMOUNT_PATTERN.findall(text)
    if amounts:
        # Take the largest amount as total
        parsed = []
        for a in amounts:
            try:
                parsed.append(float(a.replace(",", "").replace(" ", "")))
            except ValueError:
                pass
        if parsed:
            fields["total_amount"] = str(max(parsed))

    # ICD codes
    icd_codes = _ICD_PATTERN.findall(text)
    if icd_codes:
        fields["diagnosis_codes"] = ",".join(icd_codes[:5])

    # CPT codes
    cpt_codes = _CPT_PATTERN.findall(text)
    if cpt_codes:
        fields["procedure_codes"] = ",".join(cpt_codes[:5])

    return fields


def _build_extraction(
    llm_response: str,
    regex_fields: dict[str, str],
    document_id: str,
    tokens: list[SpatialToken],
) -> ExtractionResult:
    """Build ExtractionResult from LLM response merged with regex fields."""
    # Parse LLM response
    json_str = llm_response.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        llm_data = json.loads(json_str)
    except json.JSONDecodeError:
        llm_data = {}

    fields: dict[str, FieldExtraction] = {}

    # Merge: regex fields take precedence for structured data
    all_field_names = set(list(regex_fields.keys()) + list(llm_data.keys()))

    for name in all_field_names:
        regex_val = regex_fields.get(name, "")
        llm_val = llm_data.get(name, "")

        # Use regex for structured fields, LLM for unstructured
        if name in ("date_of_service", "total_amount", "diagnosis_codes", "procedure_codes"):
            value = regex_val or str(llm_val)
            confidence = 0.9 if regex_val else 0.7
        else:
            value = str(llm_val) if llm_val else regex_val
            confidence = 0.75

        if isinstance(value, list):
            value = ",".join(str(v) for v in value)

        fields[name] = FieldExtraction(
            field_name=name,
            value=value,
            confidence=confidence,
        )

    confidences = [f.confidence for f in fields.values() if f.value]
    overall = sum(confidences) / max(len(confidences), 1)

    return ExtractionResult(
        claim_id="",
        document_id=document_id,
        fields=fields,
        overall_confidence=overall,
        processing_path=ProcessingPath.CHEAP_RAIL,
    )


def _extraction_to_claim(extraction: ExtractionResult) -> InsuranceClaim:
    """Build a temporary InsuranceClaim from extraction for Neo4j validation."""
    fields = extraction.fields

    claim = InsuranceClaim(
        document_id=extraction.document_id,
        patient_name=fields.get("patient_name", FieldExtraction(field_name="", value="", confidence=0.0)).value,
        patient_id=fields.get("patient_id", FieldExtraction(field_name="", value="", confidence=0.0)).value,
        provider_name=fields.get("provider_name", FieldExtraction(field_name="", value="", confidence=0.0)).value,
        provider_id=fields.get("provider_id", FieldExtraction(field_name="", value="", confidence=0.0)).value,
    )

    diag_raw = fields.get("diagnosis_codes", FieldExtraction(field_name="", value="", confidence=0.0)).value
    if diag_raw:
        claim.diagnosis_codes = [c.strip() for c in diag_raw.split(",") if c.strip()]

    proc_raw = fields.get("procedure_codes", FieldExtraction(field_name="", value="", confidence=0.0)).value
    if proc_raw:
        claim.procedure_codes = [c.strip() for c in proc_raw.split(",") if c.strip()]

    dos_raw = fields.get("date_of_service", FieldExtraction(field_name="", value="", confidence=0.0)).value
    if dos_raw:
        try:
            from datetime import date
            claim.date_of_service = date.fromisoformat(dos_raw)
        except ValueError:
            pass

    amount_raw = fields.get("total_amount", FieldExtraction(field_name="", value="", confidence=0.0)).value
    if amount_raw:
        try:
            claim.total_amount = Decimal(amount_raw.replace(",", ""))
        except Exception:
            pass

    return claim
