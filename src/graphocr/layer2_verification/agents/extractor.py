"""Extractor Agent — pulls structured claim fields from spatial tokens.

Uses DSPy-optimized prompts when available, falling back to LangChain.
Preserves source_tokens provenance for every extracted field.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import ProcessingPath
from graphocr.models.agent_state import RedTeamState
from graphocr.models.extraction import ExtractionResult, FieldExtraction
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are an expert insurance claim data extractor.
You process OCR-extracted tokens from medical insurance claims (Arabic and English).

Each token includes its text, bounding box coordinates, reading order, language, and confidence score.
Extract the following structured fields from the tokens. For each field, list the token_ids that sourced the value.

{policy_context}

REQUIRED FIELDS:
- patient_name
- patient_id
- patient_dob (YYYY-MM-DD)
- provider_name
- provider_id
- diagnosis_codes (list of ICD-10 codes)
- procedure_codes (list of CPT codes)
- medications (list with name, dosage, frequency)
- date_of_service (YYYY-MM-DD)
- total_amount (numeric)
- currency
- line_items (list with description, code, amount, quantity)

RULES:
- If a field cannot be confidently extracted, set its value to "" and confidence to 0.0
- Always include source_tokens — the token_ids that contributed to each field
- For Arabic text, provide both the original and transliterated values where applicable
- Pay attention to reading order — tokens may come from multi-column layouts

Return ONLY valid JSON in the specified format. No explanations.

TOKENS:
{tokens}

OUTPUT FORMAT:
{{
  "fields": {{
    "field_name": {{
      "value": "...",
      "confidence": 0.0-1.0,
      "source_tokens": ["token_id1", "token_id2"]
    }}
  }}
}}"""


def _format_tokens_for_prompt(tokens: list[SpatialToken]) -> str:
    """Format tokens into a structured string for the LLM prompt."""
    lines = []
    for t in tokens:
        b = t.bbox
        lines.append(
            f"[{t.token_id[:8]}] order={t.reading_order} "
            f"text='{t.text}' lang={t.language.value} conf={t.confidence:.2f} "
            f"bbox=({b.x_min:.0f},{b.y_min:.0f})-({b.x_max:.0f},{b.y_max:.0f}) "
            f"page={b.page_number}"
            + (f" zone={t.zone_label.value}" if t.zone_label else "")
            + (" [handwritten]" if t.is_handwritten else "")
        )
    return "\n".join(lines)


async def _try_dspy_extraction(
    token_text: str,
    policy_context: str,
    tokens: list[SpatialToken],
) -> str | None:
    """Attempt extraction using a DSPy-optimized module if one exists on disk.

    Returns the raw JSON string from the DSPy module, or None to fall back
    to the LangChain prompt path.
    """
    optimized_path = Path("optimized_modules/ClaimFieldExtractor_latest")
    if not optimized_path.exists():
        return None

    try:
        from graphocr.dspy_layer.optimizers import load_optimized_module

        module = load_optimized_module("ClaimFieldExtractor", str(optimized_path))

        # Detect primary language
        ar_count = sum(1 for t in tokens if t.language.value == "ar")
        en_count = sum(1 for t in tokens if t.language.value == "en")
        if ar_count > en_count:
            doc_lang = "ar"
        elif en_count > ar_count:
            doc_lang = "en"
        else:
            doc_lang = "mixed"

        result = module.forward(
            spatial_tokens_text=token_text,
            document_language=doc_lang,
            context_hints=policy_context,
        )

        raw_json = result.claim_fields_json
        # Wrap in expected format if needed
        if not raw_json.strip().startswith("{"):
            return None

        # Ensure it has the "fields" wrapper
        parsed = json.loads(raw_json)
        if "fields" not in parsed:
            parsed = {"fields": parsed}
            raw_json = json.dumps(parsed)

        logger.info("dspy_extraction_used", path=str(optimized_path))
        return raw_json

    except Exception as e:
        logger.warning("dspy_extraction_fallback", error=str(e))
        return None


async def extractor_node(state: RedTeamState) -> dict:
    """LangGraph node: Extract structured fields from spatial tokens.

    Reads spatial_tokens from state, calls LLM, returns extraction result.
    """
    settings = get_settings()
    agent_config = settings.agents.get("extractor", {})

    # Choose model based on processing path
    if state.get("round_number", 0) > 0:
        # Retry rounds use the heavier model
        model_name = agent_config.get("llm", {}).get("vlm_consensus", "llama-3.1-70b-instruct")
    else:
        model_name = agent_config.get("llm", {}).get("cheap_rail", "qwen2.5-7b-instruct")

    llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=model_name,
        temperature=agent_config.get("temperature", 0.1),
        max_tokens=agent_config.get("max_tokens", 4096),
    )

    tokens = state["spatial_tokens"]
    token_text = _format_tokens_for_prompt(tokens)

    # Inject policy context from RAG if available
    policy_context = state.get("policy_context", "")
    if not policy_context:
        policy_context = "No policy context retrieved. Extract all visible fields."

    # Try DSPy optimized module first, fall back to LangChain
    raw_output = await _try_dspy_extraction(token_text, policy_context, tokens)

    if not raw_output:
        prompt = EXTRACTION_PROMPT.format(tokens=token_text, policy_context=policy_context)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_output = response.content

    # Parse the LLM response
    extraction = _parse_extraction_response(
        raw_output,
        document_id=state["document_id"],
        tokens=tokens,
    )

    logger.info(
        "extractor_complete",
        document_id=state["document_id"],
        fields_extracted=len(extraction.fields),
        overall_confidence=extraction.overall_confidence,
    )

    return {
        "extraction": extraction,
        "messages": [AIMessage(content=f"[Extractor] Extracted {len(extraction.fields)} fields with confidence {extraction.overall_confidence:.2f}")],
    }


def _parse_extraction_response(
    raw: str,
    document_id: str,
    tokens: list[SpatialToken],
) -> ExtractionResult:
    """Parse LLM JSON response into ExtractionResult."""
    # Extract JSON from response (handle markdown code blocks)
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error("extractor_json_parse_failed", raw=raw[:200])
        return ExtractionResult(
            claim_id="",
            document_id=document_id,
            overall_confidence=0.0,
        )

    fields: dict[str, FieldExtraction] = {}
    raw_fields = data.get("fields", {})

    for field_name, field_data in raw_fields.items():
        if isinstance(field_data, dict):
            fields[field_name] = FieldExtraction(
                field_name=field_name,
                value=str(field_data.get("value", "")),
                source_tokens=field_data.get("source_tokens", []),
                confidence=float(field_data.get("confidence", 0.0)),
            )

    # Compute overall confidence
    confidences = [f.confidence for f in fields.values() if f.value]
    overall = sum(confidences) / max(len(confidences), 1)

    return ExtractionResult(
        claim_id="",  # Assigned later
        document_id=document_id,
        fields=fields,
        overall_confidence=overall,
    )
