"""Adversarial Challenger Agent — actively tries to break the extraction.

Uses a DIFFERENT model (Llama-3.1-70B) from the Extractor (Qwen2.5)
to ensure genuine model diversity and avoid shared blind spots.

When page_images are available, high-confidence challenges are verified
visually by cropping the affected region and sending it to a VLM.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from PIL import Image

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.models.agent_state import RedTeamState
from graphocr.models.failure import Challenge

logger = get_logger(__name__)

CHALLENGER_PROMPT = """You are an adversarial insurance claim auditor. Your job is to FIND ERRORS
in the extraction below. You must actively try to break the extraction.

EXTRACTED FIELDS:
{fields_json}

VALIDATION ISSUES ALREADY FOUND:
{validation_issues}

ORIGINAL OCR TOKENS (sample):
{token_sample}

YOUR ADVERSARIAL STRATEGIES:
1. Arabic Character Confusion: Could ع be confused with غ? Could ي be confused with ى?
   Arabic OCR commonly confuses similar characters.
2. Digit OCR Errors: Could a 3 be an 8? A 1 be a 7? A 0 be an O?
3. Stamp Obscuration: Could a stamp or seal have obscured part of a field value?
4. Merged Line Items: Could two separate line items have been merged into one?
5. Date Format Ambiguity: Is DD/MM/YYYY vs MM/DD/YYYY ambiguous for this date?
6. Currency Symbol Misread: Could SAR be read as USD? Could ر.س be misread?
7. Handwriting Ambiguity: For handwritten fields, what other readings are plausible?

For each challenge you raise:
- State the target field
- State your hypothesis (what might be wrong)
- Provide evidence from the tokens
- Propose an alternative interpretation
- Rate your confidence (0.0-1.0) that this IS an error

Return ONLY valid JSON:
[
  {{
    "target_field": "...",
    "hypothesis": "...",
    "evidence": "...",
    "proposed_alternative": "...",
    "confidence": 0.0-1.0,
    "affected_tokens": ["token_id1", ...]
  }}
]

Be aggressive but specific. Raise at most {max_challenges} challenges.
Return empty list [] ONLY if you genuinely cannot find any plausible errors."""

VLM_VERIFY_PROMPT = """You are verifying a challenged field extraction from a scanned insurance claim.

CHALLENGED FIELD: {target_field}
CURRENT EXTRACTION: {current_value}
PROPOSED ALTERNATIVE: {proposed_alternative}
HYPOTHESIS: {hypothesis}

Look at the cropped image region and determine:
1. What text is actually visible in this region?
2. Does the current extraction or the proposed alternative better match what you see?

Return ONLY valid JSON:
{{
  "visible_text": "...",
  "supports_challenge": true/false,
  "confidence": 0.0-1.0,
  "explanation": "..."
}}"""


async def challenger_node(state: RedTeamState) -> dict:
    """LangGraph node: Generate adversarial challenges against the extraction."""
    settings = get_settings()
    agent_config = settings.agents.get("challenger", {})
    extraction = state.get("extraction")

    if not extraction:
        return {
            "challenges": [],
            "messages": [AIMessage(content="[Challenger] No extraction to challenge.")],
        }

    # Use a DIFFERENT model than the extractor for diversity
    llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=agent_config.get("llm", "llama-3.1-70b-instruct"),
        temperature=agent_config.get("temperature", 0.3),
        max_tokens=agent_config.get("max_tokens", 3072),
    )

    # Prepare context
    fields_json = json.dumps(
        {k: {"value": v.value, "confidence": v.confidence} for k, v in extraction.fields.items()},
        indent=2,
        ensure_ascii=False,
    )

    validation_issues = state.get("validation_issues", [])
    issues_str = "\n".join(f"- {issue}" for issue in validation_issues) or "None found."

    # Include a sample of original tokens for the challenger to reference
    tokens = state["spatial_tokens"]
    token_sample = "\n".join(
        f"[{t.token_id[:8]}] '{t.text}' conf={t.confidence:.2f} lang={t.language.value}"
        for t in tokens[:50]  # First 50 tokens
    )

    max_challenges = agent_config.get("max_challenges_per_round", 5)

    prompt = CHALLENGER_PROMPT.format(
        fields_json=fields_json,
        validation_issues=issues_str,
        token_sample=token_sample,
        max_challenges=max_challenges,
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    challenges = _parse_challenges(response.content)

    # Vision-based verification for high-confidence challenges when images available
    page_images = state.get("page_images", {})
    if page_images and challenges:
        high_conf = [c for c in challenges if c.confidence > 0.7]
        verified = await _vlm_verify_challenges(
            high_conf[:3], extraction, tokens, page_images, settings,
        )
        # Update challenge confidence based on VLM verification
        verified_map = {v["target_field"]: v for v in verified}
        for challenge in challenges:
            if challenge.target_field in verified_map:
                v = verified_map[challenge.target_field]
                if v["supports_challenge"]:
                    challenge.confidence = min(1.0, challenge.confidence + 0.1)
                else:
                    challenge.confidence = max(0.0, challenge.confidence - 0.3)

    logger.info(
        "challenger_complete",
        document_id=state["document_id"],
        challenges_raised=len(challenges),
        high_confidence_challenges=sum(1 for c in challenges if c.confidence > 0.7),
        vlm_verified=bool(page_images),
    )

    return {
        "challenges": challenges,
        "messages": [
            AIMessage(
                content=f"[Challenger] Raised {len(challenges)} challenges. "
                f"High-confidence: {sum(1 for c in challenges if c.confidence > 0.7)}"
                f"{' (VLM-verified)' if page_images else ''}"
            )
        ],
    }


def _parse_challenges(raw: str) -> list[Challenge]:
    """Parse challenger LLM response into Challenge objects."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []

        challenges = []
        for item in data:
            if not isinstance(item, dict):
                continue
            challenges.append(
                Challenge(
                    target_field=item.get("target_field", "unknown"),
                    hypothesis=item.get("hypothesis", ""),
                    evidence=item.get("evidence", ""),
                    proposed_alternative=item.get("proposed_alternative", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    affected_tokens=item.get("affected_tokens", []),
                )
            )
        return challenges

    except json.JSONDecodeError:
        logger.error("challenger_json_parse_failed", raw=raw[:200])
        return []


async def _vlm_verify_challenges(
    challenges: list[Challenge],
    extraction,
    tokens: list,
    page_images: dict[int, str],
    settings,
) -> list[dict]:
    """Verify high-confidence challenges by sending affected regions to VLM.

    Crops the region around affected tokens and asks the VLM to confirm
    whether the challenge is valid.
    """
    if not challenges:
        return []

    vlm_config = settings.agents.get("vlm_rescanner", {})
    client = AsyncOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
    )
    vlm_model = vlm_config.get("model", "qwen2-vl-7b-instruct")

    results = []
    for challenge in challenges:
        # Find affected tokens to determine crop region
        affected = [t for t in tokens if t.token_id[:8] in challenge.affected_tokens]
        if not affected:
            # Fall back to first page
            affected = tokens[:1]

        page_num = affected[0].bbox.page_number if affected else 1
        page_path = page_images.get(page_num)
        if not page_path or not Path(page_path).exists():
            continue

        # Compute bounding box around affected tokens with padding
        x_min = min(t.bbox.x_min for t in affected) - 20
        y_min = min(t.bbox.y_min for t in affected) - 20
        x_max = max(t.bbox.x_max for t in affected) + 20
        y_max = max(t.bbox.y_max for t in affected) + 20

        try:
            image = Image.open(page_path)
            cropped = image.crop((
                int(max(0, x_min)),
                int(max(0, y_min)),
                int(min(image.width, x_max)),
                int(min(image.height, y_max)),
            ))

            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            current_value = ""
            if extraction and challenge.target_field in extraction.fields:
                current_value = extraction.fields[challenge.target_field].value

            prompt = VLM_VERIFY_PROMPT.format(
                target_field=challenge.target_field,
                current_value=current_value,
                proposed_alternative=challenge.proposed_alternative,
                hypothesis=challenge.hypothesis,
            )

            response = await client.chat.completions.create(
                model=vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=512,
                temperature=0.1,
            )

            raw = response.choices[0].message.content or "{}"
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            result = json.loads(raw)
            result["target_field"] = challenge.target_field
            results.append(result)

            logger.info(
                "vlm_challenge_verified",
                target_field=challenge.target_field,
                supports_challenge=result.get("supports_challenge", False),
                vlm_confidence=result.get("confidence", 0),
            )
        except Exception as e:
            logger.warning(
                "vlm_challenge_verify_failed",
                target_field=challenge.target_field,
                error=str(e),
            )

    return results
