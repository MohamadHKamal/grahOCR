"""VLM targeted rescanner — re-reads specific document regions using Qwen2-VL.

When the knowledge graph or challenger agent flags a conflicting region,
this module crops that region from the original page image and sends it
to a Vision-Language Model for focused re-extraction.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)

VLM_RESCAN_PROMPT = """You are examining a cropped region from a scanned insurance claim document.
This region was flagged as potentially misread by OCR.

Extract ALL text visible in this image region. For each text element, provide:
- The text content (preserve Arabic characters exactly)
- Your confidence (0.0-1.0)
- Approximate position (top/middle/bottom, left/center/right)

Pay special attention to:
- Arabic characters that may have been confused (ع vs غ, ي vs ى, etc.)
- Digits that may have been misread (3 vs 8, 1 vs 7, 0 vs O)
- Text obscured by stamps or seals
- Handwritten text

Return ONLY valid JSON:
[
  {
    "text": "...",
    "confidence": 0.0-1.0,
    "position": "top-left|top-center|...",
    "is_handwritten": true/false,
    "language": "ar|en"
  }
]"""


async def rescan_region(
    page_image_path: str,
    region: BoundingBox,
    vlm_model: str = "qwen2-vl-7b-instruct",
) -> list[SpatialToken]:
    """Crop and re-scan a specific region using a VLM.

    Args:
        page_image_path: Path to the full page image.
        region: Bounding box of the region to re-scan.
        vlm_model: VLM model name for the vLLM endpoint.

    Returns:
        New SpatialTokens extracted from the re-scanned region.
    """
    settings = get_settings()

    # Crop the region
    image = Image.open(page_image_path)
    cropped = image.crop((
        int(max(0, region.x_min)),
        int(max(0, region.y_min)),
        int(min(image.width, region.x_max)),
        int(min(image.height, region.y_max)),
    ))

    # Encode to base64
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Call VLM via OpenAI-compatible API
    client = AsyncOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
    )

    response = await client.chat.completions.create(
        model=vlm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_RESCAN_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    raw_output = response.choices[0].message.content or "[]"
    tokens = _parse_vlm_response(raw_output, region)

    logger.info(
        "vlm_rescan_complete",
        page=region.page_number,
        region_area=region.area,
        tokens_extracted=len(tokens),
    )
    return tokens


def _parse_vlm_response(raw: str, region: BoundingBox) -> list[SpatialToken]:
    """Parse VLM response into SpatialTokens positioned within the region."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])

    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []
    except json.JSONDecodeError:
        logger.error("vlm_rescan_parse_failed", raw=raw[:200])
        return []

    tokens: list[SpatialToken] = []
    region_height = region.y_max - region.y_min
    region_width = region.x_max - region.x_min

    for idx, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("text"):
            continue

        # Approximate position within the region based on position string
        pos = item.get("position", "middle-center")
        y_offset, x_offset = _position_to_offset(pos, region_width, region_height)

        from graphocr.core.types import Language

        lang_str = item.get("language", "unknown")
        language = Language.ARABIC if lang_str == "ar" else Language.ENGLISH if lang_str == "en" else Language.UNKNOWN

        token = SpatialToken(
            text=item["text"],
            bbox=BoundingBox(
                x_min=region.x_min + x_offset,
                y_min=region.y_min + y_offset,
                x_max=region.x_min + x_offset + region_width * 0.3,
                y_max=region.y_min + y_offset + region_height * 0.1,
                page_number=region.page_number,
            ),
            reading_order=idx,
            confidence=float(item.get("confidence", 0.8)),
            ocr_engine="vlm_rescan",
            language=language,
            is_handwritten=item.get("is_handwritten", False),
        )
        tokens.append(token)

    return tokens


def _position_to_offset(
    position: str,
    region_width: float,
    region_height: float,
) -> tuple[float, float]:
    """Convert position string to pixel offsets within region."""
    parts = position.lower().split("-")
    y_part = parts[0] if parts else "middle"
    x_part = parts[1] if len(parts) > 1 else "center"

    y_map = {"top": 0.1, "middle": 0.4, "bottom": 0.7}
    x_map = {"left": 0.1, "center": 0.35, "right": 0.6}

    y_offset = region_height * y_map.get(y_part, 0.4)
    x_offset = region_width * x_map.get(x_part, 0.35)

    return y_offset, x_offset
