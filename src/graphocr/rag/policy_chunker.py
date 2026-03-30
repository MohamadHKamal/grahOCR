"""Policy document chunker — splits policies into metadata-rich chunks.

Unlike naive text splitting, this chunker:
1. Splits on section boundaries (coverage, exclusions, limits, definitions)
2. Propagates temporal metadata (effective_date, expiry_date) to every chunk
3. Preserves hierarchical context (rider -> parent policy link)
4. Handles bilingual (Arabic/English) policy documents
"""

from __future__ import annotations

import re

from graphocr.core.logging import get_logger
from graphocr.models.policy import PolicyChunk, PolicyDocument

logger = get_logger(__name__)

# Section header patterns (English + Arabic)
_SECTION_PATTERNS = [
    # English section headers
    (r"(?:^|\n)(#{1,3}\s+.+?)(?:\n|$)", "heading"),
    (r"(?:^|\n)((?:Section|Article|Part)\s+\d+[:\.\s]+.+?)(?:\n|$)", "section"),
    (r"(?:^|\n)((?:Coverage|Benefits?|Exclusions?|Limitations?|Definitions?|"
     r"Deductible|Copay|Preauthorization|Eligibility)[:\s].+?)(?:\n|$)", "keyword_section"),
    # Arabic section headers
    (r"(?:^|\n)((?:القسم|المادة|الباب|التغطية|الفوائد|الاستثناءات|"
     r"التعريفات|الخصم|التفويض المسبق)[:\s].+?)(?:\n|$)", "arabic_section"),
]

# Section type classification
_SECTION_TYPE_KEYWORDS = {
    "coverage": ["coverage", "benefit", "covered", "تغطية", "فوائد", "مغطى"],
    "exclusion": ["exclusion", "excluded", "not covered", "استثناء", "غير مغطى"],
    "benefit_limit": ["limit", "maximum", "annual", "lifetime", "حد", "أقصى", "سنوي"],
    "deductible": ["deductible", "copay", "co-pay", "خصم", "مشاركة"],
    "definition": ["definition", "means", "refers to", "تعريف", "يعني"],
    "preauth": ["preauthorization", "pre-auth", "prior approval", "تفويض مسبق", "موافقة مسبقة"],
    "eligibility": ["eligibility", "eligible", "qualification", "أهلية", "مؤهل"],
}


def chunk_policy(
    policy: PolicyDocument,
    max_chunk_size: int = 1000,
    overlap: int = 100,
) -> list[PolicyChunk]:
    """Split a policy document into metadata-rich chunks.

    Args:
        policy: The policy document to chunk.
        max_chunk_size: Maximum characters per chunk.
        overlap: Character overlap between chunks.

    Returns:
        List of PolicyChunks with temporal and hierarchical metadata.
    """
    text = policy.full_text
    text_ar = policy.full_text_ar

    if not text and not text_ar:
        return []

    # Split by sections first
    sections = _split_into_sections(text or text_ar)

    chunks: list[PolicyChunk] = []

    for section_title, section_text, section_type in sections:
        # Sub-split long sections
        sub_chunks = _split_section(section_text, max_chunk_size, overlap)

        for sub_text in sub_chunks:
            chunk = PolicyChunk(
                policy_id=policy.policy_id,
                policy_number=policy.policy_number,
                policy_type=policy.policy_type,
                policy_version=policy.version,
                effective_date=policy.effective_date,
                expiry_date=policy.expiry_date,
                text=sub_text,
                text_ar="",  # TODO: align Arabic chunks
                section_title=section_title,
                section_type=section_type,
                parent_policy_id=policy.parent_policy_id,
                jurisdiction=policy.jurisdiction,
            )
            chunks.append(chunk)

    logger.info(
        "policy_chunked",
        policy_id=policy.policy_id,
        policy_number=policy.policy_number,
        chunks=len(chunks),
        version=policy.version,
    )
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str, str]]:
    """Split text into sections based on headers.

    Returns list of (section_title, section_text, section_type).
    """
    # Find all section boundaries
    boundaries: list[tuple[int, str]] = []

    for pattern, _ in _SECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            boundaries.append((match.start(), match.group(1).strip()))

    if not boundaries:
        # No sections found — treat entire text as one section
        section_type = _classify_section_type("", text)
        return [("General", text, section_type)]

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    sections: list[tuple[str, str, str]] = []

    # Text before first section
    if boundaries[0][0] > 0:
        preamble = text[: boundaries[0][0]].strip()
        if preamble:
            sections.append(("Preamble", preamble, "general"))

    # Extract each section
    for i, (pos, title) in enumerate(boundaries):
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end_pos].strip()

        # Remove the title from the body
        body = section_text[len(title):].strip()
        section_type = _classify_section_type(title, body)

        sections.append((title, body, section_type))

    return sections


def _classify_section_type(title: str, text: str) -> str:
    """Classify a section based on its title and content."""
    combined = (title + " " + text[:200]).lower()

    for section_type, keywords in _SECTION_TYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return section_type

    return "general"


def _split_section(text: str, max_size: int, overlap: int) -> list[str]:
    """Split a section into chunks respecting sentence boundaries."""
    if len(text) <= max_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?。])\s+", text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: keep the last bit of the previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
