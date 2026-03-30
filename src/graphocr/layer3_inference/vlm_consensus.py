"""VLM Consensus pipeline — expensive path for high-uncertainty documents.

Runs the full multi-agent red team (Layer 2) with optional multi-VLM
voting on disagreed fields.
"""

from __future__ import annotations

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import ProcessingPath
from graphocr.layer2_verification.agents.graph_builder import run_red_team
from graphocr.models.extraction import ExtractionResult
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)


async def process_vlm_consensus(
    document_id: str,
    tokens: list[SpatialToken],
    page_images: dict[int, str] | None = None,
    policy_context_extractor: str = "",
    policy_context_validator: str = "",
    policy_context_challenger: str = "",
    retrieval_method: str = "",
    retrieval_warnings: list[str] | None = None,
) -> ExtractionResult:
    """Process a document through the full VLM consensus pipeline.

    This is the expensive path (~10% of traffic). Runs:
    1. Full multi-agent red team (Extractor -> Validator -> Challenger)
    2. Self-healing with VLM re-scan on conflicts
    3. Up to max_rounds of retry
    4. Escalation if consensus not reached
    """
    settings = get_settings()
    max_rounds = settings.pipeline_max_agent_rounds

    result_state = await run_red_team(
        document_id=document_id,
        spatial_tokens=tokens,
        page_images=page_images,
        max_rounds=max_rounds,
        policy_context=policy_context_extractor,
        policy_context_validator=policy_context_validator,
        policy_context_challenger=policy_context_challenger,
        retrieval_method=retrieval_method,
        retrieval_warnings=retrieval_warnings,
    )

    extraction = result_state.get("final_result")

    if extraction is None:
        extraction = ExtractionResult(
            claim_id="",
            document_id=document_id,
            overall_confidence=0.0,
            escalated=True,
        )

    extraction.processing_path = ProcessingPath.VLM_CONSENSUS

    logger.info(
        "vlm_consensus_complete",
        document_id=document_id,
        rounds=extraction.rounds_taken,
        escalated=extraction.escalated,
        confidence=extraction.overall_confidence,
    )
    return extraction
