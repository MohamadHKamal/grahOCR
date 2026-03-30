"""LangGraph state machine wiring all red-team agents.

Defines the cyclic agent graph:
  Extractor -> Validator -> Challenger -> Consensus Check
    -> (agree) -> Output Assembly
    -> (disagree) -> Self-Healing -> Extractor (retry, max 2 rounds)
    -> (exhausted) -> Escalate
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graphocr.core.logging import get_logger
from graphocr.monitoring.langfuse_tracer import create_trace, record_score
from graphocr.layer2_verification.agents.challenger import challenger_node
from graphocr.layer2_verification.agents.extractor import extractor_node
from graphocr.layer2_verification.agents.postmortem import postmortem_node
from graphocr.layer2_verification.agents.validator import validator_node
from graphocr.layer2_verification.self_healing.conflict_detector import detect_conflicting_regions
from graphocr.layer2_verification.self_healing.feedback_loop import patch_tokens
from graphocr.layer2_verification.self_healing.vlm_rescanner import rescan_region
from graphocr.models.agent_state import RedTeamState

logger = get_logger(__name__)


def consensus_check_node(state: RedTeamState) -> dict:
    """Check if agents have reached consensus.

    Consensus is reached when:
    1. No high-confidence challenges exist (conf > 0.7)
    2. No critical graph violations exist (severity > 0.8)
    """
    challenges = state.get("challenges", [])
    violations = state.get("graph_violations", [])

    high_conf_challenges = [c for c in challenges if c.confidence > 0.7]
    critical_violations = [v for v in violations if v.severity > 0.8]

    if not high_conf_challenges and not critical_violations:
        logger.info("consensus_reached", document_id=state["document_id"])
        return {
            "consensus_reached": True,
            "final_result": state.get("extraction"),
        }

    logger.info(
        "consensus_not_reached",
        document_id=state["document_id"],
        high_conf_challenges=len(high_conf_challenges),
        critical_violations=len(critical_violations),
    )
    return {"consensus_reached": False}


async def self_healing_node(state: RedTeamState) -> dict:
    """Run the self-healing back-propagation loop.

    1. Detect conflicting regions from challenges + violations
    2. VLM re-scan each conflicting region using page images
    3. Patch token stream with rescan results
    4. Increment round counter for retry
    """
    current_round = state.get("round_number", 0)
    extraction = state.get("extraction")
    challenges = state.get("challenges", [])
    violations = state.get("graph_violations", [])
    tokens = state.get("spatial_tokens", [])
    page_images = state.get("page_images", {})

    if not extraction:
        logger.warning("self_healing_skipped", reason="no extraction to heal")
        return {"round_number": current_round + 1}

    # Step 1: Detect conflicting regions
    regions = detect_conflicting_regions(extraction, challenges, violations, tokens)

    if not regions:
        logger.info("self_healing_no_conflicts", document_id=state["document_id"])
        return {"round_number": current_round + 1}

    logger.info(
        "self_healing_started",
        document_id=state["document_id"],
        round=current_round,
        regions=len(regions),
    )

    # Step 2: VLM re-scan each conflicting region
    all_rescan_tokens = []
    for region in regions:
        page_image = page_images.get(region.page_number)
        if not page_image:
            logger.warning(
                "self_healing_no_page_image",
                page=region.page_number,
            )
            continue
        try:
            rescan_tokens = await rescan_region(page_image, region)
            all_rescan_tokens.extend(rescan_tokens)
        except Exception as e:
            logger.error(
                "self_healing_rescan_failed",
                page=region.page_number,
                error=str(e),
            )

    # Step 3: Patch token stream
    patched_tokens = patch_tokens(tokens, regions, all_rescan_tokens)

    logger.info(
        "self_healing_complete",
        document_id=state["document_id"],
        round=current_round,
        rescan_tokens=len(all_rescan_tokens),
        patched_total=len(patched_tokens),
    )

    return {
        "spatial_tokens": patched_tokens,
        "rescan_regions": regions,
        "rescan_results": all_rescan_tokens,
        "round_number": current_round + 1,
        "self_healing_applied": True,
    }


def escalate_node(state: RedTeamState) -> dict:
    """Mark the claim for human review escalation."""
    logger.warning(
        "claim_escalated",
        document_id=state["document_id"],
        rounds=state.get("round_number", 0),
        challenges=len(state.get("challenges", [])),
        violations=len(state.get("graph_violations", [])),
    )

    extraction = state.get("extraction")
    if extraction:
        extraction.escalated = True

    return {
        "final_result": extraction,
        "consensus_reached": False,
    }


def output_assembly_node(state: RedTeamState) -> dict:
    """Package the final result with audit trail."""
    extraction = state.get("final_result") or state.get("extraction")
    if extraction:
        extraction.rounds_taken = state.get("round_number", 0) + 1
        extraction.agent_consensus = {
            "extractor": "completed",
            "validator": f"{len(state.get('validation_issues', []))} issues",
            "challenger": f"{len(state.get('challenges', []))} challenges",
        }

    logger.info(
        "output_assembled",
        document_id=state["document_id"],
        rounds=extraction.rounds_taken if extraction else 0,
        escalated=extraction.escalated if extraction else False,
    )

    return {"final_result": extraction}


def _route_after_consensus(state: RedTeamState) -> str:
    """Route based on consensus check result."""
    if state.get("consensus_reached"):
        return "output_assembly"
    return "self_healing"


def _route_after_healing(state: RedTeamState) -> str:
    """Route based on whether we have retries left."""
    max_rounds = state.get("max_rounds", 2)
    current_round = state.get("round_number", 0)

    if current_round < max_rounds:
        return "retry"
    return "escalate"


def build_red_team_graph() -> StateGraph:
    """Build and compile the multi-agent red team graph.

    Returns a compiled LangGraph that can be invoked with an initial state.
    """
    graph = StateGraph(RedTeamState)

    # Add nodes
    graph.add_node("extractor", extractor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("challenger", challenger_node)
    graph.add_node("consensus_check", consensus_check_node)
    graph.add_node("self_healing", self_healing_node)
    graph.add_node("output_assembly", output_assembly_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("postmortem", postmortem_node)

    # Linear flow: extractor -> validator -> challenger -> consensus
    graph.add_edge("extractor", "validator")
    graph.add_edge("validator", "challenger")
    graph.add_edge("challenger", "consensus_check")

    # Conditional: consensus reached or not
    graph.add_conditional_edges(
        "consensus_check",
        _route_after_consensus,
        {
            "output_assembly": "output_assembly",
            "self_healing": "self_healing",
        },
    )

    # Conditional: retry or escalate
    graph.add_conditional_edges(
        "self_healing",
        _route_after_healing,
        {
            "retry": "extractor",
            "escalate": "escalate",
        },
    )

    # Terminal nodes -> postmortem -> END
    graph.add_edge("output_assembly", "postmortem")
    graph.add_edge("escalate", "postmortem")
    graph.add_edge("postmortem", END)

    # Entry point
    graph.set_entry_point("extractor")

    return graph.compile()


async def run_red_team(
    document_id: str,
    spatial_tokens: list,
    page_images: dict[int, str] | None = None,
    max_rounds: int = 2,
    policy_context: str = "",
    policy_context_validator: str = "",
    policy_context_challenger: str = "",
    retrieval_method: str = "",
    retrieval_warnings: list[str] | None = None,
) -> RedTeamState:
    """Run the full red-team pipeline on a document.

    Args:
        document_id: Unique document identifier.
        spatial_tokens: List of SpatialTokens from Layer 1.
        page_images: Dict of page_number -> image path for VLM re-scan.
        max_rounds: Maximum self-healing retry rounds.
        policy_context: RAG-retrieved policy text for the extractor.
        policy_context_validator: Policy text formatted for the validator.
        policy_context_challenger: Policy text formatted for the challenger.
        retrieval_method: How the policy was retrieved.
        retrieval_warnings: Warnings from the policy retriever.

    Returns:
        Final RedTeamState with extraction result and audit trail.
    """
    compiled_graph = build_red_team_graph()

    # Create Langfuse trace for this red team run
    trace = create_trace(
        document_id=document_id,
        processing_path="vlm_consensus",
        metadata={"max_rounds": max_rounds},
    )

    initial_state: RedTeamState = {
        "document_id": document_id,
        "spatial_tokens": spatial_tokens,
        "page_images": page_images or {},
        # RAG context
        "policy_context": policy_context,
        "policy_context_validator": policy_context_validator,
        "policy_context_challenger": policy_context_challenger,
        "retrieval_method": retrieval_method,
        "retrieval_warnings": retrieval_warnings or [],
        # Agent outputs
        "extraction": None,
        "validation_issues": [],
        "graph_violations": [],
        "challenges": [],
        "rescan_regions": [],
        "rescan_results": [],
        "failure_classifications": [],
        "self_healing_applied": False,
        "round_number": 0,
        "max_rounds": max_rounds,
        "consensus_reached": False,
        "final_result": None,
        "messages": [],
    }

    result = await compiled_graph.ainvoke(initial_state)

    # Record scores on Langfuse trace
    if trace:
        final = result.get("final_result")
        if final:
            record_score(trace, "overall_confidence", final.overall_confidence)
            record_score(trace, "rounds_taken", float(final.rounds_taken))
        record_score(
            trace, "challenges",
            float(len(result.get("challenges", []))),
        )
        record_score(
            trace, "graph_violations",
            float(len(result.get("graph_violations", []))),
        )

    return result
