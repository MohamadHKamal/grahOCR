"""Traffic Controller — uncertainty-based routing for claims.

Routes ~90% of documents to the cheap rail (fast, single-model extraction)
and ~10% to the VLM-consensus pipeline (multi-model, adversarial checks).

The uncertainty score U is a weighted linear combination:

    U = (1 - C̄) + w_hw · R_hw + w_mix · R_mix + w_fail · S_fail + w_ent · H_norm

Where:
    C̄     = mean OCR confidence across all tokens
    R_hw  = fraction of handwritten tokens (w_hw = 0.15)
    R_mix = language mixing entropy ratio (w_mix = 0.02)
    S_fail = max failure classification severity (w_fail = 0.10)
    H_norm = normalized Shannon entropy of confidence distribution (w_ent = 0.10)

The threshold T splits traffic into two rails:
    U ≤ T  →  cheap rail (fast, single-model)
    U > T  →  VLM consensus (multi-model, adversarial)

Threshold selection:
    T is set via ROC analysis on a labeled calibration dataset
    (scripts/calibrate_threshold.py). The optimal T maximizes the Youden
    index J = sensitivity + specificity - 1. Default T = 0.35 corresponds
    to cheap_rail_confidence_threshold = 0.65 in config.

    Run `python scripts/calibrate_threshold.py --synthetic 1000` to
    recalibrate on your own data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.core.types import Language, ProcessingPath
from graphocr.models.failure import FailureClassification
from graphocr.models.token import SpatialToken

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Result of the traffic controller's routing decision."""

    path: ProcessingPath
    uncertainty_score: float
    confidence_mean: float
    handwriting_ratio: float
    language_mixing_ratio: float
    failure_severity: float
    confidence_entropy: float
    reason: str


def route_document(
    tokens: list[SpatialToken],
    failures: list[FailureClassification] | None = None,
) -> RoutingDecision:
    """Compute uncertainty score and route to cheap or expensive rail.

    The uncertainty score U is computed as:
        U = 1 - (w1 * conf_mean - w2 * hw_ratio - w3 * mix_ratio - w4 * fail_sev + w5 * (1 - entropy_norm))

    Where weights are configurable and the threshold is based on the
    desired traffic split (~90% cheap / ~10% expensive).

    Args:
        tokens: SpatialTokens from Layer 1.
        failures: Failure classifications from Layer 1.

    Returns:
        RoutingDecision with path, score, and components.
    """
    settings = get_settings()
    monitoring = settings.monitoring
    tc_config = monitoring.get("traffic_controller", {})

    if not tokens:
        return RoutingDecision(
            path=ProcessingPath.VLM_CONSENSUS,
            uncertainty_score=1.0,
            confidence_mean=0.0,
            handwriting_ratio=0.0,
            language_mixing_ratio=0.0,
            failure_severity=0.0,
            confidence_entropy=0.0,
            reason="No tokens to process",
        )

    # Compute components
    confidences = [t.confidence for t in tokens]
    conf_mean = sum(confidences) / len(confidences)
    hw_ratio = sum(1 for t in tokens if t.is_handwritten) / len(tokens)
    mix_ratio = _language_mixing_ratio(tokens)
    fail_sev = max((f.severity for f in (failures or [])), default=0.0)
    entropy = _confidence_entropy(confidences)
    entropy_norm = entropy / math.log(len(confidences)) if len(confidences) > 1 else 0.0

    # Compute uncertainty score
    # Higher score = more uncertain = needs expensive pipeline
    hw_penalty = tc_config.get("handwriting_penalty", 0.15)
    mix_penalty = tc_config.get("mixed_language_penalty", 0.02)
    fail_penalty = tc_config.get("failure_classification_penalty", 0.10)

    uncertainty = (
        (1.0 - conf_mean)
        + hw_penalty * hw_ratio
        + mix_penalty * mix_ratio
        + fail_penalty * fail_sev
        + 0.1 * entropy_norm
    )
    uncertainty = max(0.0, min(1.0, uncertainty))

    # Route based on threshold
    threshold = tc_config.get("cheap_rail_confidence_threshold", 0.65)
    # Invert: if mean confidence > threshold and uncertainty is low -> cheap rail
    cheap_threshold = 1.0 - threshold  # 0.35

    if uncertainty <= cheap_threshold:
        path = ProcessingPath.CHEAP_RAIL
        reason = f"Low uncertainty ({uncertainty:.3f} <= {cheap_threshold:.3f})"
    else:
        path = ProcessingPath.VLM_CONSENSUS
        reason = f"High uncertainty ({uncertainty:.3f} > {cheap_threshold:.3f})"

    decision = RoutingDecision(
        path=path,
        uncertainty_score=uncertainty,
        confidence_mean=conf_mean,
        handwriting_ratio=hw_ratio,
        language_mixing_ratio=mix_ratio,
        failure_severity=fail_sev,
        confidence_entropy=entropy,
        reason=reason,
    )

    logger.info(
        "routing_decision",
        path=path.value,
        uncertainty=uncertainty,
        conf_mean=conf_mean,
        hw_ratio=hw_ratio,
    )
    return decision


def _language_mixing_ratio(tokens: list[SpatialToken]) -> float:
    """Compute how mixed the languages are (0 = single language, 1 = highly mixed)."""
    if not tokens:
        return 0.0

    lang_counts: dict[Language, int] = {}
    for t in tokens:
        lang_counts[t.language] = lang_counts.get(t.language, 0) + 1

    # Remove unknown
    lang_counts.pop(Language.UNKNOWN, None)

    if len(lang_counts) <= 1:
        return 0.0

    total = sum(lang_counts.values())
    # Entropy-based mixing: max when languages are equally distributed
    entropy = -sum(
        (c / total) * math.log(c / total)
        for c in lang_counts.values()
        if c > 0
    )
    max_entropy = math.log(len(lang_counts))

    return entropy / max_entropy if max_entropy > 0 else 0.0


def _confidence_entropy(confidences: list[float]) -> float:
    """Compute entropy of the confidence distribution.

    High entropy = confidences are spread out (uncertain).
    Low entropy = confidences are clustered (consistent).
    """
    if not confidences:
        return 0.0

    # Discretize into 10 bins
    bins = [0] * 10
    for c in confidences:
        idx = min(int(c * 10), 9)
        bins[idx] += 1

    total = len(confidences)
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy
