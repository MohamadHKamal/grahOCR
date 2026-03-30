"""Textual gradient monitor — detects prompt optimization drift.

Monitors the direction and magnitude of prompt changes across optimization
runs. When gradients are unstable (prompt oscillating between strategies),
signals the supervisor to intervene.

This is NOT "better prompting" — it's a deterministic, automated monitoring
system based on textual gradient analysis.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

from graphocr.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GradientSnapshot:
    """A snapshot of prompt state at a point in time."""

    timestamp: datetime
    module_name: str
    prompt_hash: str  # Hash of the full prompt text
    prompt_text: str
    metric_score: float
    optimization_step: int


@dataclass
class GradientAnalysis:
    """Analysis of gradient stability over a window of snapshots."""

    module_name: str
    stability_score: float  # 0.0 = chaotic, 1.0 = stable
    is_diverging: bool
    direction_consistency: float  # How consistent are the changes
    magnitude_trend: str  # "increasing", "decreasing", "stable", "oscillating"
    recommendation: str
    window_size: int
    metric_trend: list[float]


class GradientMonitor:
    """Monitors textual gradient stability across DSPy optimization runs.

    Tracks how prompt instructions change over successive optimization runs.
    Detects oscillation (prompt flipping between strategies) and divergence
    (metric getting worse despite optimization).
    """

    def __init__(
        self,
        stability_threshold: float = 0.7,
        window_size: int = 5,
        divergence_alert_threshold: float = 0.3,
    ):
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.divergence_alert_threshold = divergence_alert_threshold
        self._history: dict[str, deque[GradientSnapshot]] = {}

    def record_snapshot(
        self,
        module_name: str,
        prompt_text: str,
        metric_score: float,
        optimization_step: int,
    ) -> None:
        """Record a prompt snapshot after an optimization run."""
        if module_name not in self._history:
            self._history[module_name] = deque(maxlen=self.window_size * 2)

        snapshot = GradientSnapshot(
            timestamp=datetime.utcnow(),
            module_name=module_name,
            prompt_hash=hashlib.md5(prompt_text.encode()).hexdigest()[:12],
            prompt_text=prompt_text,
            metric_score=metric_score,
            optimization_step=optimization_step,
        )
        self._history[module_name].append(snapshot)

        logger.info(
            "gradient_snapshot_recorded",
            module=module_name,
            step=optimization_step,
            score=metric_score,
            hash=snapshot.prompt_hash,
        )

    def analyze(self, module_name: str) -> GradientAnalysis:
        """Analyze gradient stability for a module.

        Returns an analysis with stability score and recommendation.
        """
        history = list(self._history.get(module_name, []))

        if len(history) < 2:
            return GradientAnalysis(
                module_name=module_name,
                stability_score=1.0,
                is_diverging=False,
                direction_consistency=1.0,
                magnitude_trend="stable",
                recommendation="Insufficient data — need at least 2 snapshots.",
                window_size=len(history),
                metric_trend=[s.metric_score for s in history],
            )

        # Use the last window_size snapshots
        recent = history[-self.window_size:]

        # 1. Compute prompt similarity chain
        similarities = []
        for i in range(1, len(recent)):
            sim = SequenceMatcher(
                None,
                recent[i - 1].prompt_text,
                recent[i].prompt_text,
            ).ratio()
            similarities.append(sim)

        # 2. Compute direction consistency
        # Direction = whether each change moves the metric in the same direction
        metric_deltas = [
            recent[i].metric_score - recent[i - 1].metric_score
            for i in range(1, len(recent))
        ]

        if len(metric_deltas) >= 2:
            # Count sign changes — more sign changes = less consistent
            sign_changes = sum(
                1
                for i in range(1, len(metric_deltas))
                if (metric_deltas[i] > 0) != (metric_deltas[i - 1] > 0)
            )
            direction_consistency = 1.0 - (sign_changes / (len(metric_deltas) - 1))
        else:
            direction_consistency = 1.0

        # 3. Compute magnitude trend
        magnitude_trend = self._classify_trend(metric_deltas)

        # 4. Compute overall stability score
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        stability_score = (avg_similarity * 0.4 + direction_consistency * 0.6)

        # 5. Detect divergence
        is_diverging = (
            stability_score < self.divergence_alert_threshold
            or (magnitude_trend == "decreasing" and len(recent) >= 3)
        )

        # 6. Generate recommendation
        recommendation = self._generate_recommendation(
            stability_score, direction_consistency, magnitude_trend, is_diverging
        )

        analysis = GradientAnalysis(
            module_name=module_name,
            stability_score=stability_score,
            is_diverging=is_diverging,
            direction_consistency=direction_consistency,
            magnitude_trend=magnitude_trend,
            recommendation=recommendation,
            window_size=len(recent),
            metric_trend=[s.metric_score for s in recent],
        )

        logger.info(
            "gradient_analysis",
            module=module_name,
            stability=stability_score,
            diverging=is_diverging,
            trend=magnitude_trend,
        )

        return analysis

    def check_stability(self, module_name: str) -> float:
        """Quick check — returns stability score (0.0-1.0)."""
        return self.analyze(module_name).stability_score

    def check_all_modules(self) -> dict[str, GradientAnalysis]:
        """Analyze all tracked modules."""
        return {name: self.analyze(name) for name in self._history}

    @staticmethod
    def _classify_trend(deltas: list[float]) -> str:
        """Classify the trend of metric changes."""
        if not deltas:
            return "stable"

        positive = sum(1 for d in deltas if d > 0.001)
        negative = sum(1 for d in deltas if d < -0.001)
        total = len(deltas)

        if positive > total * 0.7:
            return "increasing"
        elif negative > total * 0.7:
            return "decreasing"
        elif positive > 0 and negative > 0 and (positive + negative) > total * 0.6:
            return "oscillating"
        else:
            return "stable"

    @staticmethod
    def _generate_recommendation(
        stability: float,
        consistency: float,
        trend: str,
        diverging: bool,
    ) -> str:
        """Generate an actionable recommendation based on the analysis."""
        if diverging:
            if trend == "oscillating":
                return (
                    "ALERT: Prompt is oscillating between strategies. "
                    "Increase training data diversity or widen the MIPRO search space. "
                    "Consider freezing the current best prompt until more data is available."
                )
            elif trend == "decreasing":
                return (
                    "ALERT: Metric is declining despite optimization. "
                    "Check for data distribution shift. Consider rolling back to the "
                    "last known-good prompt and investigating the training data quality."
                )
            else:
                return (
                    "WARNING: Low stability detected. "
                    "Review recent training examples for label noise or contradictions."
                )

        if stability < 0.5:
            return (
                "CAUTION: Moderate instability. Monitor closely. "
                "May benefit from more training examples in the weak areas."
            )

        if trend == "increasing":
            return "HEALTHY: Optimization is improving. Continue current approach."

        return "STABLE: No intervention needed."
