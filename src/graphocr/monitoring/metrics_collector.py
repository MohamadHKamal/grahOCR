"""Metrics collector — aggregates pipeline metrics for monitoring dashboards."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics."""

    # Throughput
    documents_processed: int = 0
    documents_per_minute: float = 0.0

    # Routing
    cheap_rail_count: int = 0
    vlm_consensus_count: int = 0
    escalated_count: int = 0

    # Quality
    accuracy: float = 0.0
    avg_confidence: float = 0.0

    # Self-healing
    healing_triggered: int = 0
    healing_successful: int = 0

    # Agent
    avg_rounds: float = 0.0
    challenges_raised: int = 0
    graph_violations_caught: int = 0

    # DSPy
    dspy_optimizations: int = 0
    gradient_alerts: int = 0

    # Latency (ms)
    avg_latency_cheap: float = 0.0
    avg_latency_vlm: float = 0.0
    p95_latency: float = 0.0


class MetricsCollector:
    """Collects and aggregates real-time pipeline metrics."""

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()

    def increment(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def record_latency(self, name: str, latency_ms: float) -> None:
        self._latencies[name].append(latency_ms)
        # Keep only last 10000 measurements
        if len(self._latencies[name]) > 10000:
            self._latencies[name] = self._latencies[name][-5000:]

    def get_metrics(self) -> PipelineMetrics:
        elapsed_minutes = (time.time() - self._start_time) / 60
        total = self._counters.get("documents_processed", 0)

        cheap_latencies = self._latencies.get("cheap_rail", [])
        vlm_latencies = self._latencies.get("vlm_consensus", [])
        all_latencies = cheap_latencies + vlm_latencies

        return PipelineMetrics(
            documents_processed=total,
            documents_per_minute=total / max(elapsed_minutes, 0.01),
            cheap_rail_count=self._counters.get("cheap_rail", 0),
            vlm_consensus_count=self._counters.get("vlm_consensus", 0),
            escalated_count=self._counters.get("escalated", 0),
            healing_triggered=self._counters.get("healing_triggered", 0),
            healing_successful=self._counters.get("healing_successful", 0),
            challenges_raised=self._counters.get("challenges_raised", 0),
            graph_violations_caught=self._counters.get("graph_violations", 0),
            dspy_optimizations=self._counters.get("dspy_optimizations", 0),
            gradient_alerts=self._counters.get("gradient_alerts", 0),
            avg_latency_cheap=_avg(cheap_latencies),
            avg_latency_vlm=_avg(vlm_latencies),
            p95_latency=_percentile(all_latencies, 0.95) if all_latencies else 0.0,
        )

    def reset(self) -> None:
        self._counters.clear()
        self._latencies.clear()
        self._start_time = time.time()


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# Global collector
metrics = MetricsCollector()
