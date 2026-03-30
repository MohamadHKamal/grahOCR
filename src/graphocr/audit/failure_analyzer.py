"""Failure Analyzer — classifies and aggregates Type A/B failures.

Queries the Redis FailureStore and Neo4j to produce a diagnostic report
that classifies failures as:
  - Type A (Input Failure): OCR spatial-blind errors (reading order, column mixing)
  - Type B (Intelligence Failure): RAG context-blind errors (wrong policy version)

Supports federated queries scoped to a jurisdiction for data sovereignty.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from graphocr.core.logging import get_logger
from graphocr.core.types import FailureType

logger = get_logger(__name__)


@dataclass
class FailureBreakdown:
    """Aggregated failure statistics for a time window."""

    window_start: datetime
    window_end: datetime
    total_reports: int = 0

    # Type A (spatial-blind OCR)
    type_a_count: int = 0
    type_a_root_causes: dict[str, int] = field(default_factory=dict)

    # Type B (context-blind RAG)
    type_b_count: int = 0
    type_b_root_causes: dict[str, int] = field(default_factory=dict)

    # By field
    affected_fields: dict[str, int] = field(default_factory=dict)

    # Resolution methods
    resolution_methods: dict[str, int] = field(default_factory=dict)

    # Training data
    training_eligible: int = 0

    @property
    def type_a_rate(self) -> float:
        return self.type_a_count / max(self.total_reports, 1)

    @property
    def type_b_rate(self) -> float:
        return self.type_b_count / max(self.total_reports, 1)

    def to_dict(self) -> dict:
        return {
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
            },
            "total_reports": self.total_reports,
            "type_a": {
                "count": self.type_a_count,
                "rate": round(self.type_a_rate, 4),
                "root_causes": self.type_a_root_causes,
            },
            "type_b": {
                "count": self.type_b_count,
                "rate": round(self.type_b_rate, 4),
                "root_causes": self.type_b_root_causes,
            },
            "affected_fields": self.affected_fields,
            "resolution_methods": self.resolution_methods,
            "training_eligible": self.training_eligible,
        }


class FailureAnalyzer:
    """Analyzes failure patterns from the FailureStore.

    Provides the diagnostic capability described in Task 1:
    distinguishing Input Failure (Type A) from Intelligence Failure (Type B).
    """

    async def analyze(
        self,
        window_hours: int = 24,
        jurisdiction: str | None = None,
    ) -> FailureBreakdown:
        """Run failure analysis over a time window.

        Args:
            window_hours: How far back to look.
            jurisdiction: If set, only analyze reports from this jurisdiction
                (federated data sovereignty).

        Returns:
            FailureBreakdown with Type A/B classification and aggregations.
        """
        now = datetime.utcnow()
        window_start = now - timedelta(hours=window_hours)

        reports = await self._fetch_reports(window_start, jurisdiction)

        breakdown = FailureBreakdown(
            window_start=window_start,
            window_end=now,
            total_reports=len(reports),
        )

        for report in reports:
            # Classify Type A vs Type B
            if report.failure_type == FailureType.TYPE_A_SPATIAL_BLIND:
                breakdown.type_a_count += 1
                breakdown.type_a_root_causes[report.root_cause] = (
                    breakdown.type_a_root_causes.get(report.root_cause, 0) + 1
                )
            elif report.failure_type == FailureType.TYPE_B_CONTEXT_BLIND:
                breakdown.type_b_count += 1
                breakdown.type_b_root_causes[report.root_cause] = (
                    breakdown.type_b_root_causes.get(report.root_cause, 0) + 1
                )

            # Track affected fields
            if report.affected_field:
                breakdown.affected_fields[report.affected_field] = (
                    breakdown.affected_fields.get(report.affected_field, 0) + 1
                )

            # Track resolution methods
            breakdown.resolution_methods[report.resolution_method] = (
                breakdown.resolution_methods.get(report.resolution_method, 0) + 1
            )

            if report.add_to_dspy_training:
                breakdown.training_eligible += 1

        logger.info(
            "failure_analysis_complete",
            total=breakdown.total_reports,
            type_a=breakdown.type_a_count,
            type_b=breakdown.type_b_count,
            jurisdiction=jurisdiction,
        )

        return breakdown

    async def get_report_by_id(self, report_id: str):
        """Fetch a single failure report."""
        try:
            from graphocr.layer2_verification.agents.failure_store import FailureStore

            store = FailureStore()
            await store.connect()
            try:
                return await store.get_report(report_id)
            finally:
                await store.close()
        except Exception as e:
            logger.warning("report_fetch_failed", error=str(e))
            return None

    async def _fetch_reports(self, since: datetime, jurisdiction: str | None):
        """Fetch failure reports from Redis, optionally scoped by jurisdiction."""
        from graphocr.models.failure import FailureReport

        try:
            from graphocr.layer2_verification.agents.failure_store import FailureStore

            store = FailureStore()
            await store.connect()
            try:
                # Fetch all training data (which includes all saved reports)
                reports = await store.get_training_data(limit=10000)

                # Filter by time window
                reports = [r for r in reports if r.created_at >= since]

                # Filter by jurisdiction if federated mode
                if jurisdiction:
                    reports = [
                        r for r in reports
                        if self._matches_jurisdiction(r, jurisdiction)
                    ]

                return reports
            finally:
                await store.close()
        except Exception as e:
            logger.warning("failure_store_fetch_failed", error=str(e))
            return []

    def _matches_jurisdiction(self, report, jurisdiction: str) -> bool:
        """Check if a report belongs to the given jurisdiction.

        In federated mode, document IDs are prefixed with jurisdiction codes.
        """
        return report.document_id.startswith(f"{jurisdiction}_") or not jurisdiction
