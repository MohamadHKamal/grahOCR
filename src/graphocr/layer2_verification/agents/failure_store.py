"""FailureReport persistence — Redis-backed store for the feedback loop.

Stores FailureReport objects so the DSPy supervisor can fetch real
training examples, and the postmortem agent can update Neo4j rules.

This is the critical missing link between:
  PostMortem Agent -> FailureReport -> DSPy Supervisor -> Re-optimization
  PostMortem Agent -> FailureReport -> Neo4j Rule Updater -> New constraints
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import redis.asyncio as aioredis

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.models.failure import FailureReport

logger = get_logger(__name__)

FAILURE_KEY_PREFIX = "graphocr:failure_report:"
TRAINING_SET_KEY = "graphocr:dspy_training_set"
FAILURE_STATS_KEY = "graphocr:failure_stats"


class FailureStore:
    """Redis-backed store for FailureReport objects.

    Provides:
    - Persistent storage of failure reports with TTL
    - Queryable training set for DSPy re-optimization
    - Failure statistics for monitoring
    - Root cause aggregation for Neo4j rule updates
    """

    def __init__(self, redis_url: str | None = None, ttl_days: int = 30):
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._ttl = timedelta(days=ttl_days)
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        await self._client.ping()
        logger.info("failure_store_connected", url=self._redis_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("FailureStore not connected. Call connect() first.")
        return self._client

    async def save_report(self, report: FailureReport) -> None:
        """Persist a failure report to Redis."""
        key = f"{FAILURE_KEY_PREFIX}{report.report_id}"
        data = report.model_dump_json()

        pipe = self.client.pipeline()
        pipe.set(key, data, ex=int(self._ttl.total_seconds()))

        # Add to training set if flagged
        if report.add_to_dspy_training:
            pipe.lpush(TRAINING_SET_KEY, data)
            # Keep training set bounded
            pipe.ltrim(TRAINING_SET_KEY, 0, 9999)

        # Update stats
        pipe.hincrby(FAILURE_STATS_KEY, f"root_cause:{report.root_cause}", 1)
        pipe.hincrby(FAILURE_STATS_KEY, f"failure_type:{report.failure_type.value}", 1)
        pipe.hincrby(FAILURE_STATS_KEY, "total", 1)

        await pipe.execute()

        logger.info(
            "failure_report_saved",
            report_id=report.report_id,
            root_cause=report.root_cause,
            add_to_training=report.add_to_dspy_training,
        )

    async def save_reports_batch(self, reports: list[FailureReport]) -> None:
        """Save multiple reports atomically."""
        for report in reports:
            await self.save_report(report)

    async def get_report(self, report_id: str) -> FailureReport | None:
        """Fetch a single report by ID."""
        key = f"{FAILURE_KEY_PREFIX}{report_id}"
        data = await self.client.get(key)
        if data is None:
            return None
        return FailureReport.model_validate_json(data)

    async def get_training_data(
        self,
        limit: int = 500,
        root_cause_filter: str | None = None,
    ) -> list[FailureReport]:
        """Fetch training data for DSPy re-optimization.

        This is the method the DSPy supervisor calls to get real examples.

        Args:
            limit: Max reports to return.
            root_cause_filter: Optional filter by root cause type.

        Returns:
            List of FailureReport objects tagged for training.
        """
        raw_items = await self.client.lrange(TRAINING_SET_KEY, 0, limit - 1)

        reports = []
        for raw in raw_items:
            try:
                report = FailureReport.model_validate_json(raw)
                if root_cause_filter and report.root_cause != root_cause_filter:
                    continue
                reports.append(report)
            except Exception:
                continue

        logger.info(
            "training_data_fetched",
            total=len(reports),
            filter=root_cause_filter,
        )
        return reports

    async def get_rule_gap_reports(self, limit: int = 100) -> list[FailureReport]:
        """Fetch reports with root_cause='rule_gap' for Neo4j updates.

        These represent cases where the knowledge graph was missing a rule
        that should have caught the error earlier.
        """
        return await self.get_training_data(limit=limit, root_cause_filter="rule_gap")

    async def get_stats(self) -> dict:
        """Get failure statistics."""
        raw = await self.client.hgetall(FAILURE_STATS_KEY)
        return {k: int(v) for k, v in raw.items()}

    async def get_training_set_size(self) -> int:
        """Get the number of training examples available."""
        return await self.client.llen(TRAINING_SET_KEY)
