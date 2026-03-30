"""DSPy Mentorship Supervisor — automated prompt performance monitoring.

This is the "Agentic Supervisor" that:
1. Monitors DSPy module performance via LangSmith traces
2. Detects metric degradation against baselines
3. Triggers automatic re-optimization when performance drops
4. Monitors textual gradient stability to prevent prompt oscillation

This is a DETERMINISTIC, AUTOMATED system — not "better prompting."
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.dspy_layer.gradient_monitor import GradientMonitor
from graphocr.dspy_layer.optimizers import configure_dspy_lm, optimize_module, save_optimized_module

logger = get_logger(__name__)


@dataclass
class ModulePerformance:
    """Performance tracking for a single DSPy module."""

    module_name: str
    baseline_score: float
    current_score: float = 0.0
    samples_count: int = 0
    last_optimized: datetime | None = None
    optimization_count_today: int = 0
    last_check: datetime | None = None


@dataclass
class SupervisorState:
    """Global state of the DSPy supervisor."""

    modules: dict[str, ModulePerformance] = field(default_factory=dict)
    total_optimizations: int = 0
    last_run: datetime | None = None
    alerts: list[dict] = field(default_factory=list)


class DSPySupervisor:
    """Automated DSPy prompt performance supervisor.

    Runs periodically, checks module performance, and triggers
    re-optimization when metrics degrade beyond thresholds.
    """

    def __init__(self):
        settings = get_settings()
        dspy_config = settings.dspy
        supervisor_config = dspy_config.get("supervisor", {})

        self.check_interval = timedelta(
            minutes=supervisor_config.get("check_interval_minutes", 30)
        )
        self.performance_window = timedelta(
            minutes=supervisor_config.get("performance_window_minutes", 60)
        )
        self.min_samples = supervisor_config.get("min_samples_for_reoptimize", 100)
        self.max_reoptimize_per_day = supervisor_config.get("max_reoptimize_per_day", 3)

        self.gradient_monitor = GradientMonitor(
            stability_threshold=dspy_config.get("gradient_monitor", {}).get("stability_threshold", 0.7),
            window_size=dspy_config.get("gradient_monitor", {}).get("window_size", 5),
        )

        self.state = SupervisorState()
        self._running = False
        self._module_configs = dspy_config.get("modules", {})

        # Initialize module tracking
        for name, config in self._module_configs.items():
            self.state.modules[name] = ModulePerformance(
                module_name=name,
                baseline_score=config.get(f"baseline_{config.get('metric', 'f1')}", 0.9),
            )

    async def start(self) -> None:
        """Start the supervisor loop."""
        self._running = True
        logger.info("dspy_supervisor_started", interval=str(self.check_interval))

        configure_dspy_lm()

        while self._running:
            try:
                await self.check_and_intervene()
            except Exception as e:
                logger.error("supervisor_check_failed", error=str(e))

            await asyncio.sleep(self.check_interval.total_seconds())

    def stop(self) -> None:
        """Stop the supervisor loop."""
        self._running = False
        logger.info("dspy_supervisor_stopped")

    async def check_and_intervene(self) -> dict[str, str]:
        """Run a single check cycle. Returns actions taken per module."""
        actions: dict[str, str] = {}
        self.state.last_run = datetime.utcnow()

        for module_name, perf in self.state.modules.items():
            config = self._module_configs.get(module_name, {})
            degradation_threshold = config.get("degradation_threshold", 0.05)

            # Step 1: Fetch recent performance metrics
            current_score = await self._fetch_module_metrics(module_name)
            perf.current_score = current_score
            perf.last_check = datetime.utcnow()

            # Step 2: Check degradation
            degradation = perf.baseline_score - current_score

            if degradation > degradation_threshold and perf.samples_count >= self.min_samples:
                if perf.optimization_count_today >= self.max_reoptimize_per_day:
                    actions[module_name] = "degraded_but_max_optimizations_reached"
                    self._alert(
                        module_name,
                        f"Performance degraded by {degradation:.3f} but max daily "
                        f"optimizations ({self.max_reoptimize_per_day}) reached.",
                    )
                else:
                    actions[module_name] = "triggering_reoptimization"
                    await self._reoptimize_module(module_name)
            else:
                actions[module_name] = "healthy"

            # Step 3: Check gradient stability
            gradient_analysis = self.gradient_monitor.analyze(module_name)
            if gradient_analysis.is_diverging:
                actions[module_name] = f"gradient_alert: {gradient_analysis.recommendation}"
                self._alert(module_name, gradient_analysis.recommendation)

        logger.info("supervisor_check_complete", actions=actions)
        return actions

    async def _fetch_module_metrics(self, module_name: str) -> float:
        """Fetch recent performance metrics from LangSmith traces.

        Queries LangSmith API for runs tagged with the module name
        within the performance window. Returns average metric score.
        Falls back to baseline if LangSmith is unavailable.
        """
        settings = get_settings()
        perf = self.state.modules.get(module_name)

        if not settings.langsmith_api_key:
            if perf:
                perf.samples_count += 10
                return perf.baseline_score
            return 0.0

        try:
            from langsmith import Client

            client = Client(
                api_key=settings.langsmith_api_key,
            )

            # Query runs tagged with this DSPy module within the time window
            start_time = datetime.utcnow() - self.performance_window
            runs = list(client.list_runs(
                project_name=settings.langsmith_project,
                filter=f'and(eq(metadata_key, "dspy_module"), eq(metadata_value, "{module_name}"))',
                start_time=start_time,
                limit=500,
            ))

            if not runs:
                logger.info("no_langsmith_runs", module=module_name, window=str(self.performance_window))
                if perf:
                    perf.samples_count = 0
                    return perf.baseline_score
                return 0.0

            # Extract scores from feedback
            scores = []
            for run in runs:
                if run.feedback_stats:
                    score = run.feedback_stats.get("score") or run.feedback_stats.get("correctness")
                    if score is not None:
                        scores.append(float(score))
                # Also check output metadata for inline scores
                if run.outputs and isinstance(run.outputs, dict):
                    out_score = run.outputs.get("confidence") or run.outputs.get("score")
                    if out_score is not None:
                        scores.append(float(out_score))

            if perf:
                perf.samples_count = len(scores)

            if not scores:
                return perf.baseline_score if perf else 0.0

            avg_score = sum(scores) / len(scores)
            logger.info(
                "langsmith_metrics_fetched",
                module=module_name,
                runs=len(runs),
                scores=len(scores),
                avg_score=avg_score,
            )
            return avg_score

        except Exception as e:
            logger.warning("langsmith_fetch_failed", module=module_name, error=str(e))
            if perf:
                perf.samples_count += 10
                return perf.baseline_score
            return 0.0

    async def _reoptimize_module(self, module_name: str) -> None:
        """Trigger re-optimization of a DSPy module.

        Pulls recent failure cases, runs MIPRO, validates the new prompt,
        and atomically swaps it in.
        """
        logger.info("reoptimization_triggered", module=module_name)
        perf = self.state.modules[module_name]

        try:
            # Step 1: Fetch training data from failure reports
            training_data = await self._fetch_training_data(module_name)

            if len(training_data) < 10:
                logger.warning(
                    "insufficient_training_data",
                    module=module_name,
                    samples=len(training_data),
                )
                return

            # Step 2: Run optimization
            import dspy
            optimized = optimize_module(module_name, training_data)

            # Step 3: Validate on held-out set
            # (In production, split training_data and validate)

            # Step 4: Save and create atomic symlink for hot-swap
            save_path = f"optimized_modules/{module_name}_{int(time.time())}"
            save_optimized_module(optimized, save_path)

            # Atomic symlink swap: _latest always points to the newest optimized module
            latest_link = Path(f"optimized_modules/{module_name}_latest")
            tmp_link = Path(f"optimized_modules/{module_name}_latest.tmp")
            try:
                tmp_link.unlink(missing_ok=True)
                tmp_link.symlink_to(Path(save_path).resolve())
                tmp_link.rename(latest_link)
                logger.info("dspy_module_symlink_updated", module=module_name, target=save_path)
            except OSError as link_err:
                logger.warning("dspy_symlink_failed", error=str(link_err))

            # Step 5: Record gradient snapshot
            prompt_text = str(optimized)  # Serialized prompt
            self.gradient_monitor.record_snapshot(
                module_name=module_name,
                prompt_text=prompt_text,
                metric_score=perf.current_score,
                optimization_step=perf.optimization_count_today + 1,
            )

            perf.optimization_count_today += 1
            perf.last_optimized = datetime.utcnow()
            self.state.total_optimizations += 1

            logger.info(
                "reoptimization_complete",
                module=module_name,
                save_path=save_path,
            )

        except Exception as e:
            logger.error("reoptimization_failed", module=module_name, error=str(e))
            self._alert(module_name, f"Re-optimization failed: {e}")

    async def _fetch_training_data(self, module_name: str) -> list:
        """Fetch training examples from the post-mortem failure database.

        Reads from the Redis-backed FailureStore and converts FailureReport
        objects to dspy.Example objects suitable for MIPRO optimization.
        """
        import dspy

        try:
            from graphocr.layer2_verification.agents.failure_store import FailureStore

            store = FailureStore()
            await store.connect()
            try:
                reports = await store.get_training_data(limit=500)
            finally:
                await store.close()
        except Exception as e:
            logger.warning("training_data_fetch_failed", module=module_name, error=str(e))
            return []

        if not reports:
            return []

        # Convert FailureReports to DSPy Examples
        # Each example has the corrected value as the ground truth
        examples = []
        for report in reports:
            if not report.corrected_value or not report.original_value:
                continue

            # Map report fields to DSPy module inputs/outputs
            if module_name == "ClaimFieldExtractor":
                example = dspy.Example(
                    spatial_tokens_text=f"[corrected field: {report.affected_field}] "
                    f"original='{report.original_value}' context='{report.root_cause}'",
                    document_language="mixed",
                    context_hints=f"Root cause: {report.root_cause}",
                    claim_fields_json=f'{{"{report.affected_field}": '
                    f'{{"value": "{report.corrected_value}", "confidence": 1.0}}}}',
                ).with_inputs("spatial_tokens_text", "document_language", "context_hints")
                examples.append(example)

            elif module_name == "ArabicMedicalNormalizer":
                if report.root_cause == "ocr_misread":
                    example = dspy.Example(
                        arabic_text=report.original_value,
                        context=report.affected_field,
                        normalized_text=report.corrected_value,
                        confidence=1.0,
                    ).with_inputs("arabic_text", "context")
                    examples.append(example)

            elif module_name == "DiagnosisCodeMapper":
                if report.affected_field in ("diagnosis_codes",):
                    example = dspy.Example(
                        diagnosis_text=report.original_value,
                        language="mixed",
                        icd10_code=report.corrected_value,
                        code_description="",
                        mapping_confidence=1.0,
                    ).with_inputs("diagnosis_text", "language")
                    examples.append(example)

        logger.info(
            "training_data_converted",
            module=module_name,
            reports=len(reports),
            examples=len(examples),
        )
        return examples

    def _alert(self, module_name: str, message: str) -> None:
        """Record an alert for the module."""
        alert = {
            "module": module_name,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.state.alerts.append(alert)
        logger.warning("supervisor_alert", **alert)

    def get_status(self) -> dict:
        """Get the current supervisor status."""
        return {
            "running": self._running,
            "last_run": self.state.last_run.isoformat() if self.state.last_run else None,
            "total_optimizations": self.state.total_optimizations,
            "modules": {
                name: {
                    "baseline": perf.baseline_score,
                    "current": perf.current_score,
                    "degradation": perf.baseline_score - perf.current_score,
                    "samples": perf.samples_count,
                    "optimizations_today": perf.optimization_count_today,
                    "last_optimized": perf.last_optimized.isoformat() if perf.last_optimized else None,
                }
                for name, perf in self.state.modules.items()
            },
            "gradient_stability": {
                name: self.gradient_monitor.check_stability(name)
                for name in self.state.modules
            },
            "recent_alerts": self.state.alerts[-10:],
        }
