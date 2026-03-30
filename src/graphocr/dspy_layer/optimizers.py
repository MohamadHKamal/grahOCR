"""DSPy optimizer configurations — MIPRO and BootstrapFewShot.

Configures prompt optimizers that automatically tune instructions
and select few-shot demonstrations for each DSPy module.
"""

from __future__ import annotations

from typing import Any, Callable

import dspy

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger
from graphocr.dspy_layer.metrics import METRICS
from graphocr.dspy_layer.modules import (
    ArabicMedicalNormalizer,
    ChallengeGenerator,
    ClaimFieldExtractor,
    DiagnosisCodeMapper,
    PolicyVersionValidator,
)

logger = get_logger(__name__)

# Module registry
MODULES = {
    "ClaimFieldExtractor": ClaimFieldExtractor,
    "ArabicMedicalNormalizer": ArabicMedicalNormalizer,
    "DiagnosisCodeMapper": DiagnosisCodeMapper,
    "ChallengeGenerator": ChallengeGenerator,
    "PolicyVersionValidator": PolicyVersionValidator,
}


def configure_dspy_lm() -> None:
    """Configure DSPy to use the vLLM endpoint."""
    settings = get_settings()
    dspy_config = settings.dspy

    lm = dspy.LM(
        model=f"openai/{dspy_config.get('lm_model', 'qwen2.5-7b-instruct')}",
        api_base=dspy_config.get("lm_endpoint", "http://localhost:8000/v1"),
        api_key=settings.vllm_api_key,
    )
    dspy.configure(lm=lm)
    logger.info("dspy_lm_configured", model=dspy_config.get("lm_model"))


def optimize_module(
    module_name: str,
    trainset: list[dspy.Example],
    valset: list[dspy.Example] | None = None,
) -> dspy.Module:
    """Run DSPy optimization on a specific module.

    Args:
        module_name: Name of the module to optimize (from MODULES registry).
        trainset: Training examples for optimization.
        valset: Optional validation set.

    Returns:
        Optimized DSPy module with tuned prompts and demonstrations.
    """
    settings = get_settings()
    dspy_config = settings.dspy
    module_config = dspy_config.get("modules", {}).get(module_name, {})

    # Instantiate module
    module_cls = MODULES.get(module_name)
    if not module_cls:
        raise ValueError(f"Unknown DSPy module: {module_name}")
    module = module_cls()

    # Get metric
    metric_name = module_config.get("metric", "field_level_f1")
    metric = METRICS.get(metric_name, METRICS["field_level_f1"])

    # Choose optimizer
    optimizer_type = module_config.get("optimizer", "mipro")

    if optimizer_type == "mipro":
        optimized = _run_mipro(module, trainset, valset or trainset, metric, module_config)
    elif optimizer_type == "bootstrap_fewshot":
        optimized = _run_bootstrap(module, trainset, metric, module_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    logger.info(
        "dspy_optimization_complete",
        module=module_name,
        optimizer=optimizer_type,
        trainset_size=len(trainset),
    )
    return optimized


def _run_mipro(
    module: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable,
    config: dict[str, Any],
) -> dspy.Module:
    """Run MIPRO optimizer."""
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="medium",
        num_threads=4,
    )

    return optimizer.compile(
        module,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
        max_labeled_demos=config.get("max_labeled_demos", 8),
    )


def _run_bootstrap(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric: Callable,
    config: dict[str, Any],
) -> dspy.Module:
    """Run BootstrapFewShot optimizer."""
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=config.get("max_bootstrapped_demos", 8),
    )

    return optimizer.compile(module, trainset=trainset)


def save_optimized_module(module: dspy.Module, path: str) -> None:
    """Save an optimized module to disk."""
    module.save(path)
    logger.info("dspy_module_saved", path=path)


def load_optimized_module(module_name: str, path: str) -> dspy.Module:
    """Load a previously optimized module from disk."""
    module_cls = MODULES.get(module_name)
    if not module_cls:
        raise ValueError(f"Unknown DSPy module: {module_name}")
    module = module_cls()
    module.load(path)
    logger.info("dspy_module_loaded", module=module_name, path=path)
    return module
