"""Configuration loader — merges YAML configs with environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class PipelineSettings(BaseSettings):
    """Top-level settings — env vars override YAML values."""

    # LLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "dummy"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphocr_dev"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "graphocr"
    minio_secret_key: str = "graphocr_dev"
    minio_bucket: str = "claims"
    minio_secure: bool = False

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "graphocr"
    langsmith_tracing: bool = True

    # Pipeline
    pipeline_max_concurrent: int = 64
    pipeline_max_agent_rounds: int = 2
    log_level: str = "INFO"

    # YAML-loaded nested configs (not from env)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    ocr: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, Any] = Field(default_factory=dict)
    neo4j_rules: dict[str, Any] = Field(default_factory=dict)
    dspy: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> PipelineSettings:
    """Load settings once, merging YAML configs."""
    yaml_pipeline = _load_yaml("pipeline.yaml")
    yaml_agents = _load_yaml("agents.yaml")
    yaml_neo4j = _load_yaml("neo4j_rules.yaml")
    yaml_dspy = _load_yaml("dspy_config.yaml")
    yaml_monitoring = _load_yaml("monitoring.yaml")

    return PipelineSettings(
        pipeline=yaml_pipeline.get("pipeline", {}),
        ocr=yaml_pipeline.get("ocr", {}),
        agents=yaml_agents.get("agents", {}),
        neo4j_rules=yaml_neo4j,
        dspy=yaml_dspy.get("dspy", {}),
        monitoring=yaml_monitoring,
    )
