"""
DyTopo Configuration
====================

YAML-based configuration with sensible defaults.
Loads from dytopo_config.yaml if present, otherwise uses built-in defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULTS = {
    "llm": {
        "base_url": "http://localhost:8008/v1",
        "model": "qwen2.5-coder-7b-instruct",
        "temperature_work": 0.7,
        "temperature_descriptor": 0.1,
        "temperature_manager": 0.3,
        "max_tokens_work": 2048,
        "max_tokens_descriptor": 256,
        "max_tokens_manager": 1024,
        "timeout_seconds": 120,
    },
    "routing": {
        "embedding_model": "all-MiniLM-L6-v2",
        "tau": 0.45,
        "K_in": 2,
        "adaptive_tau": False,
        "broadcast_round_1": True,
    },
    "orchestration": {
        "T_max": 3,
        "descriptor_mode": "combined",
        "state_strategy": "full",
        "convergence_threshold": 0.75,
        "fallback_on_isolation": True,
        "max_agent_context_tokens": 12288,
    },
    "logging": {
        "log_dir": "~/dytopo-logs",
        "save_similarity_matrices": True,
        "save_raw_responses": False,
        "console_verbosity": "info",
    },
    "concurrency": {
        "backend": "llama-cpp",
        "max_concurrent": 1,
        "llm_base_url": "http://localhost:8008/v1",
        "connect_timeout": 10.0,
        "read_timeout": 120.0,
    },
    "traces": {
        "enabled": False,
        "qdrant_url": "http://localhost:6333",
        "collection": "swarm_traces",
        "boost_weight": 0.15,
        "half_life_hours": 168.0,
        "top_k": 5,
        "min_quality": 0.5,
        "prune_max_age_hours": 720.0,
    },
    "health_monitor": {
        "check_interval_seconds": 30,
        "max_restart_attempts": 3,
        "crash_window_minutes": 15,
        "alert_cooldown_minutes": 30,
        "log_dir": "~/anyloom-logs",
    },
    "checkpoint": {
        "enabled": True,
        "checkpoint_dir": "~/dytopo-checkpoints",
        "save_per_agent": False,
    },
    "verification": {
        "enabled": False,
        "max_retries": 1,
        "specs": {
            "developer": {"type": "syntax_check", "timeout_seconds": 10},
            "researcher": {"type": "schema_validation", "required_fields": ["sources", "summary"]},
            "solver": {"type": "syntax_check", "timeout_seconds": 30},
        },
    },
}


def load_config(path: str | Path = "dytopo_config.yaml") -> dict:
    """Load configuration from YAML file, merging with defaults.

    Args:
        path: Path to YAML config file. If relative, resolved from CWD.

    Returns:
        Merged config dict with all sections populated.
    """
    config = {k: dict(v) for k, v in _DEFAULTS.items()}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
    return config
