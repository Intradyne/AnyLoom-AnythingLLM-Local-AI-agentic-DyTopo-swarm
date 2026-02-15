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
        "base_url": "http://localhost:1234/v1",
        "model": "qwen3-30b-a3b-instruct-2507",
        "temperature_work": 0.3,
        "temperature_descriptor": 0.1,
        "temperature_manager": 0.1,
        "max_tokens_work": 4096,
        "max_tokens_descriptor": 256,
        "max_tokens_manager": 2000,
        "timeout_seconds": 300,
    },
    "routing": {
        "embedding_model": "all-MiniLM-L6-v2",
        "tau": 0.3,
        "K_in": 3,
        "adaptive_tau": False,
        "broadcast_round_1": True,
    },
    "orchestration": {
        "T_max": 5,
        "descriptor_mode": "combined",
        "state_strategy": "full",
        "convergence_threshold": 0.9,
        "fallback_on_isolation": True,
    },
    "logging": {
        "log_dir": "~/dytopo-logs",
        "save_similarity_matrices": True,
        "save_raw_responses": False,
        "console_verbosity": "info",
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
