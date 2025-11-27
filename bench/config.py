"""
Configuration loading and management.
"""

from pathlib import Path
from typing import Any
import yaml
import os


DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4-turbo-preview",
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-ada-002",
    },
    "adapters": {
        "krnx": {
            "image": "krnx:latest",
            "port": 8100,
            "redis_port": 6379,
            "health_endpoint": "/health",
            "timeout": 60,
        },
        "naive_rag": {
            "image": "qdrant/qdrant:latest",
            "port": 6333,
            "top_k": 10,
            "timeout": 30,
        },
        "baseline": {
            "type": "none",
        },
    },
    "defaults": {
        "trials": 50,
        "timeout": 300,
    },
    "docker": {
        "network": "krnx-bench-network",
        "remove_on_exit": True,
    },
}


def load_config(config_dir: Path = None) -> dict:
    """
    Load configuration from YAML files.
    
    Searches in order:
    1. Provided config_dir
    2. ./config/
    3. Falls back to defaults
    """
    
    if config_dir is None:
        config_dir = Path("config")
    
    config = DEFAULT_CONFIG.copy()
    
    # Load default.yaml
    default_yaml = config_dir / "default.yaml"
    if default_yaml.exists():
        with open(default_yaml) as f:
            user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
    
    # Load scenarios.yaml
    scenarios_yaml = config_dir / "scenarios.yaml"
    if scenarios_yaml.exists():
        with open(scenarios_yaml) as f:
            scenarios_config = yaml.safe_load(f) or {}
            config["scenarios"] = scenarios_config
    
    # Override with environment variables
    config = _apply_env_overrides(config)
    
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base"""
    
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides"""
    
    # LLM provider
    if os.getenv("KRNX_BENCH_LLM_PROVIDER"):
        config["llm"]["provider"] = os.getenv("KRNX_BENCH_LLM_PROVIDER")
    
    if os.getenv("KRNX_BENCH_LLM_MODEL"):
        config["llm"]["model"] = os.getenv("KRNX_BENCH_LLM_MODEL")
    
    # Trials
    if os.getenv("KRNX_BENCH_TRIALS"):
        config["defaults"]["trials"] = int(os.getenv("KRNX_BENCH_TRIALS"))
    
    return config


def get_scenario_config(config: dict, scenario_name: str) -> dict:
    """Get configuration for a specific scenario"""
    
    scenarios = config.get("scenarios", {})
    
    # Find the scenario in the nested structure
    for guarantee, scenario_configs in scenarios.items():
        if scenario_name in scenario_configs:
            return scenario_configs[scenario_name]
    
    # Return empty config if not found
    return {}


def get_adapter_config(config: dict, adapter_name: str) -> dict:
    """Get configuration for a specific adapter"""
    
    adapters = config.get("adapters", {})
    return adapters.get(adapter_name, {})
