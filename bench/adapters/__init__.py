"""
Memory system adapters for benchmarking.
"""

from .base import MemoryAdapter, NotSupported, AdapterError
from .krnx import KRNXDockerAdapter
from .naive_rag import NaiveRAGAdapter
from .baseline import BaselineAdapter

ADAPTERS = {
    "krnx": KRNXDockerAdapter,
    "naive_rag": NaiveRAGAdapter,
    "baseline": BaselineAdapter,
}


def get_adapter(name: str, config: dict) -> MemoryAdapter:
    """Get an adapter instance by name"""
    
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    
    adapter_config = config.get("adapters", {}).get(name, {})
    return ADAPTERS[name](adapter_config)


__all__ = [
    "MemoryAdapter",
    "NotSupported", 
    "AdapterError",
    "KRNXDockerAdapter",
    "NaiveRAGAdapter",
    "BaselineAdapter",
    "ADAPTERS",
    "get_adapter",
]
