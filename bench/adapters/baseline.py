"""
Baseline adapter with no memory.

This represents the absolute baseline: raw LLM with no external memory.
Used to establish lower bounds for accuracy comparisons.
"""

import time
import logging
from typing import Optional

from .base import BaseAdapter, NotSupported
from ..models import Event, QueryResult

logger = logging.getLogger(__name__)


class BaselineAdapter(BaseAdapter):
    """
    No-memory baseline adapter.
    
    This adapter represents using an LLM with zero external memory:
    - Events are "written" but not stored
    - Queries go directly to the LLM with no context
    - Used to establish accuracy baselines
    
    All extended operations raise NotSupported.
    """
    
    name = "baseline"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._event_count = 0
    
    def setup(self) -> None:
        """No-op setup"""
        logger.info("Setting up Baseline adapter (no memory)...")
        self._setup_complete = True
        self._event_count = 0
    
    def teardown(self) -> None:
        """No-op teardown"""
        self._setup_complete = False
        self._event_count = 0
    
    def clear(self) -> None:
        """Reset event counter"""
        self._event_count = 0
    
    def write_event(self, event: Event) -> str:
        """
        Pretend to write an event.
        
        Returns a fake hash but doesn't actually store anything.
        This simulates the overhead of a write operation.
        """
        self._ensure_setup()
        
        self._event_count += 1
        
        # Return a fake hash
        return f"baseline-{self._event_count}"
    
    def query(self, query: str, llm: "LLMClient") -> QueryResult:
        """
        Query the LLM with no context.
        
        This establishes the baseline accuracy when the LLM has
        no access to stored information.
        """
        self._ensure_setup()
        
        start = time.time()
        
        # Simple prompt with no context
        prompt = f"Question: {query}\n\nAnswer:"
        
        response = llm.complete(prompt)
        
        return QueryResult(
            response=response.text,
            context_events=[],  # No context provided
            context_tokens=response.prompt_tokens,
            query_time_ms=0,  # No retrieval
            llm_time_ms=(time.time() - start) * 1000,
            total_time_ms=(time.time() - start) * 1000,
        )
    
    # =========================================================================
    # All extended operations are unsupported
    # =========================================================================
    
    def get_event(self, event_hash: str):
        raise NotSupported("Baseline adapter has no memory")
    
    def replay_to(self, timestamp: float):
        raise NotSupported("Baseline adapter has no memory")
    
    def get_provenance(self, event_hash: str):
        raise NotSupported("Baseline adapter has no memory")
    
    def kill(self) -> None:
        raise NotSupported("Baseline adapter has no infrastructure to kill")
    
    def restart(self) -> None:
        raise NotSupported("Baseline adapter has no infrastructure to restart")
    
    def is_alive(self) -> bool:
        """Baseline is always "alive" """
        return self._setup_complete
    
    def supports(self, capability: str) -> bool:
        """Baseline supports nothing"""
        return False
