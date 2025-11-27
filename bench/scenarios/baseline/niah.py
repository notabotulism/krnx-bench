"""
Needle in a Haystack (NIAH) Scenario

Tests: Baseline - Basic retrieval sanity check

This is a sanity check to verify that KRNX performs at least as well
as naive RAG on standard retrieval tasks.
"""

import time
import random
import logging

from ..base import BaseScenario
from ...models import Event, TrialResult
from ...adapters.base import MemoryAdapter
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)

# Filler content for haystack
HAYSTACK_TOPICS = [
    "The history of computing dates back to ancient times when humans first developed methods for counting.",
    "Weather patterns are influenced by a complex interplay of atmospheric conditions and ocean currents.",
    "Modern architecture has evolved significantly since the industrial revolution began.",
    "The development of agriculture transformed human societies from nomadic to settled communities.",
    "Ocean ecosystems support an incredible diversity of marine life forms.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Renewable energy sources are becoming increasingly important for sustainable development.",
    "The printing press revolutionized the spread of information throughout human history.",
    "Mountain ranges form through tectonic plate movements over millions of years.",
    "The development of antibiotics transformed modern medicine in the twentieth century.",
]


class NIAHScenario(BaseScenario):
    """
    Needle in a Haystack Scenario
    
    Tests basic retrieval by:
    1. Creating a corpus of filler content
    2. Hiding a unique fact (the needle) at a specific position
    3. Querying for the needle
    
    This is a sanity check - all memory systems should pass this.
    """
    
    name = "niah"
    description = "Needle-in-haystack basic retrieval test"
    guarantee = "baseline"
    supported_adapters = ["krnx", "naive_rag", "baseline"]
    
    def __init__(self):
        super().__init__()
        self.corpus_sizes = [100, 1000]
        self.needle_positions = ["front", "middle", "end"]
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.corpus_sizes = config.get("corpus_sizes", [100, 1000])
        self.needle_positions = config.get("needle_positions", ["front", "middle", "end"])
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run a single NIAH trial"""
        
        start = time.time()
        
        # Random configuration for this trial
        corpus_size = random.choice(self.corpus_sizes)
        position = random.choice(self.needle_positions)
        
        # Generate unique needle
        needle_value = f"SECRET_{trial_id}_{random.randint(1000, 9999)}"
        needle_content = f"The secret code is {needle_value}. Remember this important information."
        
        # Calculate needle position
        if position == "front":
            needle_idx = corpus_size // 10
        elif position == "middle":
            needle_idx = corpus_size // 2
        else:  # end
            needle_idx = corpus_size - (corpus_size // 10)
        
        # Write haystack with needle
        for i in range(corpus_size):
            if i == needle_idx:
                content = needle_content
                event_type = "needle"
            else:
                content = random.choice(HAYSTACK_TOPICS)
                event_type = "haystack"
            
            event = Event(
                content=content,
                event_type=event_type,
                metadata={"index": i, "position": "needle" if i == needle_idx else "haystack"},
            )
            adapter.write_event(event)
        
        write_time = (time.time() - start) * 1000
        
        # Query for the needle
        query = "What is the secret code?"
        
        query_start = time.time()
        result = adapter.query(query, llm)
        query_time = (time.time() - query_start) * 1000
        
        # Check if needle was found
        found = needle_value in result.response
        
        return TrialResult(
            trial_id=trial_id,
            success=found,
            metrics={
                "corpus_size": corpus_size,
                "needle_position": position,
                "needle_index": needle_idx,
                "needle_value": needle_value,
                "needle_found": found,
                "context_events": len(result.context_events),
                "context_tokens": result.context_tokens,
                "write_time_ms": write_time,
                "query_time_ms": query_time,
            },
            raw_output=result.response,
            timing_ms=(time.time() - start) * 1000,
        )
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        if not results:
            return {}
        
        valid = [r for r in results if "needle_found" in r.metrics]
        if not valid:
            return {"error": "no_valid_trials"}
        
        # Overall accuracy
        found = sum(1 for r in valid if r.metrics["needle_found"])
        
        # By position
        by_position = {}
        for pos in self.needle_positions:
            pos_results = [r for r in valid if r.metrics.get("needle_position") == pos]
            if pos_results:
                pos_found = sum(1 for r in pos_results if r.metrics["needle_found"])
                by_position[pos] = pos_found / len(pos_results)
        
        # By corpus size
        by_size = {}
        for size in self.corpus_sizes:
            size_results = [r for r in valid if r.metrics.get("corpus_size") == size]
            if size_results:
                size_found = sum(1 for r in size_results if r.metrics["needle_found"])
                by_size[size] = size_found / len(size_results)
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid),
            "accuracy": found / len(valid),
            "by_position": by_position,
            "by_corpus_size": by_size,
            "mean_timing_ms": sum(r.timing_ms for r in valid) / len(valid),
        }
