"""
Temporal Versioning Scenario

Tests: Consistency - Query facts at specific timestamps

This scenario tests whether the memory system can return the correct
version of a fact at a specific point in time (not just the latest).

This is a KRNX-only capability.
"""

import time
import logging
from typing import Optional, Callable

from ..base import BaseScenario
from ...models import Event, TrialResult
from ...adapters.base import MemoryAdapter, NotSupported
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class TemporalVersioningScenario(BaseScenario):
    """
    Temporal Versioning Scenario
    
    Tests temporal query capability by:
    1. Writing fact v1 at timestamp T1
    2. Writing fact v2 at timestamp T2
    3. Writing fact v3 at timestamp T3
    4. Querying: "What was the value at T2?"
    5. Should return v2 (not v3)
    
    This tests a capability that naive RAG cannot support.
    
    Metrics:
    - Temporal accuracy (correct version for time-scoped query)
    """
    
    name = "temporal_versioning"
    description = "Test retrieval of fact versions at specific timestamps"
    guarantee = "consistency"
    supported_adapters = ["krnx"]  # Only KRNX supports this
    
    def __init__(self):
        super().__init__()
        self.versions = 5
        self.query_points = 3  # Number of timestamps to query
        self.delay_between_versions = 1.0  # Seconds
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.versions = config.get("versions", 5)
        self.query_points = config.get("query_points", 3)
        self.delay_between_versions = config.get("delay_between_versions", 1.0)
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run a single temporal versioning trial"""
        
        start = time.time()
        
        # Check if adapter supports replay
        if not adapter.supports("replay"):
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "not_supported"},
                error=f"{adapter.name} does not support temporal replay",
                timing_ms=0,
            )
        
        # Generate fact values
        fact_values = [f"temporal_value_v{i}" for i in range(1, self.versions + 1)]
        
        # Write facts with delays to establish clear timestamps
        version_timestamps = []
        
        for i, value in enumerate(fact_values):
            # Record timestamp before writing
            ts = time.time()
            version_timestamps.append(ts)
            
            fact_event = Event(
                content=f"The current value is {value}",
                event_type="fact_update",
                timestamp=ts,
                metadata={
                    "version": i + 1,
                    "value": value,
                },
            )
            adapter.write_event(fact_event)
            
            # Delay between versions
            if i < len(fact_values) - 1:
                time.sleep(self.delay_between_versions)
        
        write_time = (time.time() - start) * 1000
        
        # Query at different timestamps
        query_results = []
        
        # Select query points (not first or last)
        query_indices = self._select_query_points()
        
        for idx in query_indices:
            target_timestamp = version_timestamps[idx]
            expected_version = idx + 1
            expected_value = fact_values[idx]
            
            # Replay to that timestamp
            try:
                state = adapter.replay_to(target_timestamp + 0.1)  # Slightly after
                
                # Check if the expected value is in the state
                found_value = None
                for event in state.events:
                    if event.event_type == "fact_update":
                        found_value = event.metadata.get("value")
                
                if found_value == expected_value:
                    result = "correct"
                elif found_value is not None:
                    result = "wrong_version"
                else:
                    result = "not_found"
                
                query_results.append({
                    "target_version": expected_version,
                    "target_timestamp": target_timestamp,
                    "expected_value": expected_value,
                    "found_value": found_value,
                    "result": result,
                })
                
            except NotSupported as e:
                query_results.append({
                    "target_version": expected_version,
                    "result": "not_supported",
                    "error": str(e),
                })
            except Exception as e:
                query_results.append({
                    "target_version": expected_version,
                    "result": "error",
                    "error": str(e),
                })
        
        total_time = (time.time() - start) * 1000
        
        # Determine overall success
        correct_count = sum(1 for r in query_results if r["result"] == "correct")
        success = correct_count == len(query_results)
        
        return TrialResult(
            trial_id=trial_id,
            success=success,
            metrics={
                "versions": self.versions,
                "query_points": len(query_results),
                "correct_queries": correct_count,
                "temporal_accuracy": correct_count / len(query_results) if query_results else 0,
                "query_results": query_results,
                "write_time_ms": write_time,
            },
            timing_ms=total_time,
        )
    
    def _select_query_points(self) -> list[int]:
        """Select which version indices to query"""
        
        # Query middle versions (not first or last)
        available = list(range(1, self.versions - 1))
        
        if len(available) <= self.query_points:
            return available
        
        # Evenly spaced selection
        step = len(available) / self.query_points
        return [available[int(i * step)] for i in range(self.query_points)]
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        """Compute aggregate statistics"""
        
        if not results:
            return {}
        
        # Filter valid trials
        valid = [r for r in results if "temporal_accuracy" in r.metrics]
        
        if not valid:
            return {
                "error": "no_valid_trials",
                "total_trials": len(results),
            }
        
        accuracies = [r.metrics["temporal_accuracy"] for r in valid]
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid),
            "mean_temporal_accuracy": sum(accuracies) / len(accuracies),
            "perfect_trials": sum(1 for a in accuracies if a == 1.0),
            "mean_timing_ms": sum(r.timing_ms for r in valid) / len(valid),
        }
