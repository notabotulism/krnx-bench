"""
Point-in-Time Replay Scenario

Tests: Replay - Reconstruct state at any timestamp

This scenario tests whether the memory system can accurately
reconstruct state at arbitrary points in time.
"""

import time
import json
import logging
from typing import Optional

from ..base import BaseScenario
from ...models import Event, TrialResult, ScenarioResult
from ...adapters.base import MemoryAdapter, NotSupported
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class PointInTimeScenario(BaseScenario):
    """
    Point-in-Time Replay Scenario
    
    Tests replay capability by:
    1. Writing N events over simulated time
    2. Recording state snapshots at checkpoints
    3. Replaying to each checkpoint
    4. Comparing reconstructed state to recorded snapshot
    
    Metrics:
    - Reconstruction accuracy
    - Replay latency vs history size
    """
    
    name = "point_in_time"
    description = "Test state reconstruction at arbitrary timestamps"
    guarantee = "replay"
    supported_adapters = ["krnx"]
    
    def __init__(self):
        super().__init__()
        self.history_sizes = [100, 1000, 10000]
        self.checkpoints_per_size = 5
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.history_sizes = config.get("history_sizes", [100, 1000, 10000])
        self.checkpoints_per_size = config.get("checkpoints_per_size", 5)
    
    def run(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trials: int,
        progress_callback: Optional[callable] = None,
    ) -> ScenarioResult:
        """Run replay trials for each history size"""
        
        from datetime import datetime
        
        started_at = datetime.now().isoformat()
        results = []
        trial_id = 0
        
        if not adapter.supports("replay"):
            return ScenarioResult(
                scenario_name=self.name,
                adapter_name=adapter.name,
                trials=[TrialResult(
                    trial_id=0,
                    success=False,
                    metrics={"error_type": "not_supported"},
                    error=f"{adapter.name} does not support replay",
                )],
                aggregate={"error": "not_supported"},
                config=self.config,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )
        
        for history_size in self.history_sizes:
            trials_per_size = max(1, trials // len(self.history_sizes))
            
            for _ in range(trials_per_size):
                adapter.clear()
                result = self._run_trial_with_size(adapter, llm, trial_id, history_size)
                results.append(result)
                trial_id += 1
                
                if progress_callback:
                    progress_callback()
        
        return ScenarioResult(
            scenario_name=self.name,
            adapter_name=adapter.name,
            trials=results,
            aggregate=self._compute_aggregate(results),
            config=self.config,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
        )
    
    def _run_trial_with_size(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
        history_size: int,
    ) -> TrialResult:
        """Run trial with specific history size"""
        
        start = time.time()
        
        # Write events and record checkpoints
        checkpoint_interval = history_size // self.checkpoints_per_size
        checkpoints = {}  # timestamp -> expected state
        
        for i in range(history_size):
            ts = time.time()
            
            event = Event(
                content=f"Event {i}: data_{i}",
                event_type="test",
                timestamp=ts,
                metadata={"index": i},
            )
            adapter.write_event(event)
            
            # Record checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoints[ts] = {
                    "event_count": i + 1,
                    "last_index": i,
                }
        
        write_time = (time.time() - start) * 1000
        
        # Test replay at each checkpoint
        replay_results = []
        
        for ts, expected in checkpoints.items():
            replay_start = time.time()
            
            try:
                state = adapter.replay_to(ts + 0.001)  # Slightly after
                
                # Verify state
                actual_count = len(state.events)
                expected_count = expected["event_count"]
                
                accuracy = min(actual_count, expected_count) / max(actual_count, expected_count)
                
                replay_results.append({
                    "timestamp": ts,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "accuracy": accuracy,
                    "latency_ms": (time.time() - replay_start) * 1000,
                })
                
            except Exception as e:
                replay_results.append({
                    "timestamp": ts,
                    "error": str(e),
                    "accuracy": 0,
                    "latency_ms": (time.time() - replay_start) * 1000,
                })
        
        # Calculate metrics
        accuracies = [r["accuracy"] for r in replay_results if "accuracy" in r]
        latencies = [r["latency_ms"] for r in replay_results]
        
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return TrialResult(
            trial_id=trial_id,
            success=mean_accuracy >= 0.99,
            metrics={
                "history_size": history_size,
                "checkpoints": len(checkpoints),
                "mean_accuracy": mean_accuracy,
                "mean_replay_latency_ms": mean_latency,
                "write_time_ms": write_time,
                "replay_results": replay_results,
            },
            timing_ms=(time.time() - start) * 1000,
        )
    
    def _run_trial(self, adapter, llm, trial_id):
        return self._run_trial_with_size(adapter, llm, trial_id, self.history_sizes[0])
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        if not results:
            return {}
        
        valid = [r for r in results if "history_size" in r.metrics]
        if not valid:
            return {"error": "no_valid_trials"}
        
        # Group by history size
        by_size = {}
        for r in valid:
            size = r.metrics["history_size"]
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(r)
        
        size_stats = []
        for size, size_results in sorted(by_size.items()):
            accuracies = [r.metrics["mean_accuracy"] for r in size_results]
            latencies = [r.metrics["mean_replay_latency_ms"] for r in size_results]
            
            size_stats.append({
                "size": size,
                "accuracy": sum(accuracies) / len(accuracies),
                "latency_ms": sum(latencies) / len(latencies),
            })
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid),
            "by_size": size_stats,
            "overall_accuracy": sum(r.metrics["mean_accuracy"] for r in valid) / len(valid),
        }
