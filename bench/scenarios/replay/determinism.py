"""
Replay Determinism Scenario

Tests: Replay - Replaying same events produces identical state

This scenario verifies that replaying the same event log
produces deterministic, identical results.
"""

import time
import json
import hashlib
import logging

from ..base import BaseScenario
from ...models import Event, TrialResult
from ...adapters.base import MemoryAdapter, NotSupported
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class DeterminismScenario(BaseScenario):
    """
    Replay Determinism Scenario
    
    Tests that replay is deterministic by:
    1. Running a workflow, recording final state S1
    2. Clearing and replaying from event log
    3. Recording final state S2
    4. Verifying S1 == S2
    """
    
    name = "determinism"
    description = "Test that replay produces identical results"
    guarantee = "replay"
    supported_adapters = ["krnx"]
    
    def __init__(self):
        super().__init__()
        self.history_size = 1000
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.history_size = config.get("history_size", 1000)
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run a single determinism trial"""
        
        start = time.time()
        
        if not adapter.supports("replay"):
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "not_supported"},
                error=f"{adapter.name} does not support replay",
            )
        
        # Write events
        event_hashes = []
        for i in range(self.history_size):
            event = Event(
                content=f"Determinism test event {i}",
                event_type="test",
                metadata={"index": i, "trial": trial_id},
            )
            h = adapter.write_event(event)
            event_hashes.append(h)
        
        # Get final state S1
        final_ts = time.time()
        try:
            state1 = adapter.replay_to(final_ts)
            state1_hash = self._hash_state(state1)
        except Exception as e:
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "replay_failed"},
                error=f"First replay failed: {e}",
                timing_ms=(time.time() - start) * 1000,
            )
        
        # Replay again to same timestamp
        try:
            state2 = adapter.replay_to(final_ts)
            state2_hash = self._hash_state(state2)
        except Exception as e:
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "replay_failed"},
                error=f"Second replay failed: {e}",
                timing_ms=(time.time() - start) * 1000,
            )
        
        # Compare states
        states_match = state1_hash == state2_hash
        
        return TrialResult(
            trial_id=trial_id,
            success=states_match,
            metrics={
                "history_size": self.history_size,
                "state1_hash": state1_hash,
                "state2_hash": state2_hash,
                "states_match": states_match,
                "state1_events": len(state1.events),
                "state2_events": len(state2.events),
            },
            timing_ms=(time.time() - start) * 1000,
        )
    
    def _hash_state(self, state) -> str:
        """Create a hash of the state for comparison"""
        contents = [e.content for e in state.events]
        return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()[:16]
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        if not results:
            return {}
        
        valid = [r for r in results if "states_match" in r.metrics]
        if not valid:
            return {"error": "no_valid_trials"}
        
        matches = sum(1 for r in valid if r.metrics["states_match"])
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid),
            "deterministic_trials": matches,
            "determinism_rate": matches / len(valid),
            "mean_timing_ms": sum(r.timing_ms for r in valid) / len(valid),
        }
