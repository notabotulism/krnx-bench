"""
Provenance Chain Scenario

Tests: Auditability - Trace any output to its causes

This scenario tests whether the memory system maintains cryptographic
provenance chains that can be verified.
"""

import time
import hashlib
import logging
from typing import Optional

from ..base import BaseScenario
from ...models import Event, TrialResult
from ...adapters.base import MemoryAdapter, NotSupported
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class ProvenanceChainScenario(BaseScenario):
    """
    Provenance Chain Scenario
    
    Tests auditability by:
    1. Creating a multi-step workflow (5-10 events)
    2. Each event references the previous (hash chain)
    3. Walking the chain backward from the final event
    4. Verifying each link
    
    Metrics:
    - Chain completeness (all steps traceable)
    - Hash verification (all hashes valid)
    - Gap detection (no missing links)
    """
    
    name = "provenance_chain"
    description = "Test hash-chain provenance verification"
    guarantee = "auditability"
    supported_adapters = ["krnx"]  # Only KRNX supports this
    
    def __init__(self):
        super().__init__()
        self.workflow_steps = 5
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.workflow_steps = config.get("workflow_steps", 5)
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run a single provenance chain trial"""
        
        start = time.time()
        
        # Check if adapter supports provenance
        if not adapter.supports("provenance"):
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "not_supported"},
                error=f"{adapter.name} does not support provenance tracking",
                timing_ms=0,
            )
        
        # Create workflow events with hash chain
        event_hashes = []
        parent_hash = None
        
        for step in range(self.workflow_steps):
            content = f"Workflow step {step + 1}: Processing data batch {step}"
            
            event = Event(
                content=content,
                event_type="workflow_step",
                metadata={
                    "step": step + 1,
                    "total_steps": self.workflow_steps,
                },
                parent_hash=parent_hash,
            )
            
            event_hash = adapter.write_event(event)
            event_hashes.append(event_hash)
            parent_hash = event_hash
        
        write_time = (time.time() - start) * 1000
        
        # Get provenance chain from final event
        final_hash = event_hashes[-1]
        
        try:
            provenance = adapter.get_provenance(final_hash)
            
            # Verify chain
            chain_complete = len(provenance.chain) == self.workflow_steps
            all_verified = provenance.verified
            gaps = provenance.gaps
            
            # Check that expected hashes are in chain
            hashes_in_chain = set(e.get("hash") or e.get("event_id") for e in provenance.chain)
            missing_hashes = [h for h in event_hashes if h not in hashes_in_chain]
            
            success = chain_complete and all_verified and not gaps and not missing_hashes
            
            return TrialResult(
                trial_id=trial_id,
                success=success,
                metrics={
                    "workflow_steps": self.workflow_steps,
                    "chain_length": len(provenance.chain),
                    "chain_complete": chain_complete,
                    "all_verified": all_verified,
                    "gaps": len(gaps),
                    "missing_hashes": len(missing_hashes),
                    "write_time_ms": write_time,
                },
                timing_ms=(time.time() - start) * 1000,
            )
            
        except NotSupported as e:
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "not_supported"},
                error=str(e),
                timing_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={"error_type": "exception"},
                error=str(e),
                timing_ms=(time.time() - start) * 1000,
            )
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        """Compute aggregate statistics"""
        
        if not results:
            return {}
        
        valid = [r for r in results if "chain_complete" in r.metrics]
        
        if not valid:
            return {"error": "no_valid_trials", "total_trials": len(results)}
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid),
            "complete_chains": sum(1 for r in valid if r.metrics["chain_complete"]),
            "verified_chains": sum(1 for r in valid if r.metrics["all_verified"]),
            "chains_with_gaps": sum(1 for r in valid if r.metrics["gaps"] > 0),
            "success_rate": sum(1 for r in valid if r.success) / len(valid),
            "mean_timing_ms": sum(r.timing_ms for r in valid) / len(valid),
        }
