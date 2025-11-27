"""
Base scenario protocol and utilities.
"""

from typing import Protocol, Optional, Callable, runtime_checkable
from abc import abstractmethod
from datetime import datetime

from ..models import ScenarioResult, TrialResult
from ..adapters.base import MemoryAdapter, NotSupported
from ..llm.client import LLMClient


@runtime_checkable
class Scenario(Protocol):
    """
    Protocol for benchmark scenarios.
    
    Each scenario tests a specific capability or guarantee.
    Scenarios must be reproducible given the same configuration.
    """
    
    # Class attributes
    name: str
    description: str
    guarantee: str  # "durability" | "consistency" | "auditability" | "replay" | "baseline"
    supported_adapters: list[str]  # ["krnx", "naive_rag", "baseline"] or subset
    
    @abstractmethod
    def configure(self, config: dict) -> None:
        """
        Apply scenario-specific configuration.
        
        Args:
            config: Configuration from scenarios.yaml
        """
        ...
    
    @abstractmethod
    def run(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trials: int,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> ScenarioResult:
        """
        Execute the scenario.
        
        Args:
            adapter: Memory adapter to test
            llm: LLM client for generation
            trials: Number of trials to run
            progress_callback: Called after each trial completes
            
        Returns:
            ScenarioResult with all trial results and aggregates
        """
        ...
    
    def supports_adapter(self, adapter_name: str) -> bool:
        """Check if this scenario can run on the given adapter"""
        return adapter_name in self.supported_adapters


class BaseScenario:
    """
    Base class for scenarios with common functionality.
    
    Subclasses should:
    1. Set class attributes (name, description, guarantee, supported_adapters)
    2. Override _run_trial() to implement the actual test logic
    3. Override _compute_aggregate() to compute summary statistics
    """
    
    name: str = "base"
    description: str = "Base scenario"
    guarantee: str = "unknown"
    supported_adapters: list[str] = ["krnx", "naive_rag", "baseline"]
    
    def __init__(self):
        self.config: dict = {}
    
    def configure(self, config: dict) -> None:
        """Apply configuration"""
        self.config = config
    
    def run(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trials: int,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> ScenarioResult:
        """Run all trials and aggregate results"""
        
        started_at = datetime.now().isoformat()
        results: list[TrialResult] = []
        
        for trial_id in range(trials):
            # Clear state between trials
            adapter.clear()
            
            # Run single trial
            try:
                result = self._run_trial(adapter, llm, trial_id)
            except NotSupported as e:
                # Adapter doesn't support required operation
                result = TrialResult(
                    trial_id=trial_id,
                    success=False,
                    metrics={"error_type": "not_supported"},
                    error=str(e),
                )
            except Exception as e:
                result = TrialResult(
                    trial_id=trial_id,
                    success=False,
                    metrics={"error_type": "exception"},
                    error=str(e),
                )
            
            results.append(result)
            
            if progress_callback:
                progress_callback()
        
        completed_at = datetime.now().isoformat()
        
        # Compute aggregates
        aggregate = self._compute_aggregate(results)
        
        return ScenarioResult(
            scenario_name=self.name,
            adapter_name=adapter.name,
            trials=results,
            aggregate=aggregate,
            config=self.config,
            started_at=started_at,
            completed_at=completed_at,
        )
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """
        Run a single trial.
        
        Subclasses must override this method.
        
        Args:
            adapter: Memory adapter to test
            llm: LLM client for generation
            trial_id: Index of this trial
            
        Returns:
            TrialResult for this trial
        """
        raise NotImplementedError("Subclasses must implement _run_trial")
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        """
        Compute aggregate statistics from trial results.
        
        Subclasses may override for custom aggregation.
        
        Default implementation computes:
        - success_rate
        - mean_timing_ms
        - error_count
        """
        
        if not results:
            return {}
        
        successes = sum(1 for r in results if r.success)
        total_timing = sum(r.timing_ms for r in results)
        errors = sum(1 for r in results if r.error)
        
        return {
            "success_rate": successes / len(results),
            "mean_timing_ms": total_timing / len(results),
            "error_count": errors,
            "total_trials": len(results),
        }
    
    def supports_adapter(self, adapter_name: str) -> bool:
        """Check if this scenario can run on the given adapter"""
        return adapter_name in self.supported_adapters
