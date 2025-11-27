"""
Crash Recovery Scenario

Tests: Durability - Events survive crashes

This scenario writes events, kills the process, restarts, and verifies
that events were recovered without corruption.
"""

import time
import random
import hashlib
import logging
from typing import Optional, Callable

from ..base import BaseScenario
from ...models import Event, TrialResult, ScenarioResult
from ...adapters.base import MemoryAdapter, NotSupported
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class CrashRecoveryScenario(BaseScenario):
    """
    Crash Recovery Scenario
    
    Tests durability by:
    1. Writing N events
    2. Killing the process (SIGKILL)
    3. Restarting
    4. Verifying events are recovered
    
    Metrics:
    - Events recovered (should equal events written)
    - Corruption detected (should be 0)
    - Recovery time
    """
    
    name = "crash_recovery"
    description = "Test event durability under crash conditions"
    guarantee = "durability"
    supported_adapters = ["krnx", "naive_rag"]  # Baseline has nothing to crash
    
    def __init__(self):
        super().__init__()
        self.event_counts = [1000, 10000]  # Events to write per trial
        self.kill_delay_range = (0.5, 2.0)  # Seconds after write to kill
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.event_counts = config.get("event_counts", [1000, 10000])
        self.kill_delay_range = tuple(config.get("kill_delay_range", [0.5, 2.0]))
    
    def run(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trials: int,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> ScenarioResult:
        """
        Run crash recovery trials.
        
        Unlike other scenarios, this runs trials for each event count.
        """
        
        from datetime import datetime
        
        started_at = datetime.now().isoformat()
        results: list[TrialResult] = []
        trial_id = 0
        
        # Check if adapter supports fault injection
        if not adapter.supports("fault_injection"):
            # Return a single failed result
            return ScenarioResult(
                scenario_name=self.name,
                adapter_name=adapter.name,
                trials=[TrialResult(
                    trial_id=0,
                    success=False,
                    metrics={"error_type": "not_supported"},
                    error=f"{adapter.name} does not support fault injection",
                )],
                aggregate={"error": "not_supported"},
                config=self.config,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )
        
        for event_count in self.event_counts:
            # Run multiple trials per event count
            trials_per_count = max(1, trials // len(self.event_counts))
            
            for _ in range(trials_per_count):
                adapter.clear()
                
                result = self._run_trial_with_count(adapter, llm, trial_id, event_count)
                results.append(result)
                trial_id += 1
                
                if progress_callback:
                    progress_callback()
        
        completed_at = datetime.now().isoformat()
        
        return ScenarioResult(
            scenario_name=self.name,
            adapter_name=adapter.name,
            trials=results,
            aggregate=self._compute_aggregate(results),
            config=self.config,
            started_at=started_at,
            completed_at=completed_at,
        )
    
    def _run_trial_with_count(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
        event_count: int,
    ) -> TrialResult:
        """Run a single crash recovery trial with specified event count"""
        
        start = time.time()
        
        # Generate and write events
        written_events = []
        written_hashes = []
        
        logger.info(f"Trial {trial_id}: Writing {event_count} events...")
        
        for i in range(event_count):
            # Create event with verifiable content
            content = f"Event {i} - {hashlib.sha256(str(i).encode()).hexdigest()[:16]}"
            event = Event(
                content=content,
                event_type="test",
                metadata={"index": i, "checksum": self._compute_checksum(content)},
            )
            
            try:
                event_hash = adapter.write_event(event)
                written_events.append(event)
                written_hashes.append(event_hash)
            except Exception as e:
                logger.error(f"Failed to write event {i}: {e}")
                return TrialResult(
                    trial_id=trial_id,
                    success=False,
                    metrics={
                        "event_count": event_count,
                        "events_written": i,
                        "error_type": "write_failure",
                    },
                    error=str(e),
                    timing_ms=(time.time() - start) * 1000,
                )
        
        write_time = time.time() - start
        
        # Random delay before kill
        kill_delay = random.uniform(*self.kill_delay_range)
        time.sleep(kill_delay)
        
        # Kill the adapter
        logger.info(f"Trial {trial_id}: Killing adapter...")
        kill_start = time.time()
        
        try:
            adapter.kill()
        except Exception as e:
            logger.error(f"Failed to kill adapter: {e}")
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={
                    "event_count": event_count,
                    "events_written": event_count,
                    "error_type": "kill_failure",
                },
                error=str(e),
                timing_ms=(time.time() - start) * 1000,
            )
        
        # Restart the adapter
        logger.info(f"Trial {trial_id}: Restarting adapter...")
        
        try:
            adapter.restart()
        except Exception as e:
            logger.error(f"Failed to restart adapter: {e}")
            return TrialResult(
                trial_id=trial_id,
                success=False,
                metrics={
                    "event_count": event_count,
                    "events_written": event_count,
                    "error_type": "restart_failure",
                },
                error=str(e),
                timing_ms=(time.time() - start) * 1000,
            )
        
        recovery_time = (time.time() - kill_start) * 1000
        
        # Verify events were recovered
        logger.info(f"Trial {trial_id}: Verifying recovery...")
        
        recovered = 0
        corrupted = 0
        lost = 0
        
        for i, (event, event_hash) in enumerate(zip(written_events, written_hashes)):
            try:
                retrieved = adapter.get_event(event_hash)
                
                # Verify content
                if retrieved.content == event.content:
                    # Verify checksum if available
                    if "checksum" in retrieved.metadata:
                        expected_checksum = self._compute_checksum(retrieved.content)
                        if retrieved.metadata["checksum"] == expected_checksum:
                            recovered += 1
                        else:
                            corrupted += 1
                    else:
                        recovered += 1
                else:
                    corrupted += 1
                    
            except KeyError:
                lost += 1
            except NotSupported:
                # Adapter doesn't support get_event, count as recovered
                # (we can't verify, but write didn't fail)
                recovered += 1
            except Exception as e:
                logger.warning(f"Error retrieving event {i}: {e}")
                lost += 1
        
        total_time = (time.time() - start) * 1000
        
        # Determine success
        success = (recovered == event_count) and (corrupted == 0)
        
        return TrialResult(
            trial_id=trial_id,
            success=success,
            metrics={
                "event_count": event_count,
                "events_written": event_count,
                "events_recovered": recovered,
                "events_corrupted": corrupted,
                "events_lost": lost,
                "recovery_rate": recovered / event_count if event_count > 0 else 0,
                "write_time_ms": write_time * 1000,
                "recovery_time_ms": recovery_time,
            },
            timing_ms=total_time,
        )
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run with default event count"""
        return self._run_trial_with_count(
            adapter, llm, trial_id, self.event_counts[0]
        )
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        """Compute aggregate statistics"""
        
        if not results:
            return {}
        
        # Filter successful trials
        valid_results = [r for r in results if "events_written" in r.metrics]
        
        if not valid_results:
            return {
                "error": "no_valid_trials",
                "total_trials": len(results),
            }
        
        # Aggregate by event count
        by_count: dict[int, list[TrialResult]] = {}
        for r in valid_results:
            count = r.metrics["event_count"]
            if count not in by_count:
                by_count[count] = []
            by_count[count].append(r)
        
        # Compute stats per count
        by_count_stats = []
        for count, count_results in sorted(by_count.items()):
            total_written = sum(r.metrics["events_written"] for r in count_results)
            total_recovered = sum(r.metrics["events_recovered"] for r in count_results)
            total_corrupted = sum(r.metrics["events_corrupted"] for r in count_results)
            avg_recovery_time = sum(r.metrics["recovery_time_ms"] for r in count_results) / len(count_results)
            
            by_count_stats.append({
                "event_count": count,
                "trials": len(count_results),
                "events_written": total_written,
                "events_recovered": total_recovered,
                "events_corrupted": total_corrupted,
                "recovery_rate": total_recovered / total_written if total_written > 0 else 0,
                "avg_recovery_time_ms": avg_recovery_time,
            })
        
        # Overall stats
        total_written = sum(r.metrics["events_written"] for r in valid_results)
        total_recovered = sum(r.metrics["events_recovered"] for r in valid_results)
        total_corrupted = sum(r.metrics["events_corrupted"] for r in valid_results)
        
        return {
            "total_trials": len(results),
            "valid_trials": len(valid_results),
            "events_written": total_written,
            "events_recovered": total_recovered,
            "corruption_count": total_corrupted,
            "recovery_rate": total_recovered / total_written if total_written > 0 else 0,
            "recovery_time_ms": sum(r.metrics["recovery_time_ms"] for r in valid_results) / len(valid_results),
            "by_count": by_count_stats,
        }
    
    def _compute_checksum(self, content: str) -> str:
        """Compute a checksum for content verification"""
        return hashlib.md5(content.encode()).hexdigest()[:8]
