"""
Markdown table generation for benchmark results.
"""

from typing import Optional
from ..models import ScenarioResult


class TableGenerator:
    """
    Generates Markdown tables from benchmark results.
    """
    
    def generate_table(
        self, 
        scenario_name: str, 
        results: list[ScenarioResult]
    ) -> Optional[str]:
        """
        Generate a Markdown table for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            results: List of results (one per adapter)
            
        Returns:
            Markdown table string, or None if no valid results
        """
        
        if not results:
            return None
        
        # Route to scenario-specific table generator
        if scenario_name == "crash_recovery":
            return self._durability_table(results)
        elif scenario_name == "fact_correction":
            return self._consistency_table(results)
        elif scenario_name == "temporal_versioning":
            return self._temporal_table(results)
        elif scenario_name == "provenance_chain":
            return self._provenance_table(results)
        elif scenario_name in ("point_in_time", "determinism"):
            return self._replay_table(results)
        elif scenario_name == "niah":
            return self._niah_table(results)
        else:
            return self._generic_table(results)
    
    def _durability_table(self, results: list[ScenarioResult]) -> str:
        """Generate durability (crash recovery) table"""
        
        lines = [
            "## Durability: Crash Recovery",
            "",
            "| Adapter | Events Written | Events Recovered | Recovery Rate | Corruption | Recovery Time |",
            "|---------|---------------|-----------------|---------------|------------|---------------|",
        ]
        
        for result in results:
            agg = result.aggregate
            if "error" in agg:
                lines.append(f"| {result.adapter_name} | N/A | N/A | N/A | N/A | N/A |")
                continue
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {agg.get('events_written', 'N/A'):,} "
                f"| {agg.get('events_recovered', 'N/A'):,} "
                f"| {agg.get('recovery_rate', 0)*100:.1f}% "
                f"| {agg.get('corruption_count', 0)} "
                f"| {agg.get('recovery_time_ms', 0):.0f}ms |"
            )
        
        return "\n".join(lines)
    
    def _consistency_table(self, results: list[ScenarioResult]) -> str:
        """Generate consistency (fact correction) table"""
        
        lines = [
            "## Consistency: Fact Correction",
            "",
            "| Adapter | Correct | Stale | Hallucinated | Trials |",
            "|---------|---------|-------|--------------|--------|",
        ]
        
        for result in results:
            agg = result.aggregate
            if "error" in agg:
                lines.append(f"| {result.adapter_name} | N/A | N/A | N/A | N/A |")
                continue
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {agg.get('correct_rate', 0)*100:.1f}% "
                f"| {agg.get('stale_rate', 0)*100:.1f}% "
                f"| {agg.get('hallucination_rate', 0)*100:.1f}% "
                f"| {agg.get('total_trials', 0)} |"
            )
        
        return "\n".join(lines)
    
    def _temporal_table(self, results: list[ScenarioResult]) -> str:
        """Generate temporal versioning table"""
        
        lines = [
            "## Consistency: Temporal Versioning",
            "",
            "| Adapter | Temporal Accuracy | Perfect Trials | Avg Time |",
            "|---------|-------------------|----------------|----------|",
        ]
        
        for result in results:
            agg = result.aggregate
            if "error" in agg:
                lines.append(f"| {result.adapter_name} | N/A | N/A | N/A |")
                continue
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {agg.get('mean_temporal_accuracy', 0)*100:.1f}% "
                f"| {agg.get('perfect_trials', 0)}/{agg.get('valid_trials', 0)} "
                f"| {agg.get('mean_timing_ms', 0):.0f}ms |"
            )
        
        return "\n".join(lines)
    
    def _provenance_table(self, results: list[ScenarioResult]) -> str:
        """Generate provenance chain table"""
        
        lines = [
            "## Auditability: Provenance Chain",
            "",
            "| Adapter | Complete Chains | Verified | Gaps Found | Success Rate |",
            "|---------|-----------------|----------|------------|--------------|",
        ]
        
        for result in results:
            agg = result.aggregate
            if "error" in agg:
                lines.append(f"| {result.adapter_name} | N/A | N/A | N/A | N/A |")
                continue
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {agg.get('complete_chains', 0)}/{agg.get('valid_trials', 0)} "
                f"| {agg.get('verified_chains', 0)}/{agg.get('valid_trials', 0)} "
                f"| {agg.get('chains_with_gaps', 0)} "
                f"| {agg.get('success_rate', 0)*100:.1f}% |"
            )
        
        return "\n".join(lines)
    
    def _replay_table(self, results: list[ScenarioResult]) -> str:
        """Generate replay table"""
        
        # Check which scenario type
        scenario_name = results[0].scenario_name if results else "replay"
        
        if scenario_name == "point_in_time":
            title = "## Replay: Point-in-Time"
            
            lines = [
                title,
                "",
                "| Adapter | History Size | Accuracy | Latency |",
                "|---------|-------------|----------|---------|",
            ]
            
            for result in results:
                agg = result.aggregate
                if "error" in agg:
                    lines.append(f"| {result.adapter_name} | N/A | N/A | N/A |")
                    continue
                
                # Show by-size breakdown
                for size_stat in agg.get("by_size", []):
                    lines.append(
                        f"| {result.adapter_name} "
                        f"| {size_stat['size']:,} "
                        f"| {size_stat['accuracy']*100:.1f}% "
                        f"| {size_stat['latency_ms']:.0f}ms |"
                    )
        
        else:  # determinism
            title = "## Replay: Determinism"
            
            lines = [
                title,
                "",
                "| Adapter | Deterministic | Total Trials | Success Rate |",
                "|---------|---------------|--------------|--------------|",
            ]
            
            for result in results:
                agg = result.aggregate
                if "error" in agg:
                    lines.append(f"| {result.adapter_name} | N/A | N/A | N/A |")
                    continue
                
                lines.append(
                    f"| {result.adapter_name} "
                    f"| {agg.get('deterministic_trials', 0)} "
                    f"| {agg.get('valid_trials', 0)} "
                    f"| {agg.get('determinism_rate', 0)*100:.1f}% |"
                )
        
        return "\n".join(lines)
    
    def _niah_table(self, results: list[ScenarioResult]) -> str:
        """Generate NIAH baseline table"""
        
        lines = [
            "## Baseline: Needle in Haystack",
            "",
            "| Adapter | Accuracy | Front | Middle | End |",
            "|---------|----------|-------|--------|-----|",
        ]
        
        for result in results:
            agg = result.aggregate
            if "error" in agg:
                lines.append(f"| {result.adapter_name} | N/A | N/A | N/A | N/A |")
                continue
            
            by_pos = agg.get("by_position", {})
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {agg.get('accuracy', 0)*100:.1f}% "
                f"| {by_pos.get('front', 0)*100:.1f}% "
                f"| {by_pos.get('middle', 0)*100:.1f}% "
                f"| {by_pos.get('end', 0)*100:.1f}% |"
            )
        
        return "\n".join(lines)
    
    def _generic_table(self, results: list[ScenarioResult]) -> str:
        """Generate generic results table"""
        
        scenario_name = results[0].scenario_name if results else "unknown"
        
        lines = [
            f"## {scenario_name}",
            "",
            "| Adapter | Success Rate | Trials | Avg Time |",
            "|---------|-------------|--------|----------|",
        ]
        
        for result in results:
            agg = result.aggregate
            success_rate = agg.get("success_rate", 0)
            total = agg.get("total_trials", len(result.trials))
            avg_time = agg.get("mean_timing_ms", 0)
            
            lines.append(
                f"| {result.adapter_name} "
                f"| {success_rate*100:.1f}% "
                f"| {total} "
                f"| {avg_time:.0f}ms |"
            )
        
        return "\n".join(lines)
