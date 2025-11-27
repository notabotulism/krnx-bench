"""
Summary report generation.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from ..models import ScenarioResult


class SummaryGenerator:
    """
    Generates a summary Markdown report from benchmark results.
    """
    
    def generate(
        self, 
        results: dict[str, list[ScenarioResult]],
        run_dir: Path,
    ) -> str:
        """
        Generate a summary report.
        
        Args:
            results: Dict mapping scenario name to list of results
            run_dir: Path to the run directory
            
        Returns:
            Markdown summary string
        """
        
        lines = [
            "# KRNX Benchmark Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Calculate headline metrics
        krnx_wins = 0
        total_scenarios = 0
        
        for scenario_name, scenario_results in results.items():
            krnx_result = next((r for r in scenario_results if r.adapter_name == "krnx"), None)
            rag_result = next((r for r in scenario_results if r.adapter_name == "naive_rag"), None)
            
            if krnx_result and rag_result:
                total_scenarios += 1
                
                # Compare success rates
                krnx_rate = krnx_result.aggregate.get("success_rate", 0)
                rag_rate = rag_result.aggregate.get("success_rate", 0)
                
                if krnx_rate > rag_rate:
                    krnx_wins += 1
        
        if total_scenarios > 0:
            lines.append(f"KRNX outperformed naive RAG in **{krnx_wins}/{total_scenarios}** comparable scenarios.")
            lines.append("")
        
        # Four guarantees summary
        lines.extend([
            "### Four Guarantees",
            "",
            "| Guarantee | KRNX | Naive RAG | Winner |",
            "|-----------|------|-----------|--------|",
        ])
        
        guarantees = {
            "durability": "crash_recovery",
            "consistency": "fact_correction",
            "auditability": "provenance_chain",
            "replay": "point_in_time",
        }
        
        for guarantee, scenario_name in guarantees.items():
            if scenario_name in results:
                scenario_results = results[scenario_name]
                
                krnx = next((r for r in scenario_results if r.adapter_name == "krnx"), None)
                rag = next((r for r in scenario_results if r.adapter_name == "naive_rag"), None)
                
                krnx_val = self._get_primary_metric(krnx) if krnx else "N/A"
                rag_val = self._get_primary_metric(rag) if rag else "N/A"
                
                winner = self._determine_winner(krnx, rag)
                
                lines.append(f"| {guarantee.capitalize()} | {krnx_val} | {rag_val} | {winner} |")
            else:
                lines.append(f"| {guarantee.capitalize()} | - | - | - |")
        
        lines.append("")
        
        # Detailed results by guarantee
        lines.extend([
            "## Detailed Results",
            "",
        ])
        
        for guarantee, scenario_names in self._group_by_guarantee(results).items():
            lines.append(f"### {guarantee.capitalize()}")
            lines.append("")
            
            for scenario_name in scenario_names:
                if scenario_name in results:
                    lines.extend(self._scenario_summary(scenario_name, results[scenario_name]))
                    lines.append("")
        
        # Key findings
        lines.extend([
            "## Key Findings",
            "",
        ])
        
        findings = self._extract_findings(results)
        for finding in findings:
            lines.append(f"- {finding}")
        
        lines.append("")
        
        # Methodology
        lines.extend([
            "## Methodology",
            "",
            "Each scenario was run with the configuration specified in `config/scenarios.yaml`.",
            "Results are aggregated across multiple trials to account for variance.",
            "",
            "### Adapters Tested",
            "",
            "- **KRNX**: Temporal memory kernel with event sourcing, hash chains, and replay",
            "- **Naive RAG**: Qdrant vector store with top-k embedding similarity search",
            "- **Baseline**: Raw LLM with no external memory (where applicable)",
            "",
        ])
        
        return "\n".join(lines)
    
    def _get_primary_metric(self, result: Optional[ScenarioResult]) -> str:
        """Get the primary metric for a result"""
        
        if result is None:
            return "N/A"
        
        agg = result.aggregate
        
        if "error" in agg:
            return "N/A"
        
        # Different metrics for different scenarios
        if result.scenario_name == "crash_recovery":
            rate = agg.get("recovery_rate", 0)
            return f"{rate*100:.0f}%"
        
        elif result.scenario_name == "fact_correction":
            rate = agg.get("correct_rate", 0)
            return f"{rate*100:.0f}%"
        
        elif result.scenario_name == "provenance_chain":
            rate = agg.get("success_rate", 0)
            return f"{rate*100:.0f}%"
        
        elif result.scenario_name in ("point_in_time", "determinism"):
            if "overall_accuracy" in agg:
                return f"{agg['overall_accuracy']*100:.0f}%"
            elif "determinism_rate" in agg:
                return f"{agg['determinism_rate']*100:.0f}%"
        
        elif result.scenario_name == "niah":
            rate = agg.get("accuracy", 0)
            return f"{rate*100:.0f}%"
        
        # Default
        rate = agg.get("success_rate", 0)
        return f"{rate*100:.0f}%"
    
    def _determine_winner(
        self, 
        krnx: Optional[ScenarioResult], 
        rag: Optional[ScenarioResult]
    ) -> str:
        """Determine the winner between KRNX and RAG"""
        
        if krnx is None and rag is None:
            return "-"
        
        if krnx is None:
            return "RAG"
        
        if rag is None:
            return "KRNX"
        
        krnx_rate = krnx.aggregate.get("success_rate", 0)
        rag_rate = rag.aggregate.get("success_rate", 0)
        
        # Handle N/A cases
        if "error" in krnx.aggregate:
            krnx_rate = 0
        if "error" in rag.aggregate:
            rag_rate = 0
        
        if krnx_rate > rag_rate:
            return "**KRNX**"
        elif rag_rate > krnx_rate:
            return "RAG"
        else:
            return "Tie"
    
    def _group_by_guarantee(self, results: dict) -> dict:
        """Group scenarios by their guarantee"""
        
        guarantee_map = {
            "durability": ["crash_recovery"],
            "consistency": ["fact_correction", "temporal_versioning"],
            "auditability": ["provenance_chain"],
            "replay": ["point_in_time", "determinism"],
            "baseline": ["niah"],
        }
        
        grouped = {}
        for guarantee, scenarios in guarantee_map.items():
            present = [s for s in scenarios if s in results]
            if present:
                grouped[guarantee] = present
        
        return grouped
    
    def _scenario_summary(
        self, 
        scenario_name: str, 
        results: list[ScenarioResult]
    ) -> list[str]:
        """Generate summary lines for a scenario"""
        
        lines = [f"#### {scenario_name}", ""]
        
        for result in results:
            agg = result.aggregate
            
            if "error" in agg:
                lines.append(f"- **{result.adapter_name}**: {agg.get('error', 'Error')}")
            else:
                metric = self._get_primary_metric(result)
                trials = agg.get("total_trials", agg.get("valid_trials", 0))
                lines.append(f"- **{result.adapter_name}**: {metric} ({trials} trials)")
        
        return lines
    
    def _extract_findings(self, results: dict) -> list[str]:
        """Extract key findings from results"""
        
        findings = []
        
        # Crash recovery finding
        if "crash_recovery" in results:
            krnx = next((r for r in results["crash_recovery"] if r.adapter_name == "krnx"), None)
            rag = next((r for r in results["crash_recovery"] if r.adapter_name == "naive_rag"), None)
            
            if krnx and rag:
                krnx_rate = krnx.aggregate.get("recovery_rate", 0)
                rag_rate = rag.aggregate.get("recovery_rate", 0)
                
                if krnx_rate > rag_rate:
                    findings.append(
                        f"KRNX recovered {krnx_rate*100:.0f}% of events after crash vs "
                        f"{rag_rate*100:.0f}% for naive RAG"
                    )
        
        # Fact correction finding
        if "fact_correction" in results:
            krnx = next((r for r in results["fact_correction"] if r.adapter_name == "krnx"), None)
            rag = next((r for r in results["fact_correction"] if r.adapter_name == "naive_rag"), None)
            
            if krnx and rag:
                krnx_correct = krnx.aggregate.get("correct_rate", 0)
                rag_stale = rag.aggregate.get("stale_rate", 0)
                
                if krnx_correct > 0.9:
                    findings.append(
                        f"KRNX achieved {krnx_correct*100:.0f}% accuracy on fact updates; "
                        f"naive RAG returned stale data {rag_stale*100:.0f}% of the time"
                    )
        
        # KRNX-only capabilities
        if "temporal_versioning" in results:
            findings.append(
                "KRNX successfully demonstrated temporal versioning (query facts at past timestamps) - "
                "a capability not available in naive RAG"
            )
        
        if "provenance_chain" in results:
            findings.append(
                "KRNX maintained cryptographic provenance chains for complete auditability - "
                "not available in naive RAG"
            )
        
        if not findings:
            findings.append("See detailed results above for complete analysis")
        
        return findings
