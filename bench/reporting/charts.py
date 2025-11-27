"""
Chart generation for benchmark results.
"""

from pathlib import Path
from typing import Optional
import logging

from ..models import ScenarioResult

logger = logging.getLogger(__name__)

# Colors for adapters
COLORS = {
    "krnx": "#2ecc71",      # Green
    "naive_rag": "#3498db", # Blue
    "baseline": "#95a5a6",  # Gray
}


class ChartGenerator:
    """
    Generates charts from benchmark results using matplotlib.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import matplotlib lazily
        self._plt = None
        self._np = None
    
    @property
    def plt(self):
        if self._plt is None:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
            self._plt = plt
        return self._plt
    
    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np
    
    def generate_all(
        self, 
        results: dict[str, list[ScenarioResult]]
    ) -> list[Path]:
        """
        Generate all charts for the benchmark results.
        
        Args:
            results: Dict mapping scenario name to list of results
            
        Returns:
            List of paths to generated charts
        """
        
        charts = []
        
        # Consistency comparison (fact correction)
        if "fact_correction" in results:
            path = self._consistency_chart(results["fact_correction"])
            if path:
                charts.append(path)
        
        # Durability comparison
        if "crash_recovery" in results:
            path = self._durability_chart(results["crash_recovery"])
            if path:
                charts.append(path)
        
        # Replay scaling
        if "point_in_time" in results:
            path = self._replay_scaling_chart(results["point_in_time"])
            if path:
                charts.append(path)
        
        # NIAH comparison
        if "niah" in results:
            path = self._niah_chart(results["niah"])
            if path:
                charts.append(path)
        
        return charts
    
    def _consistency_chart(self, results: list[ScenarioResult]) -> Optional[Path]:
        """Generate fact correction accuracy chart"""
        
        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            
            adapters = []
            correct = []
            stale = []
            hallucinated = []
            
            for result in results:
                agg = result.aggregate
                if "error" in agg:
                    continue
                
                adapters.append(result.adapter_name)
                correct.append(agg.get("correct_rate", 0) * 100)
                stale.append(agg.get("stale_rate", 0) * 100)
                hallucinated.append(agg.get("hallucination_rate", 0) * 100)
            
            if not adapters:
                return None
            
            x = self.np.arange(len(adapters))
            width = 0.25
            
            bars1 = ax.bar(x - width, correct, width, label='Correct', color='#2ecc71')
            bars2 = ax.bar(x, stale, width, label='Stale', color='#f39c12')
            bars3 = ax.bar(x + width, hallucinated, width, label='Hallucinated', color='#e74c3c')
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Fact Correction Accuracy by Adapter')
            ax.set_xticks(x)
            ax.set_xticklabels(adapters)
            ax.legend()
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            
            self.plt.tight_layout()
            
            path = self.output_dir / "consistency_comparison.png"
            self.plt.savefig(path, dpi=150)
            self.plt.close()
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to generate consistency chart: {e}")
            return None
    
    def _durability_chart(self, results: list[ScenarioResult]) -> Optional[Path]:
        """Generate crash recovery chart"""
        
        try:
            fig, ax = self.plt.subplots(figsize=(8, 6))
            
            adapters = []
            recovery_rates = []
            colors = []
            
            for result in results:
                agg = result.aggregate
                if "error" in agg:
                    continue
                
                adapters.append(result.adapter_name)
                recovery_rates.append(agg.get("recovery_rate", 0) * 100)
                colors.append(COLORS.get(result.adapter_name, "#95a5a6"))
            
            if not adapters:
                return None
            
            bars = ax.bar(adapters, recovery_rates, color=colors)
            
            ax.set_ylabel('Recovery Rate (%)')
            ax.set_title('Crash Recovery: Event Durability')
            ax.set_ylim(0, 105)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
            
            self.plt.tight_layout()
            
            path = self.output_dir / "durability_comparison.png"
            self.plt.savefig(path, dpi=150)
            self.plt.close()
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to generate durability chart: {e}")
            return None
    
    def _replay_scaling_chart(self, results: list[ScenarioResult]) -> Optional[Path]:
        """Generate replay latency scaling chart"""
        
        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            
            for result in results:
                agg = result.aggregate
                if "error" in agg or "by_size" not in agg:
                    continue
                
                sizes = [s["size"] for s in agg["by_size"]]
                latencies = [s["latency_ms"] for s in agg["by_size"]]
                
                color = COLORS.get(result.adapter_name, "#95a5a6")
                ax.plot(sizes, latencies, 'o-', label=result.adapter_name, 
                       color=color, linewidth=2, markersize=8)
            
            # Add O(n) reference line
            if results and "by_size" in results[0].aggregate:
                sizes = [s["size"] for s in results[0].aggregate["by_size"]]
                if sizes:
                    # Normalize to first point of first result
                    first_lat = results[0].aggregate["by_size"][0]["latency_ms"]
                    reference = [first_lat * (s / sizes[0]) for s in sizes]
                    ax.plot(sizes, reference, '--', color='gray', alpha=0.5, 
                           label='O(n) reference')
            
            ax.set_xlabel('History Size (events)')
            ax.set_ylabel('Replay Latency (ms)')
            ax.set_title('Replay Latency vs History Size')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.plt.tight_layout()
            
            path = self.output_dir / "replay_scaling.png"
            self.plt.savefig(path, dpi=150)
            self.plt.close()
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to generate replay scaling chart: {e}")
            return None
    
    def _niah_chart(self, results: list[ScenarioResult]) -> Optional[Path]:
        """Generate NIAH baseline chart"""
        
        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            
            adapters = []
            front = []
            middle = []
            end = []
            
            for result in results:
                agg = result.aggregate
                if "error" in agg:
                    continue
                
                by_pos = agg.get("by_position", {})
                adapters.append(result.adapter_name)
                front.append(by_pos.get("front", 0) * 100)
                middle.append(by_pos.get("middle", 0) * 100)
                end.append(by_pos.get("end", 0) * 100)
            
            if not adapters:
                return None
            
            x = self.np.arange(len(adapters))
            width = 0.25
            
            bars1 = ax.bar(x - width, front, width, label='Front', color='#3498db')
            bars2 = ax.bar(x, middle, width, label='Middle', color='#9b59b6')
            bars3 = ax.bar(x + width, end, width, label='End', color='#1abc9c')
            
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Needle-in-Haystack: Retrieval by Position')
            ax.set_xticks(x)
            ax.set_xticklabels(adapters)
            ax.legend()
            ax.set_ylim(0, 100)
            
            self.plt.tight_layout()
            
            path = self.output_dir / "niah_comparison.png"
            self.plt.savefig(path, dpi=150)
            self.plt.close()
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to generate NIAH chart: {e}")
            return None
