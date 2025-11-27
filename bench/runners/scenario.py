"""
Single scenario runner.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

from ..config import get_scenario_config, get_adapter_config
from ..adapters import get_adapter
from ..scenarios import get_scenario
from ..llm.client import LLMClient
from ..models import ScenarioResult

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Runs a single benchmark scenario against an adapter.
    """
    
    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        
        # Set up logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create LLM client
        self.llm = LLMClient(config.get("llm", {}))
    
    def run(
        self,
        scenario_name: str,
        adapter_name: str,
        trials: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> ScenarioResult:
        """
        Run a scenario against an adapter.
        
        Args:
            scenario_name: Name of the scenario to run
            adapter_name: Name of the adapter to test
            trials: Number of trials (overrides config)
            progress_callback: Called after each trial
            
        Returns:
            ScenarioResult with all trial data
        """
        
        # Get scenario class and config
        scenario_cls = get_scenario(scenario_name)
        scenario_config = get_scenario_config(self.config, scenario_name)
        
        # Create and configure scenario
        scenario = scenario_cls()
        scenario.configure(scenario_config)
        
        # Check adapter compatibility
        if not scenario.supports_adapter(adapter_name):
            logger.warning(f"Scenario {scenario_name} does not support adapter {adapter_name}")
            # Return empty result
            return ScenarioResult(
                scenario_name=scenario_name,
                adapter_name=adapter_name,
                trials=[],
                aggregate={"error": "adapter_not_supported"},
                config=scenario_config,
            )
        
        # Get adapter
        adapter = get_adapter(adapter_name, self.config)
        
        # Determine trial count
        if trials is None:
            trials = self.config.get("defaults", {}).get("trials", 10)
        
        logger.info(f"Running {scenario_name} on {adapter_name} for {trials} trials")
        
        # Run with adapter lifecycle management
        try:
            adapter.setup()
            
            result = scenario.run(
                adapter=adapter,
                llm=self.llm,
                trials=trials,
                progress_callback=progress_callback,
            )
            
            return result
            
        finally:
            adapter.teardown()
    
    def save_result(self, result: ScenarioResult, output_dir: Path) -> Path:
        """
        Save scenario result to output directory.
        
        Args:
            result: The scenario result
            output_dir: Directory to save to
            
        Returns:
            Path to the saved file
        """
        
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL (one line per trial)
        filename = f"{result.scenario_name}_{result.adapter_name}.jsonl"
        filepath = raw_dir / filename
        
        with open(filepath, "w") as f:
            f.write(result.to_jsonl())
        
        logger.info(f"Saved results to {filepath}")
        
        # Also save full result as JSON
        json_path = raw_dir / f"{result.scenario_name}_{result.adapter_name}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
