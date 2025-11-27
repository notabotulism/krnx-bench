"""
Full benchmark suite runner.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..config import load_config, get_scenario_config
from ..adapters import get_adapter, ADAPTERS
from ..scenarios import ALL_SCENARIOS, get_scenario
from ..llm.client import LLMClient
from ..models import ScenarioResult, SuiteResult

logger = logging.getLogger(__name__)
console = Console()


class SuiteRunner:
    """
    Runs the complete benchmark suite.
    """
    
    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        
        # Set up logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level)
        
        # Create LLM client
        self.llm = LLMClient(config.get("llm", {}))
    
    def run(
        self,
        output_dir: Path,
        skip_baseline: bool = False,
        adapter_filter: Optional[list[str]] = None,
        scenario_filter: Optional[list[str]] = None,
    ) -> SuiteResult:
        """
        Run the complete benchmark suite.
        
        Args:
            output_dir: Directory to save results
            skip_baseline: Skip NIAH baseline test
            adapter_filter: Only run these adapters
            scenario_filter: Only run these scenarios
            
        Returns:
            SuiteResult with all scenario results
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        started_at = datetime.now().isoformat()
        results: list[ScenarioResult] = []
        
        # Determine scenarios to run
        scenarios = ALL_SCENARIOS
        if skip_baseline:
            scenarios = [s for s in scenarios if s.guarantee != "baseline"]
        if scenario_filter:
            scenarios = [s for s in scenarios if s.name in scenario_filter]
        
        # Determine adapters to run
        adapters = list(ADAPTERS.keys())
        if adapter_filter:
            adapters = [a for a in adapters if a in adapter_filter]
        
        # Calculate total tasks
        total_tasks = 0
        for scenario_cls in scenarios:
            for adapter_name in adapters:
                if scenario_cls().supports_adapter(adapter_name):
                    total_tasks += 1
        
        console.print(f"Running {total_tasks} scenario/adapter combinations...")
        
        # Write manifest
        manifest = {
            "started_at": started_at,
            "config": self.config,
            "scenarios": [s.name for s in scenarios],
            "adapters": adapters,
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Run each scenario/adapter combination
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Suite progress", total=total_tasks)
            
            for scenario_cls in scenarios:
                scenario = scenario_cls()
                scenario_config = get_scenario_config(self.config, scenario.name)
                scenario.configure(scenario_config)
                
                for adapter_name in adapters:
                    if not scenario.supports_adapter(adapter_name):
                        continue
                    
                    task_desc = f"{scenario.name} on {adapter_name}"
                    progress.update(main_task, description=task_desc)
                    
                    try:
                        result = self._run_scenario(
                            scenario=scenario,
                            adapter_name=adapter_name,
                        )
                        results.append(result)
                        
                        # Save individual result
                        self._save_result(result, output_dir)
                        
                        progress.update(main_task, description=f"[green]✓[/green] {task_desc}")
                        
                    except Exception as e:
                        logger.error(f"Failed: {task_desc}: {e}")
                        progress.update(main_task, description=f"[red]✗[/red] {task_desc}")
                        
                        if self.verbose:
                            console.print_exception()
                    
                    progress.advance(main_task)
        
        completed_at = datetime.now().isoformat()
        
        # Create suite result
        suite_result = SuiteResult(
            results=results,
            started_at=started_at,
            completed_at=completed_at,
            config=self.config,
        )
        
        # Save suite result
        with open(output_dir / "suite_result.json", "w") as f:
            json.dump(suite_result.to_dict(), f, indent=2)
        
        return suite_result
    
    def _run_scenario(
        self,
        scenario,
        adapter_name: str,
    ) -> ScenarioResult:
        """Run a single scenario against an adapter"""
        
        adapter = get_adapter(adapter_name, self.config)
        trials = self.config.get("defaults", {}).get("trials", 10)
        
        try:
            adapter.setup()
            
            result = scenario.run(
                adapter=adapter,
                llm=self.llm,
                trials=trials,
            )
            
            return result
            
        finally:
            adapter.teardown()
    
    def _save_result(self, result: ScenarioResult, output_dir: Path) -> None:
        """Save individual scenario result"""
        
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        filename = f"{result.scenario_name}_{result.adapter_name}.json"
        with open(raw_dir / filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save as JSONL
        jsonl_filename = f"{result.scenario_name}_{result.adapter_name}.jsonl"
        with open(raw_dir / jsonl_filename, "w") as f:
            f.write(result.to_jsonl())


def load_results(run_dir: Path) -> dict[str, list[ScenarioResult]]:
    """
    Load results from a run directory.
    
    Args:
        run_dir: Path to the run output directory
        
    Returns:
        Dict mapping scenario name to list of results (one per adapter)
    """
    
    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"
    
    if not raw_dir.exists():
        return {}
    
    results: dict[str, list[ScenarioResult]] = {}
    
    for json_file in raw_dir.glob("*.json"):
        if json_file.name == "manifest.json":
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            result = ScenarioResult.from_dict(data)
            
            if result.scenario_name not in results:
                results[result.scenario_name] = []
            results[result.scenario_name].append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    return results
