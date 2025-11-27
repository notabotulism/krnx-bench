#!/usr/bin/env python3
"""
KRNX Benchmark Harness CLI

Usage:
    krnx-bench run <scenario> [--adapter=all] [--trials=10]
    krnx-bench suite [--output=<dir>]
    krnx-bench report <run_dir>
    krnx-bench scenarios
    krnx-bench adapters
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

import typer
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from bench.config import load_config
from bench.runners.scenario import ScenarioRunner
from bench.runners.suite import SuiteRunner
from bench.reporting.tables import TableGenerator
from bench.reporting.charts import ChartGenerator
from bench.reporting.summary import SummaryGenerator
from bench.scenarios import SCENARIOS, get_scenario

app = typer.Typer(
    name="krnx-bench",
    help="KRNX Memory Kernel Benchmark Harness",
    add_completion=False
)
console = Console()


@app.command()
def run(
    scenario: str = typer.Argument(..., help="Scenario name (e.g., fact_correction)"),
    adapter: str = typer.Option("all", "--adapter", "-a", help="Adapter: krnx, naive_rag, baseline, or all"),
    trials: int = typer.Option(None, "--trials", "-t", help="Number of trials (overrides config)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run a single benchmark scenario"""
    
    config = load_config()
    
    if trials:
        config["defaults"]["trials"] = trials
    
    if output is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output = Path(f"outputs/runs/{timestamp}")
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Resolve adapters
    if adapter == "all":
        adapters = ["krnx", "naive_rag", "baseline"]
    else:
        adapters = [a.strip() for a in adapter.split(",")]
    
    # Get scenario
    try:
        scenario_cls = get_scenario(scenario)
    except KeyError:
        console.print(f"[red]Error:[/red] Unknown scenario: {scenario}")
        console.print("Run 'krnx-bench scenarios' to see available scenarios.")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]KRNX Benchmark[/bold]\n\n"
        f"Scenario: [cyan]{scenario}[/cyan]\n"
        f"Adapters: [cyan]{', '.join(adapters)}[/cyan]\n"
        f"Trials: [cyan]{config['defaults']['trials']}[/cyan]\n"
        f"Output: [dim]{output}[/dim]",
        title="Configuration"
    ))
    
    runner = ScenarioRunner(config, verbose=verbose)
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        for adp in adapters:
            task = progress.add_task(
                f"Running [cyan]{scenario}[/cyan] on [green]{adp}[/green]...",
                total=config["defaults"]["trials"]
            )
            
            try:
                result = runner.run(
                    scenario_name=scenario,
                    adapter_name=adp,
                    progress_callback=lambda: progress.advance(task)
                )
                results.append(result)
                runner.save_result(result, output)
                progress.update(task, description=f"[green]✓[/green] {scenario} on {adp}")
            
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {scenario} on {adp}: {e}")
                if verbose:
                    console.print_exception()
    
    # Print summary
    console.print()
    _print_results_table(results)
    
    console.print(f"\n[green]✓[/green] Results saved to [dim]{output}[/dim]")


@app.command()
def suite(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    skip_baseline: bool = typer.Option(False, "--skip-baseline", help="Skip NIAH baseline test"),
    adapters: str = typer.Option("all", "--adapters", "-a", help="Adapters to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run complete benchmark suite"""
    
    config = load_config()
    
    if output is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output = Path(f"outputs/runs/{timestamp}")
    
    output.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel(
        f"[bold]KRNX Benchmark Suite[/bold]\n\n"
        f"Output: [dim]{output}[/dim]\n"
        f"Skip baseline: [cyan]{skip_baseline}[/cyan]",
        title="Configuration"
    ))
    
    runner = SuiteRunner(config, verbose=verbose)
    
    try:
        result = runner.run(
            output_dir=output,
            skip_baseline=skip_baseline,
            adapter_filter=None if adapters == "all" else adapters.split(",")
        )
        
        console.print(f"\n[green]✓[/green] Suite complete: {len(result.results)} scenario runs")
        console.print(f"[green]✓[/green] Results saved to [dim]{output}[/dim]")
        
        # Auto-generate report
        console.print("\nGenerating report...")
        _generate_report(output)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="Path to run output directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Report output directory")
):
    """Generate tables and charts from benchmark run"""
    
    if not run_dir.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_dir}")
        raise typer.Exit(1)
    
    _generate_report(run_dir, output)


def _generate_report(run_dir: Path, output: Optional[Path] = None):
    """Generate report from run directory"""
    
    if output is None:
        output = run_dir.parent.parent / "reports" / run_dir.name
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Load results
    from bench.runners.suite import load_results
    results = load_results(run_dir)
    
    if not results:
        console.print("[yellow]Warning:[/yellow] No results found in run directory")
        return
    
    # Generate tables
    table_gen = TableGenerator()
    tables_dir = output / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    console.print("Generating tables...")
    
    for scenario_name, scenario_results in results.items():
        table = table_gen.generate_table(scenario_name, scenario_results)
        if table:
            table_path = tables_dir / f"{scenario_name}.md"
            table_path.write_text(table)
            console.print(f"  [green]✓[/green] {table_path.name}")
    
    # Generate charts
    chart_gen = ChartGenerator(output / "figures")
    
    console.print("Generating charts...")
    
    charts = chart_gen.generate_all(results)
    for chart_path in charts:
        console.print(f"  [green]✓[/green] {chart_path.name}")
    
    # Generate summary
    summary_gen = SummaryGenerator()
    summary = summary_gen.generate(results, run_dir)
    (output / "summary.md").write_text(summary)
    console.print(f"  [green]✓[/green] summary.md")
    
    console.print(f"\n[green]✓[/green] Report generated: [dim]{output}[/dim]")


@app.command()
def scenarios():
    """List available benchmark scenarios"""
    
    console.print(Panel("[bold]Available Scenarios[/bold]", expand=False))
    
    for guarantee, scenario_list in SCENARIOS.items():
        table = Table(title=f"[cyan]{guarantee.upper()}[/cyan]", show_header=True)
        table.add_column("Name", style="green")
        table.add_column("Description")
        table.add_column("Adapters", style="dim")
        
        for s in scenario_list:
            adapters = ", ".join(s.supported_adapters) if hasattr(s, 'supported_adapters') else "all"
            table.add_row(s.name, s.description, adapters)
        
        console.print(table)
        console.print()


@app.command()
def adapters():
    """List available memory adapters"""
    
    table = Table(title="[bold]Available Adapters[/bold]", show_header=True)
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Docker", style="cyan")
    table.add_column("Capabilities")
    
    table.add_row(
        "krnx",
        "KRNX temporal memory kernel",
        "Yes",
        "write, query, replay, provenance"
    )
    table.add_row(
        "naive_rag",
        "Qdrant vector store (top-k)",
        "Yes",
        "write, query"
    )
    table.add_row(
        "baseline",
        "No memory (raw LLM)",
        "No",
        "query only"
    )
    
    console.print(table)


@app.command()
def clean(
    runs: bool = typer.Option(False, "--runs", help="Clean run outputs"),
    reports: bool = typer.Option(False, "--reports", help="Clean reports"),
    all_outputs: bool = typer.Option(False, "--all", help="Clean all outputs"),
    force: bool = typer.Option(False, "--force", "-f", help="Don't ask for confirmation")
):
    """Clean output directories"""
    
    import shutil
    
    to_clean = []
    
    if all_outputs or runs:
        to_clean.append(Path("outputs/runs"))
    if all_outputs or reports:
        to_clean.append(Path("outputs/reports"))
    
    if not to_clean:
        console.print("Specify --runs, --reports, or --all")
        return
    
    existing = [p for p in to_clean if p.exists()]
    
    if not existing:
        console.print("Nothing to clean")
        return
    
    console.print("Will delete:")
    for p in existing:
        console.print(f"  [red]{p}[/red]")
    
    if not force:
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Aborted")
            return
    
    for p in existing:
        shutil.rmtree(p)
        console.print(f"[green]✓[/green] Deleted {p}")


def _print_results_table(results):
    """Print a summary table of results"""
    
    if not results:
        return
    
    table = Table(title="Results Summary", show_header=True)
    table.add_column("Adapter", style="cyan")
    table.add_column("Trials")
    table.add_column("Success Rate", style="green")
    table.add_column("Avg Latency")
    
    for result in results:
        success_rate = sum(1 for t in result.trials if t.success) / len(result.trials)
        avg_latency = sum(t.timing_ms for t in result.trials) / len(result.trials)
        
        table.add_row(
            result.adapter_name,
            str(len(result.trials)),
            f"{success_rate*100:.1f}%",
            f"{avg_latency:.0f}ms"
        )
    
    console.print(table)


if __name__ == "__main__":
    app()
