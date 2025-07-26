#!/usr/bin/env python3
"""Command-line interface for CloudTrain.

This module provides a CLI for interacting with the CloudTrain universal
cloud training API, allowing users to submit jobs, monitor status, and
manage configurations from the command line.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cloudtrain import CloudProvider, CloudTrainingAPI, JobStatus
from cloudtrain.config import ConfigManager
from cloudtrain.schemas import TrainingJobSpec, JobStatusUpdate

console: Console = Console()


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """CloudTrain - Universal Cloud Training API CLI.

    Submit and manage machine learning training jobs across multiple cloud providers.
    """
    ctx.ensure_object(dict)

    # Set up configuration
    config_manager: ConfigManager = (
        ConfigManager(config_file=config) if config else ConfigManager()
    )
    ctx.obj["config_manager"] = config_manager
    ctx.obj["verbose"] = verbose

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


@cli.command()
@click.pass_context
def providers(ctx: click.Context) -> None:
    """List available cloud providers."""

    async def list_providers() -> None:
        config_manager: ConfigManager = ctx.obj["config_manager"]

        async with CloudTrainingAPI(config_manager=config_manager) as api:
            available: List[CloudProvider] = api.get_available_providers()

            if not available:
                console.print("[yellow]No providers are currently available.[/yellow]")
                console.print("Please check your configuration and credentials.")
                return

            table: Table = Table(title="Available Cloud Providers")
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Region", style="dim")

            for provider in available:
                config: Any = config_manager.get_provider_config(provider)
                region: str = getattr(config, "region", "N/A") if config else "N/A"
                table.add_row(provider.value.upper(), "✅ Available", region)

            console.print(table)

    asyncio.run(list_providers())


@cli.command()
@click.argument("job_spec_file", type=click.Path(exists=True))
@click.option(
    "--provider",
    "-p",
    type=click.Choice([p.value for p in CloudProvider]),
    help="Cloud provider to use",
)
@click.option("--dry-run", is_flag=True, help="Validate job spec without submitting")
@click.pass_context
def submit(
    ctx: click.Context, job_spec_file: str, provider: Optional[str], dry_run: bool
) -> None:
    """Submit a training job from a specification file."""

    async def submit_job() -> None:
        config_manager: ConfigManager = ctx.obj["config_manager"]

        # Load job specification
        job_spec_path: Path = Path(job_spec_file)

        try:
            with open(job_spec_path, "r") as f:
                if job_spec_path.suffix.lower() == ".json":
                    job_data: Dict[str, Any] = json.load(f)
                else:
                    import yaml

                    job_data = yaml.safe_load(f)

            job_spec: TrainingJobSpec = TrainingJobSpec(**job_data)

        except Exception as e:
            console.print(f"[red]Error loading job specification: {e}[/red]")
            sys.exit(1)

        async with CloudTrainingAPI(config_manager=config_manager) as api:
            available_providers: List[CloudProvider] = api.get_available_providers()

            if not available_providers:
                console.print(
                    "[red]No providers are available. Please check your configuration.[/red]"
                )
                sys.exit(1)

            # Select provider
            if provider:
                selected_provider: CloudProvider = CloudProvider(provider)
                if selected_provider not in available_providers:
                    console.print(f"[red]Provider {provider} is not available.[/red]")
                    console.print(
                        f"Available providers: {[p.value for p in available_providers]}"
                    )
                    sys.exit(1)
            else:
                # Use first available provider
                selected_provider = available_providers[0]
                console.print(
                    f"[dim]No provider specified, using {selected_provider.value}[/dim]"
                )

            # Submit job
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Submitting job...", total=None)

                    result = await api.submit_job(
                        selected_provider, job_spec, dry_run=dry_run
                    )

                    progress.update(task, description="Job submitted!")

                if dry_run:
                    console.print("[green]✅ Job specification is valid![/green]")
                else:
                    # Display result
                    panel = Panel.fit(
                        f"[green]Job ID:[/green] {result.job_id}\n"
                        f"[green]Provider:[/green] {result.provider.value}\n"
                        f"[green]Status:[/green] {result.status.value}\n"
                        f"[green]Submitted:[/green] {result.submission_time}",
                        title="Job Submitted Successfully",
                        border_style="green",
                    )
                    console.print(panel)

                    if result.provider_job_url:
                        console.print(
                            f"[dim]View in provider console: {result.provider_job_url}[/dim]"
                        )

            except Exception as e:
                console.print(f"[red]Error submitting job: {e}[/red]")
                sys.exit(1)

    asyncio.run(submit_job())


@cli.command()
@click.argument("job_id")
@click.option(
    "--provider",
    "-p",
    type=click.Choice([p.value for p in CloudProvider]),
    required=True,
    help="Cloud provider where the job is running",
)
@click.option(
    "--follow", "-f", is_flag=True, help="Follow job progress until completion"
)
@click.pass_context
def status(ctx: click.Context, job_id: str, provider: str, follow: bool) -> None:
    """Get the status of a training job."""

    async def get_status() -> None:
        config_manager: ConfigManager = ctx.obj["config_manager"]
        selected_provider: CloudProvider = CloudProvider(provider)

        async with CloudTrainingAPI(config_manager=config_manager) as api:
            try:
                if follow:
                    # Follow job progress
                    console.print(f"Following job {job_id} on {provider}...")
                    console.print("[dim]Press Ctrl+C to stop following[/dim]\n")

                    while True:
                        job_status = await api.get_job_status(selected_provider, job_id)

                        # Clear previous output and display current status
                        console.clear()
                        display_job_status(job_status)

                        if job_status.status.is_terminal():
                            console.print(
                                f"\n[green]Job finished with status: {job_status.status.value}[/green]"
                            )
                            break

                        await asyncio.sleep(10)

                else:
                    # Get status once
                    job_status = await api.get_job_status(selected_provider, job_id)
                    display_job_status(job_status)

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped following job[/yellow]")
            except Exception as e:
                console.print(f"[red]Error getting job status: {e}[/red]")
                sys.exit(1)

    asyncio.run(get_status())


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice([p.value for p in CloudProvider]),
    help="Cloud provider to list jobs from",
)
@click.option(
    "--status-filter",
    type=click.Choice([s.value for s in JobStatus]),
    help="Filter jobs by status",
)
@click.option("--limit", type=int, default=10, help="Maximum number of jobs to list")
@click.pass_context
def list_jobs(ctx, provider: Optional[str], status_filter: Optional[str], limit: int):
    """List training jobs."""

    async def list_provider_jobs():
        config_manager = ctx.obj["config_manager"]

        async with CloudTrainingAPI(config_manager=config_manager) as api:
            available_providers = api.get_available_providers()

            if provider:
                providers_to_check = [CloudProvider(provider)]
            else:
                providers_to_check = available_providers

            status_enum = JobStatus(status_filter) if status_filter else None

            for prov in providers_to_check:
                try:
                    jobs = await api.list_jobs(prov, status_enum, limit)

                    if jobs:
                        table = Table(title=f"Jobs on {prov.value.upper()}")
                        table.add_column("Job ID", style="cyan")
                        table.add_column("Status", style="green")
                        table.add_column("Progress", style="yellow")
                        table.add_column("Updated", style="dim")

                        for job in jobs:
                            progress_str = (
                                f"{job.progress_percentage:.1f}%"
                                if job.progress_percentage
                                else "N/A"
                            )
                            table.add_row(
                                (
                                    job.job_id[:20] + "..."
                                    if len(job.job_id) > 20
                                    else job.job_id
                                ),
                                job.status.value,
                                progress_str,
                                job.updated_time.strftime("%Y-%m-%d %H:%M"),
                            )

                        console.print(table)
                    else:
                        console.print(f"[dim]No jobs found on {prov.value}[/dim]")

                except Exception as e:
                    console.print(
                        f"[red]Error listing jobs from {prov.value}: {e}[/red]"
                    )

    asyncio.run(list_provider_jobs())


@cli.command()
@click.argument("job_id")
@click.option(
    "--provider",
    "-p",
    type=click.Choice([p.value for p in CloudProvider]),
    required=True,
    help="Cloud provider where the job is running",
)
@click.confirmation_option(prompt="Are you sure you want to cancel this job?")
@click.pass_context
def cancel(ctx: click.Context, job_id: str, provider: str) -> None:
    """Cancel a training job."""

    async def cancel_job() -> None:
        config_manager: ConfigManager = ctx.obj["config_manager"]
        selected_provider: CloudProvider = CloudProvider(provider)

        async with CloudTrainingAPI(config_manager=config_manager) as api:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Cancelling job...", total=None)

                    success = await api.cancel_job(selected_provider, job_id)

                    if success:
                        progress.update(task, description="Job cancelled!")
                        console.print(
                            f"[green]✅ Job {job_id} cancelled successfully[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]⚠️ Job {job_id} could not be cancelled[/yellow]"
                        )

            except Exception as e:
                console.print(f"[red]Error cancelling job: {e}[/red]")
                sys.exit(1)

    asyncio.run(cancel_job())


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""

    config_manager: ConfigManager = ctx.obj["config_manager"]

    # Validate configuration
    validation: Dict[str, Any] = config_manager.validate_configuration()

    console.print(
        Panel.fit(
            f"[green]Configuration Sources:[/green] {', '.join(config_manager.config_sources)}\n"
            f"[green]Valid:[/green] {'✅ Yes' if validation['valid'] else '❌ No'}",
            title="CloudTrain Configuration",
            border_style="blue",
        )
    )

    # Show provider status
    table: Table = Table(title="Provider Configuration")
    table.add_column("Provider", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Valid", style="yellow")
    table.add_column("Errors", style="red")

    for provider_name, provider_info in validation["providers"].items():
        enabled: str = "✅" if provider_info["enabled"] else "❌"
        valid: str = "✅" if provider_info["valid"] else "❌"
        errors: str = (
            ", ".join(provider_info["errors"]) if provider_info["errors"] else "None"
        )

        table.add_row(provider_name.upper(), enabled, valid, errors)

    console.print(table)

    if validation["errors"]:
        console.print("\n[red]Configuration Errors:[/red]")
        for error in validation["errors"]:
            console.print(f"  • {error}")


def display_job_status(job_status: JobStatusUpdate) -> None:
    """Display job status in a formatted way."""

    status_color: Dict[JobStatus, str] = {
        JobStatus.PENDING: "yellow",
        JobStatus.STARTING: "blue",
        JobStatus.RUNNING: "green",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.STOPPED: "orange",
        JobStatus.STOPPING: "orange",
        JobStatus.UNKNOWN: "dim",
    }

    color: str = status_color.get(job_status.status, "white")

    info: str = f"[{color}]Status:[/{color}] {job_status.status.value}\n"
    info += f"[cyan]Job ID:[/cyan] {job_status.job_id}\n"
    info += f"[cyan]Updated:[/cyan] {job_status.updated_time}\n"

    if job_status.progress_percentage is not None:
        info += f"[cyan]Progress:[/cyan] {job_status.progress_percentage:.1f}%\n"

    if job_status.current_epoch and job_status.total_epochs:
        info += f"[cyan]Epoch:[/cyan] {job_status.current_epoch}/{job_status.total_epochs}\n"

    if job_status.metrics:
        metrics_str: str = ", ".join(
            [f"{k}: {v:.4f}" for k, v in job_status.metrics.items()]
        )
        info += f"[cyan]Metrics:[/cyan] {metrics_str}\n"

    if job_status.error_message:
        info += f"[red]Error:[/red] {job_status.error_message}\n"

    panel: Panel = Panel.fit(info.strip(), title="Job Status", border_style=color)
    console.print(panel)

    if job_status.logs:
        console.print("\n[dim]Recent logs:[/dim]")
        for log_line in job_status.logs[-5:]:  # Show last 5 log lines
            if log_line.strip():
                console.print(f"  {log_line}")


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
