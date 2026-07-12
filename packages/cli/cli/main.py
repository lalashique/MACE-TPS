from __future__ import annotations

from pathlib import Path

import click

from core.gates import WaitingForApproval
from core.schemas import WorkflowConfig
from core.state import WorkflowState
from workflow.engine import WorkflowEngine


@click.group()
def cli():
    """mtps - drive the MACE-TPS generational active-learning workflow."""


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--max-steps", default=10_000, help="Safety cap on stage transitions per invocation.")
def run(config_path: Path, max_steps: int):
    """Advance the workflow as far as it will go without hitting a manual gate."""
    config = WorkflowConfig.from_json(config_path)
    engine = WorkflowEngine(config)

    try:
        for _ in range(max_steps):
            if engine.is_done():
                click.echo(f"All {config.max_generations} generations complete.")
                return
            record = engine.step()
            click.echo(f"gen {engine.state.current_generation}: {record.stage.value}")
    except WaitingForApproval as e:
        click.echo(f"\nWaiting for approval: [{e.gate_id}]")
        click.echo(e.instructions)
        click.echo(f"\nOnce done, run: mtps approve --config {config_path} {e.gate_id}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
def status(config_path: Path):
    """Show the current state of every generation."""
    config = WorkflowConfig.from_json(config_path)
    state = WorkflowState.load_or_create(config.root_dir)
    click.echo(f"current_generation: {state.current_generation}/{config.max_generations}")
    for record in state.generations:
        flag = " [BLOCKED]" if record.blocked else ""
        click.echo(f"  gen {record.index}: {record.stage.value}{flag}")
        if record.blocked:
            click.echo(f"    gate: {record.pending_gate}")
            click.echo(f"    instructions: {record.pending_instructions}")
        if record.error:
            click.echo(f"    error: {record.error}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.argument("gate_id")
def approve(config_path: Path, gate_id: str):
    """Approve a gate that `mtps run` reported as blocking, then resume with `mtps run`."""
    config = WorkflowConfig.from_json(config_path)
    engine = WorkflowEngine(config)
    engine.gates.approve(gate_id)
    click.echo(f"Approved {gate_id}. Run `mtps run --config {config_path}` to continue.")


if __name__ == "__main__":
    cli()
