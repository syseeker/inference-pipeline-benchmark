"""Run a smoke-test scenario end-to-end against a live (or stubbed) backend.

    # offline (stub reasoner) — works without credentials
    python -m examples.run_scenario 01_clash_of_clans_start_attack

    # live NIM
    NIM_API_KEY=... python -m examples.run_scenario 02_catan_open_menu --backend nim

    # list available scenarios
    python -m examples.run_scenario --list

Prints the actual ActionSequence next to the gold one, plus the
validator verdict and per-stage latency. Useful for eyeballing model
behaviour on a known visual without spinning up the full benchmark
runner.
"""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from tests.smoke.scenarios.loader import list_scenarios, load_scenario
from vlm_pipeline import Pipeline
from vlm_pipeline.config import PipelineConfig
from vlm_pipeline.schemas import ModelMeta

app = typer.Typer(add_completion=False)
console = Console()


class _StubReasoner:
    """Echoes the scenario's gold output (offline default)."""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def generate(self, **_: object) -> tuple[str, ModelMeta, float | None]:
        return self._raw, ModelMeta(framework="stub", model_id="gold"), None


@app.command()
def main(
    scenario: str = typer.Argument(None, help="Scenario name, e.g. 01_clash_of_clans_start_attack"),
    backend: str = typer.Option("stub", help="stub | nim"),
    list_sc: bool = typer.Option(False, "--list", help="List available scenarios and exit"),
) -> None:
    if list_sc:
        for name in list_scenarios():
            console.print(f"- {name}")
        return

    if not scenario:
        console.print("[red]scenario name required (or pass --list)[/red]")
        raise typer.Exit(2)

    sc = load_scenario(scenario)
    cfg = PipelineConfig.from_env()

    if backend == "nim":
        if not cfg.nim.api_key:
            console.print("[red]NIM_API_KEY not set[/red]")
            raise typer.Exit(2)
        from vlm_pipeline.reasoners.nim_qwen_vl import NimQwenVlReasoner

        reasoner = NimQwenVlReasoner(cfg.nim)
    elif backend == "stub":
        reasoner = _StubReasoner(sc.expected.actions.model_dump_json())
    else:
        console.print(f"[red]unknown backend: {backend}[/red]")
        raise typer.Exit(2)

    pipe = Pipeline(reasoner=reasoner, config=cfg)
    console.rule(f"[bold]{sc.name}[/bold]  ({backend})")
    console.print(f"[dim]{sc.spec.description}[/dim]")
    console.print(f"[bold]instruction:[/bold] {sc.spec.instruction}")
    if sc.spec.context_history:
        console.print("[bold]history:[/bold]")
        for turn in sc.spec.context_history:
            console.print(f"  [{turn.role}] {turn.text}")

    resp = pipe.run(sc.pipeline_request())

    console.print()
    console.print("[bold]GOLD actions:[/bold]")
    console.print(json.dumps(sc.expected.actions.model_dump(), indent=2))
    console.print("[bold]ACTUAL actions:[/bold]")
    console.print(
        json.dumps(resp.actions.model_dump(), indent=2) if resp.actions else "<none>"
    )

    table = Table(title="Verdict")
    table.add_column("field")
    table.add_column("value")
    table.add_row("schema_valid", str(resp.validation.schema_valid))
    table.add_row("safe", str(resp.validation.safe))
    table.add_row("was_executed", str(resp.was_executed))
    table.add_row("error", resp.error or "-")
    table.add_row("ttft_ms", _fmt(resp.latency.reasoner_ttft_ms))
    table.add_row("reasoner_total_ms", _fmt(resp.latency.reasoner_total_ms))
    table.add_row("validator_ms", _fmt(resp.latency.validator_ms))
    table.add_row("total_ms", _fmt(resp.latency.total_ms))
    console.print(table)


def _fmt(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "-"


if __name__ == "__main__":
    sys.exit(app() or 0)
