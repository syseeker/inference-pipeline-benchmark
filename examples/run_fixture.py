"""Run a smoke-test fixture end-to-end against a live (or stubbed) backend.

    # offline (stub reasoner) — works without credentials
    python -m examples.run_fixture 01_click_start_button

    # live NIM
    NIM_API_KEY=... python -m examples.run_fixture 02_dismiss_update_popup --backend nim

    # list available fixtures
    python -m examples.run_fixture --list

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

from tests.smoke.fixtures.loader import list_fixtures, load_fixture
from vlm_pipeline import Pipeline
from vlm_pipeline.config import PipelineConfig
from vlm_pipeline.schemas import ModelMeta

app = typer.Typer(add_completion=False)
console = Console()


class _StubReasoner:
    """Echoes the fixture's gold output (offline default)."""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def generate(self, **_: object) -> tuple[str, ModelMeta, float | None]:
        return self._raw, ModelMeta(framework="stub", model_id="gold"), None


@app.command()
def main(
    fixture: str = typer.Argument(None, help="Fixture name, e.g. 01_click_start_button"),
    backend: str = typer.Option("stub", help="stub | nim"),
    list_fx: bool = typer.Option(False, "--list", help="List available fixtures and exit"),
) -> None:
    if list_fx:
        for name in list_fixtures():
            console.print(f"- {name}")
        return

    if not fixture:
        console.print("[red]fixture name required (or pass --list)[/red]")
        raise typer.Exit(2)

    fx = load_fixture(fixture)
    cfg = PipelineConfig.from_env()

    if backend == "nim":
        if not cfg.nim.api_key:
            console.print("[red]NIM_API_KEY not set[/red]")
            raise typer.Exit(2)
        from vlm_pipeline.reasoners.nim_qwen_vl import NimQwenVlReasoner

        reasoner = NimQwenVlReasoner(cfg.nim)
    elif backend == "stub":
        reasoner = _StubReasoner(fx.expected.actions.model_dump_json())
    else:
        console.print(f"[red]unknown backend: {backend}[/red]")
        raise typer.Exit(2)

    pipe = Pipeline(reasoner=reasoner, config=cfg)
    console.rule(f"[bold]{fx.name}[/bold]  ({backend})")
    console.print(f"[dim]{fx.input.description}[/dim]")
    console.print(f"[bold]instruction:[/bold] {fx.input.instruction}")
    if fx.input.context_history:
        console.print("[bold]history:[/bold]")
        for turn in fx.input.context_history:
            console.print(f"  [{turn.role}] {turn.text}")

    resp = pipe.run(fx.request)

    console.print()
    console.print("[bold]GOLD actions:[/bold]")
    console.print(json.dumps(fx.expected.actions.model_dump(), indent=2))
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
