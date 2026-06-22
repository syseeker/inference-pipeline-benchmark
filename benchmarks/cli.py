"""`bench` — one CLI for the inference-pipeline-benchmark harness.

Wraps the existing scripts (gpu_probe.sh, build_nitrogen_scenarios.py,
run_all_scenarios.sh, benchmarks.runner, benchmarks.summary) behind a stable
command surface so agents (Claude Code skills, Codex AGENTS.md, Cursor rules)
have one thing to call instead of memorising five entry points.

Every command supports `--json`. With it, the only thing on stdout is one
line of structured status:

    {
      "status": "ok" | "error" | "skipped",
      "command": "probe" | "setup" | ...,
      "next_action": "human hint or null",
      "artifacts": [paths the command wrote/used],
      "error": {"code": int, "remediation": str} | null,
      "data": { ...command-specific... }
    }

Exit codes the agent can branch on:
    0  ok
    1  generic error
    2  unsupported combo (per yaml `unsupported_backends`)
    3  runtime incompat / OOM / server crash
    4  missing credential / license / dependency

Human (non-json) mode prints the same information as colour-free text and
is the default. The underlying scripts keep working unchanged — this is
strictly a wrapper.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_UNSUPPORTED = 2
EXIT_RUNTIME = 3
EXIT_MISSING_DEP = 4

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inference-pipeline-benchmark CLI. Subcommands: probe, setup, scenarios, smoke, sweep, summary.",
)
scenarios_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build/list benchmark scenarios for any registered source.",
)
app.add_typer(scenarios_app, name="scenarios")


# --------------------------------------------------------------------------- #
# JSON / text status emitter                                                  #
# --------------------------------------------------------------------------- #


def emit(
    *,
    command: str,
    status: str,
    data: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    next_action: str | None = None,
    error: dict[str, Any] | None = None,
    json_out: bool = False,
    exit_code: int = EXIT_OK,
) -> None:
    """Print the structured status line and exit with `exit_code`.

    In JSON mode the status line is the ONLY thing on stdout — text/log
    output from wrapped tools should go to stderr (or be captured via the
    `data` field). In text mode the status is printed as a short banner.
    """
    payload = {
        "status": status,
        "command": command,
        "next_action": next_action,
        "artifacts": artifacts or [],
        "error": error,
        "data": data or {},
    }
    if json_out:
        typer.echo(json.dumps(payload, default=str))
    else:
        if status == "ok":
            typer.echo(f"[ok] {command}: {next_action or 'done'}")
        elif status == "skipped":
            typer.echo(f"[skip] {command}: {next_action or '-'}")
        else:
            msg = error.get("remediation", "see logs") if error else "see logs"
            typer.echo(f"[error] {command}: {msg}", err=True)
        if artifacts:
            for p in artifacts:
                typer.echo(f"  - {p}")
    raise typer.Exit(exit_code)


def _run(
    cmd: list[str],
    *,
    capture: bool = False,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess. Inherits stdio by default so the user sees progress
    live; `capture=True` collects both streams as text (used by `--json`
    callers that need structured output to be the only thing on stdout)."""
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env={**os.environ, **(env or {})},
        check=False,
        capture_output=capture,
        text=True,
    )


# --------------------------------------------------------------------------- #
# bench probe                                                                 #
# --------------------------------------------------------------------------- #


@app.command(help="Probe GPU + driver + per-venv backend versions. Wraps scripts/gpu_probe.sh.")
def probe(
    out: Path = typer.Option(
        None, help="Output path for the host JSON. Default: benchmarks/results/host_<hostname>.json."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit one-line JSON status to stdout."),
) -> None:
    script = REPO_ROOT / "scripts" / "gpu_probe.sh"
    if not script.exists():
        emit(
            command="probe",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": f"missing {script}"},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )
    out_path = Path(out) if out else (RESULTS_ROOT / f"host_{os.uname().nodename}.json")
    cmd = ["bash", str(script), str(out_path)]
    res = _run(cmd, capture=json_out)
    if res.returncode != 0:
        emit(
            command="probe",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": (res.stderr or "gpu_probe.sh failed").strip()[:400]},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )
    data: dict[str, Any] = {}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            pass
    emit(
        command="probe",
        status="ok",
        data=data,
        artifacts=[str(out_path)],
        next_action=f"GPU={data.get('gpu', 'unknown')} driver={data.get('driver', 'unknown')}",
        json_out=json_out,
    )


# --------------------------------------------------------------------------- #
# bench setup --backend <name>                                                #
# --------------------------------------------------------------------------- #

_BACKEND_EXTRAS = {
    "vllm": ["vllm", "dev"],
    "sglang": ["sglang", "dev"],
    "trtllm": ["dev"],          # tensorrt-llm wheel installed separately (NVIDIA index)
    "nitrogen": ["nitrogen", "dataset", "dev"],
    "nim": ["nim", "dev"],
}


@app.command(help="Idempotent per-backend venv + dependency install.")
def setup(
    backend: str = typer.Option(..., help=f"One of: {', '.join(_BACKEND_EXTRAS)}"),
    force: bool = typer.Option(False, help="Recreate the venv even if present."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    if backend not in _BACKEND_EXTRAS:
        emit(
            command="setup",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": f"unknown backend {backend!r}; expected one of {sorted(_BACKEND_EXTRAS)}"},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )
    venv = REPO_ROOT / f".venv-{backend}"
    artifacts: list[str] = [str(venv)]
    extras = _BACKEND_EXTRAS[backend]
    extra_spec = ".[" + ",".join(extras) + "]"

    if venv.exists() and not force:
        emit(
            command="setup",
            status="skipped",
            artifacts=artifacts,
            next_action=f"venv exists at {venv}; pass --force to rebuild",
            data={"backend": backend, "venv": str(venv), "extras": extras},
            json_out=json_out,
        )

    if force and venv.exists():
        import shutil

        shutil.rmtree(venv)

    # Create venv and install harness extras. The trtllm wheel comes from
    # NVIDIA's index, not regular PyPI — caller is responsible for that
    # second pip install per INFERENCE_BACKENDS.md. The nitrogen package
    # comes from `pip install -e ../NitroGen`, also a separate step.
    steps = [
        ["python3", "-m", "venv", str(venv)],
        [str(venv / "bin" / "pip"), "install", "--upgrade", "pip"],
        [str(venv / "bin" / "pip"), "install", "-e", extra_spec],
    ]
    for cmd in steps:
        res = _run(cmd, capture=json_out)
        if res.returncode != 0:
            emit(
                command="setup",
                status="error",
                error={
                    "code": EXIT_MISSING_DEP,
                    "remediation": (res.stderr or f"step failed: {' '.join(cmd)}").strip()[:500],
                },
                artifacts=artifacts,
                json_out=json_out,
                exit_code=EXIT_MISSING_DEP,
            )

    next_hint = {
        "trtllm": "now install: source .venv-trtllm/bin/activate && pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com",
        "nitrogen": "now install NitroGen: pip install -e ../NitroGen (and hf download nvidia/NitroGen ng.pt)",
    }.get(backend, f"venv ready at {venv}")

    emit(
        command="setup",
        status="ok",
        artifacts=artifacts,
        next_action=next_hint,
        data={"backend": backend, "venv": str(venv), "extras": extras},
        json_out=json_out,
    )


# --------------------------------------------------------------------------- #
# bench scenarios build / list                                                #
# --------------------------------------------------------------------------- #


@scenarios_app.command("build", help="Build scenarios from a registered source.")
def scenarios_build(
    source: str = typer.Option("nitrogen", help="Source name from the pipeline_bench.scenario_sources entry-point group. Built-in: nitrogen. Customer sources are auto-discovered."),
    n: int = typer.Option(3, help="Number of scenarios to produce."),
    out: Path = typer.Option(REPO_ROOT / "tests" / "smoke" / "scenarios_nitrogen", help="Output directory."),
    actions_root: Path = typer.Option(
        None, help="For source=nitrogen: path to the extracted actions/ tree. Required unless $NITROGEN_ACTIONS_ROOT is set."
    ),
    synthetic_frames: bool = typer.Option(
        False,
        "--synthetic-frames",
        help="Skip yt-dlp video fetch; emit placeholder frames. Use on cloud IPs where source videos are unreachable.",
    ),
    game_mapping: Path = typer.Option(None, help="Optional ng.pt / .json / .parquet mapping; usually omit for the released unconditional checkpoint."),
    cache_dir: Path = typer.Option(REPO_ROOT / ".cache" / "nitrogen_videos", help="Video-download cache."),
    deadline_ms: int = typer.Option(1500),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from benchmarks.sources import discover, get

    try:
        builder = get(source)
    except KeyError as e:
        emit(
            command="scenarios.build",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": str(e)},
            data={"registered_sources": sorted(discover())},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )

    try:
        count = builder(
            n=n,
            out=out,
            actions_root=actions_root,
            synthetic_frames=synthetic_frames,
            game_mapping=game_mapping,
            cache_dir=cache_dir,
            deadline_ms=deadline_ms,
        )
    except FileNotFoundError as e:
        emit(
            command="scenarios.build",
            status="error",
            error={"code": EXIT_MISSING_DEP, "remediation": str(e)},
            json_out=json_out,
            exit_code=EXIT_MISSING_DEP,
        )
    except Exception as e:
        emit(
            command="scenarios.build",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": str(e)[:500]},
            artifacts=[str(out)] if out.exists() else [],
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )

    built = sorted(p.name for p in out.iterdir() if p.is_dir() and (p / "request.json").exists()) if out.exists() else []
    emit(
        command="scenarios.build",
        status="ok",
        artifacts=[str(out)],
        next_action=f"built {count}/{n} scenarios; run: bench smoke --backend nitrogen-eager --model nitrogen-500m-bf16 --scenarios-dir {out}",
        data={"source": source, "out_dir": str(out), "count": count, "scenarios": built},
        json_out=json_out,
    )


@scenarios_app.command("sources", help="List registered scenario sources (built-in + customer entry-points).")
def scenarios_sources(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from benchmarks.sources import discover

    sources = sorted(discover())
    emit(
        command="scenarios.sources",
        status="ok",
        next_action=f"{len(sources)} source(s) registered",
        data={"sources": sources},
        json_out=json_out,
    )


@scenarios_app.command("list", help="List scenarios under a directory.")
def scenarios_list(
    scenarios_dir: Path = typer.Option(REPO_ROOT / "tests" / "smoke" / "scenarios", help="Scenario root."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    if not scenarios_dir.exists():
        emit(
            command="scenarios.list",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": f"not a directory: {scenarios_dir}"},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )
    entries = []
    for p in sorted(scenarios_dir.iterdir()):
        if not p.is_dir() or not (p / "request.json").exists():
            continue
        entries.append(
            {
                "name": p.name,
                "has_expected": (p / "expected.json").exists(),
                "has_gold_action": (p / "gold_action.json").exists(),
            }
        )
    emit(
        command="scenarios.list",
        status="ok",
        artifacts=[str(scenarios_dir)],
        next_action=f"{len(entries)} scenarios under {scenarios_dir}",
        data={"scenarios_dir": str(scenarios_dir), "scenarios": entries},
        json_out=json_out,
    )


# --------------------------------------------------------------------------- #
# bench sweep / smoke / summary                                               #
# --------------------------------------------------------------------------- #


@app.command(help="Run the sweep orchestrator. Wraps scripts/run_all_scenarios.sh.")
def sweep(
    gpu: str = typer.Option(..., help="GPU profile under benchmarks/configs/."),
    sweep_name: str = typer.Option(None, "--sweep", help="Named sweep from the yaml (e.g. nitrogen-backends)."),
    backends: str = typer.Option(None, help="Quoted, space-separated backend whitelist (e.g. 'nitrogen-eager nitrogen-compile')."),
    model: str = typer.Option(None, help="Override single-round model id."),
    variants: str = typer.Option(None, help="Variant set (e.g. 'baseline eager')."),
    scenarios_dir: Path = typer.Option(None, help="Override scenarios source dir."),
    nitrogen_ckpt_path: Path = typer.Option(None, help="Pre-downloaded ng.pt path (exports NITROGEN_CKPT_PATH for the run)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    script = REPO_ROOT / "scripts" / "run_all_scenarios.sh"
    cmd = ["bash", str(script), "--gpu", gpu]
    if sweep_name:
        cmd += ["--sweep", sweep_name]
    if backends:
        cmd += ["--backends", backends]
    if model:
        cmd += ["--model", model]
    if variants:
        cmd += ["--variants", variants]
    if scenarios_dir:
        cmd += ["--scenarios-dir", str(scenarios_dir)]

    env: dict[str, str] = {}
    if nitrogen_ckpt_path:
        env["NITROGEN_CKPT_PATH"] = str(nitrogen_ckpt_path)

    res = _run(cmd, capture=json_out, env=env)
    summary_path = RESULTS_ROOT / gpu / "summary.md"
    if res.returncode != 0:
        emit(
            command="sweep",
            status="error",
            error={
                "code": EXIT_RUNTIME,
                "remediation": (res.stderr or "run_all_scenarios.sh failed; see benchmarks/results/<gpu>/server-logs/").strip()[:500],
            },
            artifacts=[str(summary_path)] if summary_path.exists() else [],
            json_out=json_out,
            exit_code=EXIT_RUNTIME,
        )
    emit(
        command="sweep",
        status="ok",
        artifacts=[str(summary_path)] if summary_path.exists() else [],
        next_action=f"bench summary --gpu {gpu}   # or read {summary_path}",
        data={"gpu": gpu, "sweep": sweep_name, "backends": backends},
        json_out=json_out,
    )


@app.command(help="Regenerate summary.md for a GPU from existing per-backend result JSONs.")
def summary(
    gpu: str = typer.Option(..., help="GPU profile under benchmarks/configs/."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    cmd = [sys.executable, "-m", "benchmarks.summary", "--gpu", gpu]
    res = _run(cmd, capture=json_out)
    summary_path = RESULTS_ROOT / gpu / "summary.md"
    if res.returncode != 0:
        emit(
            command="summary",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": (res.stderr or "summary regen failed").strip()[:400]},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )
    emit(
        command="summary",
        status="ok",
        artifacts=[str(summary_path)] if summary_path.exists() else [],
        next_action=f"read {summary_path}",
        data={"gpu": gpu, "summary_path": str(summary_path)},
        json_out=json_out,
    )


@app.command(help="Run one scenario end-to-end against a backend (single round).")
def smoke(
    gpu: str = typer.Option(..., help="GPU profile under benchmarks/configs/."),
    backend: str = typer.Option(..., help="Backend id from the GPU yaml (e.g. nitrogen-eager)."),
    model: str = typer.Option(None, help="Model id from the GPU yaml. Defaults to the yaml's default_model."),
    scenarios_dir: Path = typer.Option(None),
    nitrogen_ckpt_path: Path = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Smoke is implemented as a one-round sweep — same launch path, same
    cleanup, same outputs. Lets agents validate a (backend, model) combo
    works on this GPU before kicking off the full sweep."""
    script = REPO_ROOT / "scripts" / "run_all_scenarios.sh"
    cmd = ["bash", str(script), "--gpu", gpu, "--backends", backend]
    if model:
        cmd += ["--model", model]
    if scenarios_dir:
        cmd += ["--scenarios-dir", str(scenarios_dir)]

    env: dict[str, str] = {}
    if nitrogen_ckpt_path:
        env["NITROGEN_CKPT_PATH"] = str(nitrogen_ckpt_path)

    res = _run(cmd, capture=json_out, env=env)
    summary_path = RESULTS_ROOT / gpu / "summary.md"
    if res.returncode != 0:
        emit(
            command="smoke",
            status="error",
            error={
                "code": EXIT_RUNTIME,
                "remediation": (res.stderr or "smoke run failed; check benchmarks/results/<gpu>/server-logs/").strip()[:500],
            },
            artifacts=[str(summary_path)] if summary_path.exists() else [],
            json_out=json_out,
            exit_code=EXIT_RUNTIME,
        )
    emit(
        command="smoke",
        status="ok",
        artifacts=[str(summary_path)] if summary_path.exists() else [],
        next_action=f"smoke ok — now run: bench sweep --gpu {gpu} --sweep <sweep-name>",
        data={"gpu": gpu, "backend": backend, "model": model},
        json_out=json_out,
    )


if __name__ == "__main__":
    app()
