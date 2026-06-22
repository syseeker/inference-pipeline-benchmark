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
    # AIPerf bundled into the HTTP-backend extras so `bench load-test`
    # (PR #6) Just Works against any venv `bench setup` produced.
    "vllm":   ["vllm", "aiperf", "dev"],
    "sglang": ["sglang", "aiperf", "dev"],
    "trtllm": ["aiperf", "dev"],   # tensorrt-llm wheel installed separately (NVIDIA index)
    "nim":    ["nim", "aiperf", "dev"],
    # NitroGen is ZMQ + single-flight today — no AIPerf install needed.
    "nitrogen": ["nitrogen", "dataset", "dev"],
    "nitrogen-quant": ["nitrogen", "nitrogen-quant", "dataset", "dev"],
}


@app.command(help="Idempotent per-backend venv + dependency install. The special backend 'profile' installs the Nsight Systems CLI for `bench profile`.")
def setup(
    backend: str = typer.Option(..., help=f"One of: {', '.join(list(_BACKEND_EXTRAS) + ['profile'])}"),
    force: bool = typer.Option(False, help="Recreate the venv even if present."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    # 'profile' isn't a venv backend — it's a system-tooling installer
    # for `bench profile` (PR #7). Bundling nsys into every backend's
    # extras would tax customers who never escalate to profiling, so we
    # gate it behind explicit `bench setup --backend profile`. The
    # `bench profile` missing-nsys path tells the customer exactly this.
    if backend == "profile":
        _setup_profiler(force=force, json_out=json_out)
        return  # _setup_profiler always emits + exits

    if backend not in _BACKEND_EXTRAS:
        emit(
            command="setup",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": f"unknown backend {backend!r}; expected one of {sorted(list(_BACKEND_EXTRAS) + ['profile'])}"},
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


# --------------------------------------------------------------------------- #
# bench load-test (PR #6 — AIPerf concurrency sweeps for HTTP backends)       #
# --------------------------------------------------------------------------- #


def _aiperf_endpoint_type_for(backend: str) -> str:
    """Map our backend names to AIPerf's --endpoint-type values.

    HTTP backends all speak the OpenAI Chat Completions surface. NitroGen
    speaks ZMQ and is single-flight by server design (see PR #6 SKILL
    doc), so we refuse it here rather than silently fall back.
    """
    if backend in ("vllm", "sglang", "trtllm", "nim"):
        return "chat"
    raise ValueError(
        f"backend {backend!r} has no AIPerf endpoint type. "
        "NitroGen rounds are ZMQ + single-flight; for concurrency curves on "
        "policy models see PR #8 (replicate-per-GPU). Today, NitroGen latency "
        "at concurrency=1 comes from `bench sweep`."
    )


@app.command(
    "load-test",
    help="AIPerf concurrency sweep against a running HTTP backend (vLLM / SGLang / TRT-LLM).",
)
def load_test(
    gpu: str = typer.Option(..., help="GPU profile under benchmarks/configs/ — used to resolve `base_url` for the named backend."),
    backend: str = typer.Option(..., help="HTTP backend name from the GPU yaml (vllm | sglang | trtllm | nim)."),
    model: str = typer.Option(..., help="Served model identifier — passed to AIPerf as `--model`."),
    concurrency: str = typer.Option(
        "1,4,16,32",
        help="Comma-separated concurrency levels to sweep. Each becomes one AIPerf phase; results combine into a single concurrency curve.",
    ),
    warmup: int = typer.Option(10, help="Warmup requests per phase (AIPerf rejects warmup=0)."),
    request_count: int = typer.Option(200, help="Measured requests per phase."),
    duration: int = typer.Option(0, help="If > 0, use a wall-clock budget (seconds) instead of `--request-count`."),
    artifact_dir: Path = typer.Option(None, help="Output dir for AIPerf artifacts. Default: benchmarks/results/<gpu>/aiperf/<backend>-<model>-<ts>/."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run AIPerf against an already-running HTTP server.

    Does NOT launch the server — point at one started by `bench smoke`
    or `bench sweep`. AIPerf writes profile_export_aiperf.{csv,json}
    under `benchmarks/results/<gpu>/aiperf/...`; the summary generator
    picks them up via the new "Concurrency profile" section.

    NitroGen: refused at the input stage with a pointer to PR #8.
    """
    if backend.startswith("nitrogen"):
        emit(
            command="load-test",
            status="error",
            error={
                "code": EXIT_UNSUPPORTED,
                "remediation": (
                    "NitroGen runs as a single-flight ZMQ server (REP socket). "
                    "AIPerf concurrency sweeps need a multi-flight endpoint — "
                    "see PR #8 (replicate-per-GPU). Today, NitroGen latency at "
                    "concurrency=1 comes from `bench sweep`."
                ),
            },
            json_out=json_out,
            exit_code=EXIT_UNSUPPORTED,
        )

    try:
        endpoint_type = _aiperf_endpoint_type_for(backend)
    except ValueError as e:
        emit(
            command="load-test", status="error",
            error={"code": EXIT_GENERIC, "remediation": str(e)},
            json_out=json_out, exit_code=EXIT_GENERIC,
        )

    # Resolve base_url from the GPU yaml so agents don't memorize ports.
    from benchmarks.scenario_config import load_gpu_config

    try:
        cfg = load_gpu_config(gpu)
    except FileNotFoundError as e:
        emit(
            command="load-test", status="error",
            error={"code": EXIT_GENERIC, "remediation": str(e)},
            json_out=json_out, exit_code=EXIT_GENERIC,
        )
    bk = cfg.get("backends", {}).get(backend)
    if not bk:
        emit(
            command="load-test", status="error",
            error={"code": EXIT_GENERIC, "remediation": f"backend {backend!r} not in benchmarks/configs/{gpu}.yaml"},
            json_out=json_out, exit_code=EXIT_GENERIC,
        )
    base_url = str(bk["base_url"]).rstrip("/")
    # AIPerf's --url is the server root (http://host:port), not the
    # endpoint path — strip /v1 if the yaml carries it.
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]

    if artifact_dir is None:
        import time

        ts = time.strftime("%Y%m%dT%H%M%S")
        artifact_dir = RESULTS_ROOT / gpu / "aiperf" / f"{backend}-{model.replace('/', '-')}-{ts}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aiperf", "profile",
        "--url", base_url,
        "--model", model,
        "--endpoint-type", endpoint_type,
        "--concurrency", concurrency,
        "--warmup-request-count", str(warmup),
        "--output-artifact-dir", str(artifact_dir),
    ]
    if duration > 0:
        cmd += ["--benchmark-duration", str(duration)]
    else:
        cmd += ["--request-count", str(request_count)]

    res = _run(cmd, capture=json_out)
    if res.returncode != 0:
        emit(
            command="load-test", status="error",
            error={
                "code": EXIT_RUNTIME,
                "remediation": (res.stderr or f"aiperf failed; see {artifact_dir}").strip()[:500],
            },
            artifacts=[str(artifact_dir)],
            json_out=json_out, exit_code=EXIT_RUNTIME,
        )

    canonical = artifact_dir / "profile_export_aiperf.json"
    emit(
        command="load-test",
        status="ok",
        artifacts=[str(artifact_dir)],
        next_action=(
            f"AIPerf wrote {canonical.name}; re-run `bench summary --gpu {gpu}` "
            "so the Concurrency profile section picks it up."
        ),
        data={
            "gpu": gpu, "backend": backend, "model": model,
            "concurrency_sweep": [int(c) for c in concurrency.split(",")],
            "artifact_dir": str(artifact_dir),
        },
        json_out=json_out,
    )


# --------------------------------------------------------------------------- #
# bench setup --backend profile (PR #7.1)                                     #
# --------------------------------------------------------------------------- #


def _setup_profiler(*, force: bool, json_out: bool) -> None:
    """Install nsight-systems-cli so `bench profile --tool nsys` works.

    Strategy:
      1. If `nsys` is already on PATH → skipped (cached).
      2. Else try `sudo apt-get install -y nsight-systems-cli` — works on
         the standard Ubuntu image we test against; needs sudo. If sudo
         is unavailable / blocked we surface the exact tarball-download
         alternative.
      3. ncu (Nsight Compute) ships with the CUDA Toolkit; we don't
         install it here — but we report whether it's available so the
         customer knows the kernel-deep-dive path is ready too.
    """
    import shutil

    nsys_path = shutil.which("nsys")
    ncu_path = shutil.which("ncu")

    if nsys_path and not force:
        emit(
            command="setup",
            status="skipped",
            artifacts=[nsys_path] + ([ncu_path] if ncu_path else []),
            next_action=(
                f"nsys already at {nsys_path}; pass --force to reinstall. "
                f"Run: bench profile --tool nsys --gpu <gpu> --backend <bk> --model <m>"
            ),
            data={"nsys": nsys_path, "ncu": ncu_path},
            json_out=json_out,
        )

    # Try the apt path. If `sudo` isn't on PATH or returns non-zero, fall
    # back to the tarball-download hint — we don't try to silently mutate
    # /opt without permission.
    if shutil.which("sudo") and shutil.which("apt-get") and shutil.which("apt-cache"):
        # NVIDIA's CUDA apt repo ships versioned packages like
        # `nsight-systems-2026.1.3`, not the generic `nsight-systems-cli`
        # most docs reference. Discover the latest available so the
        # customer doesn't have to know the exact version string.
        pkg_name = _latest_nsight_systems_apt_pkg()
        if not pkg_name:
            emit(
                command="setup",
                status="error",
                error={
                    "code": EXIT_MISSING_DEP,
                    "remediation": (
                        "apt-cache returned no `nsight-systems-*` packages — the NVIDIA CUDA "
                        "apt repo may not be configured. Add it:\n"
                        "  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \\\n"
                        "    -o /tmp/cuda-keyring.deb && sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update\n"
                        "Then re-run: bench setup --backend profile"
                    ),
                },
                data={"nsys": None, "ncu": ncu_path},
                json_out=json_out,
                exit_code=EXIT_MISSING_DEP,
            )
        cmd = ["sudo", "apt-get", "install", "-y", pkg_name]
        print(f">> {' '.join(cmd)}")
        res = _run(cmd, capture=json_out)
        # NVIDIA's deb package installs to /opt/nvidia/nsight-systems/<ver>/
        # with root-only perms (no `other` read+exec). Re-open it for the
        # non-root user that will run `bench profile`. We do this after
        # every apt path, idempotent — no-op if the dir doesn't exist.
        if res.returncode == 0:
            chmod_cmd = ["sudo", "chmod", "-R", "o+rX", "/opt/nvidia/nsight-systems"]
            print(f">> {' '.join(chmod_cmd)}  (open install dir for non-root use)")
            _run(chmod_cmd, capture=json_out)
            # Also drop a /usr/local/bin/nsys symlink so non-login shells
            # find it without an explicit PATH munge.
            link_target = subprocess.run(  # noqa: S603
                ["bash", "-lc", "ls -1d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if link_target:
                _run(["sudo", "ln", "-sf", link_target, "/usr/local/bin/nsys"], capture=json_out)
        if res.returncode == 0 and shutil.which("nsys"):
            nsys_path = shutil.which("nsys")
            emit(
                command="setup",
                status="ok",
                artifacts=[nsys_path] + ([ncu_path] if ncu_path else []),
                next_action=(
                    f"nsys installed at {nsys_path}. "
                    f"Run: bench profile --tool nsys --gpu <gpu> --backend <bk> --model <m>"
                ),
                data={"nsys": nsys_path, "ncu": ncu_path, "via": "apt"},
                json_out=json_out,
            )
        # apt failed (no sudo, no network, missing repo). Surface the
        # remediation rather than fail silently. Common reasons: sudo
        # prompted for a password we couldn't supply, or the
        # nsight-systems-cli package isn't in the configured apt sources.
        manual_hint = _profiler_manual_install_hint()
        emit(
            command="setup",
            status="error",
            error={
                "code": EXIT_MISSING_DEP,
                "remediation": (
                    "Tried `sudo apt-get install nsight-systems-cli` and it failed. "
                    f"apt output (last lines): {(res.stderr or res.stdout or '').strip()[-400:]}\n\n"
                    f"Manual install:\n{manual_hint}"
                ),
            },
            data={"nsys": None, "ncu": ncu_path},
            json_out=json_out,
            exit_code=EXIT_MISSING_DEP,
        )

    # No sudo / no apt → tarball-only path. Print the URL + extract steps.
    emit(
        command="setup",
        status="error",
        error={
            "code": EXIT_MISSING_DEP,
            "remediation": (
                "Cannot run apt-get (sudo not available / apt not on PATH). "
                f"Manual install required:\n{_profiler_manual_install_hint()}"
            ),
        },
        data={"nsys": None, "ncu": ncu_path},
        json_out=json_out,
        exit_code=EXIT_MISSING_DEP,
    )


def _latest_nsight_systems_apt_pkg() -> str | None:
    """Find the newest `nsight-systems-<version>` package in apt-cache.

    NVIDIA's CUDA repo ships versioned packages (e.g. 2025.1.3, 2025.3.2,
    2026.1.3) rather than the generic `nsight-systems-cli`. We pick the
    lexicographically-greatest version — version strings sort right
    here because they're date-based (YYYY.X.Y).
    """
    import re

    res = subprocess.run(  # noqa: S603
        ["apt-cache", "search", "-n", "^nsight-systems-20"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return None
    pkgs: list[str] = []
    for line in res.stdout.splitlines():
        m = re.match(r"^(nsight-systems-\d+\.\d+\.\d+)\s", line)
        if m:
            pkgs.append(m.group(1))
    return max(pkgs) if pkgs else None


def _profiler_manual_install_hint() -> str:
    """One-stop install instructions for nsight-systems-cli, no sudo needed."""
    return (
        "  1. Sign in to https://developer.nvidia.com/nsight-systems/get-started\n"
        "  2. Download the linux-x86_64 .tar.gz for your CUDA version\n"
        "  3. Extract to /opt/nvidia/nsight-systems/:\n"
        "       tar -xf nsight-systems-*.tar.gz -C /opt/nvidia/\n"
        "  4. Add to PATH (in ~/.bashrc):\n"
        "       export PATH=/opt/nvidia/nsight-systems/<version>/bin:$PATH\n"
        "  5. Verify: nsys --version\n"
        "  6. Re-run: bench setup --backend profile"
    )


# --------------------------------------------------------------------------- #
# bench profile (PR #7 — Nsight Systems / Compute opt-in profiling)           #
# --------------------------------------------------------------------------- #
#
# Wraps `nsys profile` (timeline) or `ncu --launch-replay` (kernel-level)
# around a single sweep round so the customer can confirm what summary.md
# only inferred. Used when section 5 flags "DRAM_ACTIVE low + e2e high"
# (launch-bound) or section 2 shows TTFT dominating e2e (capture/JIT
# overhead) — the agent surfaces this command so the customer can SEE
# the gaps in a Nsight timeline rather than trust an inferred label.
#
# Not auto-run on every sweep: nsys adds ~5–10% overhead, ncu serializes
# kernels (10x+ slowdown). Strictly an escalation step.


_PROFILE_TOOLS = ("nsys", "ncu")


def _profile_tool_check(tool: str) -> str | None:
    """Locate `nsys` or `ncu` on PATH. Returns the absolute path, or None
    if not installed (caller surfaces an install hint)."""
    import shutil

    if tool not in _PROFILE_TOOLS:
        raise ValueError(f"unknown profile tool: {tool!r}; expected one of {_PROFILE_TOOLS}")
    return shutil.which(tool)


def _profile_install_hint(tool: str) -> str:
    """Customer-facing install instructions when nsys/ncu is missing."""
    if tool == "nsys":
        return (
            "nsys (Nsight Systems) not on PATH. Run the one-stop installer:\n"
            "    bench setup --backend profile\n"
            "(tries `sudo apt-get install nsight-systems-cli`, falls back to a "
            "manual tarball-install hint when apt isn't available)."
        )
    return (
        "ncu (Nsight Compute) not on PATH — usually ships with the CUDA Toolkit\n"
        "under /usr/local/cuda-*/bin/. Add that directory to PATH, or run:\n"
        "    bench setup --backend profile\n"
        "(also probes for ncu and reports per-tool availability)."
    )


@app.command(
    "profile",
    help="One-shot Nsight profiling pass over a single round (escalation tool for under-performers in summary.md).",
)
def profile(
    tool: str = typer.Option(
        "nsys",
        help="Which Nsight tool: `nsys` (timeline; default, ~5–10% overhead) or `ncu` (kernel deep-dive; serializes kernels, 10x+ slowdown — use only for one or two kernels).",
    ),
    gpu: str = typer.Option(..., help="GPU profile under benchmarks/configs/."),
    backend: str = typer.Option(..., help="Backend id from the yaml (e.g. nitrogen-eager)."),
    model: str = typer.Option(None, help="Model id; defaults to the yaml's default_model."),
    scenarios_dir: Path = typer.Option(None, help="Scenarios to profile. Defaults to the yaml's standard set."),
    nitrogen_ckpt_path: Path = typer.Option(None, help="ng.pt path; needed for nitrogen backends."),
    output: Path = typer.Option(None, help="Output path for the profile report. Default: benchmarks/results/<gpu>/profiles/<backend>-<model>-<ts>.<ext>."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Wrap `nsys profile` (or `ncu`) around a single `bench smoke`-style
    round and write the report under `benchmarks/results/<gpu>/profiles/`.

    The output is a binary report you open in the Nsight Systems / Compute
    desktop UI. For an inline summary the skill points at the auto-emitted
    `<output>.summary.md` (nsys-only — from `nsys stats`).
    """
    tool_path = _profile_tool_check(tool)
    if tool_path is None:
        emit(
            command="profile",
            status="error",
            error={"code": EXIT_MISSING_DEP, "remediation": _profile_install_hint(tool)},
            json_out=json_out,
            exit_code=EXIT_MISSING_DEP,
        )

    import time

    ts = time.strftime("%Y%m%dT%H%M%S")
    profile_dir = RESULTS_ROOT / gpu / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{backend}-{(model or 'default').replace('/', '-')}-{ts}"
    if output is None:
        suffix = ".nsys-rep" if tool == "nsys" else ".ncu-rep"
        output = profile_dir / f"{out_stem}{suffix}"

    # Build the sweep command (single round) that nsys/ncu wraps. We use
    # `bash scripts/run_all_scenarios.sh` for identical setup to the
    # measured sweep, so timeline labels line up with the rows in
    # summary.md. nsys can launch the bash script directly; ncu must
    # attach to a child process — we wrap the same script and let ncu
    # follow forks via --target-processes=all.
    inner_script = REPO_ROOT / "scripts" / "run_all_scenarios.sh"
    inner_cmd = [
        "bash", str(inner_script),
        "--gpu", gpu,
        "--backends", backend,
    ]
    if model:
        inner_cmd += ["--model", model]
    if scenarios_dir:
        inner_cmd += ["--scenarios-dir", str(scenarios_dir)]

    env: dict[str, str] = {}
    if nitrogen_ckpt_path:
        env["NITROGEN_CKPT_PATH"] = str(nitrogen_ckpt_path)

    if tool == "nsys":
        # nsys profile -o <out> <cmd...> — the canonical wrap. We capture
        # NVTX + CUDA + OS signposts; kernel-level data needs ncu.
        cmd = [
            tool_path, "profile",
            "--output", str(output.with_suffix("")),  # nsys appends .nsys-rep
            "--trace=cuda,nvtx,osrt",
            "--force-overwrite=true",
            "--",  # stop nsys flag parsing
            *inner_cmd,
        ]
    else:  # ncu
        cmd = [
            tool_path,
            "--target-processes=all",
            "--export", str(output.with_suffix("")),  # ncu appends .ncu-rep
            "--force-overwrite",
            "--set", "basic",  # minimal metric set; user can re-run with --set full
            "--",
            *inner_cmd,
        ]

    res = _run(cmd, capture=True, env=env)
    report_path = output
    if not report_path.exists():
        # Some versions write .qdrep or different extensions; widen the
        # match so we surface whatever was produced.
        candidates = sorted(profile_dir.glob(f"{out_stem}.*"))
        if candidates:
            report_path = candidates[0]

    if res.returncode != 0:
        # Decode the common known failures into actionable remediation
        # text. ncu writes profiler errors to STDOUT (not stderr), and
        # nsys produces messages on both — scan everything.
        merged = (res.stdout or "") + "\n" + (res.stderr or "")
        remediation: str
        if "ERR_NVGPUCTRPERM" in merged or "permission to access NVIDIA GPU Performance Counters" in merged:
            remediation = (
                f"{tool} can't read GPU performance counters (driver-level permission). Fix: "
                "set `NVreg_RestrictProfilingToAdminUsers=0` and reload the nvidia kernel "
                "module, OR run the bench profile command as root. See "
                "https://developer.nvidia.com/ERR_NVGPUCTRPERM for the full procedure."
            )
        else:
            remediation = (
                merged.strip()[-500:]
                or f"{tool} failed; check the server log under benchmarks/results/{gpu}/server-logs/"
            )
        emit(
            command="profile",
            status="error",
            error={"code": EXIT_RUNTIME, "remediation": remediation},
            artifacts=[str(report_path)] if report_path.exists() else [],
            json_out=json_out,
            exit_code=EXIT_RUNTIME,
        )

    summary_md = _maybe_emit_profile_summary(tool_path, tool, report_path)
    artifacts = [str(report_path)]
    if summary_md and summary_md.exists():
        artifacts.append(str(summary_md))

    emit(
        command="profile",
        status="ok",
        artifacts=artifacts,
        next_action=(
            f"open {report_path.name} in the Nsight {'Systems' if tool == 'nsys' else 'Compute'} UI; "
            + (f"read {summary_md.name} for a text narrative; " if summary_md else "")
            + "the report is GPU-specific — keep it next to the summary.md row it diagnoses."
        ),
        data={
            "tool": tool, "gpu": gpu, "backend": backend, "model": model,
            "report": str(report_path),
            "summary_md": str(summary_md) if summary_md else None,
        },
        json_out=json_out,
    )


def _maybe_emit_profile_summary(tool_path: str, tool: str, report_path: Path) -> Path | None:
    """Best-effort text narrative of the report for agent consumption.

    For nsys: `nsys stats --report cuda_gpu_kern_sum,nvtx_sum` over the
    .nsys-rep gives a kernel-time top-N and NVTX-region timing — enough
    for an agent to say "kernel X took Y% of GPU time, NVTX region Z
    dominates." For ncu: `ncu --import <report>` re-prints the per-kernel
    metric block which is already markdown-shaped.

    Returns the path to the .summary.md or None on failure (we don't
    fail the parent `bench profile` over a missing narrative).
    """
    out = report_path.with_suffix(".summary.md")
    try:
        if tool == "nsys":
            res = subprocess.run(  # noqa: S603
                [tool_path, "stats", "--report", "cuda_gpu_kern_sum,nvtx_sum", "--format=table", str(report_path)],
                capture_output=True, text=True, check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                out.write_text(
                    f"# nsys stats — {report_path.name}\n\n"
                    f"Auto-generated by `bench profile`. The full timeline lives in `{report_path.name}`; open it in Nsight Systems for the visual.\n\n"
                    f"```\n{res.stdout.rstrip()}\n```\n"
                )
                return out
        else:
            res = subprocess.run(  # noqa: S603
                [tool_path, "--import", str(report_path), "--page", "details", "--print-summary", "per-kernel"],
                capture_output=True, text=True, check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                out.write_text(
                    f"# ncu summary — {report_path.name}\n\n"
                    f"Auto-generated by `bench profile --tool ncu`. The per-kernel details live in `{report_path.name}`; open it in Nsight Compute for the metric drill-down.\n\n"
                    f"```\n{res.stdout.rstrip()}\n```\n"
                )
                return out
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# bench install-skill                                                         #
# --------------------------------------------------------------------------- #

# How each agent host discovers its skill/rule files. `dest_fn` returns the
# install path; `source_root` is what we install from.
_AGENT_TARGETS = {
    "claude": {
        # Claude Code reads <project>/.claude/skills/<name>/SKILL.md (project-
        # local) or ~/.claude/skills/<name>/SKILL.md (--user). The repo-root
        # `skills/` is the source layout — install copies/links each subdir.
        "source": REPO_ROOT / "skills",
        "needs_install": True,
    },
    "cursor": {
        # .cursor/rules/<name>.mdc is the file Cursor loads. We render those
        # at repo root from skills/ via scripts/render_agent_docs.py — they
        # ARE the installed form. No copy needed for a project-local clone.
        "source": REPO_ROOT / ".cursor" / "rules",
        "needs_install": False,
    },
    "codex": {
        # AGENTS.md at repo root is rendered from skills/. Codex auto-loads
        # it from the project. No copy needed.
        "source": REPO_ROOT / "AGENTS.md",
        "needs_install": False,
    },
}


def _detect_agents() -> list[str]:
    """Best-effort detection of which agents are running this clone.

    Heuristics:
      claude  -> $CLAUDECODE or $CLAUDE_PROJECT_DIR
      cursor  -> $CURSOR_PROJECT or a `.cursor/` dir already present
      codex   -> $CODEX_PROJECT or $OPENAI_AGENTS_HOME

    Empty -> caller asked auto and we couldn't tell; install all three.
    """
    found: list[str] = []
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_PROJECT_DIR"):
        found.append("claude")
    if os.environ.get("CURSOR_PROJECT") or (REPO_ROOT / ".cursor").is_dir():
        found.append("cursor")
    if os.environ.get("CODEX_PROJECT") or os.environ.get("OPENAI_AGENTS_HOME"):
        found.append("codex")
    return found


def _install_claude(*, user_scope: bool, copy: bool, dry_run: bool) -> tuple[list[str], list[str]]:
    """Install skills/<name>/ into the agent's expected location.

    Returns (installed_paths, skipped_reasons).
    """
    src_root: Path = _AGENT_TARGETS["claude"]["source"]  # type: ignore[assignment]
    if not src_root.is_dir():
        return [], [f"source dir missing: {src_root}"]

    if user_scope:
        dest_root = Path.home() / ".claude" / "skills"
    else:
        dest_root = REPO_ROOT / ".claude" / "skills"

    installed: list[str] = []
    skipped: list[str] = []
    for skill_dir in sorted(src_root.iterdir()):
        if not (skill_dir / "SKILL.md").exists():
            continue
        dest = dest_root / skill_dir.name
        if dest.exists() or dest.is_symlink():
            skipped.append(f"{dest} exists (pass --force to replace)")
            continue
        if dry_run:
            installed.append(str(dest))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            import shutil

            shutil.copytree(skill_dir, dest)
        else:
            # Relative symlink for project-scope; absolute for user-scope so
            # the link doesn't break if the user moves their home.
            target = skill_dir if user_scope else Path(os.path.relpath(skill_dir, dest.parent))
            dest.symlink_to(target, target_is_directory=True)
        installed.append(str(dest))
    return installed, skipped


@app.command("install-skill", help="Install the bundled skills into your agent's expected location.")
def install_skill(
    agent: str = typer.Option(
        "auto",
        help="One of: claude | cursor | codex | all | auto. `auto` detects from env + repo state.",
    ),
    user: bool = typer.Option(
        False, "--user", help="Install at ~/.<agent>/ instead of <repo>/.<agent>/ (claude only)."
    ),
    copy: bool = typer.Option(
        False, "--copy", help="Copy files instead of symlinking. Detaches the install from this clone."
    ),
    force: bool = typer.Option(False, "--force", help="Replace existing install."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change; do not write."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Render-then-install for asymmetric agent hosts.

    Cursor and Codex auto-discover files already at repo root (`.cursor/rules/`,
    `AGENTS.md`) — for those, this command is a no-op that just confirms the
    files exist. Claude Code does NOT auto-load `skills/<name>/SKILL.md` from
    the repo root, so for `claude` we install (symlink by default) into
    `.claude/skills/<name>/` (project) or `~/.claude/skills/<name>/` (--user).
    """
    if agent == "auto":
        detected = _detect_agents()
        agents = detected or ["claude", "cursor", "codex"]
    elif agent == "all":
        agents = ["claude", "cursor", "codex"]
    elif agent in _AGENT_TARGETS:
        agents = [agent]
    else:
        emit(
            command="install-skill",
            status="error",
            error={"code": EXIT_GENERIC, "remediation": f"unknown agent {agent!r}; expected: claude | cursor | codex | all | auto"},
            json_out=json_out,
            exit_code=EXIT_GENERIC,
        )

    results: dict[str, dict[str, Any]] = {}
    all_artifacts: list[str] = []
    for a in agents:
        target = _AGENT_TARGETS[a]
        if not target["needs_install"]:
            source: Path = target["source"]  # type: ignore[assignment]
            ok = source.exists()
            results[a] = {
                "status": "ok" if ok else "missing",
                "scope": "project (auto-loaded)",
                "source": str(source),
                "note": f"{a} reads {source.name} directly — no install needed"
                if ok
                else f"missing: {source}. Run `python scripts/render_agent_docs.py` first.",
            }
            if ok:
                all_artifacts.append(str(source))
            continue

        if force and not dry_run:
            # Remove existing installs only for the agents we're acting on.
            scope_root = (
                Path.home() / ".claude" / "skills"
                if user
                else REPO_ROOT / ".claude" / "skills"
            )
            if scope_root.exists():
                import shutil

                shutil.rmtree(scope_root)

        installed, skipped = _install_claude(user_scope=user, copy=copy, dry_run=dry_run)
        results[a] = {
            "status": "ok",
            "scope": "user (~/.claude)" if user else "project (.claude)",
            "method": "copy" if copy else "symlink",
            "installed": installed,
            "skipped": skipped,
        }
        all_artifacts.extend(installed)

    emit(
        command="install-skill",
        status="ok",
        artifacts=all_artifacts,
        next_action=(
            "skill files in place. For claude: open a new Claude Code session in this repo; "
            "it'll discover `.claude/skills/`. For cursor/codex: nothing to do — already auto-loaded."
        ),
        data={"agents": results, "dry_run": dry_run},
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
