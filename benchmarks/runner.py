"""Benchmark runner CLI.

    python -m benchmarks.runner --framework vllm --gpu rtx_pro6000 --model qwen3-vl-8b

Today this loads a GPU yaml, picks an adapter, and emits a stub
`BenchmarkResult`. The actual timing loop is intentionally not yet
implemented — see `_run_loop` for the contract.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import typer
import yaml

from benchmarks.frameworks.base import BenchmarkAdapter
from benchmarks.frameworks.modelopt_bench import ModelOptAdapter
from benchmarks.frameworks.sglang_bench import SglangAdapter
from benchmarks.frameworks.triton_bench import TritonAdapter
from benchmarks.frameworks.trtllm_bench import TrtLlmAdapter
from benchmarks.frameworks.vllm_bench import VllmAdapter
from benchmarks.metrics import BenchmarkResult, utc_now_iso

ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    "vllm": VllmAdapter,
    "sglang": SglangAdapter,
    "trtllm": TrtLlmAdapter,
    "modelopt": ModelOptAdapter,
    "triton": TritonAdapter,
}

app = typer.Typer(add_completion=False)


@app.command()
def main(
    framework: str = typer.Option(..., help=f"One of: {', '.join(ADAPTERS)}"),
    gpu: str = typer.Option(..., help="GPU profile name; resolves to benchmarks/configs/<gpu>.yaml"),
    model: str = typer.Option(..., help="Model id (e.g. qwen3-vl-8b)"),
    quantization: str = typer.Option(None, help="Optional: bf16 | fp8 | int8 | awq"),
    concurrency: int = typer.Option(1, min=1),
    n_requests: int = typer.Option(64, min=1),
    out_dir: Path = typer.Option(Path("benchmarks/results"), help="Where to write the result row."),
    dry_run: bool = typer.Option(True, help="Skip actually calling the backend (placeholder run)."),
) -> None:
    cfg_path = Path(__file__).parent / "configs" / f"{gpu}.yaml"
    if not cfg_path.exists():
        typer.echo(f"missing GPU config: {cfg_path}", err=True)
        raise typer.Exit(2)
    gpu_cfg = yaml.safe_load(cfg_path.read_text()) or {}

    adapter_cls = ADAPTERS.get(framework)
    if adapter_cls is None:
        typer.echo(f"unknown framework: {framework}", err=True)
        raise typer.Exit(2)
    adapter = adapter_cls()

    knobs = gpu_cfg.get("frameworks", {}).get(framework, {})
    adapter.setup(model=model, quantization=quantization, knobs=knobs)

    run_id = uuid.uuid4().hex[:12]
    started_at = utc_now_iso()

    if not dry_run:
        result = _run_loop(adapter, knobs=knobs, concurrency=concurrency, n_requests=n_requests)
    else:
        result = _stub_result(
            framework=framework,
            adapter=adapter,
            gpu=gpu,
            gpu_cfg=gpu_cfg,
            model=model,
            quantization=quantization,
            concurrency=concurrency,
            n_requests=n_requests,
            knobs=knobs,
        )

    result.run_id = run_id
    result.started_at = started_at

    out_path = out_dir / gpu / f"{framework}-{model}-{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    typer.echo(f"wrote {out_path}")


def _run_loop(
    adapter: BenchmarkAdapter,
    *,
    knobs: dict[str, Any],
    concurrency: int,
    n_requests: int,
) -> BenchmarkResult:
    """The real timing loop. Not yet implemented.

    Contract once filled in:
    - Warm up (cuda graph capture, prefix cache prime).
    - Drive `n_requests` through `adapter.call` at `concurrency`.
    - Record per-call latencies into `LatencySamples`.
    - Score each result through the validator.
    - Emit summarised `BenchmarkResult`.
    """
    raise NotImplementedError(
        "_run_loop: real timing loop deferred until adapter.call is implemented."
    )


def _stub_result(
    *,
    framework: str,
    adapter: BenchmarkAdapter,
    gpu: str,
    gpu_cfg: dict,
    model: str,
    quantization: str | None,
    concurrency: int,
    n_requests: int,
    knobs: dict[str, Any],
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id="",
        started_at="",
        framework=framework,
        framework_version=adapter.framework_version(),
        gpu=gpu_cfg.get("display_name", gpu),
        driver=gpu_cfg.get("driver", "unknown"),
        cuda=gpu_cfg.get("cuda", "unknown"),
        model=model,
        quantization=quantization,
        tensor_parallel=int(gpu_cfg.get("tensor_parallel", 1)),
        concurrency=concurrency,
        n_requests=n_requests,
        framework_knobs=knobs,
        notes=["dry-run; real timing loop not yet implemented"],
    )


if __name__ == "__main__":
    sys.exit(app() or 0)
