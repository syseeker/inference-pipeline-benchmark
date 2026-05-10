"""Resolve a launch round from benchmarks/configs/<gpu>.yaml.

Two surfaces — Python (used by `benchmarks.runner`) and CLI (used by
`scripts/run_all_scenarios.sh`).

Schema (see e.g. benchmarks/configs/rtx_pro6000.yaml):

    models:
      <id>:
        hf_id: "..."
        family: "..."
        quantization: "..."
        ready_timeout_s: 1800               # optional, overrides runner default
        backend_args:                       # optional
          vllm:   ["--quantization=fp8"]
          sglang: []
          trtllm: []
        unsupported_backends:               # optional, dict[backend -> reason]
          trtllm: "TRT-LLM 1.2.1 fused-MoE backend needs SM_90 (DeepGEMM)"
          # Sweep mode silently skips these. resolve_round() raises a
          # ValueError with the reason for direct (non-sweep) invocations.
    default_model: <id>
    backends:
      vllm:
        base_url: "..."
        port: 8000
        extra_args: [...]
        variants:
          eager: ["--enforce-eager"]
          chunked_off: ["--no-enable-chunked-prefill"]
      trtllm:
        base_url: "..."
        port: 8002
        backend: pytorch                    # pytorch | trtllm | _autodeploy
        extra_args: [...]
    sweeps:
      <name>:
        backends: [vllm, sglang, trtllm]    # optional; defaults to all
        rounds:
          - {model: <id>, variant: <name>, backends: [vllm]}
          - ...

Resolution rules:
- A "round" = (backend, model_id, variant?) → concrete launch params.
- launch_args = backends.<bk>.extra_args
              + backends.<bk>.variants.<variant> (if variant)
              + models.<id>.backend_args.<bk>     (if present)
- model_id default: yaml's `default_model`.
- For trtllm, `backend.backend` (pytorch | trtllm | _autodeploy) is also
  carried; the launcher translates `trtllm` → `--backend tensorrt`.

CLI:
    # Resolve one field for one (backend, model, variant)
    python -m benchmarks.scenario_config \
        --gpu rtx_pro6000 --backend vllm --field hf_id

    python -m benchmarks.scenario_config \
        --gpu rtx_pro6000 --backend vllm --variant eager --field launch_args --list

    # Iterate a sweep — newline-delimited JSON, one round per line
    python -m benchmarks.scenario_config \
        --gpu rtx_pro6000 --sweep models --emit-rounds

    # Probe whether a sweep / variant exists (rc 0/2)
    python -m benchmarks.scenario_config \
        --gpu rtx_pro6000 --has-sweep models
    python -m benchmarks.scenario_config \
        --gpu rtx_pro6000 --backend vllm --has-variant eager
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import typer
import yaml

app = typer.Typer(add_completion=False)


CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class Round:
    """A concrete (backend, model, variant) → launch params resolution."""

    backend: str               # vllm | sglang | trtllm
    model_id: str              # logical id, key in `models:`
    hf_id: str                 # HF Hub id passed to the launcher
    family: str                # qwen3-vl | qwen3.5 | qwen3.6 | nemotron | ...
    quantization: str          # bf16 | fp8 | nvfp4 (recorded in BenchmarkResult)
    base_url: str              # OpenAI-compatible client base URL
    port: int                  # server port
    launch_args: list[str] = field(default_factory=list)
    variant: str | None = None
    # trtllm-only — pytorch | trtllm | _autodeploy. None for non-trtllm.
    trtllm_backend: str | None = None
    # Per-model override for the runner's server-readiness wait. None = use
    # the runner's global default. Set on models with cold-cache loads that
    # exceed the default (e.g. Nemotron-Omni: ~280s download + ~150s load).
    ready_timeout_s: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gpu_config(gpu: str) -> dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{gpu}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing GPU config: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def resolve_round(
    cfg: dict[str, Any],
    backend: str,
    model_id: str | None = None,
    variant: str | None = None,
) -> Round:
    """Return a `Round` for (backend, model_id|default, variant|none)."""
    models = cfg.get("models") or {}
    backends = cfg.get("backends") or {}

    if backend not in backends:
        raise ValueError(f"unknown backend {backend!r}; defined: {sorted(backends)}")
    bk = backends[backend]

    mid = model_id or cfg.get("default_model")
    if mid is None:
        raise ValueError("default_model is unset and --model not given")
    if mid not in models:
        raise ValueError(f"unknown model {mid!r}; defined: {sorted(models)}")
    model = models[mid]

    # Refuse hardware/version-incompatible (backend, model) pairs up front so
    # that direct `--backend X --model Y` invocations don't get past server
    # launch. Sweep iteration filters silently in `iter_sweep`.
    unsupported = (model.get("unsupported_backends") or {})
    if backend in unsupported:
        raise ValueError(
            f"backend {backend!r} is not supported for model {mid!r}: "
            f"{unsupported[backend]}"
        )

    variant_args: list[str] = []
    if variant is not None:
        variants = bk.get("variants") or {}
        if variant not in variants:
            raise ValueError(
                f"unknown variant {variant!r} for backend {backend!r}; "
                f"defined: {sorted(variants)}"
            )
        variant_args = list(variants[variant] or [])

    launch_args = list(bk.get("extra_args") or []) + variant_args
    backend_args_per_model = (model.get("backend_args") or {}).get(backend) or []
    launch_args.extend(backend_args_per_model)

    trtllm_backend = bk.get("backend") if backend == "trtllm" else None

    ready_timeout_s = model.get("ready_timeout_s")
    ready_timeout_s = int(ready_timeout_s) if ready_timeout_s is not None else None

    return Round(
        backend=backend,
        model_id=mid,
        hf_id=str(model["hf_id"]),
        family=str(model.get("family", "")),
        quantization=str(model.get("quantization", "bf16")),
        base_url=str(bk["base_url"]),
        port=int(bk["port"]),
        launch_args=launch_args,
        variant=variant,
        trtllm_backend=trtllm_backend,
        ready_timeout_s=ready_timeout_s,
    )


def iter_sweep(cfg: dict[str, Any], sweep_name: str) -> Iterator[Round]:
    """Yield one `Round` per (round, backend) combination in the named sweep."""
    sweeps = cfg.get("sweeps") or {}
    if sweep_name not in sweeps:
        raise ValueError(f"unknown sweep {sweep_name!r}; defined: {sorted(sweeps)}")
    sweep = sweeps[sweep_name]

    all_backends = list((cfg.get("backends") or {}).keys())
    sweep_backends = list(sweep.get("backends") or all_backends)

    models_cfg = cfg.get("models") or {}
    for round_spec in sweep.get("rounds") or []:
        rs: dict[str, Any] = dict(round_spec) if round_spec else {}
        round_backends = list(rs.get("backends") or sweep_backends)
        mid = rs.get("model") or cfg.get("default_model")
        unsupported = (models_cfg.get(mid, {}).get("unsupported_backends") or {})
        for bk in round_backends:
            if bk in unsupported:
                # Tell the operator why we're skipping; the sweep continues.
                print(
                    f">> sweep skip {bk}/{mid}: {unsupported[bk]}",
                    file=sys.stderr,
                )
                continue
            yield resolve_round(
                cfg,
                backend=bk,
                model_id=rs.get("model"),
                variant=rs.get("variant"),
            )


# ─── CLI ──────────────────────────────────────────────────────────────


_SCALAR_FIELDS = {
    "backend",
    "model_id",
    "hf_id",
    "family",
    "quantization",
    "base_url",
    "port",
    "variant",
    "trtllm_backend",
    "ready_timeout_s",
}
_LIST_FIELDS = {"launch_args"}


@app.command()
def main(
    gpu: str = typer.Option(..., help="GPU profile name (resolves benchmarks/configs/<gpu>.yaml)"),
    backend: str = typer.Option(None, help="vllm | sglang | trtllm. Required unless --emit-rounds."),
    model: str = typer.Option(None, help="Model id from `models:`. Defaults to `default_model`."),
    variant: str = typer.Option(None, help="Variant name from `backends.<backend>.variants`."),
    field_name: str = typer.Option(
        None, "--field",
        help=f"One of: {sorted(_SCALAR_FIELDS | _LIST_FIELDS)}",
    ),
    list_mode: bool = typer.Option(
        False, "--list", help="Treat `--field launch_args` as a list (one per line)."
    ),
    emit_rounds: str = typer.Option(
        None, "--emit-rounds",
        help="Print newline-delimited JSON rounds for the named sweep. Passed value is the sweep name.",
    ),
    has_sweep: str = typer.Option(
        None, "--has-sweep",
        help="rc 0 if the named sweep exists, rc 2 otherwise.",
    ),
    has_variant: str = typer.Option(
        None, "--has-variant",
        help="rc 0 if the named variant exists for --backend, rc 2 otherwise.",
    ),
    unsupported_reason: bool = typer.Option(
        False, "--unsupported-reason",
        help=(
            "Print the `unsupported_backends.<backend>` reason for "
            "(--backend, --model) and exit 0. Empty stdout = supported. "
            "Lets bash callers cheaply check before launching a server."
        ),
    ),
) -> None:
    try:
        cfg = load_gpu_config(gpu)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)

    # Probe modes ──────────────────────────────────────────────────
    if has_sweep is not None:
        sweeps = cfg.get("sweeps") or {}
        raise typer.Exit(0 if has_sweep in sweeps else 2)

    if has_variant is not None:
        if not backend:
            typer.echo("--has-variant requires --backend", err=True)
            raise typer.Exit(2)
        backends = cfg.get("backends") or {}
        if backend not in backends:
            raise typer.Exit(2)
        variants = backends[backend].get("variants") or {}
        raise typer.Exit(0 if has_variant in variants else 2)

    if unsupported_reason:
        if not backend:
            typer.echo("--unsupported-reason requires --backend", err=True)
            raise typer.Exit(2)
        models = cfg.get("models") or {}
        mid = model or cfg.get("default_model")
        m = models.get(mid) or {}
        reason = (m.get("unsupported_backends") or {}).get(backend, "")
        typer.echo(reason)
        return

    # Sweep iteration ──────────────────────────────────────────────
    if emit_rounds is not None:
        try:
            for r in iter_sweep(cfg, emit_rounds):
                typer.echo(json.dumps(r.to_dict()))
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(2)
        return

    # Single-field resolution ──────────────────────────────────────
    if not backend or not field_name:
        typer.echo(
            "single-resolve mode needs --backend and --field "
            "(or use --emit-rounds <sweep> / --has-sweep <name>)",
            err=True,
        )
        raise typer.Exit(2)

    try:
        r = resolve_round(cfg, backend=backend, model_id=model, variant=variant)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)

    d = r.to_dict()
    if field_name not in d:
        typer.echo(
            f"unknown --field {field_name!r}; expected one of "
            f"{sorted(_SCALAR_FIELDS | _LIST_FIELDS)}",
            err=True,
        )
        raise typer.Exit(2)

    value = d[field_name]
    if list_mode or field_name in _LIST_FIELDS:
        if value is None:
            return
        if not isinstance(value, list):
            typer.echo(f"value at {field_name!r} is not a list", err=True)
            raise typer.Exit(2)
        for item in value:
            typer.echo(str(item))
        return

    if isinstance(value, list):
        typer.echo(f"value at {field_name!r} is a list; pass --list", err=True)
        raise typer.Exit(2)
    typer.echo("" if value is None else str(value))


if __name__ == "__main__":
    sys.exit(app() or 0)
