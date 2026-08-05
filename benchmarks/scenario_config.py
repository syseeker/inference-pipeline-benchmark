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
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

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
    # Transport for the backend: "http" (vllm/sglang/trtllm/nim) or "zmq"
    # (nitrogen). The launcher picks how to start/talk to the server from this.
    transport: str = "http"
    # Checkpoint identity for non-HTTP policy backends (nitrogen). None for
    # HTTP backends, which resolve the served model from `hf_id`.
    ckpt: str | None = None
    # Per-model override for the runner's server-readiness wait. None = use
    # the runner's global default. Set on models with cold-cache loads that
    # exceed the default (e.g. Nemotron-Omni: ~280s download + ~150s load).
    ready_timeout_s: int | None = None
    # Video sweep knob: how many frames to extract per video. None = use the
    # VideoTextConfig default (8). Set per sweep round via `num_frames:` in the yaml.
    num_frames: int | None = None

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

    # Per-model launch flags: keyed by the exact backend, else by the backend's
    # `family` (e.g. all nitrogen-* engine backends share `backend_args.nitrogen`,
    # so a model writes its precision/steps once rather than per engine).
    model_backend_args = model.get("backend_args") or {}
    family_key = str(bk.get("family", "")) or None
    backend_args_per_model = (
        model_backend_args.get(backend)
        or (model_backend_args.get(family_key) if family_key else None)
        or []
    )
    launch_args = list(bk.get("extra_args") or []) + variant_args
    launch_args.extend(backend_args_per_model)

    trtllm_backend = bk.get("backend") if backend == "trtllm" else None

    ready_timeout_s = model.get("ready_timeout_s")
    ready_timeout_s = int(ready_timeout_s) if ready_timeout_s is not None else None

    transport = str(bk.get("transport", "http"))
    # Checkpoint identity for policy backends (zmq) lives on the MODEL; fall back
    # to a backend-level ckpt for compatibility.
    ckpt = model.get("ckpt") or bk.get("ckpt")
    # HTTP backends require an hf_id; policy backends (zmq) identify by ckpt.
    if "hf_id" in model:
        hf_id = str(model["hf_id"])
    elif ckpt is not None:
        hf_id = str(ckpt)
    else:
        hf_id = mid

    return Round(
        backend=backend,
        model_id=mid,
        hf_id=hf_id,
        family=str(model.get("family", "")),
        quantization=str(model.get("quantization", "bf16")),
        base_url=str(bk["base_url"]),
        port=int(bk["port"]),
        launch_args=launch_args,
        variant=variant,
        trtllm_backend=trtllm_backend,
        ready_timeout_s=ready_timeout_s,
        transport=transport,
        ckpt=str(ckpt) if ckpt is not None else None,
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
            r = resolve_round(
                cfg,
                backend=bk,
                model_id=rs.get("model"),
                variant=rs.get("variant"),
            )
            if "num_frames" in rs:
                r.num_frames = int(rs["num_frames"])
            yield r


# ─── Colocations (multi-model contention) ─────────────────────────────
#
# A `Colocation` is a set of tenants that share the GPU for one timed
# window. It reuses `resolve_round()` per tenant, so backend_args fan-out
# via `family:` and `unsupported_backends` gating work exactly as they do
# for single-model sweeps.
#
# See skills/gpu-contention-benchmark/reference/design-decisions.md.


# Which load generator drives which backend. This is not a preference:
# AIPerf dropped GenAI-Perf's `kserve`/`dynamic_grpc` endpoint types and
# CANNOT drive Triton, so a Triton tenant must use perf_analyzer. Encoded
# here so a yaml typo fails loudly instead of silently producing a run
# with no load on one tenant.
_DRIVER_FOR_TRANSPORT = {"http": "aiperf", "zmq": "zmq_client", "triton": "perf_analyzer"}


# `rps_sweep: {tenant: "*"}` — raise EVERY tenant's rate together.
#
# A cross-* experiment holds the subject's load fixed and sweeps the
# neighbour's, so that the subject's latency curve is attributable to the
# neighbour. A SAME-category experiment is asking a different question — where
# does this pair saturate the card? — and that is a curve in aggregate offered
# load. Sweeping one tenant there would quietly turn it into a cross-*
# experiment against a fixed neighbour and answer neither question.
ALL_TENANTS = "*"


# ─── VRAM cap sizing ──────────────────────────────────────────────────
#
# `kv_budget_gb` is a COLOCATION-level setting, not a per-tenant one, and
# that placement is the whole point: it is the quantity that must not vary
# across the comparison set. vLLM reserves `gpu_memory_utilization` × VRAM
# and everything the weights don't take becomes KV cache, which is what
# sets how many requests can be in flight — i.e. how fast the model runs
# (docs/contention.md §2b). Size a tenant's cap proportionally to its
# model and the size ladder measures memory allocation instead of
# contention. So the budget is fixed once per colocation and each tenant's
# cap absorbs only its own weights; the cap varies, the KV cache does not.
#
# A tenant MAY override it with its own `kv_budget_gb`, for the one case the
# rule above does not cover: a colocation whose swept variable is the KV
# split itself. `cross-memory-pressure-*` moves total KV from 3 to 29 GB at a
# fixed ~2:1 anchor:neighbour ratio, which a single shared value cannot
# express. Overriding is the exception; inheriting is the default, and
# anything comparing across models should keep inheriting.
DEFAULT_KV_BUDGET_GB = 20.0

# CUDA context, activation buffers and allocator fragmentation live inside
# the reservation too — without this term the KV cache silently eats the
# difference and is no longer the constant we claimed it was.
DEFAULT_CAP_OVERHEAD_GB = 2.0


def derive_cap(
    weights_gb: float, kv_budget_gb: float, vram_gb: float,
    overhead_gb: float = DEFAULT_CAP_OVERHEAD_GB, *, model_id: str = "?",
) -> float:
    """`gpu_memory_utilization` for a model at a fixed KV budget.

    Rounded to 2 dp because that is the precision the backends' flags are
    written at, and a cap that reproduces exactly is a cap a reader can
    check against the yaml.
    """
    total = weights_gb + kv_budget_gb + overhead_gb
    cap = round(total / vram_gb, 2)
    if cap > 1.0:
        raise ValueError(
            f"model {model_id!r} does not fit at kv_budget_gb={kv_budget_gb}: "
            f"weights {weights_gb} + KV {kv_budget_gb} + overhead {overhead_gb} "
            f"= {total} GB on a {vram_gb} GB card → cap {cap} > 1.0. Lower the "
            "KV budget for the whole colocation (never for one tenant — the "
            "budget must stay constant across the comparison), quantize the "
            "model, or move it to more GPUs."
        )
    return cap


@dataclass
class LoadSpec:
    """Offered load for one tenant.

    Open-loop (`rps` set) is the default and the only shape that yields a
    valid degradation ratio: a closed-loop client throttles itself in
    proportion to the slowdown being measured, so the ratio would describe
    the harness rather than the GPU.
    """

    pattern: str = "poisson"          # poisson | constant | gamma | closed
    rps: float | None = None
    output_tokens: int | None = None

    @property
    def is_open_loop(self) -> bool:
        return self.pattern != "closed" and self.rps is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Widest single-node box we plan for. An index outside this range is a yaml
# typo, and a typo that silently placed a tenant on a non-existent GPU would
# only surface as a CUDA error minutes into a run.
MAX_GPUS = 8


def _normalise_devices(device: Any, *, tenant_name: str) -> list[int]:
    """Validated, sorted list of GPU indices a tenant occupies.

    `None` means "unspecified" and resolves to GPU 0 — every colocation written
    before device placement existed assumed a single card, and those configs
    must keep running unchanged. A list means tensor parallel: the tenant sits
    on ALL of those cards at once, which is why the VRAM pre-flight charges its
    fraction to each of them.
    """
    if device is None:
        return [0]
    # bool is an int subclass; `device: true` is a typo, not GPU 1.
    raw = [device] if isinstance(device, int) and not isinstance(device, bool) else device
    if isinstance(raw, (list, tuple)):
        raw = list(raw)
    else:
        raise ValueError(
            f"tenant {tenant_name!r}: device must be an int or a list of ints, "
            f"got {device!r}"
        )
    if not raw:
        raise ValueError(
            f"tenant {tenant_name!r}: device list is empty — omit `device` to default to GPU 0."
        )
    out: list[int] = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"tenant {tenant_name!r}: device index {v!r} is not an integer "
                f"(valid GPU indices are 0..{MAX_GPUS - 1})."
            )
        if not 0 <= v < MAX_GPUS:
            raise ValueError(
                f"tenant {tenant_name!r}: device index {v} is out of range — "
                f"valid GPU indices are 0..{MAX_GPUS - 1}."
            )
        out.append(v)
    if len(set(out)) != len(out):
        raise ValueError(
            f"tenant {tenant_name!r}: device list {raw} has duplicate indices — "
            "a tensor-parallel tenant occupies each GPU once."
        )
    return sorted(out)


@dataclass
class Tenant:
    """One model in a colocation, plus how it is loaded."""

    name: str                          # role label, e.g. "victim_llm"
    round: Round                       # resolved launch params
    driver: str                        # aiperf | perf_analyzer | zmq_client
    load: LoadSpec
    workload: str | None = None        # key into the yaml `workloads:` block
    # The resolved `workloads:` entry itself (prompts / data / output_tokens),
    # carried so the run-time layer can materialise the tenant's aiperf input
    # file without re-loading the GPU yaml. Config only — nothing here touches
    # the filesystem; coloc.materialise_workload_input owns that.
    workload_spec: dict[str, Any] = field(default_factory=dict)
    gpu_memory_utilization: float | None = None
    # The KV budget this tenant's cap was derived from, recorded so a result
    # can prove the budget really was constant across the comparison set.
    # None ⇒ the cap came from the yaml verbatim, not from a derivation.
    kv_budget_gb: float | None = None
    triton_backend: str | None = None  # tensorrt | onnx | python
    # int → one card; list → tensor parallel across those cards; None → GPU 0,
    # which is what every pre-existing colocation means by saying nothing.
    device: int | list[int] | None = None

    @property
    def devices(self) -> list[int]:
        """Every GPU index this tenant occupies, ascending. A tensor-parallel
        tenant occupies all of them — placement is not a single number."""
        return _normalise_devices(self.device, tenant_name=self.name)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["round"] = self.round.to_dict()
        d["load"] = self.load.to_dict()
        # Both forms: `device` as written in the yaml, `devices` normalised, so
        # a manifest can be grouped per GPU without re-deriving the default.
        d["devices"] = self.devices
        return d


@dataclass
class Colocation:
    """One timed window: N tenants sharing a GPU, or 1 tenant solo."""

    id: str                            # yaml key, e.g. "mix-llm-cv"
    tenants: list[Tenant]
    duration_s: int = 120
    isolation: str = "mps"             # none | mps | mig | separate-gpu
    phase: int | None = None
    is_solo: bool = False
    # 1-based index within a `repetitions:` set. 1 for every colocation that
    # does not ask for repeats, so the field is always meaningful on a result.
    repetition: int = 1

    @property
    def run_label(self) -> str:
        """Pairs a contention row with its baseline in summary.py, which
        keys cross-run deltas off `run_label`. Repetitions deliberately SHARE
        one label: they are samples of the same experiment, and summary.py
        should aggregate them rather than treat them as three experiments."""
        return "solo" if self.is_solo else f"coloc:{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "duration_s": self.duration_s,
            "isolation": self.isolation,
            "is_solo": self.is_solo,
            "repetition": self.repetition,
            "run_label": self.run_label,
            "n_tenants": len(self.tenants),
            "tenants": [t.to_dict() for t in self.tenants],
        }


def _merge_extends(colos: dict[str, Any], name: str, _seen: set[str] | None = None) -> dict[str, Any]:
    """Resolve `extends:` into a flat spec. Child keys win; `tenants` are
    merged by tenant `name` so a child can override one field of one tenant
    without restating the whole roster."""
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"circular extends: {' -> '.join([*_seen, name])}")
    if name not in colos:
        raise ValueError(f"unknown colocation {name!r}; defined: {sorted(colos)}")

    spec = dict(colos[name] or {})
    parent_name = spec.pop("extends", None)
    if not parent_name:
        return spec

    parent = _merge_extends(colos, str(parent_name), _seen | {name})
    merged = {**parent, **spec}

    parent_tenants = {t["name"]: dict(t) for t in (parent.get("tenants") or [])}
    for child in spec.get("tenants") or []:
        nm = child["name"]
        parent_tenants[nm] = {**parent_tenants.get(nm, {}), **child}
    if parent_tenants:
        merged["tenants"] = list(parent_tenants.values())
    return merged


def _resolve_tenant(
    cfg: dict[str, Any], tspec: dict[str, Any], workloads: dict[str, Any],
    *, kv_budget_gb: float | None = None,
) -> Tenant:
    backend = tspec["backend"]
    model_id = tspec.get("model")
    r = resolve_round(cfg, backend=backend, model_id=model_id, variant=tspec.get("variant"))

    wl_name = tspec.get("workload")
    wl = (workloads.get(wl_name) or {}) if wl_name else {}
    if wl_name and wl_name not in workloads:
        raise ValueError(f"unknown workload {wl_name!r}; defined: {sorted(workloads)}")

    # A model can be incompatible with a workload rather than a backend —
    # e.g. an image-only VLM asked to serve a video workload. Same
    # fail-safely contract as unsupported_backends: raise with the reason
    # so the caller skips the row instead of crashing mid-run.
    models = cfg.get("models") or {}
    unsupported_wl = (models.get(r.model_id, {}).get("unsupported_workloads") or {})
    if wl_name and wl_name in unsupported_wl:
        raise ValueError(
            f"model {r.model_id!r} does not support workload {wl_name!r}: "
            f"{unsupported_wl[wl_name]}"
        )

    load_spec = dict(tspec.get("load") or {})
    load = LoadSpec(
        pattern=str(load_spec.get("pattern", "poisson")),
        rps=float(load_spec["rps"]) if load_spec.get("rps") is not None else None,
        # Output length lives on the workload, not the load spec — it is a
        # property of what we're asking for, not how fast we ask.
        output_tokens=(
            int(load_spec["output_tokens"]) if load_spec.get("output_tokens") is not None
            else (int(wl["output_tokens"]) if wl.get("output_tokens") is not None else None)
        ),
    )

    transport = "triton" if backend == "triton" or r.transport == "triton" else r.transport
    default_driver = _DRIVER_FOR_TRANSPORT.get(transport, "aiperf")
    driver = str(tspec.get("driver") or default_driver)
    if transport == "triton" and driver == "aiperf":
        raise ValueError(
            f"tenant {tspec['name']!r}: aiperf cannot drive Triton (no kserve / "
            "dynamic_grpc endpoint type). Use driver: perf_analyzer."
        )

    # Validate placement here, at parse time, so a bad index fails the plan
    # rather than the run. Store the normalised (sorted) form.
    device = _normalise_devices(tspec.get("device"), tenant_name=str(tspec["name"]))
    if tspec.get("device") is None:
        device = None                  # keep "unspecified" distinguishable in the manifest
    elif len(device) == 1 and isinstance(tspec["device"], int):
        device = device[0]

    # An explicit yaml cap always wins — it is the escape hatch, and every
    # colocation written before derivation existed must keep its numbers.
    # Derivation only fills the gap, and only where we know the weights:
    # a Triton CV tenant has no GPU fraction to set, and a model with no
    # `weights_gb` would have its KV budget guessed rather than held.
    cap = (
        float(tspec["gpu_memory_utilization"])
        if tspec.get("gpu_memory_utilization") is not None else None
    )
    kv_used: float | None = None
    weights_gb = (models.get(r.model_id, {}) or {}).get("weights_gb")
    vram_gb = cfg.get("vram_gb")
    # A tenant may override the colocation budget. The colocation-level
    # setting stays the default precisely because it holds KV constant across
    # a comparison set (see "VRAM cap sizing" above) — but a colocation whose
    # variable IS the KV split needs to state each tenant's share, and
    # cross-memory-pressure-* sweeps exactly that at a fixed ~2:1 ratio.
    # Without this, both tenants silently take the inherited default and the
    # rung measures nothing it claims to.
    tenant_kv = tspec.get("kv_budget_gb")
    if tenant_kv is not None:
        kv_budget_gb = float(tenant_kv)
    if (
        cap is None and kv_budget_gb is not None and transport != "triton"
        and weights_gb is not None and vram_gb
    ):
        kv_used = float(kv_budget_gb)
        cap = derive_cap(
            float(weights_gb), kv_used, float(vram_gb), model_id=r.model_id,
        )

    return Tenant(
        name=str(tspec["name"]),
        round=r,
        driver=driver,
        load=load,
        workload=wl_name,
        workload_spec=dict(wl),
        gpu_memory_utilization=cap,
        kv_budget_gb=kv_used,
        triton_backend=tspec.get("triton_backend"),
        device=device,
    )


def _assign_distinct_ports(tenants: list[Tenant]) -> list[Tenant]:
    """Give every HTTP tenant in a window a port of its own.

    `backends.<b>.port` is a BACKEND-wide default, so two tenants on the same
    backend inherit the same port. The second server then takes the endpoint
    and each driver is answered by the other tenant's model: mix-full logged
    332/700 and 83/178 requests failing with HTTP 404 "The model ... does not
    exist", and the survivors were whichever server happened to own the port at
    that moment. Triton already avoids this with a per-device stride; HTTP
    tenants had no equivalent.

    Ports are bumped only on collision, so the first tenant on a backend keeps
    the configured port and every single-HTTP-tenant colocation — including
    every solo baseline — is byte-for-byte unchanged. That matters: the port
    reaches the launch command, and shifting it for a baseline would make it a
    different deployment from the window it is the reference for.

    Triton tenants are untouched; they are addressed per device, not per tenant.
    """
    used: set[int] = set()
    out: list[Tenant] = []
    for t in tenants:
        if t.round.transport == "triton":
            out.append(t)
            continue
        port = t.round.port
        while port in used:
            port += 1
        used.add(port)
        if port == t.round.port:
            out.append(t)
            continue
        base = t.round.base_url.replace(f":{t.round.port}", f":{port}")
        out.append(replace(t, round=replace(t.round, port=port, base_url=base)))
    return out


def _solo_key(t: Tenant) -> tuple:
    """Identity of a solo baseline. A baseline is only valid for a
    contention run at the SAME offered load, so load is part of the key —
    but two contention runs sharing a tenant config share one baseline,
    which is why we dedupe rather than re-running it per colocation.

    Placement is part of the identity too: a baseline taken on GPU 0 does not
    describe a tenant pinned to GPU 3 (different card, possibly different
    clocks), and a TP-2 tenant is a different deployment from a TP-1 one.

    The VRAM cap is part of it for the same reason (§2b): the cap sets the
    KV cache, so the same model at 0.45 and at 0.70 are two different
    deployments. Leave it out and a 2-tenant window and a 4-tenant window
    would share one baseline, and one of them would be compared against a
    reference that never existed.

    `kv_budget_gb` is in the key too, and the cap is no longer sufficient on
    its own. Once a tenant states its cache absolutely, the cap becomes a
    derived, 2-dp-rounded consequence — so two genuinely different deployments
    can round to the same number. cross-memory-pressure's p25 and p50
    neighbours (0.64 and 1.28 GiB of KV) both derive 0.19, and would have
    shared one baseline: the p50 contention run would then have been scored
    against a reference recorded at HALF its cache, reading a 2x KV difference
    as contention. That is the exact confound this family exists to avoid."""
    return (t.round.backend, t.round.model_id, t.workload, t.load.pattern, t.load.rps,
            tuple(t.devices), t.gpu_memory_utilization, t.kv_budget_gb)


def iter_colocation(cfg: dict[str, Any], name: str) -> Iterator[Colocation]:
    """Yield every run for a named colocation: solo baselines first, then the
    co-resident window(s).

    Expansion rules, applied in order:
      extends     — inherit a base colocation, merging tenants by name
      rps_sweep   — one colocation per rate for the named tenant, or for
                    every tenant at once with `tenant: "*"`
      vary        — one colocation per value of a named tenant field
      repetitions — each resulting contention window emitted N times

    Baselines come first so that a partial run still produces the reference
    numbers the ratios need; a contention row without its baseline is
    uninterpretable.

    Baselines are deduped WITHIN a colocation but not across them — the same
    (backend, model, workload, load) recurs in several. That is deliberate:
    each colocation is independently runnable. The orchestrator is expected
    to cache by `_solo_key` across a session so a full study doesn't re-run
    identical baselines (~40 of the 69 emitted runs are baselines, most of
    them repeats).
    """
    colos = cfg.get("colocations") or {}
    spec = _merge_extends(colos, name)
    workloads = cfg.get("workloads") or {}

    base_tenants = spec.get("tenants") or []
    if len(base_tenants) < 1:
        raise ValueError(f"colocation {name!r} has no tenants")

    # Build the variant list of tenant-spec rosters.
    rosters: list[list[dict[str, Any]]] = [[dict(t) for t in base_tenants]]

    sweep = spec.get("rps_sweep")
    if sweep:
        target, values = str(sweep["tenant"]), list(sweep["values"])
        names = {str(t["name"]) for t in base_tenants}
        if target != ALL_TENANTS and target not in names:
            raise ValueError(
                f"colocation {name!r}: rps_sweep names tenant {target!r}, which is "
                f"not in this roster ({sorted(names)}). Use {ALL_TENANTS!r} to sweep "
                "every tenant together."
            )
        expanded = []
        for v in values:
            roster = [dict(t) for t in base_tenants]
            for t in roster:
                if target == ALL_TENANTS or t["name"] == target:
                    t["load"] = {**(t.get("load") or {}), "rps": v}
            expanded.append(roster)
        rosters = expanded

    vary = spec.get("vary")
    if vary:
        target, fld, values = str(vary["tenant"]), str(vary["field"]), list(vary["values"])
        expanded = []
        for roster in rosters:
            for v in values:
                new = [dict(t) for t in roster]
                for t in new:
                    if t["name"] == target:
                        t[fld] = v
                expanded.append(new)
        rosters = expanded

    duration_s = int(spec.get("duration_s", 120))
    isolation = str(spec.get("isolation", "mps"))
    phase = int(spec["phase"]) if spec.get("phase") is not None else None
    want_solo = str(spec.get("solo_baselines", "auto")) == "auto"
    # Read once per colocation, applied to every tenant that omits an
    # explicit cap. Resolution happens BEFORE the solo baselines are cut,
    # so a baseline is built from the same Tenant object as the contention
    # run and therefore carries the same cap — which is the only way its
    # KV cache matches (docs/contention.md §2b).
    kv_budget_gb = float(spec.get("kv_budget_gb", DEFAULT_KV_BUDGET_GB))

    # `repetitions:` — run each contention window N times. Only ask for this
    # where the quantity being measured is not unimodal: near the VRAM ceiling
    # a model either fits or thrashes, and the mean of those two states
    # describes neither, so the spread across repeats IS the finding.
    # Baselines are NOT repeated: they are deduped by `_solo_key` here and
    # cached again per session by the orchestrator, so a repeat would be
    # dropped anyway — and the bimodality lives in the co-resident window.
    repetitions = int(spec.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError(
            f"colocation {name!r}: repetitions must be >= 1, got {repetitions}"
        )

    seen_solo: set[tuple] = set()
    pending: list[Colocation] = []

    for roster in rosters:
        try:
            tenants = [
                _resolve_tenant(cfg, t, workloads, kv_budget_gb=kv_budget_gb) for t in roster
            ]
        except ValueError as e:
            # Skip the whole window: a 2-tenant contention test cannot run
            # with one tenant missing, and a silently-degraded roster would
            # produce a ratio against the wrong baseline.
            print(f">> colocation skip {name}: {e}", file=sys.stderr)
            continue

        if want_solo:
            for t in tenants:
                k = _solo_key(t)
                if k in seen_solo:
                    continue
                seen_solo.add(k)
                yield Colocation(
                    id=name, tenants=[t], duration_s=duration_s,
                    isolation=isolation, phase=phase, is_solo=True,
                )

        window_tenants = _assign_distinct_ports(tenants)
        for rep in range(1, repetitions + 1):
            pending.append(
                Colocation(
                    id=name, tenants=window_tenants, duration_s=duration_s,
                    isolation=isolation, phase=phase, is_solo=False,
                    repetition=rep,
                )
            )

    yield from pending


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
    "transport",
    "ckpt",
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
    emit_colocations: str = typer.Option(
        None, "--emit-colocations",
        help=(
            "Print newline-delimited JSON colocations for the named entry in "
            "`colocations:`. Solo baselines are emitted first, then the "
            "co-resident window(s). Expands extends / rps_sweep / vary."
        ),
    ),
    list_colocations: bool = typer.Option(
        False, "--list-colocations",
        help="List defined colocations with their phase number.",
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

    # Colocation iteration ─────────────────────────────────────────
    if emit_colocations is not None:
        try:
            for c in iter_colocation(cfg, emit_colocations):
                typer.echo(json.dumps(c.to_dict()))
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(2)
        return

    if list_colocations:
        for nm, spec in sorted((cfg.get("colocations") or {}).items()):
            phase = (spec or {}).get("phase")
            typer.echo(f"{nm}\tphase={phase if phase is not None else '-'}")
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
