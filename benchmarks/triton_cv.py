"""Triton model-repository tooling for the CV tenants (step 7).

The contention study serves CV models (YOLOv8, DINOv2) on Triton while LLM/VLM
tenants run on native vLLM/SGLang (serving-topology.md: CV exports cleanly, and
Triton's per-request queue/compute decomposition is exactly the attribution a
contention study needs). This module builds the pieces Triton needs — the model
repository layout and each model's `config.pbtxt` — from a small CV model
registry, plus the ONNX export path.

Split of concerns:
  - config.pbtxt generation + repo layout + the model registry are PURE and
    unit-tested here.
  - the actual weight export (ultralytics / torch) needs the CV deps and runs in
    scripts/build_triton_cv_repo.py; launching the Triton container is in the
    orchestrator (benchmarks/coloc.py).

Triton backends we target (build-plan §1.14): `onnxruntime` (portable baseline),
`tensorrt` (optimised .plan), `python` (models with no clean export — kosmos-2.5,
paddleocr). One TensorRT + CUDA environment for all CV models, driven by
`perf_analyzer` — never AIPerf, which cannot drive Triton.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Triton container: Dynamo-Triton 26.07 (build-plan §1.14). server runs
# tritonserver; the -sdk image carries perf_analyzer.
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-py3"
TRITON_SDK_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-py3-sdk"
# Python-backend models import torch/transformers, which the stock image does
# not carry. Built from docker/Dockerfile.triton-python; only repos holding a
# python-backend model need it, so TensorRT-only repos stay on the stock image.
TRITON_PYTHON_IMAGE = "inference-bench/tritonserver:26.07-py3-transformers"

# Where the hand-authored model.py files live, one directory per model.
PYTHON_MODEL_SRC = REPO_ROOT / "benchmarks" / "triton_python_models"

# Triton `backend:` value + the model filename it expects in the version dir.
_BACKEND_TABLE = {
    "onnx": ("onnxruntime", "model.onnx"),
    "tensorrt": ("tensorrt", "model.plan"),
    "python": ("python", "model.py"),
}


@dataclass
class CVModelSpec:
    """Everything the config.pbtxt + export need for one CV model.

    dims are CHW WITHOUT the batch dimension — Triton prepends batch via
    `max_batch_size`, so the tensor dims here must not include it.
    """

    name: str                       # logical id, matches the GPU yaml model id
    hf_id: str
    exporter: str                   # "ultralytics" | "torch-hf" | "none" (python backend)
    input_name: str
    input_dims: list[int]           # CHW, no batch
    output_name: str
    output_dims: list[int]          # no batch
    weights: str = ""               # ultralytics weight stem, e.g. "yolov8l"
    # Must match `models.<id>.quantization` in the GPU yaml — asserted by
    # test_cv_spec_precision_matches_the_gpu_yaml, because the two registries
    # are separate and nothing else would notice them drifting apart.
    precision: str = "fp16"         # fp16 | fp32
    # Python-backend models do not take a raw pixel tensor and do not return
    # one, so they set their Triton I/O dtypes explicitly instead of deriving
    # them from `precision` (which still describes the weights).
    input_dtype_override: str | None = None
    output_dtype_override: str | None = None

    @property
    def torch_dtype_is_half(self) -> bool:
        return self.precision == "fp16"

    @property
    def input_dtype(self) -> str:
        return self.input_dtype_override or _TRITON_DTYPE[self.precision]

    @property
    def output_dtype(self) -> str:
        return self.output_dtype_override or _TRITON_DTYPE[self.precision]

    @property
    def is_python_backend(self) -> bool:
        """No export path — served by a hand-authored model.py.

        These need torch + transformers inside the Triton container, which the
        stock image does not carry (it ships numpy and nothing else), so they
        also need the derived image built by docker/Dockerfile.triton-python.
        """
        return self.exporter == "none"


# TensorRT v11 makes every network strongly typed and has REMOVED trtexec's
# --fp16: the builder no longer coerces precision, it obeys the ONNX graph. So
# an fp16 engine has to come from an fp16 export, and the config.pbtxt I/O dtype
# has to agree with it or Triton rejects the model. Precision is therefore a
# property of the spec, not a build flag.
_TRITON_DTYPE = {"fp16": "TYPE_FP16", "fp32": "TYPE_FP32", "bf16": "TYPE_BF16"}


# CV models referenced by the rtx_pro6000 colocations. YOLOv8-l is the primary
# tenant; dinov2 is the bandwidth-bound neighbour; kosmos-2.5 / paddleocr use the
# Python backend (no clean ONNX/TensorRT export) and are staged in step 7b.
CV_MODELS: dict[str, CVModelSpec] = {
    # output0 anchor axis is dynamic under ultralytics dynamic=True export
    # (shape [-1, 84, -1]); -1 tells Triton the dim is variable. Hardcoding
    # 8400 makes Triton reject the model with a shape mismatch (live-verified).
    "yolov8-n": CVModelSpec(
        "yolov8-n", "ultralytics/yolov8n", "ultralytics",
        "images", [3, 640, 640], "output0", [84, -1], weights="yolov8n",
    ),
    "yolov8-l": CVModelSpec(
        "yolov8-l", "ultralytics/yolov8l", "ultralytics",
        "images", [3, 640, 640], "output0", [84, -1], weights="yolov8l",
    ),
    "dinov2-base": CVModelSpec(
        "dinov2-base", "facebook/dinov2-base", "torch-hf",
        "pixel_values", [3, 224, 224], "last_hidden_state", [257, 768],
    ),
    "dinov2-large": CVModelSpec(
        "dinov2-large", "facebook/dinov2-large", "torch-hf",
        "pixel_values", [3, 224, 224], "last_hidden_state", [257, 1024],
    ),
    # Python backend: Kosmos2_5ForConditionalGeneration is absent from vLLM's
    # and SGLang's registries (verified against
    # ModelRegistry.get_supported_archs()), so Triton is the only path.
    #
    # The input is a trigger, not the image. kosmos-2.5's processor emits
    # pix2struct-style `flattened_patches` of (4096, 770); random values there
    # are not a document, and the model would generate degenerately against
    # them — the tenant would be "running" while measuring nothing like the
    # real workload. model.py instead preprocesses the workload's actual
    # document and prompts once at load, and each request replays them. That
    # also satisfies the design rule to hold a CV tenant's input fixed so the
    # variance measured is contention rather than content.
    "kosmos-2.5": CVModelSpec(
        "kosmos-2.5", "microsoft/kosmos-2.5", "none",
        "TRIGGER", [1], "TEXT", [1],
        precision="bf16",
        input_dtype_override="TYPE_INT32",
        output_dtype_override="TYPE_STRING",
    ),
}


def image_for_models(models: list[str] | None) -> str:
    """Which Triton image can serve this set of models.

    A repo holding any python-backend model needs the derived image; anything
    else runs on the stock one. Chosen per container rather than globally so a
    GPU serving only TensorRT plans does not pull several GB it cannot use.
    """
    for name in models or []:
        spec = CV_MODELS.get(name)
        if spec is not None and spec.is_python_backend:
            return TRITON_PYTHON_IMAGE
    return TRITON_IMAGE


def python_model_source(name: str) -> Path | None:
    """The checked-in model.py for a python-backend model, if there is one."""
    p = PYTHON_MODEL_SRC / name / "model.py"
    return p if p.exists() else None


def triton_backend_of(triton_backend: str) -> str:
    """Map our short name (onnx|tensorrt|python) to Triton's `backend:` value."""
    if triton_backend not in _BACKEND_TABLE:
        raise ValueError(
            f"unknown triton_backend {triton_backend!r}; expected one of {sorted(_BACKEND_TABLE)}"
        )
    return _BACKEND_TABLE[triton_backend][0]


def model_filename(triton_backend: str) -> str:
    """The weight filename Triton expects in the version dir for this backend."""
    if triton_backend not in _BACKEND_TABLE:
        raise ValueError(f"unknown triton_backend {triton_backend!r}")
    return _BACKEND_TABLE[triton_backend][1]


@dataclass
class RepoLayout:
    """Resolved paths for one model in a Triton model repository.

        <repo_root>/<name>/
          config.pbtxt
          <version>/
            model.onnx | model.plan | model.py
    """

    repo_root: Path
    name: str
    triton_backend: str
    version: int = 1

    @property
    def model_dir(self) -> Path:
        return self.repo_root / self.name

    @property
    def config_pbtxt(self) -> Path:
        return self.model_dir / "config.pbtxt"

    @property
    def version_dir(self) -> Path:
        return self.model_dir / str(self.version)

    @property
    def weight_file(self) -> Path:
        return self.version_dir / model_filename(self.triton_backend)


def build_config_pbtxt(
    spec: CVModelSpec, triton_backend: str, *, max_batch_size: int = 8,
    instance_count: int = 1, dynamic_batching: bool = True,
    params: dict[str, str] | None = None,
) -> str:
    """Render a Triton config.pbtxt for a CV model.

    Batch is expressed via `max_batch_size` (Triton prepends the batch dim), so
    the input/output `dims` are the per-sample shape. `dynamic_batching {}` lets
    Triton coalesce concurrent requests — that batching is part of what a CV
    tenant contributes to GPU contention, so it is on by default.
    """
    backend = triton_backend_of(triton_backend)
    in_dims = ", ".join(str(d) for d in spec.input_dims)
    out_dims = ", ".join(str(d) for d in spec.output_dims)
    params = params or {}
    lines = [
        f'name: "{spec.name}"',
        f'backend: "{backend}"',
        f"max_batch_size: {max_batch_size}",
        "input [",
        "  {",
        f'    name: "{spec.input_name}"',
        f"    data_type: {spec.input_dtype}",
        f"    dims: [ {in_dims} ]",
        "  }",
        "]",
        "output [",
        "  {",
        f'    name: "{spec.output_name}"',
        f"    data_type: {spec.output_dtype}",
        f"    dims: [ {out_dims} ]",
        "  }",
        "]",
        "instance_group [",
        "  {",
        "    kind: KIND_GPU",
        f"    count: {instance_count}",
        "  }",
        "]",
    ]
    if dynamic_batching:
        lines.append("dynamic_batching { }")
    # `parameters` is a protobuf map: one repeated `parameters { }` entry per
    # key, NOT a bracketed list.
    for key, value in sorted(params.items()):
        lines.append("parameters {")
        lines.append(f'  key: "{key}"')
        lines.append(f'  value: {{ string_value: "{value}" }}')
        lines.append("}")
    return "\n".join(lines) + "\n"


def write_model_repo(
    repo_root: Path, spec: CVModelSpec, triton_backend: str, *,
    max_batch_size: int = 8, instance_count: int = 1, version: int = 1,
    params: dict[str, str] | None = None,
) -> RepoLayout:
    """Create the model dir + version dir + config.pbtxt (NOT the weights).

    The weight file is written by the exporter (needs the CV deps); this lays out
    the structure Triton requires and drops the config in place.

    A python-backend model has no exporter, so its `model.py` IS its weight file
    and is copied from benchmarks/triton_python_models/<name>/ here — otherwise
    the repo looks complete while Triton reports the model UNAVAILABLE.
    """
    layout = RepoLayout(repo_root=repo_root, name=spec.name,
                        triton_backend=triton_backend, version=version)
    layout.version_dir.mkdir(parents=True, exist_ok=True)
    layout.config_pbtxt.write_text(
        build_config_pbtxt(spec, triton_backend, max_batch_size=max_batch_size,
                           instance_count=instance_count, params=params)
    )
    if triton_backend == "python":
        src = python_model_source(spec.name)
        if src is None:
            raise FileNotFoundError(
                f"{spec.name} is a python-backend model but has no model.py at "
                f"{PYTHON_MODEL_SRC / spec.name / 'model.py'}"
            )
        shutil.copyfile(src, layout.weight_file)
    return layout


def resolve_spec(model_id: str) -> CVModelSpec:
    if model_id not in CV_MODELS:
        raise ValueError(
            f"no CV spec for {model_id!r}; known: {sorted(CV_MODELS)}. "
            "kosmos-2.5 / paddleocr use the Python backend and are staged separately."
        )
    return CV_MODELS[model_id]


# ─────────────────────────── multi-GPU placement ───────────────────────────
#
# One Triton container per GPU that has CV tenants on it. A single shared
# container would force every Triton-served model onto one card, which makes two
# of the three four-tenant/two-GPU pairings in the Phase-5 placement study
# (docs/contention-coverage.md) inexpressible.
#
# Ports: two containers cannot bind the same host port, so each device's block
# is offset from the configured base by a fixed stride:
#
#     http    = base_http + STRIDE * device      # 8100, 8110, 8120, …
#     grpc    = http + 1
#     metrics = http + 2
#
# Device 0 therefore keeps EXACTLY the ports every existing single-GPU config
# already uses (base, base+1, base+2), so nothing that omits `device:` moves.
# The stride is 10, not 3: it leaves slack so one device's listeners can never
# run into the next device's block if Triton ever grows a fourth port, and it
# keeps the mapping readable at a glance (GPU n ⇒ …n0).
TRITON_PORT_STRIDE_PER_DEVICE = 10

# Device 0 keeps the historical bare name; other devices are suffixed. The name
# is also the orchestrator's launch mutex, so it must be per-device or two
# tenants on different cards would silently share one container.
TRITON_CONTAINER_PREFIX = "triton-cv"

# Placement is per-card, and a machine we would run this on tops out at 8.
MAX_GPU_INDEX = 7


def resolve_triton_device(device: int | list[int] | None) -> int:
    """Normalise a Triton tenant's placement to the ONE GPU index it runs on.

    A Triton tenant is single-GPU by construction: the CV models are exported as
    whole-graph ONNX/TensorRT engines and placed with `instance_group
    { kind: KIND_GPU }`, which replicates a model per card — it does not shard
    one. A `device:` list means tensor parallelism, which none of these models
    are built for, so it is rejected rather than half-honoured by picking the
    first index. `None` is the pre-placement default: GPU 0.
    """
    if device is None:
        return 0
    if isinstance(device, bool):        # bool is an int subclass; `device: true` is a typo
        raise ValueError(f"triton device must be an int, got {device!r}")
    if isinstance(device, int):
        dev = device
    elif isinstance(device, (list, tuple)):
        if len(device) != 1:
            raise ValueError(
                f"Triton tenants are single-GPU — device={list(device)} requests tensor "
                "parallelism, which the CV models are not exported for. Give one index."
            )
        return resolve_triton_device(device[0])
    else:
        raise ValueError(f"triton device must be an int or a 1-element list, got {device!r}")
    if not 0 <= dev <= MAX_GPU_INDEX:
        raise ValueError(f"triton device index {dev} out of range (0..{MAX_GPU_INDEX})")
    return dev


def triton_ports(base_http_port: int = 8000, device: int | list[int] | None = None) -> tuple[int, int, int]:
    """(http, grpc, metrics) for the container on `device`. See the stride note above."""
    dev = resolve_triton_device(device)
    http = base_http_port + TRITON_PORT_STRIDE_PER_DEVICE * dev
    return http, http + 1, http + 2


def triton_container_name(device: int | list[int] | None = None,
                          *, prefix: str = TRITON_CONTAINER_PREFIX) -> str:
    """Container name for the Triton server on `device` — `triton-cv` on GPU 0,
    `triton-cv-gpu<N>` elsewhere. Doubles as the per-device launch mutex."""
    dev = resolve_triton_device(device)
    return prefix if dev == 0 else f"{prefix}-gpu{dev}"


# ─────────────────────────── container commands ────────────────────────────

def build_triton_serve_cmd(
    repo_root: Path, *, device: int | list[int] | None = None,
    http_port: int = 8000, grpc_port: int = 8001,
    metrics_port: int = 8002, shm_size: str = "1g",
    container_name: str = TRITON_CONTAINER_PREFIX, image: str | None = None,
    mps_pipe_dir: str | None = None, models: list[str] | None = None,
    extra_mounts: list[tuple[str, str]] | None = None,
) -> list[str]:
    """`docker run` for the Triton server hosting the CV model repo on ONE GPU.

    Notes baked in:
      - `--gpus device=<N>` pins the container to a single card. That is the
        docker form for one index (`--gpus all` would expose every GPU and let
        Triton place instances wherever it likes, which is exactly what the
        placement study must control). `device` is the physical index; unlike
        the HTTP tenants we do NOT use CUDA_VISIBLE_DEVICES, because Docker
        already hides the other cards from the container.
      - `--allow-client-shm=true` — off by default since Triton 26.04; without
        it large CV tensors read as a model regression that is really
        serialization overhead (serving-topology.md gotcha).
      - MPS: the pipe dir is passed through so Triton's CUDA context joins the
        same MPS control the LLM tenants use (isolation is fixed MPS-on).
      - `models`: with per-GPU containers, a card must load ONLY the models
        placed on it — a stray model costs VRAM on a card that does not need it
        and perturbs the very contention we are measuring. The per-device repo
        already holds only those models; `--model-control-mode=explicit` plus
        one `--load-model` each makes that a server-side guarantee too, which
        matters because GPU 0's repo is also the shared staging dir the weight
        exporter writes into and may carry other devices' models on disk.
      - `--shm-size` is Docker's; the Triton flags follow the image.
    """
    dev = resolve_triton_device(device)
    # Per-container image: a repo holding a python-backend model needs torch and
    # transformers, which only the derived image carries.
    image = image or image_for_models(models)
    cmd = [
        "docker", "run", "--rm", "-d", "--name", container_name,
        "--gpus", f"device={dev}", "--shm-size", shm_size, "--network", "host",
        "-v", f"{Path(repo_root).resolve()}:/models",
    ]
    # Workload payloads a python-backend model.py reads at load (the document
    # image, the prompt jsonl). Mounted read-only at the same absolute path the
    # host uses, so config.pbtxt parameters need no translation.
    for src, dst in (extra_mounts or []):
        cmd += ["-v", f"{src}:{dst}:ro"]
    # A python-backend model loads its weights from the HuggingFace cache at
    # startup, inside the container. Without the cache mounted it has nowhere to
    # read them from and — running as the caller's uid, with no HOME — nowhere
    # to download them to either, so the model never loads and the container
    # exits before Triton is ready.
    if any((CV_MODELS.get(m) or CVModelSpec("", "", "", "", [], "", [])).is_python_backend
           for m in (models or [])):
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cmd += ["-e", f"HF_HOME={hf_home}", "-v", f"{hf_home}:{hf_home}"]
        token = os.environ.get("HF_TOKEN")
        if token:
            cmd += ["-e", f"HF_TOKEN={token}"]

    if mps_pipe_dir:
        # Share the host MPS control pipe so kernels co-schedule with the LLM.
        #
        # `--user` is not optional here. MPS servers are per-UID and a non-root
        # control daemon cannot spawn one for a different UID, so the container's
        # default root would fail with CUDA error 805 ("MPS client failed to
        # connect"): the daemon is normally started by the ordinary user running
        # the study. Matching the caller's uid:gid makes the container a client
        # of the daemon that already exists. Live-verified on PRO 6000 — as root
        # every model reports "unable to get number of CUDA devices"; with
        # --user, Triton is READY in ~2s and appears in get_client_list.
        #
        # `--ipc=host` is NOT required — verified by running the same probe with
        # and without it — so it is deliberately omitted rather than relaxing a
        # namespace on a guess.
        cmd += ["-e", f"CUDA_MPS_PIPE_DIRECTORY={mps_pipe_dir}",
                "-v", f"{mps_pipe_dir}:{mps_pipe_dir}",
                "--user", f"{os.getuid()}:{os.getgid()}"]
    cmd += [
        image, "tritonserver", "--model-repository=/models",
        f"--http-port={http_port}", f"--grpc-port={grpc_port}",
        f"--metrics-port={metrics_port}", "--allow-client-shm=true",
    ]
    if models:
        cmd.append("--model-control-mode=explicit")
        cmd += [f"--load-model={m}" for m in models]
    return cmd


def triton_ready_url(http_port: int = 8000, host: str = "localhost") -> str:
    """Triton's readiness endpoint — 200 once every model it was told to load has.

    Per-device containers each expose their own; the caller passes the port for
    the device it is waiting on (see triton_ports)."""
    return f"http://{host}:{http_port}/v2/health/ready"


def wrap_perf_analyzer_docker(
    inner_cmd: list[str], *, sdk_image: str = TRITON_SDK_IMAGE,
    mounts: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Wrap a bare perf_analyzer command to run inside the SDK container.

    perf_analyzer ships in the -sdk image, not on the host. It needs host
    networking to reach Triton and the input image mounted into the container.
    `inner_cmd[0]` is expected to be `perf_analyzer`.
    """
    cmd = ["docker", "run", "--rm", "--network", "host"]
    for host_path, cont_path in (mounts or []):
        cmd += ["-v", f"{host_path}:{cont_path}"]
    cmd += [sdk_image, *inner_cmd]
    return cmd
