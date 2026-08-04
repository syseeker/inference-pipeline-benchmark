"""Triton CV model-repo tooling (benchmarks/triton_cv.py) — pure logic.

The weight export and container launch need CV deps / Docker; the config.pbtxt
generation, backend mapping and repo layout are pure and are where a wrong dim,
backend name or filename makes Triton refuse to load the model. Tested here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks import triton_cv as tc


def test_resolve_spec_known_and_unknown():
    spec = tc.resolve_spec("yolov8-l")
    assert spec.hf_id == "ultralytics/yolov8l"
    assert spec.input_dims == [3, 640, 640]
    with pytest.raises(ValueError):
        tc.resolve_spec("no-such-model")


def test_backend_mapping():
    assert tc.triton_backend_of("onnx") == "onnxruntime"
    assert tc.triton_backend_of("tensorrt") == "tensorrt"
    assert tc.triton_backend_of("python") == "python"


def test_backend_mapping_rejects_unknown():
    try:
        tc.triton_backend_of("openvino")
        assert False
    except ValueError:
        pass


def test_model_filename_per_backend():
    assert tc.model_filename("onnx") == "model.onnx"
    assert tc.model_filename("tensorrt") == "model.plan"
    assert tc.model_filename("python") == "model.py"


def test_config_pbtxt_onnx_shape():
    spec = tc.resolve_spec("yolov8-l")
    cfg = tc.build_config_pbtxt(spec, "onnx", max_batch_size=8, instance_count=1)
    assert 'name: "yolov8-l"' in cfg
    assert 'backend: "onnxruntime"' in cfg
    assert "max_batch_size: 8" in cfg
    assert 'name: "images"' in cfg
    assert "dims: [ 3, 640, 640 ]" in cfg          # per-sample, batch excluded
    assert 'name: "output0"' in cfg
    assert "dims: [ 84, -1 ]" in cfg          # anchor axis dynamic (dynamic=True export)
    assert "kind: KIND_GPU" in cfg
    assert "dynamic_batching { }" in cfg


def test_config_pbtxt_tensorrt_backend():
    cfg = tc.build_config_pbtxt(tc.resolve_spec("yolov8-l"), "tensorrt")
    assert 'backend: "tensorrt"' in cfg


def test_config_pbtxt_dynamic_batching_toggle():
    cfg = tc.build_config_pbtxt(tc.resolve_spec("dinov2-base"), "onnx", dynamic_batching=False)
    assert "dynamic_batching" not in cfg


def test_config_pbtxt_dinov2_dims():
    cfg = tc.build_config_pbtxt(tc.resolve_spec("dinov2-base"), "onnx")
    assert 'name: "pixel_values"' in cfg
    assert "dims: [ 3, 224, 224 ]" in cfg
    assert "dims: [ 257, 768 ]" in cfg


def test_repo_layout_paths():
    from pathlib import Path
    lay = tc.RepoLayout(repo_root=Path("/repo"), name="yolov8-l", triton_backend="onnx")
    assert lay.config_pbtxt == Path("/repo/yolov8-l/config.pbtxt")
    assert lay.version_dir == Path("/repo/yolov8-l/1")
    assert lay.weight_file == Path("/repo/yolov8-l/1/model.onnx")


def test_repo_layout_tensorrt_weight_file():
    from pathlib import Path
    lay = tc.RepoLayout(repo_root=Path("/repo"), name="yolov8-l", triton_backend="tensorrt")
    assert lay.weight_file == Path("/repo/yolov8-l/1/model.plan")


def test_triton_serve_cmd_has_shm_and_repo(tmp_path):
    cmd = tc.build_triton_serve_cmd(tmp_path, http_port=8000)
    assert cmd[0] == "docker" and "run" in cmd
    assert "--allow-client-shm=true" in cmd          # 26.04 gotcha
    assert "--model-repository=/models" in cmd
    assert any(a == f"{tmp_path.resolve()}:/models" for a in cmd)
    assert "tritonserver" in cmd


def test_triton_serve_cmd_passes_mps_pipe(tmp_path):
    cmd = tc.build_triton_serve_cmd(tmp_path, mps_pipe_dir="/tmp/nvidia-mps")
    assert "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" in cmd
    assert "/tmp/nvidia-mps:/tmp/nvidia-mps" in cmd


def test_triton_ready_url():
    assert tc.triton_ready_url(8000) == "http://localhost:8000/v2/health/ready"


# ── per-GPU placement ────────────────────────────────────────────────────────
#
# One container per card. What must hold: device 0 is byte-for-byte the
# single-GPU behaviour these configs already have, and two devices never
# collide on a name or a port.

def test_default_device_is_gpu0_everywhere():
    """No `device:` anywhere in the existing colocations, so the unspecified
    case has to resolve to the historical container name and ports."""
    assert tc.resolve_triton_device(None) == 0
    assert tc.triton_container_name() == "triton-cv"
    assert tc.triton_ports(8100) == (8100, 8101, 8102)


def test_triton_serve_cmd_default_is_todays_single_gpu_layout(tmp_path):
    cmd = tc.build_triton_serve_cmd(
        tmp_path, http_port=8100, grpc_port=8101, metrics_port=8102,
        container_name=tc.triton_container_name(0),
    )
    assert cmd[cmd.index("--name") + 1] == "triton-cv"
    assert "--http-port=8100" in cmd
    assert "--grpc-port=8101" in cmd
    assert "--metrics-port=8102" in cmd
    assert any(a == f"{tmp_path.resolve()}:/models" for a in cmd)
    assert cmd[cmd.index("--gpus") + 1] == "device=0"
    # No model list ⇒ no explicit control; the container loads the whole repo,
    # exactly as the single-GPU path always did.
    assert not any(a.startswith("--load-model") for a in cmd)
    assert "--model-control-mode=explicit" not in cmd


def test_ports_are_offset_per_device():
    assert tc.triton_ports(8100, 0) == (8100, 8101, 8102)
    assert tc.triton_ports(8100, 1) == (8110, 8111, 8112)
    assert tc.triton_ports(8100, 3) == (8130, 8131, 8132)
    # No two devices' blocks may overlap, or the second container fails to bind.
    blocks = [p for d in range(8) for p in tc.triton_ports(8100, d)]
    assert len(set(blocks)) == len(blocks)


def test_container_names_are_distinct_per_device():
    names = {tc.triton_container_name(d) for d in range(8)}
    assert len(names) == 8
    assert tc.triton_container_name(0) == "triton-cv"
    assert tc.triton_container_name(1) == "triton-cv-gpu1"


def test_serve_cmd_pins_a_single_gpu(tmp_path):
    cmd = tc.build_triton_serve_cmd(tmp_path, device=1)
    assert cmd[cmd.index("--gpus") + 1] == "device=1"
    assert "all" not in cmd


def test_serve_cmd_loads_only_its_own_models(tmp_path):
    cmd = tc.build_triton_serve_cmd(tmp_path, device=1, models=["yolov8-l", "dinov2-base"])
    assert "--model-control-mode=explicit" in cmd
    assert "--load-model=yolov8-l" in cmd
    assert "--load-model=dinov2-base" in cmd


def test_multi_device_triton_tenant_is_rejected():
    """CV models are not tensor-parallel; a device list must fail loudly rather
    than be half-honoured by taking the first index."""
    with pytest.raises(ValueError, match="single-GPU"):
        tc.resolve_triton_device([0, 1])
    with pytest.raises(ValueError, match="single-GPU"):
        tc.build_triton_serve_cmd(Path("/repo"), device=[0, 1])
    # A one-element list is just that one card.
    assert tc.resolve_triton_device([2]) == 2


def test_device_index_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        tc.resolve_triton_device(8)


def test_ready_url_follows_the_device_port():
    assert tc.triton_ready_url(tc.triton_ports(8100, 1)[0]) == (
        "http://localhost:8110/v2/health/ready"
    )


def test_wrap_perf_analyzer_docker():
    inner = ["perf_analyzer", "-m", "yolov8-l", "--service-kind", "triton"]
    wrapped = tc.wrap_perf_analyzer_docker(inner, mounts=[("/data", "/data")])
    assert wrapped[0] == "docker"
    assert "--network" in wrapped and "host" in wrapped
    assert "/data:/data" in wrapped
    assert wrapped[-4:] == ["-m", "yolov8-l", "--service-kind", "triton"]
    assert "perf_analyzer" in wrapped


def test_write_model_repo_creates_structure(tmp_path):
    spec = tc.resolve_spec("yolov8-l")
    lay = tc.write_model_repo(tmp_path, spec, "onnx", max_batch_size=4)
    assert lay.config_pbtxt.exists()
    assert lay.version_dir.is_dir()
    assert "max_batch_size: 4" in lay.config_pbtxt.read_text()
    # Weights are NOT written by write_model_repo (that's the exporter's job).
    assert not lay.weight_file.exists()


# ── precision ───────────────────────────────────────────────────────────────
#
# Regression: the CV plans were built fp32 while both models declared
# quantization: "fp16". Nothing caught it, because precision lived in neither
# registry — the pipeline exported fp32 ONNX and relied on `trtexec --fp16` to
# coerce it. TensorRT v11 removed that flag (networks are strongly typed and
# obey the ONNX), so the coercion silently stopped happening. dinov2-base is
# the study's designated bandwidth aggressor, so running it fp32 doubles the
# memory traffic of the single variable the study is built to measure.

def test_config_pbtxt_io_dtype_follows_precision():
    spec = tc.resolve_spec("yolov8-l")
    assert spec.precision == "fp16"
    cfg = tc.build_config_pbtxt(spec, "tensorrt")
    assert "TYPE_FP16" in cfg
    assert "TYPE_FP32" not in cfg, "I/O dtype must match the exported graph or Triton rejects the model"


def test_config_pbtxt_io_dtype_follows_fp32_precision():
    from dataclasses import replace
    spec = replace(tc.resolve_spec("yolov8-l"), precision="fp32")
    cfg = tc.build_config_pbtxt(spec, "tensorrt")
    assert "TYPE_FP32" in cfg
    assert "TYPE_FP16" not in cfg


@pytest.mark.parametrize("name", ["yolov8-n", "yolov8-l", "dinov2-base", "dinov2-large"])
def test_cv_spec_precision_matches_the_gpu_yaml(name):
    """CVModelSpec and the GPU yaml are separate registries with no link between
    them. If they disagree the export silently produces an engine at a precision
    the config never asked for — exactly the failure this test exists to stop."""
    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "benchmarks/configs/rtx_pro6000.yaml").read_text()
    )
    declared = (cfg.get("models") or {}).get(name, {}).get("quantization")
    if declared is None:
        pytest.skip(f"{name} is not registered in rtx_pro6000.yaml")
    assert tc.resolve_spec(name).precision == declared, (
        f"{name}: CVModelSpec says {tc.resolve_spec(name).precision!r}, "
        f"rtx_pro6000.yaml says {declared!r}"
    )


# ── MPS in the container ────────────────────────────────────────────────────
#
# Regression: passing the pipe dir alone made Triton FAIL where it had
# previously (wrongly) succeeded outside MPS — the container defaults to root,
# MPS servers are per-UID, and a non-root control daemon cannot spawn a server
# for a different UID. Live symptom on PRO 6000: every model UNAVAILABLE with
# "unable to get number of CUDA devices: MPS client failed to connect" and
# cuInit returning 805. With --user, Triton is READY in ~2s.

def test_mps_pipe_also_sets_user_to_match_the_daemon_uid():
    cmd = tc.build_triton_serve_cmd(Path("/repo"), mps_pipe_dir="/tmp/nvidia-mps")
    assert "--user" in cmd, "pipe dir without --user makes the container fail CUDA init entirely"
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_no_user_flag_when_mps_is_absent():
    # No MPS, no reason to override the image's user.
    cmd = tc.build_triton_serve_cmd(Path("/repo"))
    assert "--user" not in cmd


def test_ipc_host_is_not_used():
    """Verified unnecessary on real hardware: the same CUDA probe passes with
    and without it. Kept as a test so it is not re-added on a guess."""
    cmd = tc.build_triton_serve_cmd(Path("/repo"), mps_pipe_dir="/tmp/nvidia-mps")
    assert "--ipc=host" not in cmd


# ── python backend (kosmos-2.5) ─────────────────────────────────────────────
#
# Regression: kosmos-2.5 was named by 4 colocations but was in neither the
# CV_MODELS registry nor on disk as a model.py, so every one of them would have
# failed at the first CV tenant. Kosmos2_5ForConditionalGeneration is in
# neither vLLM's nor SGLang's registry, so Triton's python backend is the only
# way to serve it — and the stock Triton image ships numpy alone, no torch or
# transformers, which is the part nothing recorded.

def test_kosmos_is_registered_as_a_python_backend_model():
    spec = tc.resolve_spec("kosmos-2.5")
    assert spec.is_python_backend
    assert spec.exporter == "none"


def test_python_backend_model_has_a_model_py_on_disk():
    """The repo is not complete without it; Triton reports UNAVAILABLE."""
    assert tc.python_model_source("kosmos-2.5") is not None


def test_python_backend_repo_gets_its_model_py_copied(tmp_path):
    spec = tc.resolve_spec("kosmos-2.5")
    layout = tc.write_model_repo(tmp_path, spec, "python", max_batch_size=1)
    assert layout.weight_file.name == "model.py"
    assert layout.weight_file.exists()
    assert "TritonPythonModel" in layout.weight_file.read_text()


def test_python_backend_repo_needs_the_derived_image():
    """torch/transformers are absent from the stock image."""
    assert tc.image_for_models(["kosmos-2.5"]) == tc.TRITON_PYTHON_IMAGE
    assert tc.image_for_models(["yolov8-l", "dinov2-base"]) == tc.TRITON_IMAGE
    # A mixed repo must still get the derived image, or the python model fails.
    assert tc.image_for_models(["yolov8-l", "kosmos-2.5"]) == tc.TRITON_PYTHON_IMAGE


def test_python_backend_io_dtypes_are_not_derived_from_precision():
    """A trigger in, a string out — neither follows the weight precision."""
    cfg = tc.build_config_pbtxt(tc.resolve_spec("kosmos-2.5"), "python", max_batch_size=1)
    assert "TYPE_INT32" in cfg and "TYPE_STRING" in cfg


def test_config_parameters_use_repeated_map_entries():
    cfg = tc.build_config_pbtxt(
        tc.resolve_spec("kosmos-2.5"), "python", max_batch_size=1,
        params={"document_path": "/x/doc.png"},
    )
    assert 'parameters {\n  key: "document_path"' in cfg
    assert "parameters [" not in cfg, "map field, not a bracketed list"


def test_missing_model_py_is_a_loud_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "python_model_source", lambda name: None)
    with pytest.raises(FileNotFoundError, match="model.py"):
        tc.write_model_repo(tmp_path, tc.resolve_spec("kosmos-2.5"), "python")
