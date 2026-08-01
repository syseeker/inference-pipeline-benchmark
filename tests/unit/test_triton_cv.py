"""Triton CV model-repo tooling (benchmarks/triton_cv.py) — pure logic.

The weight export and container launch need CV deps / Docker; the config.pbtxt
generation, backend mapping and repo layout are pure and are where a wrong dim,
backend name or filename makes Triton refuse to load the model. Tested here.
"""

from __future__ import annotations

from benchmarks import triton_cv as tc


def test_resolve_spec_known_and_unknown():
    spec = tc.resolve_spec("yolov8-l")
    assert spec.hf_id == "ultralytics/yolov8l"
    assert spec.input_dims == [3, 640, 640]
    try:
        tc.resolve_spec("kosmos-2.5")
        assert False, "kosmos-2.5 has no CV spec (python backend)"
    except ValueError as e:
        assert "kosmos" in str(e).lower() or "no CV spec" in str(e)


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
