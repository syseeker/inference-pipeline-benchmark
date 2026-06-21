"""Unit tests for NitroGen execution-backend planning (CPU-only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from benchmarks.nitrogen_exec import (
    ExecBackend,
    ExecPlan,
    Precision,
    parse_exec_plan,
    requires_blackwell,
)

# Load the serve launcher (under scripts/) to test its pure build_plan helper.
_SERVE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "serve_nitrogen.py"
_spec = importlib.util.spec_from_file_location("serve_nitrogen", _SERVE_PATH)
serve_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = serve_mod
_spec.loader.exec_module(serve_mod)  # type: ignore[union-attr]


def test_default_plan_is_eager_bf16_16step():
    plan = parse_exec_plan([])
    assert plan.exec_backend is ExecBackend.EAGER
    assert plan.precision is Precision.BF16
    assert plan.steps == 16 and plan.cfg_scale == 1.0


def test_parse_equals_and_aliases():
    plan = parse_exec_plan(["--exec=trt", "--precision=fp8", "--steps=4", "--seed=7"])
    assert plan.exec_backend is ExecBackend.TENSORRT
    assert plan.precision is Precision.FP8
    assert plan.steps == 4 and plan.seed == 7
    assert plan.is_quantized
    assert plan.label() == "tensorrt-fp8-4step"


def test_parse_space_separated_and_compile_alias():
    plan = parse_exec_plan(["--exec", "compile", "--cfg", "1.5"])
    assert plan.exec_backend is ExecBackend.TORCH_COMPILE
    assert plan.cfg_scale == 1.5
    assert plan.label() == "torch_compile-bf16-16step-cfg1.5"


def test_unknown_values_raise():
    with pytest.raises(ValueError):
        parse_exec_plan(["--exec=quantum"])
    with pytest.raises(ValueError):
        parse_exec_plan(["--precision=int3"])
    with pytest.raises(ValueError):
        parse_exec_plan(["--steps=0"])


def test_torch_dtype_and_quantized_flags():
    assert parse_exec_plan(["--precision=bf16"]).torch_dtype_name == "bfloat16"
    assert parse_exec_plan(["--precision=fp16"]).torch_dtype_name == "float16"
    # FP8/NVFP4 keep a bf16 compute dtype; quantization is applied separately.
    fp8 = parse_exec_plan(["--precision=fp8"])
    assert fp8.torch_dtype_name == "bfloat16" and fp8.is_quantized
    assert requires_blackwell(parse_exec_plan(["--precision=nvfp4"]))
    assert not requires_blackwell(fp8)


def test_to_knobs_roundtrip():
    plan = ExecPlan(exec_backend=ExecBackend.ONNXRUNTIME, precision=Precision.FP16, steps=8)
    knobs = plan.to_knobs()
    assert knobs == {
        "exec_backend": "onnxruntime",
        "precision": "fp16",
        "denoise_steps": 8,
        "cfg_scale": 1.0,
        "seed": 0,
    }


def test_serve_launcher_build_plan():
    parser = serve_mod._build_parser()
    args = parser.parse_args(["ng.pt", "--exec", "onnx", "--precision", "fp8", "--steps", "2"])
    plan = serve_mod.build_plan(args)
    assert plan.exec_backend is ExecBackend.ONNXRUNTIME
    assert plan.precision is Precision.FP8 and plan.steps == 2
