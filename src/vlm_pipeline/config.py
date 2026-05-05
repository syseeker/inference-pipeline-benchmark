"""Runtime configuration for the pipeline.

Loads from environment variables (and optionally a YAML file). Centralised
here so each stage reads one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class NimConfig:
    api_key: str | None = None
    base_url: str = "https://integrate.api.nvidia.com/v1"
    # NIM cloud's catalogue does NOT expose Qwen2.5-VL or Qwen3-VL today.
    # The only multimodal Qwen NIM is qwen/qwen3.5-397b-a17b (400B MoE).
    # For real Qwen3-VL benchmarks, self-host via INFERENCE_BACKENDS Mode B and
    # point NIM_BASE_URL at http://localhost:8001/v1 with NIM_MODEL set
    # to the local container's served id.
    model: str = "qwen/qwen3.5-397b-a17b"
    timeout_s: float = 60.0


@dataclass
class VllmConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class SglangConfig:
    base_url: str = "http://localhost:30000/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class TrtLlmConfig:
    base_url: str = "http://localhost:8002/v1"
    engine_dir: str = "trt_engines/qwen3-vl-8b"
    tokenizer_dir: str = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class TritonConfig:
    grpc_url: str = "localhost:8001"
    ensemble_name: str = "vlm_pipeline_ensemble"


@dataclass
class PipelineConfig:
    backend: str = "nim"  # nim | vllm | sglang | trtllm | triton
    deadline_ms: int = 1500  # interactive budget per request
    max_history_turns: int = 6
    nim: NimConfig = field(default_factory=NimConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    sglang: SglangConfig = field(default_factory=SglangConfig)
    trtllm: TrtLlmConfig = field(default_factory=TrtLlmConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)

    @classmethod
    def from_env(cls, yaml_path: str | Path | None = None) -> "PipelineConfig":
        cfg = cls()
        if yaml_path:
            data = yaml.safe_load(Path(yaml_path).read_text()) or {}
            cfg = _merge(cfg, data)

        # Env overrides — keep this list explicit so it's grep-able.
        cfg.nim.api_key = os.getenv("NIM_API_KEY", cfg.nim.api_key)
        cfg.nim.base_url = os.getenv("NIM_BASE_URL", cfg.nim.base_url)
        cfg.nim.model = os.getenv("NIM_MODEL", cfg.nim.model)
        cfg.vllm.base_url = os.getenv("VLLM_BASE_URL", cfg.vllm.base_url)
        cfg.vllm.model = os.getenv("VLLM_MODEL", cfg.vllm.model)
        cfg.sglang.base_url = os.getenv("SGLANG_BASE_URL", cfg.sglang.base_url)
        cfg.sglang.model = os.getenv("SGLANG_MODEL", cfg.sglang.model)
        cfg.trtllm.base_url = os.getenv("TRTLLM_BASE_URL", cfg.trtllm.base_url)
        cfg.triton.grpc_url = os.getenv("TRITON_GRPC_URL", cfg.triton.grpc_url)
        if backend := os.getenv("PIPELINE_BACKEND"):
            cfg.backend = backend
        return cfg


def _merge(cfg: PipelineConfig, data: dict) -> PipelineConfig:
    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        attr = getattr(cfg, k)
        if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
            for sk, sv in v.items():
                if hasattr(attr, sk):
                    setattr(attr, sk, sv)
        else:
            setattr(cfg, k, v)
    return cfg
