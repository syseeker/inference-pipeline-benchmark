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
    """TRT-LLM PyTorch backend via `trtllm-serve` over HTTP (OpenAI-compatible).
    Same client surface as vLLM/SGLang."""

    base_url: str = "http://localhost:8002/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"


@dataclass
class TritonConfig:
    grpc_url: str = "localhost:8001"
    ensemble_name: str = "vlm_pipeline_ensemble"


@dataclass
class NitrogenConfig:
    """NitroGen diffusion-policy backend (ZMQ, not HTTP/OpenAI).

    The server (`scripts/serve_nitrogen.py` / NitroGen's `serve.py`) holds the
    checkpoint + execution backend + precision + denoise steps; this client
    only needs the ZMQ endpoint and the per-request game id. `seed` is sent so
    the server can pin denoising noise — making FP8-vs-BF16 action deltas
    reflect precision, not sampling randomness.
    """

    base_url: str = "tcp://localhost:5555"  # ZMQ REQ endpoint
    model_id: str = "nvidia/NitroGen"       # checkpoint identity (for ModelMeta)
    game_id: str | None = None              # default game; per-request overrides
    seed: int = 0                           # pinned denoising seed
    move_scale: int = 512                   # joystick -> MOVE px scale (lossy adapter)
    timeout_ms: int = 30000


@dataclass
class PipelineConfig:
    backend: str = "nim"  # nim | vllm | sglang | trtllm | triton | nitrogen
    deadline_ms: int = 1500  # interactive budget per request
    max_history_turns: int = 6
    nim: NimConfig = field(default_factory=NimConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    sglang: SglangConfig = field(default_factory=SglangConfig)
    trtllm: TrtLlmConfig = field(default_factory=TrtLlmConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)
    nitrogen: NitrogenConfig = field(default_factory=NitrogenConfig)

    @classmethod
    def from_env(cls, yaml_path: str | Path | None = None) -> PipelineConfig:
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
        cfg.trtllm.model = os.getenv("TRTLLM_MODEL", cfg.trtllm.model)
        cfg.triton.grpc_url = os.getenv("TRITON_GRPC_URL", cfg.triton.grpc_url)
        cfg.nitrogen.base_url = os.getenv("NITROGEN_ZMQ_URL", cfg.nitrogen.base_url)
        cfg.nitrogen.model_id = os.getenv("NITROGEN_MODEL", cfg.nitrogen.model_id)
        cfg.nitrogen.game_id = os.getenv("NITROGEN_GAME_ID", cfg.nitrogen.game_id)
        if seed := os.getenv("NITROGEN_SEED"):
            cfg.nitrogen.seed = int(seed)
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
