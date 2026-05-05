"""Live smoke tests for local vLLM / SGLang / TRT-LLM backends.

Each test is parametrised over the three game scenarios. The server must be
running before the test suite is collected — if the port is not reachable
the whole parametrised group is skipped rather than failed.

Usage
-----
    # vLLM — start server first (Shell 1), then in Shell 2:
    source .venv-vllm/bin/activate
    pytest -m vllm tests/smoke/test_local_backends.py -v

    # SGLang
    source .venv-sglang/bin/activate
    pytest -m sglang tests/smoke/test_local_backends.py -v

    # TRT-LLM (trtllm-serve on port 8002)
    source .venv-trtllm/bin/activate
    pytest -m trtllm tests/smoke/test_local_backends.py -v

    # All backends that happen to be up right now
    pytest tests/smoke/test_local_backends.py -v

Override URLs / model ids via environment variables:
    VLLM_BASE_URL, VLLM_MODEL, SGLANG_BASE_URL, SGLANG_MODEL, TRTLLM_BASE_URL
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from tests.smoke.scenarios.loader import list_scenarios, load_scenario
from vlm_pipeline import Pipeline
from vlm_pipeline.config import PipelineConfig

# ── server reachability (evaluated once at collection time) ───────────────────


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


_cfg = PipelineConfig.from_env()

_VLLM_UP = _reachable(f"{_cfg.vllm.base_url.rstrip('/')}/models")
_SGLANG_UP = _reachable(f"{_cfg.sglang.base_url.rstrip('/')}/models")
_TRTLLM_UP = _reachable(f"{_cfg.trtllm.base_url.rstrip('/')}/models")

_SCENARIOS = list_scenarios()

# ── vLLM ──────────────────────────────────────────────────────────────────────


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.smoke
@pytest.mark.skipif(not _VLLM_UP, reason="vLLM server not reachable — start it first (see INFERENCE_BACKENDS B.1)")
@pytest.mark.parametrize("scenario_name", _SCENARIOS)
def test_vllm_scenario(scenario_name: str) -> None:
    from vlm_pipeline.reasoners.vllm_backend import VllmReasoner

    sc = load_scenario(scenario_name)
    pipe = Pipeline(reasoner=VllmReasoner(_cfg.vllm), config=_cfg)
    resp = pipe.run(sc.pipeline_request())

    assert resp.latency.total_ms > 0, "no timing recorded"
    assert resp.model_meta is not None, "model_meta missing"
    assert resp.model_meta.framework == "vllm"


# ── SGLang ────────────────────────────────────────────────────────────────────


@pytest.mark.sglang
@pytest.mark.gpu
@pytest.mark.smoke
@pytest.mark.skipif(not _SGLANG_UP, reason="SGLang server not reachable — start it first (see INFERENCE_BACKENDS B.2)")
@pytest.mark.parametrize("scenario_name", _SCENARIOS)
def test_sglang_scenario(scenario_name: str) -> None:
    from vlm_pipeline.reasoners.sglang_backend import SglangReasoner

    sc = load_scenario(scenario_name)
    pipe = Pipeline(reasoner=SglangReasoner(_cfg.sglang), config=_cfg)
    resp = pipe.run(sc.pipeline_request())

    assert resp.latency.total_ms > 0, "no timing recorded"
    assert resp.model_meta is not None, "model_meta missing"
    assert resp.model_meta.framework == "sglang"


# ── TRT-LLM ───────────────────────────────────────────────────────────────────


@pytest.mark.trtllm
@pytest.mark.gpu
@pytest.mark.smoke
@pytest.mark.skipif(not _TRTLLM_UP, reason="TRT-LLM server not reachable — run trtllm-serve on port 8002 first (see INFERENCE_BACKENDS B.3)")
@pytest.mark.parametrize("scenario_name", _SCENARIOS)
def test_trtllm_scenario(scenario_name: str) -> None:
    from vlm_pipeline.reasoners.trtllm_backend import TrtLlmReasoner

    sc = load_scenario(scenario_name)
    pipe = Pipeline(reasoner=TrtLlmReasoner(_cfg.trtllm), config=_cfg)
    resp = pipe.run(sc.pipeline_request())

    assert resp.latency.total_ms > 0, "no timing recorded"
    assert resp.model_meta is not None, "model_meta missing"
    assert resp.model_meta.framework == "trtllm"
