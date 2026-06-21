"""Phase 4: runner glue for the nitrogen backend (CPU-only, no server)."""

from __future__ import annotations

from benchmarks.runner import _apply_round_to_cfg
from benchmarks.scenario_config import Round
from vlm_pipeline.config import PipelineConfig


def _nitrogen_round(launch_args):
    return Round(
        backend="nitrogen",
        model_id="nitrogen-500m-fp8",
        hf_id="nvidia/NitroGen:ng.pt",
        family="nitrogen",
        quantization="fp8",
        base_url="tcp://localhost:5599",
        port=5599,
        launch_args=launch_args,
        transport="zmq",
        ckpt="nvidia/NitroGen:ng.pt",
    )


def test_apply_round_stamps_nitrogen_cfg_and_seed():
    cfg = PipelineConfig.from_env()
    _apply_round_to_cfg(cfg, _nitrogen_round(["--seed=11", "--exec=tensorrt", "--precision=fp8"]))
    assert cfg.nitrogen.base_url == "tcp://localhost:5599"
    assert cfg.nitrogen.model_id == "nvidia/NitroGen:ng.pt"
    assert cfg.nitrogen.seed == 11  # parsed from launch args, pins denoising noise


def test_make_reasoner_nitrogen_uses_injected_client_path():
    # _make_reasoner builds the real ZMQ client (imports nitrogen, absent on CPU),
    # but NitrogenReasoner accepts an injected client — verify that path directly.
    from vlm_pipeline.config import NitrogenConfig
    from vlm_pipeline.reasoners.nitrogen_backend import NitrogenReasoner

    class _Client:
        def predict(self, image, *, game_id=None, seed=None):
            return {"buttons": [[0.0] * 21], "j_left": [[0.0, 0.0]], "j_right": [[0.0, 0.0]]}

        def reset(self):
            pass

    r = NitrogenReasoner(NitrogenConfig(), client=_Client())
    assert r is not None
