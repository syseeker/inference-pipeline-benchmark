"""NitroGen quantization via NVIDIA modelopt.

Wraps `modelopt.torch.quantization.mtq.quantize` with a recipe tuned for
the NitroGen flow-matching policy. The wins from PTQ are in the
**weight-heavy Linear layers** of the DiT and the SigLIP-2 vision tower;
the action-decoder head, AdaLN norms, and timestep embeddings stay at
the compute dtype because they're (a) tiny in flops and (b) more
sensitive to amax error (per the docstring note we promise in the
docs/nitrogen.md write-up).

Returns the quantized PyTorch model. Export to ONNX / TensorRT lives in
`benchmarks/nitrogen_export.py` so the calibration phase is testable
without modelopt's optional `onnx` + `tensorrt` deps.

Usage:
    from benchmarks.nitrogen_quant import quantize_for_serving
    quantize_for_serving(session.model, precision="fp8", calib_images=imgs)
    # session.model is now amax-calibrated + fake-quantized
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# NitroGen's action decoder / output-projection layer names — we exclude
# these from quantization to preserve action-vector fidelity. The DiT and
# vision tower are quantized; everything else stays in compute dtype.
#
# Names verified against /home/ubuntu/NitroGen/nitrogen/flow_matching_transformer/nitrogen.py
# — `action_head`, `proprio_emb`, `time_emb`, action token embeddings.
_EXCLUDE_PATTERNS = [
    "*action_head*",
    "*action_decoder*",
    "*proprio_emb*",
    "*time_emb*",
    "*timestep*",
    "*norm*",     # AdaLN layers — these are LayerNorm/RMSNorm, often
                   # not quantized by default but we belt-and-braces it
    "*ada_ln*",
    "*pos_embed*",
]


def _build_quant_cfg(precision: str) -> dict[str, Any]:
    """Return a modelopt quant config for `precision`, with NitroGen exclusions appended.

    `precision` is the harness's Precision enum string value (`"fp8"` or
    `"nvfp4"`). Other values are rejected — modelopt PTQ doesn't help bf16/fp16.
    """
    import modelopt.torch.quantization as mtq

    base = {
        "fp8":   mtq.FP8_DEFAULT_CFG,
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    }.get(precision)
    if base is None:
        raise ValueError(
            f"unsupported precision {precision!r} for modelopt PTQ; expected 'fp8' or 'nvfp4'"
        )

    cfg = copy.deepcopy(base)
    # modelopt 0.44+ schema: quant_cfg is a list of rules in priority order.
    # Append our exclusions at the end so they override defaults for
    # action-decoder / norm / embedding patterns.
    for pattern in _EXCLUDE_PATTERNS:
        cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    return cfg


def quantize_for_serving(
    model: Any,
    *,
    precision: str,
    calib_images: Iterable[Any],
    predict_fn: Any | None = None,
    batch_size: int = 1,
) -> Any:
    """Calibrate + apply PTQ to a NitroGen model in place.

    `model` is the `nitrogen.flow_matching_transformer.nitrogen.NitroGen`
    instance from `session.model`. After this call the model has fake-quant
    nodes inserted around its weight-heavy Linears and amax statistics
    learned from the calibration data. Export to a hardware-FP8 runtime
    happens separately (`benchmarks/nitrogen_export.py`).

    `predict_fn(model, image)` is the forward shim used during calibration.
    Defaults to running through `InferenceSession.predict` via a temp session
    wrapping the model — but tests can pass in a stub.

    `calib_images`: iterable of HxWx3 RGB numpy arrays (game frames). A few
    dozen real-distribution samples is typical; more = tighter amax. With
    only a handful of samples (today's synthetic-frames mode) the amax is
    biased — flag that in the model card.
    """
    import modelopt.torch.quantization as mtq

    quant_cfg = _build_quant_cfg(precision)

    # The calibration loop modelopt expects: drive the model through enough
    # real inputs that every quantized op sees its true activation range.
    images = list(calib_images)
    if not images:
        raise ValueError(
            "calib_images is empty — refuse to ptq-calibrate without data. "
            "Pass at least a few representative frames; ~32 is a reasonable floor."
        )

    if predict_fn is None:
        from nitrogen.inference_client import InferenceSession  # type: ignore[import-not-found]

        # We need to drive `model.forward` (or the get_action equivalent)
        # through real inputs. The InferenceSession orchestrates the
        # pre/post-process around `model`, so the cleanest path is to wrap
        # it. But the calibration only cares about activations through
        # quantized layers — any forward that exercises them is fine.
        def _drive(image: Any) -> None:
            # Caller must already have set `session.model = model` so the
            # session uses our (about-to-be-quantized) instance. We delegate
            # to the same predict path the serve loop uses.
            _ = InferenceSession  # noqa: F841 (anchor for linters)
            raise NotImplementedError(
                "Pass `predict_fn(model, image)` explicitly — the default "
                "session-based path needs the caller to thread the session "
                "through. See scripts/serve_nitrogen.py for the wiring."
            )

        predict_fn = _drive

    def _forward_loop(_model: Any) -> None:
        # modelopt calls this once after instrumenting the model; we
        # iterate `images` and let predict_fn drive the forward path.
        for img in images:
            predict_fn(_model, img)

    mtq.quantize(model, quant_cfg, forward_loop=_forward_loop)
    return model


def load_calib_images_from_scenarios(*scenarios_dirs: Path) -> list[Any]:
    """Pull `screen.png` from every scenario across `scenarios_dirs`.

    Real-distribution calibration is better than synthetic noise — the
    amax values learned from uniform noise are systematically off the
    distribution of game frames. The bundled VLM scenarios under
    `tests/smoke/scenarios/` are real game screenshots and worth
    pulling alongside the policy scenarios under `scenarios_nitrogen/`,
    especially when the latter were built with `--synthetic-frames`.

    Returns RGB numpy arrays at the on-disk resolution; the caller is
    expected to feed them through NitroGen's image processor (which
    resizes to 256×256). Caller is responsible for ordering and
    deduplication. Missing dirs are skipped silently — pass at least
    one valid dir.
    """
    import numpy as np
    from PIL import Image

    images: list[Any] = []
    seen = False
    for sd in scenarios_dirs:
        if not Path(sd).is_dir():
            continue
        seen = True
        for sc_dir in sorted(Path(sd).iterdir()):
            if not sc_dir.is_dir():
                continue
            for ext in ("screen.png", "screen.jpeg", "screen.jpg"):
                png = sc_dir / ext
                if png.exists():
                    with Image.open(png) as im:
                        images.append(np.array(im.convert("RGB"), dtype=np.uint8))
                    break
    if not seen:
        raise FileNotFoundError(
            f"no scenarios dir among {[str(d) for d in scenarios_dirs]} exists"
        )
    return images
