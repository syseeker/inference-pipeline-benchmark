#!/usr/bin/env python3
"""Export a CV model and stage it into a Triton model repository (step 7).

Produces the on-disk layout Triton loads:

    <repo-root>/<model>/
      config.pbtxt          # from benchmarks.triton_cv (pure)
      1/
        model.onnx          # onnx backend  (this script exports it)
        model.plan          # tensorrt backend — built IN the Triton container
        model.py            # python backend — hand-authored (kosmos/paddleocr)

Usage:
    # ONNX backend (portable baseline) — fully handled here
    python scripts/build_triton_cv_repo.py --model yolov8-l --triton-backend onnx \
        --repo-root benchmarks/results/rtx_pro6000/triton_repo

    # TensorRT backend — exports ONNX, then prints the in-container trtexec step.
    # The .plan MUST be built with the container's TensorRT to load, so we don't
    # build it on the host.

Needs the CV deps (requirements-contention.txt) in the active venv. Run under a
torch-capable venv, e.g. .venv-vllm/bin/python.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.triton_cv import (  # noqa: E402
    TRITON_IMAGE, model_filename, resolve_spec, write_model_repo,
)


def export_onnx(spec, dest: Path, *, max_batch_size: int, imgsz: int) -> None:
    """Export the model's weights to ONNX at `dest`."""
    if spec.exporter == "ultralytics":
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(f"{spec.weights}.pt")   # downloads the .pt on first use
        # dynamic=True → dynamic batch dim so Triton max_batch_size works.
        out = model.export(format="onnx", dynamic=True, batch=max_batch_size, imgsz=imgsz)
        shutil.copyfile(out, dest)
    elif spec.exporter == "torch-hf":
        import torch  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(spec.hf_id).eval()
        c, h, w = spec.input_dims
        dummy = torch.randn(1, c, h, w)
        torch.onnx.export(
            model, dummy, str(dest),
            input_names=[spec.input_name], output_names=[spec.output_name],
            dynamic_axes={spec.input_name: {0: "batch"}, spec.output_name: {0: "batch"}},
            opset_version=17,
        )
    else:
        raise ValueError(
            f"model {spec.name!r} has exporter {spec.exporter!r} — no ONNX export path "
            "(Python-backend model; author model.py by hand)."
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage a CV model into a Triton model repo.")
    p.add_argument("--model", required=True, help="CV model id (yolov8-l, dinov2-base, …).")
    p.add_argument("--triton-backend", choices=("onnx", "tensorrt", "python"), default="onnx")
    p.add_argument("--repo-root", required=True, type=Path, help="Triton model repository root.")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640, help="YOLO export input size.")
    p.add_argument("--version", type=int, default=1)
    args = p.parse_args(argv)

    spec = resolve_spec(args.model)
    layout = write_model_repo(
        args.repo_root, spec, args.triton_backend,
        max_batch_size=args.max_batch_size, version=args.version,
    )
    print(f">> wrote {layout.config_pbtxt}")

    if args.triton_backend == "onnx":
        print(f">> exporting {spec.name} → {layout.weight_file}")
        export_onnx(spec, layout.weight_file, max_batch_size=args.max_batch_size, imgsz=args.imgsz)
        print(f">> done: {layout.weight_file} ({layout.weight_file.stat().st_size} bytes)")
    elif args.triton_backend == "tensorrt":
        # The plan must be built with the container's TensorRT or it won't load.
        onnx_tmp = layout.version_dir / "model.onnx"
        print(f">> exporting {spec.name} → {onnx_tmp} (intermediate ONNX)")
        export_onnx(spec, onnx_tmp, max_batch_size=args.max_batch_size, imgsz=args.imgsz)
        plan = layout.version_dir / model_filename("tensorrt")
        print(
            "\n>> Build the TensorRT plan INSIDE the Triton container so the TRT "
            "version matches (host trtexec would produce an unloadable plan):\n"
            f"   docker run --rm --gpus all -v {layout.version_dir}:/w {TRITON_IMAGE} \\\n"
            f"     trtexec --onnx=/w/model.onnx --saveEngine=/w/{plan.name} \\\n"
            f"       --minShapes={spec.input_name}:1x{'x'.join(map(str, spec.input_dims))} \\\n"
            f"       --optShapes={spec.input_name}:{args.max_batch_size}x{'x'.join(map(str, spec.input_dims))} \\\n"
            f"       --maxShapes={spec.input_name}:{args.max_batch_size}x{'x'.join(map(str, spec.input_dims))} --fp16\n"
        )
    else:  # python
        print(
            f">> Python backend: author {layout.version_dir / 'model.py'} by hand "
            f"(TritonPythonModel wrapping {spec.hf_id})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
