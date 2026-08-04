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
    """Export the model's weights to ONNX at `dest`, at the spec's precision.

    Precision is baked into the graph here rather than requested at plan-build
    time: TensorRT v11 networks are strongly typed and `trtexec --fp16` no
    longer exists, so an fp32 export can only ever produce an fp32 engine no
    matter what the GPU yaml says. Exporting fp16 needs a GPU on both paths —
    half-precision ops are largely unimplemented on CPU.
    """
    half = spec.torch_dtype_is_half
    if spec.exporter == "ultralytics":
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(f"{spec.weights}.pt")   # downloads the .pt on first use
        # dynamic=True → dynamic batch dim so Triton max_batch_size works.
        # quantize=16|32 replaces the deprecated half= arg (ultralytics 8.4.x).
        # fp16 needs a CUDA device; ultralytics refuses it on CPU.
        out = model.export(format="onnx", dynamic=True, batch=max_batch_size,
                           imgsz=imgsz, quantize=16 if half else 32,
                           device=0 if half else "cpu")
        # move, not copy: ultralytics writes the .onnx next to the .pt in cwd,
        # which is the repo root. Copying leaves an 84 MB duplicate behind.
        shutil.move(out, dest)
    elif spec.exporter == "torch-hf":
        import torch  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(spec.hf_id).eval()
        c, h, w = spec.input_dims
        dummy = torch.randn(1, c, h, w)
        if half:
            model, dummy = model.half().cuda(), dummy.half().cuda()
        torch.onnx.export(
            model, dummy, str(dest),
            input_names=[spec.input_name], output_names=[spec.output_name],
            dynamic_axes={spec.input_name: {0: "batch"}, spec.output_name: {0: "batch"}},
            # 18, not 17: torch has no 17 implementation for these ops, so it
            # exports at 18 and then tries to down-convert — which fails on
            # Resize and prints a RuntimeError traceback while silently keeping
            # the 18 graph. Asking for 18 skips the round trip and the noise.
            opset_version=18,
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
        # .resolve(): docker parses a non-absolute -v source as a NAMED VOLUME,
        # and a volume name may not contain "/", so a relative --repo-root (the
        # form the quickstart uses) yields a command that cannot run.
        print(
            "\n>> Build the TensorRT plan INSIDE the Triton container so the TRT "
            "version matches (host trtexec would produce an unloadable plan):\n"
            f"   docker run --rm --gpus all -v {layout.version_dir.resolve()}:/w {TRITON_IMAGE} \\\n"
            f"     trtexec --onnx=/w/model.onnx --saveEngine=/w/{plan.name} \\\n"
            f"       --minShapes={spec.input_name}:1x{'x'.join(map(str, spec.input_dims))} \\\n"
            f"       --optShapes={spec.input_name}:{args.max_batch_size}x{'x'.join(map(str, spec.input_dims))} \\\n"
            f"       --maxShapes={spec.input_name}:{args.max_batch_size}x{'x'.join(map(str, spec.input_dims))}\n"
            f"\n   The plan will be {spec.precision} — taken from the ONNX above, not "
            "from a flag.\n   TensorRT v11 networks are strongly typed and --fp16 has been "
            "REMOVED; passing\n   it fails with 'Unknown option'. To change precision, "
            "re-export, don't re-build.\n"
        )
    else:  # python
        from benchmarks.triton_cv import TRITON_PYTHON_IMAGE
        print(f">> Python backend: staged {layout.weight_file}")
        print(
            "\n>> This model runs INSIDE the Triton container and imports torch +\n"
            "   transformers, which the stock image does not ship. Build the derived\n"
            "   image once (~10 min, 11 GB); the tenant will not start without it:\n"
            f"     docker build -f docker/Dockerfile.triton-python -t {TRITON_PYTHON_IMAGE} .\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
