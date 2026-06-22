#!/usr/bin/env python3
"""One-shot builder: calibrate + export every NitroGen quant artifact.

Used to populate the HF model repo `syseeker/nitrogen-quant`. Customers
never run this — they download the pre-built artifacts via the manifest
(benchmarks/nitrogen_artifacts.yaml).

Reads the same `(precision, steps)` matrix the nitrogen-backends sweep
expects. Skips entries already present in the cache so re-runs are cheap.

Run:
    bench setup --backend nitrogen-quant     # if not already done
    python scripts/build_nitrogen_artifacts.py \\
        --ckpt /path/to/ng.pt \\
        --out  /ephemeral/cache/nitrogen-engines
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# (precision, steps) pairs the harness expects. Matches the rounds in
# benchmarks/configs/*.yaml's nitrogen-backends sweep.
TARGETS = [
    ("fp8",   16),
    ("fp8",   4),
    ("nvfp4", 16),
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to ng.pt")
    ap.add_argument("--out", default="/ephemeral/cache/nitrogen-engines", help="Cache root")
    ap.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = ap.parse_args()

    # Lazy imports so `--help` works on a CPU-only box.
    import torch  # noqa: F401
    from nitrogen.inference_session import InferenceSession

    from benchmarks.nitrogen_export import cache_paths, export_dit_to_onnx
    from benchmarks.nitrogen_quant import (
        load_calib_images_from_scenarios,
        quantize_for_serving,
    )

    repo_root = Path(__file__).resolve().parents[1]
    scen_roots = [
        repo_root / "tests" / "smoke" / "scenarios",
        repo_root / "tests" / "smoke" / "scenarios_nitrogen",
    ]
    calib = load_calib_images_from_scenarios(*scen_roots)
    print(f"[build] {len(calib)} calibration frames from {[str(r) for r in scen_roots]}")

    manifest: dict[str, dict] = {"artifacts": {}}
    cache_root = Path(args.out)

    for precision, steps in TARGETS:
        key = f"{precision}-steps{steps}"
        paths = cache_paths(precision, steps, cache_root)
        already = paths["onnx"].exists() and not args.force
        if already:
            print(f"[build] {key}: cached at {paths['onnx']} — reusing")
        else:
            print(f"[build] {key}: loading ckpt + calibrating + exporting...")
            t0 = time.time()
            # Fresh session per (precision, steps) — modelopt's quantize is
            # stateful and we don't want fp8 amax to leak into the nvfp4 run.
            session = InferenceSession.from_ckpt(args.ckpt, cfg_scale=1.0, context_length=1)
            session.selected_game = None

            def _drive(_m, frame):
                session.predict(frame)

            quantize_for_serving(
                session.model, precision=precision,
                calib_images=calib, predict_fn=_drive,
            )
            export_dit_to_onnx(session.model, precision=precision, steps=steps, cache_root=cache_root)
            print(f"[build] {key}: built in {time.time()-t0:.1f}s")
            del session
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Record manifest entry (sha256 + size for both .onnx and .onnx.data).
        onnx_data = paths["onnx"].with_suffix(".onnx.data")
        entry: dict[str, object] = {
            "onnx":        paths["onnx"].name,
            "onnx_sha256": _sha256(paths["onnx"]),
            "onnx_bytes":  paths["onnx"].stat().st_size,
        }
        if onnx_data.exists():
            entry["data"]        = onnx_data.name
            entry["data_sha256"] = _sha256(onnx_data)
            entry["data_bytes"]  = onnx_data.stat().st_size
        if paths["meta"].exists():
            entry["meta"] = json.loads(paths["meta"].read_text())
        manifest["artifacts"][key] = entry
        print(f"[build] {key}: sha256={entry['onnx_sha256'][:12]}… size={entry['onnx_bytes']:,}B"
              f"{' + data ' + entry['data_sha256'][:12] + '… ' + format(entry['data_bytes'], ',') + 'B' if 'data' in entry else ''}")

    out_manifest = repo_root / "benchmarks" / "nitrogen_artifacts.yaml"
    try:
        import yaml
        out_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
    except ImportError:
        out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[build] manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
