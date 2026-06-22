"""Pre-built NitroGen quant artifact loader.

The calibration + ONNX export from `nitrogen_quant.py` + `nitrogen_export.py`
is a one-time deterministic operation (same ckpt + same calib set + same
modelopt version → bit-identical ONNX). We do it once on a known-good box,
upload the result to HuggingFace (`syseeker-at-nv/nitrogen-quant`), and the
customer downloads it instead of re-running calibration.

Manifest schema: see `benchmarks/nitrogen_artifacts.yaml`.

Usage:
    from benchmarks.nitrogen_artifacts import ensure_artifact
    onnx_path = ensure_artifact("fp8", 16)       # downloads if not cached
    # Now `onnx_path` is on disk with sha256-verified contents.

Force re-calibration (skip the artifact cache, e.g. for a customer with
a meaningfully different production frame distribution):
    NITROGEN_FORCE_RECALIBRATE=1
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "nitrogen_artifacts.yaml"

# HF repo that hosts the calibrated ONNX bundles. Public, no auth needed.
# Override via env if you fork the artifact set (e.g. you re-calibrated
# on your own frame distribution and uploaded to your own repo).
DEFAULT_HF_REPO = os.environ.get("NITROGEN_QUANT_HF_REPO", "syseeker-at-nv/nitrogen-quant")
DEFAULT_HF_REVISION = os.environ.get("NITROGEN_QUANT_HF_REVISION", "main")


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"artifacts": {}}
    try:
        import yaml

        return yaml.safe_load(MANIFEST_PATH.read_text()) or {"artifacts": {}}
    except ImportError:
        return json.loads(MANIFEST_PATH.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(precision: str, steps: int) -> str:
    return f"{precision}-steps{steps}"


def _cache_dir(precision: str, steps: int) -> Path:
    """Same layout as `nitrogen_export.cache_paths` so a downloaded
    artifact is indistinguishable from a locally-built one."""
    from benchmarks.nitrogen_export import cache_paths  # avoid cycle

    return cache_paths(precision, steps)["dir"]


def _hf_filename(precision: str, steps: int, basename: str) -> str:
    """The path inside the HF repo. Keep the same layout as on disk so
    `hf download` can be used with `--include` patterns if you only want
    one precision."""
    return f"{_key(precision, steps)}/{basename}"


def ensure_artifact(precision: str, steps: int) -> Path:
    """Return the local path to the calibrated ONNX, fetching from HF if needed.

    Resolution order:
      1. If the cache already has a file with the manifest's expected
         sha256, return it. No-op.
      2. Else if the manifest has an entry, `hf download` it into the
         cache dir, sha256-verify, return the path. Errors surface as
         RuntimeError with the offending key.
      3. Else (no manifest entry) raise FileNotFoundError pointing at the
         build script.

    Caller is responsible for the runtime swap — this only delivers the bytes.

    `$NITROGEN_FORCE_RECALIBRATE=1` skips the artifact path entirely; the
    caller is expected to fall back to calibrating + exporting locally.
    """
    if os.environ.get("NITROGEN_FORCE_RECALIBRATE", "").strip() in {"1", "true", "yes"}:
        raise FileNotFoundError(
            "NITROGEN_FORCE_RECALIBRATE set — bypassing the artifact cache. "
            "Caller must fall back to local calibration."
        )

    manifest = _load_manifest()
    key = _key(precision, steps)
    entry = manifest.get("artifacts", {}).get(key)
    if not entry:
        raise FileNotFoundError(
            f"no manifest entry for {key!r} in {MANIFEST_PATH.relative_to(REPO_ROOT)}. "
            f"Either add one (one-time bench team workflow) or run "
            f"`scripts/build_nitrogen_artifacts.py` locally."
        )

    cache_dir = _cache_dir(precision, steps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / entry["onnx"]
    data_path = cache_dir / entry["data"] if entry.get("data") else None

    # Fast path: cached file with the expected sha256.
    if onnx_path.exists():
        if _sha256(onnx_path) == entry["onnx_sha256"] and (
            data_path is None
            or (data_path.exists() and _sha256(data_path) == entry["data_sha256"])
        ):
            return onnx_path
        # Stale or partial — drop it.
        onnx_path.unlink()
        if data_path and data_path.exists():
            data_path.unlink()

    # Slow path: download via hf CLI (lazy import so the module loads on
    # CPU-only boxes that don't have huggingface-hub).
    from huggingface_hub import hf_hub_download

    print(f"[ng-artifacts] downloading {key} from {DEFAULT_HF_REPO}@{DEFAULT_HF_REVISION}...")
    downloaded_onnx = Path(
        hf_hub_download(
            repo_id=DEFAULT_HF_REPO,
            filename=_hf_filename(precision, steps, entry["onnx"]),
            revision=DEFAULT_HF_REVISION,
            local_dir=cache_dir.parent,  # honors the (precision-steps-sha) layout
        )
    )
    # hf_hub_download nests under the local_dir; normalise into our cache.
    if downloaded_onnx != onnx_path:
        downloaded_onnx.replace(onnx_path)

    if entry.get("data"):
        downloaded_data = Path(
            hf_hub_download(
                repo_id=DEFAULT_HF_REPO,
                filename=_hf_filename(precision, steps, entry["data"]),
                revision=DEFAULT_HF_REVISION,
                local_dir=cache_dir.parent,
            )
        )
        assert data_path is not None
        if downloaded_data != data_path:
            downloaded_data.replace(data_path)

    # Verify.
    got = _sha256(onnx_path)
    if got != entry["onnx_sha256"]:
        onnx_path.unlink()
        raise RuntimeError(
            f"{key} sha256 mismatch: expected {entry['onnx_sha256'][:12]}…, "
            f"got {got[:12]}…. Either the upstream repo was retagged or the "
            f"manifest is stale; re-pin or rebuild locally."
        )
    if data_path is not None:
        got_data = _sha256(data_path)
        if got_data != entry["data_sha256"]:
            data_path.unlink()
            raise RuntimeError(
                f"{key} data sha256 mismatch: expected {entry['data_sha256'][:12]}…, got {got_data[:12]}…"
            )

    print(f"[ng-artifacts] {key} ready at {onnx_path}")
    return onnx_path
