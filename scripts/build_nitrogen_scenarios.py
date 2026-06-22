#!/usr/bin/env python3
"""Build benchmark scenarios from the nvidia/NitroGen dataset.

The HF dataset (`nvidia/NitroGen`) ships **action annotations only** — no
pixels. Each chunk is:

    actions/SHARD_####/<video_id>/<video_id>_chunk_####/
        actions_processed.parquet   # 17 buttons + j_left/j_right per frame
        actions_raw.parquet
        metadata.json               # url, game, resolution, frame indices, bboxes

This converter turns selected chunks into the harness's on-disk scenario
format (mirroring tests/smoke/scenarios/), so NitroGen can be benchmarked on
real frames with an accuracy-vs-gold ground truth:

    <out>/<name>/
        screen.png         # decoded frame, cropped to game area, resized 256x256
        request.json       # ScenarioRequest (game_id set, instruction="")
        expected.json      # ScenarioExpected (lossy ActionSequence + verdict)
        gold_action.json   # SIDECAR: faithful gamepad action + provenance
                           #          -> source of truth for the accuracy metric

To get the frame we must fetch the source video from `metadata.json:url`
(no download script is provided by the dataset) and decode it at the chunk's
frame index. Source videos rot, so the build loop **over-samples**: it walks
candidate chunks and keeps going past failures until `--n` scenarios succeed.

Heavy / platform deps (`yt-dlp`, `av`, `polars`) are imported lazily inside
the functions that need them, so the pure helpers below import and unit-test
on any box (no GPU, no network).

Example (on a networked instance):
    python scripts/build_nitrogen_scenarios.py \
        --actions-root /data/NitroGen/actions \
        --out tests/smoke/scenarios_nitrogen \
        --n 3 --game-mapping /path/to/ng.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pure-core imports only (always available once the package is installed).
from vlm_pipeline.adapters import Gamepad

DEFAULT_FRAME_SIZE = 256


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested on CPU)                                           #
# --------------------------------------------------------------------------- #


@dataclass
class ChunkMeta:
    """The fields we need out of a chunk's metadata.json."""

    url: str
    game: str
    width: int
    height: int
    frame_indices: list[int] = field(default_factory=list)
    game_area_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)

    @property
    def sample_frame_index(self) -> int:
        """A representative frame for the chunk (the midpoint)."""
        if not self.frame_indices:
            return 0
        return self.frame_indices[len(self.frame_indices) // 2]


def parse_metadata(meta: dict[str, Any]) -> ChunkMeta:
    """Extract the fields we use from a chunk metadata.json (tolerant to layout).

    Real schema (nvidia/NitroGen): video info nested under `original_video` with
    `resolution: [H, W]`, `url`, `start_frame`, `end_frame`. Top-level `game` and
    optional `bbox_game_area`. We accept the flat layout too for robustness.
    """

    ov = meta.get("original_video") or {}
    url = str(meta.get("url") or ov.get("url") or "")

    res = meta.get("resolution") or ov.get("resolution") or {}
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        # Dataset stores [H, W]; ChunkMeta tracks (width, height).
        height, width = int(res[0]), int(res[1])
    elif isinstance(res, dict):
        width = int(meta.get("width", res.get("width", 0)) or 0)
        height = int(meta.get("height", res.get("height", 0)) or 0)
    else:
        width = int(meta.get("width", 0) or 0)
        height = int(meta.get("height", 0) or 0)

    frames = meta.get("frame_indices") or meta.get("frames")
    if frames is None and ov:
        start = ov.get("start_frame")
        end = ov.get("end_frame")
        if start is not None and end is not None:
            frames = {"start": int(start), "end": int(end) + 1}
    frames = frames or []
    if isinstance(frames, dict):  # {"start": a, "end": b}
        start, end = int(frames.get("start", 0)), int(frames.get("end", 0))
        frames = list(range(start, end)) if end > start else [start]
    frame_indices = [int(f) for f in frames]

    bbox = meta.get("game_area_bbox") or meta.get("game_bbox") or meta.get("bbox_game_area")
    if isinstance(bbox, dict):
        # Some chunks ship bbox_game_area as {x, y, w, h}.
        bbox = [bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)]
    game_area_bbox = tuple(int(v) for v in bbox) if bbox else None  # type: ignore[assignment]

    return ChunkMeta(
        url=url,
        game=str(meta.get("game", "unknown")),
        width=width,
        height=height,
        frame_indices=frame_indices,
        game_area_bbox=game_area_bbox,  # type: ignore[arg-type]
    )


def resolve_game_id(game_label: str, game_mapping: dict[str, Any] | None) -> str:
    """Map a dataset `game` string to the game id the checkpoint expects.

    Identity fallback when no mapping is supplied (lets the converter run
    CPU-side without the checkpoint's game_mapping parquet).
    """

    if not game_mapping:
        return game_label
    if game_label in game_mapping:
        return str(game_mapping[game_label])
    # Try a normalized lookup before giving up.
    norm = game_label.strip().lower().replace(" ", "_")
    for k, v in game_mapping.items():
        if str(k).strip().lower().replace(" ", "_") == norm:
            return str(v)
    raise KeyError(
        f"game '{game_label}' not found in game_mapping ({len(game_mapping)} entries). "
        "Conditioning would be wrong; fix the mapping or exclude this game."
    )


def build_scenario_payloads(
    *,
    name: str,
    description: str,
    game_id: str,
    pad: Gamepad,
    deadline_ms: int,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the (request, gold) JSON payloads for one policy scenario.

    Pure: takes a Gamepad, returns dicts. No filesystem, no image.

    Note: we deliberately do NOT synthesise an `expected.json` ActionSequence
    here. A NitroGen-dataset scenario carries only the gamepad ground truth;
    text-VLM grading needs a human-authored instruction + expected command
    list, which lives in a separately authored sibling file (see the
    `author-vlm-grading` workflow in BENCHMARK_GUIDE.md). The runner's
    grader dispatches on which ground-truth files are present.
    """

    request = {
        "name": name,
        "description": description,
        "image_path": "screen.png",
        "instruction": "",  # NitroGen is conditioned on game_id, not text
        "context_history": [],
        "deadline_ms": deadline_ms,
        "game_id": game_id,
    }
    gold = {
        "game_id": game_id,
        "buttons": pad.buttons,
        "j_left": list(pad.j_left),
        "j_right": list(pad.j_right),
        "provenance": provenance,
    }
    return request, gold


def write_scenario(
    out_dir: Path,
    *,
    request: dict[str, Any],
    gold: dict[str, Any],
    image: Any,  # PIL.Image.Image
) -> None:
    """Write screen.png + request.json + gold_action.json for one policy scenario."""

    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "screen.png")
    (out_dir / "request.json").write_text(json.dumps(request, indent=2))
    (out_dir / "gold_action.json").write_text(json.dumps(gold, indent=2))


# --------------------------------------------------------------------------- #
# I/O layer (lazy heavy imports; exercised on networked / GPU instances)      #
# --------------------------------------------------------------------------- #


def iter_chunks(actions_root: Path) -> Iterator[Path]:
    """Yield chunk directories (those containing metadata.json) in sorted order."""

    for meta in sorted(actions_root.glob("SHARD_*/*/*/metadata.json")):
        yield meta.parent


def load_chunk_action(chunk_dir: Path, frame_index: int) -> Gamepad:
    """Read actions_processed.parquet and return the Gamepad at `frame_index`."""

    import polars as pl  # lazy

    df = pl.read_parquet(chunk_dir / "actions_processed.parquet")
    row_idx = min(frame_index, df.height - 1) if df.height else 0
    row = df.row(row_idx, named=True)
    return Gamepad.from_dataset_row(row)


def fetch_and_decode_frame(
    meta: ChunkMeta, frame_index: int, *, cache_dir: Path, frame_size: int
) -> Any:
    """Download the source video (cached) and decode one frame to a 256x256 PIL image.

    Raises on dead URL / decode failure so the caller can skip to the next chunk.
    """

    import av  # lazy
    from PIL import Image  # available, but kept local for symmetry

    video_path = _ensure_video(meta.url, cache_dir)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        target = None
        for i, frame in enumerate(container.decode(stream)):
            if i >= frame_index:
                target = frame
                break
        if target is None:
            raise RuntimeError(f"frame {frame_index} not found in {video_path}")
        img = target.to_image()  # PIL RGB

    if meta.game_area_bbox:
        x, y, w, h = meta.game_area_bbox
        img = img.crop((x, y, x + w, y + h))
    return img.convert("RGB").resize((frame_size, frame_size), Image.BICUBIC)


_FAILED_URLS: set[str] = set()


def _synthetic_frame(frame_size: int, *, seed: int) -> Any:
    """Deterministic placeholder frame for offline / dead-URL benchmark runs.

    Returns a `frame_size x frame_size` RGB PIL image with a per-seed noise tile so
    the vision encoder still sees varying inputs across scenarios. NitroGen will
    happily process it — the resulting gamepad prediction won't match real
    gameplay (that's what the real frame is for), but latency, throughput, GPU
    util and the bit-level FP8-vs-BF16 accuracy delta are still measurable.
    """

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(frame_size, frame_size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _ensure_video(url: str, cache_dir: Path) -> Path:
    """Download `url` into cache_dir via yt-dlp if not already present.

    Per-URL failure cache: once a URL fails we raise on subsequent calls
    without invoking yt-dlp — videos have ~1200 chunks each, no point
    retrying a dead URL 1200 times.
    """

    import subprocess

    if not url:
        raise RuntimeError("empty url")
    if url in _FAILED_URLS:
        raise RuntimeError(f"url previously failed: {url}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic cache key from the URL.
    import hashlib

    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    out_tmpl = str(cache_dir / f"{key}.%(ext)s")
    existing = [p for p in cache_dir.glob(f"{key}.*") if p.suffix not in {".part", ".ytdl"}]
    if existing:
        return existing[0]

    res = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[height<=720]/best", "-o", out_tmpl, url],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        _FAILED_URLS.add(url)
        raise RuntimeError(f"yt-dlp failed for {url}: {res.stderr.strip()[:300]}")
    found = [p for p in cache_dir.glob(f"{key}.*") if p.suffix not in {".part", ".ytdl"}]
    if not found:
        _FAILED_URLS.add(url)
        raise RuntimeError(f"yt-dlp produced no file for {url}")
    return found[0]


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def build(args: argparse.Namespace) -> int:
    actions_root = Path(args.actions_root)
    out_root = Path(args.out)
    cache_dir = Path(args.cache_dir)

    game_mapping: dict[str, Any] | None = None
    if args.game_mapping:
        game_mapping = _load_game_mapping(Path(args.game_mapping))

    built = 0
    attempted = 0
    seen_video_ids: set[str] = set()
    for chunk_dir in iter_chunks(actions_root):
        if built >= args.n:
            break
        # Pick one chunk per source video so N scenarios = N different videos.
        # chunk_dir layout: SHARD_xxxx/<video_id>/<video_id>_chunk_yyyy
        video_id = chunk_dir.parent.name
        if video_id in seen_video_ids:
            continue
        attempted += 1
        try:
            meta = parse_metadata(json.loads((chunk_dir / "metadata.json").read_text()))
            game_id = resolve_game_id(meta.game, game_mapping)
            frame_index = meta.sample_frame_index
            if args.synthetic_frames:
                image = _synthetic_frame(args.frame_size, seed=attempted)
            else:
                image = fetch_and_decode_frame(
                    meta, frame_index, cache_dir=cache_dir, frame_size=args.frame_size
                )
            pad = load_chunk_action(chunk_dir, frame_index)
            name = f"{built:02d}_{_slug(meta.game)}_{chunk_dir.name}"
            request, gold = build_scenario_payloads(
                name=name,
                description=f"NitroGen dataset frame from '{meta.game}' ({chunk_dir.name}).",
                game_id=game_id,
                pad=pad,
                deadline_ms=args.deadline_ms,
                provenance={
                    "chunk": str(chunk_dir.relative_to(actions_root)),
                    "url": meta.url,
                    "frame_index": frame_index,
                    "game": meta.game,
                },
            )
            write_scenario(out_root / name, request=request, gold=gold, image=image)
            built += 1
            seen_video_ids.add(video_id)
            print(f"[ok] {name}  (attempt {attempted})")
        except Exception as exc:  # skip dead URLs / decode errors, try the next chunk
            # Mark the whole video bad so we don't try its 1199 other chunks.
            seen_video_ids.add(video_id)
            print(f"[skip] {chunk_dir.name}: {exc}", file=sys.stderr)
            continue

    print(f"\nBuilt {built}/{args.n} scenarios from {attempted} candidate chunks -> {out_root}")
    return 0 if built == args.n else 1


def _load_game_mapping(path: Path) -> dict[str, Any]:
    """Load game_label -> game_id mapping from the checkpoint or a parquet/json file."""

    if path.suffix == ".pt":
        import torch  # lazy

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("ckpt_config", {}) or {}
        tok_cfg = cfg.get("tokenizer_cfg", {}) or {}
        gm_cfg = tok_cfg.get("game_mapping_cfg")
        # The released ng.pt sets game_mapping_cfg=None → unconditional model.
        # Return {} so resolve_game_id falls through to identity (game_id=label).
        if not isinstance(gm_cfg, dict):
            return {}
        gm = gm_cfg.get("mapping") or gm_cfg.get("game_mapping") or {}
        return {str(k): str(v) for k, v in gm.items()} if isinstance(gm, dict) else {}
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".parquet":
        import polars as pl  # lazy

        df = pl.read_parquet(path)
        return {
            str(r["game_label"]): str(r.get("game_id", r["game_label"]))
            for r in df.iter_rows(named=True)
        }
    raise ValueError(f"unsupported game_mapping file: {path}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s.strip().lower()).strip("_") or "game"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--actions-root", required=True, help="Path to the dataset's actions/ tree.")
    p.add_argument("--out", default="tests/smoke/scenarios_nitrogen", help="Output scenarios dir.")
    p.add_argument("--n", type=int, default=3, help="Scenarios to build (with fallback).")
    p.add_argument(
        "--cache-dir", default=".cache/nitrogen_videos", help="Source-video download cache."
    )
    p.add_argument(
        "--frame-size", type=int, default=DEFAULT_FRAME_SIZE, help="Output frame size (square)."
    )
    p.add_argument("--deadline-ms", type=int, default=1500, help="Scenario deadline_ms.")
    p.add_argument(
        "--game-mapping", default=None, help="Checkpoint .pt / .json / .parquet game mapping."
    )
    p.add_argument(
        "--synthetic-frames",
        action="store_true",
        help=(
            "Skip yt-dlp / video fetch and write a placeholder frame. Use on offline "
            "boxes or when source URLs are dead. Gold gamepad actions still come from "
            "the real parquet, so latency/throughput/accuracy-vs-gold remain measurable."
        ),
    )
    return build(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
