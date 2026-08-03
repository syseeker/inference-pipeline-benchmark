#!/usr/bin/env python3
"""Emit the contention study's prompt files from experiment_config.json.

The customer's prompts live in `workspace/contention/experiment_config.json`
under the top-level `prompts` key. The GPU yaml's `workloads:` block points at
`workspace/contention/prompts/<name>.jsonl` — this script is what puts those
files on disk, in the shape aiperf's `single_turn` custom dataset loader wants:

    {"text": "What is CUDA?"}

One JSON object per line, TEXT ONLY. The media file (video clip / document
image) is NOT baked in here: `vlm_video_short` and `vlm_video_long` share one
prompts file and differ only by their `data:` clip, so the text+media pairing
has to happen per run. benchmarks/coloc.py does that combining into the run's
artifact dir (materialise_workload_input).

Idempotent: re-running rewrites the same bytes, so the generated .jsonl can be
checked in and a regeneration shows up as an empty diff.

Usage:
    python3 scripts/build_contention_prompts.py            # write + report
    python3 scripts/build_contention_prompts.py --check    # verify only, no write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "workspace" / "contention" / "experiment_config.json"
OUT_DIR = REPO_ROOT / "workspace" / "contention" / "prompts"

# Rough tokens-per-character for English prose. Only used for the report that
# lets a human check the yaml's "~N tok" comments — never for anything the run
# depends on, so a real tokenizer (and its model download) is not worth it.
CHARS_PER_TOKEN = 4.0


def load_prompts(config_path: Path) -> dict[str, list[str]]:
    """The `prompts` block, validated as name -> non-empty list of strings."""
    data = json.loads(config_path.read_text())
    prompts = data.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise SystemExit(f"{config_path}: no top-level `prompts` object")
    out: dict[str, list[str]] = {}
    for name, entries in prompts.items():
        if not isinstance(entries, list) or not entries or not all(
            isinstance(e, str) and e.strip() for e in entries
        ):
            raise SystemExit(f"{config_path}: prompts.{name} must be a non-empty list of strings")
        out[name] = [e.strip() for e in entries]
    return out


def render_jsonl(entries: list[str]) -> str:
    """The exact bytes for one prompt file. `ensure_ascii=False` keeps any
    non-ASCII prompt readable in the committed artefact; aiperf reads UTF-8."""
    return "".join(json.dumps({"text": e}, ensure_ascii=False) + "\n" for e in entries)


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="Fail if any file is missing or stale; write nothing.")
    args = ap.parse_args(argv)

    prompts = load_prompts(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    for name, entries in sorted(prompts.items()):
        path = args.out_dir / f"{name}.jsonl"
        body = render_jsonl(entries)
        current = path.read_text() if path.exists() else None
        if current != body:
            if args.check:
                stale.append(str(path.relative_to(REPO_ROOT)))
            else:
                path.write_text(body)
        toks = [est_tokens(e) for e in entries]
        print(
            f"{path.relative_to(REPO_ROOT)}: {len(entries)} prompt(s), "
            f"~{min(toks)}-{max(toks)} tok each (~{sum(toks)} tok total)"
        )

    if stale:
        print(
            "stale or missing prompt files: " + ", ".join(stale)
            + " — re-run scripts/build_contention_prompts.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
