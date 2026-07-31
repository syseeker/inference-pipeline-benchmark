"""Built-in `video-text` scenario source.

Unlike the `nitrogen` source (which downloads + transforms a HF dataset),
video-text scenarios are pre-authored by the customer: each folder under
tests/smoke/scenarios_video/ already contains a request.json and the
customer's video file. This source just validates the directory and reports
the count — no transformation needed.

Customer workflow:
  1. Drop your video files into the scenario folders (see QUICKSTART_VLM.md).
  2. Fill in request.json (description, prompt) and expected.json (key_phrases).
  3. Run: bench scenarios build --source video-text --out tests/smoke/scenarios_video/
     (or skip build entirely and pass --scenarios-dir directly to bench sweep).
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "smoke" / "scenarios_video"


def build(
    *,
    n: int,
    out: Path,
    **_: object,
) -> int:
    """Validate that `out` contains video-text scenarios and return the count.

    For the video-text source, 'build' does no generation — the customer has
    already placed videos + request.json files into the scenario directories.
    This call validates the structure and tells the CLI how many are ready.
    """
    out = Path(out)
    if not out.exists():
        raise FileNotFoundError(
            f"scenarios_video directory not found: {out}\n"
            "Create it and populate scenario folders per QUICKSTART_VLM.md."
        )

    ready = []
    missing_video = []
    for p in sorted(out.iterdir()):
        if not p.is_dir() or not (p / "request.json").exists():
            continue
        import json

        req = json.loads((p / "request.json").read_text())
        video_file = req.get("video_path")
        if video_file and not (p / video_file).exists() and not req.get("video_url"):
            missing_video.append(p.name)
        else:
            ready.append(p.name)

    if missing_video:
        raise FileNotFoundError(
            f"Video files missing in {len(missing_video)} scenario(s): {missing_video}\n"
            "Upload your .mp4 files into the scenario folders — see QUICKSTART_VLM.md "
            "'Step 1: Upload your videos'."
        )

    return min(len(ready), n)
