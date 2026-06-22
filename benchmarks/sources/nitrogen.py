"""Built-in `nitrogen` scenario source.

Registered as an entry-point under `pipeline_bench.scenario_sources` so
the discovery path is uniform — built-in and customer sources go through
the same surface. Thin wrapper that shells out to the existing
`scripts/build_nitrogen_scenarios.py` (which already has the parsing,
yt-dlp handling, synthetic-frame mode, and dataset-shape patches from PR #1).

Customer sources should follow the same `build(*, n, out, **kwargs) -> int`
contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "build_nitrogen_scenarios.py"


def build(
    *,
    n: int,
    out: Path,
    actions_root: Path | str | None = None,
    synthetic_frames: bool = False,
    game_mapping: Path | str | None = None,
    cache_dir: Path | str | None = None,
    deadline_ms: int = 1500,
    **_: object,
) -> int:
    """Build `n` scenarios under `out` from the nvidia/NitroGen dataset.

    Returns the count actually written (the builder over-samples and
    may produce fewer than `n` if dead URLs / missing parquets exhaust
    the shard).
    """
    actions_root = actions_root or os.environ.get("NITROGEN_ACTIONS_ROOT")
    if not actions_root or not Path(actions_root).exists():
        raise FileNotFoundError(
            "nitrogen source requires --actions-root or $NITROGEN_ACTIONS_ROOT pointing "
            "at the extracted actions/ tree from `hf download nvidia/NitroGen --repo-type dataset`."
        )

    cmd = [
        sys.executable, str(_SCRIPT),
        "--actions-root", str(actions_root),
        "--out", str(out),
        "--n", str(n),
        "--deadline-ms", str(deadline_ms),
    ]
    if cache_dir:
        cmd += ["--cache-dir", str(cache_dir)]
    if game_mapping:
        cmd += ["--game-mapping", str(game_mapping)]
    if synthetic_frames:
        cmd += ["--synthetic-frames"]

    res = subprocess.run(cmd, check=False)  # noqa: S603
    if res.returncode != 0:
        raise RuntimeError(f"nitrogen scenario builder failed (exit {res.returncode})")

    out = Path(out)
    if not out.exists():
        return 0
    return sum(1 for p in out.iterdir() if p.is_dir() and (p / "request.json").exists())
