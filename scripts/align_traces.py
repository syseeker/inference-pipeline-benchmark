#!/usr/bin/env python3
"""Align all tenant traces in a coloc run to the shared t0 and summarise.

Reads one coloc run directory:

    benchmarks/results/<gpu>/coloc/<colocation>/<run_label>/
      manifest.json         # t0_epoch_ms, tenant specs, gpu_sampler aggregate
      <tenant>.ndjson       # per-request rows (LLM/VLM) or aggregate (CV)
      gpu.ndjson            # optional per-row sampler (may not exist)

Verifies that tenants were genuinely concurrent (overlap window > 0), reports
per-tenant latency and throughput within the window, and summarises GPU load
from the manifest's gpu_sampler aggregate.

Usage:
    python scripts/align_traces.py <run_dir>
    python scripts/align_traces.py <run_dir> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────── loading ───────────────────────────────────────

def load_manifest(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"no manifest.json in {run_dir}")
    return json.loads(p.read_text())


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# ─────────────────────────── stats helpers ─────────────────────────────────

def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def _trace_window(records: list[dict[str, Any]]) -> tuple[float, float] | None:
    """[min t_start_ms, max t_end_ms] from per-request records. None for CV aggregate."""
    starts = [r["t_start_ms"] for r in records if r.get("t_start_ms") is not None]
    ends = [r["t_end_ms"] for r in records if r.get("t_end_ms") is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _tenant_stats(records: list[dict[str, Any]], offered_rps: float | None,
                  achieved_rps: float | None) -> dict[str, Any]:
    """Compute per-tenant stats. Handles both aiperf and perf_analyzer record shapes."""
    is_cv = bool(records) and "measured_rps" in records[0]

    if is_cv:
        r = records[0]
        return {
            "is_cv": True,
            "n_requests": None,
            "e2e_p50_ms": r.get("p50_ms"),
            "e2e_p95_ms": r.get("p95_ms"),
            "e2e_avg_ms": r.get("e2e_ms"),
            "ttft_p95_ms": None,
            "offered_rps": offered_rps,
            "achieved_rps": r.get("measured_rps") or achieved_rps,
            "window_ms": None,
        }

    e2e_vals = [r["e2e_ms"] for r in records if r.get("e2e_ms") is not None]
    ttft_vals = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    window = _trace_window(records)
    return {
        "is_cv": False,
        "n_requests": len(records),
        "e2e_p50_ms": _pct(e2e_vals, 50),
        "e2e_p95_ms": _pct(e2e_vals, 95),
        "e2e_avg_ms": (sum(e2e_vals) / len(e2e_vals)) if e2e_vals else None,
        "ttft_p95_ms": _pct(ttft_vals, 95),
        "offered_rps": offered_rps,
        "achieved_rps": achieved_rps,
        "window_ms": window,
    }


def _overlap_window(
    tenant_stats: dict[str, dict[str, Any]],
    t0_epoch_ms: float,
    duration_ms: float,
) -> tuple[float, float] | None:
    """Absolute ms overlap window where ALL tenants had active requests.

    CV tenants get the full measurement window [t0, t0+duration] as a fallback
    since perf_analyzer gives no per-request timestamps.
    """
    starts, ends = [], []
    for name, s in tenant_stats.items():
        w = s["window_ms"]
        if w is None:
            # CV aggregate — assume it covered the full window
            starts.append(t0_epoch_ms)
            ends.append(t0_epoch_ms + duration_ms)
        else:
            starts.append(w[0])
            ends.append(w[1])
    if not starts:
        return None
    lo = max(starts)
    hi = min(ends)
    return (lo, hi) if lo < hi else None


# ─────────────────────────── formatting ────────────────────────────────────

def _fmt(v: float | None, suffix: str = "", decimals: int = 1) -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "n/a"


def _fmt_ms(v: float | None) -> str:
    return _fmt(v, " ms")


def _ms_to_epoch(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def _summarise_text(run_dir: Path, manifest: dict[str, Any],
                    tenant_stats: dict[str, dict[str, Any]],
                    overlap: tuple[float, float] | None) -> str:
    t0 = manifest["t0_epoch_ms"]
    duration_s = manifest["duration_s"]
    gpu_sm = manifest.get("gpu_sampler") or {}
    throttle = manifest.get("throttle_reasons") or []

    lines: list[str] = []
    lines.append(f"Run:      {manifest['colocation_id']}  ({manifest['gpu']}, {manifest['isolation_mode']})")
    lines.append(f"Dir:      {run_dir}")
    lines.append(f"t0:       {_ms_to_epoch(t0)}")
    lines.append(f"Duration: {duration_s} s  |  is_solo: {manifest['is_solo']}")
    lines.append("")

    lines.append("Tenant alignment  (timestamps relative to t0)")
    lines.append("-" * 78)
    for name, s in tenant_stats.items():
        if s["is_cv"]:
            lines.append(
                f"  {name:<10} [aggregate — CV]  "
                f"offered {_fmt(s['offered_rps'])} rps | "
                f"achieved {_fmt(s['achieved_rps'])} rps | "
                f"e2e avg {_fmt_ms(s['e2e_avg_ms'])} p95 {_fmt_ms(s['e2e_p95_ms'])}"
            )
        else:
            w = s["window_ms"]
            if w:
                first_rel = (w[0] - t0) / 1000.0
                last_rel = (w[1] - t0) / 1000.0
                span = f"+{first_rel:.1f}s → +{last_rel:.1f}s"
            else:
                span = "no timestamps"
            lines.append(
                f"  {name:<10} {span:<24}  "
                f"n={s['n_requests'] or '?'}  "
                f"e2e p50 {_fmt_ms(s['e2e_p50_ms'])} p95 {_fmt_ms(s['e2e_p95_ms'])} | "
                f"ttft p95 {_fmt_ms(s['ttft_p95_ms'])} | "
                f"offered {_fmt(s['offered_rps'])} rps | achieved {_fmt(s['achieved_rps'])} rps"
            )
    lines.append("")

    if overlap is None:
        lines.append("Overlap window: NONE — tenants did not serve simultaneously!")
        lines.append("  Check: (a) did both servers start? (b) was load high enough?")
    else:
        dur_s = (overlap[1] - overlap[0]) / 1000.0
        lines.append(
            f"Overlap window: +{(overlap[0]-t0)/1000:.1f}s → +{(overlap[1]-t0)/1000:.1f}s  "
            f"({dur_s:.1f} s of {duration_s} s = {100*dur_s/duration_s:.0f}%)"
        )
        if dur_s / duration_s >= 0.80:
            lines.append("  Overlap coverage: OK (≥80%)")
        else:
            lines.append("  Overlap coverage: LOW (<80%) — partial co-residency; interpret with care")
    lines.append("")

    lines.append("GPU (whole window, from manifest aggregate)")
    lines.append(
        f"  util p50 {_fmt(gpu_sm.get('gpu_util_pct_p50'), '%', 0)} | "
        f"util peak {_fmt(gpu_sm.get('gpu_util_pct_peak'), '%', 0)} | "
        f"mem-bw p50 {_fmt(gpu_sm.get('mem_bw_util_pct_p50'), '%', 0)} | "
        f"power avg {_fmt(gpu_sm.get('power_avg_w'), ' W', 0)} | "
        f"peak VRAM {_fmt(gpu_sm.get('fb_used_peak_gb'), ' GB')}"
    )
    lines.append("")

    if throttle:
        lines.append(f"Throttle: {', '.join(throttle)}  ← clock integrity violation (§4.2)")
    else:
        lines.append("Throttle: none")

    return "\n".join(lines)


def _summarise_json(manifest: dict[str, Any],
                    tenant_stats: dict[str, dict[str, Any]],
                    overlap: tuple[float, float] | None) -> str:
    t0 = manifest["t0_epoch_ms"]
    out: dict[str, Any] = {
        "colocation_id": manifest["colocation_id"],
        "is_solo": manifest["is_solo"],
        "gpu": manifest["gpu"],
        "isolation_mode": manifest["isolation_mode"],
        "t0_epoch_ms": t0,
        "duration_s": manifest["duration_s"],
        "tenants": tenant_stats,
        "overlap_window": (
            {"start_ms": overlap[0], "end_ms": overlap[1],
             "duration_s": (overlap[1] - overlap[0]) / 1000.0,
             "start_rel_s": (overlap[0] - t0) / 1000.0,
             "end_rel_s": (overlap[1] - t0) / 1000.0}
            if overlap else None
        ),
        "gpu_sampler": manifest.get("gpu_sampler") or {},
        "throttle_reasons": manifest.get("throttle_reasons") or [],
    }
    return json.dumps(out, indent=2)


# ─────────────────────────── main ──────────────────────────────────────────

def analyse(run_dir: Path, *, json_out: bool = False) -> str:
    manifest = load_manifest(run_dir)
    t0 = manifest["t0_epoch_ms"]
    duration_ms = manifest["duration_s"] * 1000.0

    t_by_name: dict[str, dict[str, Any]] = {t["name"]: t for t in manifest["tenants"]}
    stats: dict[str, dict[str, Any]] = {}
    for tname, tdict in t_by_name.items():
        records = load_ndjson(run_dir / f"{tname}.ndjson")
        stats[tname] = _tenant_stats(
            records,
            offered_rps=tdict.get("offered_rps"),
            achieved_rps=tdict.get("achieved_rps"),
        )

    overlap = _overlap_window(stats, t0, duration_ms)
    if json_out:
        return _summarise_json(manifest, stats, overlap)
    return _summarise_text(run_dir, manifest, stats, overlap)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Align coloc traces to t0 and summarise.")
    p.add_argument("run_dir", type=Path, help="Coloc run directory containing manifest.json.")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = p.parse_args(argv)

    try:
        print(analyse(args.run_dir, json_out=args.json))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
