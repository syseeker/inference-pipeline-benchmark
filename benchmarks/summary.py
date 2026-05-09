"""Aggregate per-GPU BenchmarkResult JSONs into summary.md.

    python -m benchmarks.summary --gpu rtx_pro6000

Scans `benchmarks/results/<gpu>/*.json` (top-level rows = aggregate
BenchmarkResult; per-scenario rows live under `<gpu>/<framework>/`) and
writes `benchmarks/results/<gpu>/summary.md` per docs/metrics.md.

Per-scenario JSONs are stamped with run_id in the filename
(`<scenario>__<run_id>.json`) so history accumulates rather than being
overwritten. The cross-run section pairs (framework, model) rows by
`run_label` to compute deltas — eager-vs-graph, fp8-vs-bf16, tp1-vs-tp2,
chunked-prefill on-vs-off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)


# ----------------------------- helpers -----------------------------------------

def _fmt(v: float | None, suffix: str = "") -> str:
    if v is None:
        return "-"
    return f"{v:.1f}{suffix}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def _fmt_int(v: int | float | None) -> str:
    if v is None:
        return "-"
    return f"{int(v)}"


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _delta_pct(baseline: float | None, variant: float | None) -> str:
    if baseline is None or variant is None or baseline == 0:
        return "-"
    return f"{(variant - baseline) / baseline * 100:+.1f}%"


def _delta_pp(baseline: float | None, variant: float | None) -> str:
    """Percentage-point delta for rates that are already in 0-1."""
    if baseline is None or variant is None:
        return "-"
    return f"{(variant - baseline) * 100:+.2f}pp"


# ----------------------------- loaders -----------------------------------------

def _load_aggregate_rows(gpu_dir: Path) -> list[dict[str, Any]]:
    """Load aggregate BenchmarkResult rows. Each file is
    `{"configs": {…}, "results": {…}}`; we merge the two halves into a
    flat dict so the rest of this module reads fields by name."""
    rows: list[dict[str, Any]] = []
    for p in sorted(gpu_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        configs = data.get("configs") or {}
        results = data.get("results") or {}
        if (
            not configs
            or not results
            or "framework" not in configs
            or "model" not in configs
            or "run_id" not in results
        ):
            continue
        flat = {**configs, **results, "_path": p.name}
        rows.append(flat)
    rows.sort(key=lambda r: (r.get("framework", ""), r.get("model", ""), r.get("run_label", ""), r.get("started_at", "")))
    return rows


def _load_per_scenario(gpu_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-scenario rows. Each file is `{"configs": {…}, "results": {…}}`;
    we merge the two halves into a flat dict so the rest of this module reads
    fields by name without caring which group they live in."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sub in sorted(p for p in gpu_dir.iterdir() if p.is_dir()):
        rows: list[dict[str, Any]] = []
        for p in sorted(sub.glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            configs = data.get("configs") or {}
            results = data.get("results") or {}
            if not configs or not results or "scenario" not in configs:
                continue
            rows.append({**configs, **results})
        if rows:
            rows.sort(key=lambda r: (r.get("run_id", ""), r.get("scenario", "")))
            grouped[sub.name] = rows
    return grouped


# ----------------------------- per-section tables ------------------------------

def _row_label(r: dict) -> str:
    return r.get("run_label") or "baseline"


def _decision_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | model | quant | run_id | n | grammar_valid | exec_accept | e2e p50 | e2e p95 | e2e p99 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {model} | {q} | {rid} | {n} | {gv} | {cs} | {p50} | {p95} | {p99} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                model=r.get("model", "-"),
                q=r.get("quantization") or "-",
                rid=r.get("run_id", "-"),
                n=r.get("n_requests", "-"),
                gv=_fmt_pct(r.get("grammar_validity_rate")),
                cs=_fmt_pct(r.get("command_success_rate")),
                p50=_fmt(r.get("valid_e2e_p50_ms"), " ms"),
                p95=_fmt(r.get("valid_e2e_p95_ms"), " ms"),
                p99=_fmt(r.get("valid_e2e_p99_ms"), " ms"),
            )
        )
    return lines


def _latency_diag_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | model | ttft p50 | ttft p95 | ttft p99 | itl p50 | itl p95 | itl p99 | prefill p50 | decode p50 | queue p50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {model} | {t50} | {t95} | {t99} | {i50} | {i95} | {i99} | {pf50} | {dc50} | {q50} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                model=r.get("model", "-"),
                t50=_fmt(r.get("ttft_p50_ms"), " ms"),
                t95=_fmt(r.get("ttft_p95_ms"), " ms"),
                t99=_fmt(r.get("ttft_p99_ms"), " ms"),
                i50=_fmt(r.get("itl_p50_ms"), " ms"),
                i95=_fmt(r.get("itl_p95_ms"), " ms"),
                i99=_fmt(r.get("itl_p99_ms"), " ms"),
                pf50=_fmt(r.get("prefill_time_p50_ms"), " ms"),
                dc50=_fmt(r.get("decode_time_p50_ms"), " ms"),
                q50=_fmt(r.get("queue_time_p50_ms"), " ms"),
            )
        )
    return lines


def _throughput_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | model | seq/s | goodput | tok/s_decode | mean prompt_toks | mean comp_toks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {model} | {tp} | {gp} | {td} | {mp} | {mc} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                model=r.get("model", "-"),
                tp=_fmt(r.get("throughput_seq_per_s")),
                gp=_fmt(r.get("goodput_seq_per_s")),
                td=_fmt(r.get("tokens_per_sec_decode")),
                mp=_fmt(r.get("mean_prompt_tokens")),
                mc=_fmt(r.get("mean_completion_tokens")),
            )
        )
    return lines


def _cache_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | model | prefix_cache_hit | kv_cache_usage | chunked_prefill | enforce_eager |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cp = r.get("chunked_prefill_enabled")
        ee = r.get("enforce_eager")
        lines.append(
            "| {fw} | {lbl} | {model} | {pc} | {ku} | {cp} | {ee} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                model=r.get("model", "-"),
                pc=_fmt_pct(r.get("prefix_cache_hit_rate")),
                ku=_fmt(r.get("kv_cache_usage_pct"), "%"),
                cp=("on" if cp is True else "off" if cp is False else "-"),
                ee=("on" if ee is True else "off" if ee is False else "-"),
            )
        )
    return lines


def _gpu_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | model | sampler | mem_bw p50 | mem_bw peak | gpu_util p50 | fb peak (GB) | power avg (W) | power peak (W) | energy/req (J) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        sb = r.get("sampler_backend") or "-"
        lines.append(
            "| {fw} | {lbl} | {model} | {sb} | {mb50} | {mbp} | {gu50} | {fb} | {pa} | {pp} | {e} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                model=r.get("model", "-"),
                sb=sb,
                mb50=_fmt(r.get("mem_bw_util_pct_p50"), "%") if sb == "dcgm" else "n/a",
                mbp=_fmt(r.get("mem_bw_util_pct_peak"), "%") if sb == "dcgm" else "n/a",
                gu50=_fmt(r.get("gpu_util_pct_p50"), "%"),
                fb=_fmt(r.get("fb_used_peak_gb")),
                pa=_fmt(r.get("power_avg_w")),
                pp=_fmt(r.get("power_peak_w")),
                e=_fmt(r.get("energy_per_request_j")),
            )
        )
    return lines


# ----------------------------- per-scenario detail -----------------------------

def _per_scenario_section(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for fw in sorted(grouped):
        lines.append(f"### {fw}")
        lines.append("")
        lines.append("| run_id | label | scenario | total_ms | ttft_ms | schema_valid | safe | executed | error |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in grouped[fw]:
            lat = r.get("latency_ms") or {}
            val = r.get("validation") or {}
            lines.append(
                "| {rid} | {lbl} | {sc} | {tot} | {ttft} | {sv} | {sf} | {ex} | {err} |".format(
                    rid=r.get("run_id", "-"),
                    lbl=r.get("run_label") or "baseline",
                    sc=r.get("scenario", "-"),
                    tot=_fmt(lat.get("total_ms"), " ms"),
                    ttft=_fmt(lat.get("reasoner_ttft_ms"), " ms"),
                    sv=val.get("schema_valid", "-"),
                    sf=val.get("safe", "-"),
                    ex=r.get("was_executed", "-"),
                    err=(r.get("error") or "-")[:60],
                )
            )
        lines.append("")
    return lines


# ----------------------------- cross-run deltas --------------------------------

# Maps a variant label to (display name, comparison kind).
# "kind" determines which fields are highlighted in the delta table.
_KNOWN_VARIANTS = {
    "eager":       "graph→eager (cuda_graph_speedup)",
    "graph":       "eager→graph (cuda_graph_speedup)",
    "chunked_on":  "chunked_prefill on/off",
    "chunked_off": "chunked_prefill on/off",
    "fp8":         "bf16→fp8",
    "int8":        "bf16→int8",
    "tp2":         "TP=1 → TP=2",
    "tp4":         "TP=1 → TP=4",
}


def _group_by_fw_model(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (str(r.get("framework", "-")), str(r.get("model", "-")))
        by.setdefault(key, []).append(r)
    return by


def _pick_latest(rows: list[dict], label: str) -> dict | None:
    """Among rows of a given label, pick the most recent by started_at."""
    cands = [r for r in rows if (r.get("run_label") or "baseline") == label]
    if not cands:
        return None
    return sorted(cands, key=lambda r: r.get("started_at") or "")[-1]


def _cross_run_section(rows: list[dict]) -> list[str]:
    """Pair runs by (framework, model), pivot on run_label, emit deltas
    relative to the most recent `baseline` row."""

    sections: list[str] = []
    by = _group_by_fw_model(rows)
    any_pair = False

    sections.append(
        "Each row is one (variant) vs the latest `baseline` run for the same "
        "(framework, model). Headline comparisons:"
    )
    sections.append("")
    sections.append(
        "- **graph → eager**: `cuda_graph_speedup = e2e_p50(eager) / e2e_p50(baseline)`."
    )
    sections.append(
        "- **bf16 → fp8/int8**: `quant_accuracy_delta = grammar_validity(baseline) − grammar_validity(variant)` (in pp)."
    )
    sections.append(
        "- **TP=1 → TP=2**: `tp_efficiency = e2e_p50(baseline) / (2 × e2e_p50(tp2))`. `> 1` is a win."
    )
    sections.append(
        "- **chunked_prefill**: ttft and decode percentage shifts."
    )
    sections.append("")

    sections.append("| framework | model | comparison | metric | baseline | variant | delta |")
    sections.append("| --- | --- | --- | --- | --- | --- | --- |")

    for (fw, model), grp in sorted(by.items()):
        baseline = _pick_latest(grp, "baseline")
        if baseline is None:
            continue
        variants = sorted(set((r.get("run_label") or "baseline") for r in grp) - {"baseline"})
        for vlabel in variants:
            variant = _pick_latest(grp, vlabel)
            if variant is None:
                continue
            any_pair = True
            comparison = _KNOWN_VARIANTS.get(vlabel, f"baseline→{vlabel}")

            # Always show e2e_p50 and grammar_validity deltas.
            sections.append(
                "| {fw} | {m} | {cmp} | e2e_p50_ms | {b} | {v} | {d} |".format(
                    fw=fw, m=model, cmp=comparison,
                    b=_fmt(baseline.get("valid_e2e_p50_ms"), " ms"),
                    v=_fmt(variant.get("valid_e2e_p50_ms"), " ms"),
                    d=_delta_pct(baseline.get("valid_e2e_p50_ms"), variant.get("valid_e2e_p50_ms")),
                )
            )
            sections.append(
                "| {fw} | {m} | {cmp} | grammar_validity | {b} | {v} | {d} |".format(
                    fw=fw, m=model, cmp=comparison,
                    b=_fmt_pct(baseline.get("grammar_validity_rate")),
                    v=_fmt_pct(variant.get("grammar_validity_rate")),
                    d=_delta_pp(baseline.get("grammar_validity_rate"), variant.get("grammar_validity_rate")),
                )
            )

            # Headline metric per known variant.
            if vlabel == "eager":
                bp = baseline.get("valid_e2e_p50_ms")
                vp = variant.get("valid_e2e_p50_ms")
                if bp and vp:
                    sections.append(
                        f"| {fw} | {model} | {comparison} | cuda_graph_speedup | - | - | {vp/bp:.2f}x |"
                    )
            elif vlabel.startswith("tp"):
                try:
                    tp_n = int(vlabel[2:])
                except ValueError:
                    tp_n = None
                bp = baseline.get("valid_e2e_p50_ms")
                vp = variant.get("valid_e2e_p50_ms")
                if tp_n and bp and vp:
                    eff = bp / (tp_n * vp)
                    sections.append(
                        f"| {fw} | {model} | {comparison} | tp{tp_n}_efficiency | - | - | {eff:.2f}x |"
                    )
            elif vlabel in ("fp8", "int8"):
                # Headline = grammar_validity drop in pp (already shown above) plus throughput uplift.
                sections.append(
                    "| {fw} | {m} | {cmp} | tok/s_decode | {b} | {v} | {d} |".format(
                        fw=fw, m=model, cmp=comparison,
                        b=_fmt(baseline.get("tokens_per_sec_decode")),
                        v=_fmt(variant.get("tokens_per_sec_decode")),
                        d=_delta_pct(baseline.get("tokens_per_sec_decode"), variant.get("tokens_per_sec_decode")),
                    )
                )
            elif vlabel in ("chunked_off", "chunked_on"):
                sections.append(
                    "| {fw} | {m} | {cmp} | ttft_p50_ms | {b} | {v} | {d} |".format(
                        fw=fw, m=model, cmp=comparison,
                        b=_fmt(baseline.get("ttft_p50_ms"), " ms"),
                        v=_fmt(variant.get("ttft_p50_ms"), " ms"),
                        d=_delta_pct(baseline.get("ttft_p50_ms"), variant.get("ttft_p50_ms")),
                    )
                )

    if not any_pair:
        return ["_no variant runs to pair with `baseline`._", ""]
    return sections


# ----------------------------- environment & avgs ------------------------------

def _env_block(rows: list[dict]) -> list[str]:
    seen: set[tuple[str, str, str, str]] = set()
    lines = ["| framework | version | driver | cuda |", "| --- | --- | --- | --- |"]
    for r in rows:
        key = (
            str(r.get("framework", "-")),
            str(r.get("framework_version", "-")),
            str(r.get("driver", "-")),
            str(r.get("cuda", "-")),
        )
        if key in seen:
            continue
        seen.add(key)
        lines.append("| {} | {} | {} | {} |".format(*key))
    return lines


# ----------------------------- main entrypoint ---------------------------------

@app.command()
def main(
    gpu: str = typer.Option(..., help="GPU profile name; reads benchmarks/results/<gpu>/"),
    results_dir: Path = typer.Option(Path("benchmarks/results"), help="Results root."),
) -> None:
    gpu_dir = results_dir / gpu
    if not gpu_dir.is_dir():
        typer.echo(f"no results directory: {gpu_dir}", err=True)
        raise typer.Exit(2)

    rows = _load_aggregate_rows(gpu_dir)
    grouped = _load_per_scenario(gpu_dir)

    if not rows and not grouped:
        typer.echo(f"no result rows found under {gpu_dir}", err=True)
        raise typer.Exit(2)

    out: list[str] = []
    out.append(f"# Benchmark summary — {gpu}")
    out.append("")
    out.append(
        "Decision metrics drive go/no-go (see docs/metrics.md). Diagnostics "
        "explain *why* a decision metric moved. Cross-run deltas pair "
        "`run_label` variants against `baseline` for the same (framework, model)."
    )
    out.append("")

    if rows:
        out.append("## 1. Decision metrics")
        out.append("")
        out.extend(_decision_table(rows))
        out.append("")

        out.append("## 2. Latency diagnostics")
        out.append("")
        out.append(
            "Server-side `prefill / decode / queue` times come from the framework's "
            "`/metrics` Prometheus endpoint (vllm/sglang). trtllm-serve has not yet "
            "exposed Prometheus-compatible metrics, so these fields are n/a for trtllm."
        )
        out.append("")
        out.extend(_latency_diag_table(rows))
        out.append("")

        out.append("## 3. Throughput & token counts")
        out.append("")
        out.append(
            "`tok/s_decode` is decode-only tokens-per-second computed from "
            "`(completion_tokens − 1) / (e2e − ttft)` per request, not wall time. "
            "`mean prompt_toks` includes vision tokens for VLMs."
        )
        out.append("")
        out.extend(_throughput_table(rows))
        out.append("")

        out.append("## 4. Cache & scheduling")
        out.append("")
        out.extend(_cache_table(rows))
        out.append("")

        out.append("## 5. GPU resource usage")
        out.append("")
        out.append(
            "`mem_bw` requires DCGM (`DCGM_FI_PROF_DRAM_ACTIVE`); falls back to "
            "n/a when only nvidia-smi is available. `energy/req` = "
            "`power_avg_w × wall_time_s / n_completed`."
        )
        out.append("")
        out.extend(_gpu_table(rows))
        out.append("")

        out.append("## 6. Cross-run deltas")
        out.append("")
        out.extend(_cross_run_section(rows))
        out.append("")

    if grouped:
        out.append("## 7. Per-scenario detail")
        out.append("")
        out.extend(_per_scenario_section(grouped))

    if rows:
        out.append("## 8. Environment")
        out.append("")
        out.extend(_env_block(rows))
        out.append("")

    summary_path = gpu_dir / "summary.md"
    summary_path.write_text("\n".join(out))
    typer.echo(f"wrote {summary_path}")


if __name__ == "__main__":
    sys.exit(app() or 0)
