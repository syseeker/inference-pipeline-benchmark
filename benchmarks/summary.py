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
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer
import yaml

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

# Framework display order within each per-model sub-table.
_FW_ORDER = {"vllm": 0, "sglang": 1, "trtllm": 2}


def _fw_rank(fw: str) -> int:
    return _FW_ORDER.get(fw, 99)


def _row_label(r: dict) -> str:
    return r.get("run_label") or "baseline"


def _e2e(r: dict, pct: str) -> float | None:
    """Read e2e percentile, falling back to the old `valid_e2e_*` field
    name so historical JSONs render until they're re-run."""
    return r.get(f"e2e_{pct}_ms") if r.get(f"e2e_{pct}_ms") is not None else r.get(f"valid_e2e_{pct}_ms")


def _group_rows_by_model(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group rows by model (alphabetical), with rows inside each group ordered
    by framework (vllm → sglang → trtllm), then run_label, then started_at."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get("model", "-")), []).append(r)
    out: list[tuple[str, list[dict]]] = []
    for model in sorted(by):
        grp = sorted(
            by[model],
            key=lambda r: (
                _fw_rank(str(r.get("framework", ""))),
                r.get("run_label") or "",
                r.get("started_at") or "",
            ),
        )
        out.append((model, grp))
    return out


def _decision_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | quant | run_id | n | grammar_valid | exec_accept | e2e p50 | e2e p95 | e2e p99 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {q} | {rid} | {n} | {gv} | {cs} | {p50} | {p95} | {p99} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                q=r.get("quantization") or "-",
                rid=r.get("run_id", "-"),
                n=r.get("n_requests", "-"),
                gv=_fmt_pct(r.get("grammar_validity_rate")),
                cs=_fmt_pct(r.get("command_success_rate")),
                p50=_fmt(_e2e(r, "p50"), " ms"),
                p95=_fmt(_e2e(r, "p95"), " ms"),
                p99=_fmt(_e2e(r, "p99"), " ms"),
            )
        )
    return lines


def _latency_diag_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | ttft p50 | ttft p95 | ttft p99 | itl p50 | itl p95 | itl p99 | prefill p50 | decode p50 | queue p50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {t50} | {t95} | {t99} | {i50} | {i95} | {i99} | {pf50} | {dc50} | {q50} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
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
        "| framework | label | seq/s | goodput | tok/s_decode | mean prompt_toks | mean comp_toks |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {fw} | {lbl} | {tp} | {gp} | {td} | {mp} | {mc} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
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
        "| framework | label | prefix_cache_hit | kv_cache_usage | chunked_prefill | enforce_eager |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cp = r.get("chunked_prefill_enabled")
        ee = r.get("enforce_eager")
        lines.append(
            "| {fw} | {lbl} | {pc} | {ku} | {cp} | {ee} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
                pc=_fmt_pct(r.get("prefix_cache_hit_rate")),
                ku=_fmt(r.get("kv_cache_usage_pct"), "%"),
                cp=("on" if cp is True else "off" if cp is False else "-"),
                ee=("on" if ee is True else "off" if ee is False else "-"),
            )
        )
    return lines


def _gpu_table(rows: list[dict]) -> list[str]:
    lines = [
        "| framework | label | sampler | mem_bw p50 | mem_bw peak | gpu_util p50 | fb peak (GB) | power avg (W) | power peak (W) | energy/req (J) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        sb = r.get("sampler_backend") or "-"
        lines.append(
            "| {fw} | {lbl} | {sb} | {mb50} | {mbp} | {gu50} | {fb} | {pa} | {pp} | {e} |".format(
                fw=r.get("framework", "-"),
                lbl=_row_label(r),
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


def _emit_per_model(
    out: list[str],
    rows: list[dict],
    table_fn,
) -> None:
    """Emit one sub-table per model, with rows ordered vllm → sglang → trtllm."""
    for model, grp in _group_rows_by_model(rows):
        out.append(f"### {model}")
        out.append("")
        out.extend(table_fn(grp))
        out.append("")


# ----------------------------- findings knowledge loader -----------------------

# `docs/findings/knowledge.yaml` is a curated 3-level (gpu → framework →
# model[/variant]) lookup table that fills in [TBD] markers in Core findings
# bullets. Lives alongside the long-form findings markdowns it cross-
# references via `ref:` fields. See that file's header for schema details.
# No LLM is required to populate or read it.

_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "docs" / "findings" / "knowledge.yaml"


@lru_cache(maxsize=1)
def _load_findings_knowledge() -> dict[str, Any]:
    if not _KNOWLEDGE_PATH.is_file():
        return {}
    try:
        with _KNOWLEDGE_PATH.open() as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _lookup_finding(
    gpu: str,
    framework: str,
    model: str,
    symptom: str,
    variant: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Look up `(why, how_to_improve, ref)` for a (gpu, framework, model[/variant], symptom) tuple.

    Tries `<model>/<variant>` first when a non-baseline variant is supplied,
    falls back to bare `<model>`. Returns `(None, None, None)` when no
    match — caller keeps its `[TBD]` placeholder.
    """
    kb = _load_findings_knowledge()
    fw_kb = (kb.get(gpu) or {}).get(framework) or {}
    if not fw_kb:
        return None, None, None
    keys: list[str] = []
    if variant and variant != "baseline":
        keys.append(f"{model}/{variant}")
    keys.append(model)
    for k in keys:
        sym_entry = (fw_kb.get(k) or {}).get(symptom)
        if isinstance(sym_entry, dict):
            return (
                sym_entry.get("why"),
                sym_entry.get("how_to_improve"),
                sym_entry.get("ref"),
            )
    return None, None, None


def _format_kb_phrases(
    why: str | None,
    how: str | None,
    ref: str | None,
    fallback_why: str,
    fallback_how: str,
) -> tuple[str, str]:
    """Squash multi-line YAML strings to single lines for inline bullet use,
    fall back to the operator-pass `[TBD]` text when not present, and append
    a `(see <ref>)` pointer when the entry has a deep-dive reference."""
    why_text = " ".join((why or "").split()) if why else fallback_why
    how_text = " ".join((how or "").split()) if how else fallback_how
    if ref:
        how_text = f"{how_text} (see [{ref}]({ref}))"
    return why_text, how_text


# ----------------------------- core findings (auto) ----------------------------

# Style guide for this section lives in
# `~/.claude/projects/-home-ubuntu/memory/feedback_summary_md_core_findings_style.md`
# and docs/metrics.md. The shape: bullet 1 = winner (model+backend+variant+why);
# ≤10 bullets total; underperformer bullets must include **Why** (mechanism) and
# **How to improve** (knob/kernel fix/next experiment). Anything that needs
# domain knowledge gets a `[TBD]` placeholder so an operator pass can fill it
# in — never invent a "why" the data doesn't support.


def _is_moe(model_id: str) -> bool:
    """Heuristic: model id contains a known MoE marker."""
    s = model_id.lower()
    return any(t in s for t in ("a3b", "moe", "mixtral", "8x7b", "8x22b"))


def _winner_bullet(best: dict, rows: list[dict]) -> str:
    fw = best.get("framework", "?")
    model = best.get("model", "?")
    label = _row_label(best)
    quant = (best.get("quantization") or "").lower()
    e2e_p50 = _e2e(best, "p50")
    ttft = best.get("ttft_p50_ms")
    seq_s = best.get("throughput_seq_per_s")
    energy = best.get("energy_per_request_j")
    mem_bw_peak = best.get("mem_bw_util_pct_peak")

    why_parts: list[str] = []
    if _is_moe(model):
        why_parts.append("MoE — sparse compute per token")
    if quant in ("fp8", "int8"):
        why_parts.append(f"{quant.upper()} weights cut bandwidth pressure")
    if mem_bw_peak is not None:
        if mem_bw_peak < 50:
            why_parts.append(
                f"DRAM_ACTIVE peak only {_fmt(mem_bw_peak, '%')} → headroom for higher concurrency"
            )
        elif mem_bw_peak >= 70:
            why_parts.append(
                f"DRAM_ACTIVE peak {_fmt(mem_bw_peak, '%')} → bw-bound, near ceiling"
            )
    if not why_parts:
        why_parts.append("[TBD — structural reason, e.g. arch/quant/saturation]")

    same_model_variants = sorted({
        _row_label(r) for r in rows if r.get("model") == model
    })
    if len(same_model_variants) <= 1:
        variant_note = (
            "Only `baseline` has been run for this model — "
            "**best variant is open** until the next sweep populates section 6."
        )
    else:
        variant_note = (
            f"Variants observed: `{'`, `'.join(same_model_variants)}` "
            "— see section 6 for delta vs baseline."
        )

    metrics: list[str] = [f"e2e p50 **{_fmt(e2e_p50, ' ms')}**"]
    if ttft is not None:
        metrics.append(f"TTFT p50 {_fmt(ttft, ' ms')}")
    if seq_s is not None:
        metrics.append(f"{_fmt(seq_s)} seq/s")
    if energy is not None:
        metrics.append(f"{_fmt(energy)} J/req")

    return (
        f"- **Best run: `{model}` on **{fw}** `{label}`** — "
        f"{', '.join(metrics)}. **Why:** {'; '.join(why_parts)}. {variant_note}"
    )


def _framework_gap_bullet(rows: list[dict], gpu: str) -> str | None:
    """Per-model fastest-vs-slowest baseline framework gap.

    The headline gap (largest ratio) carries the Why / How-to-improve;
    smaller gaps are listed without an explanation to keep the bullet
    tight. The Why/How for the headline pair comes from
    `docs/findings/knowledge.yaml` under
    `<gpu> → <slower-framework> → <model> → slow_baseline`, falling
    back to a `[TBD]` operator-pass marker when there's no match.
    """
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if _row_label(r) != "baseline" or _e2e(r, "p50") is None:
            continue
        by_model.setdefault(str(r.get("model", "-")), []).append(r)

    gaps: list[tuple[str, str, str, float, float, float]] = []
    for m, grp in by_model.items():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=lambda r: _e2e(r, "p50"))
        fastest, slowest = grp[0], grp[-1]
        bp, sp = _e2e(fastest, "p50"), _e2e(slowest, "p50")
        if not bp or not sp:
            continue
        ratio = sp / bp
        if ratio > 1.1:
            gaps.append((m, fastest["framework"], slowest["framework"], ratio, bp, sp))

    if not gaps:
        return None
    gaps.sort(key=lambda x: -x[3])
    parts = [
        f"`{m}`: {fast} {ratio:.1f}× faster than {slow} ({_fmt(bp, ' ms')} vs {_fmt(sp, ' ms')})"
        for m, fast, slow, ratio, bp, sp in gaps[:4]
    ]

    # Headline pair drives the Why / How lookup.
    head_model, _, head_slow_fw, *_ = gaps[0]
    why, how, ref = _lookup_finding(gpu, head_slow_fw, head_model, symptom="slow_baseline")
    why_text, how_text = _format_kb_phrases(
        why, how, ref,
        fallback_why=(
            "[TBD — cite per-framework ITL / DRAM_ACTIVE / "
            "scheduler-or-kernel underutilization from sections 2 + 5]"
        ),
        fallback_how=(
            "[TBD — knob to flip, kernel/arch fix to upstream, or "
            "next sweep variant to diagnose]"
        ),
    )

    return (
        "- **Framework gap on shared models (baseline only):** "
        + "; ".join(parts)
        + f". **Why {head_slow_fw} lags on `{head_model}`:** {why_text} "
        f"**How to improve:** {how_text}"
    )


def _outlier_bullet(best: dict, e2e_rows: list[dict], gpu: str) -> str | None:
    """If the slowest run is 5×+ the best, call it out as non-competitive.

    The Why / How-to-improve come from `docs/findings/knowledge.yaml`,
    keyed on the *outlier's* (gpu, framework, model[/variant]) and
    `ttft_dominance` symptom when TTFT eats >70% of e2e, otherwise
    `slow_baseline`. Falls back to operator-pass `[TBD]` when no entry.
    """
    if len(e2e_rows) < 2:
        return None
    worst = e2e_rows[-1]
    bp = _e2e(best, "p50")
    wp = _e2e(worst, "p50")
    if not bp or not wp or wp < 5 * bp:
        return None
    ttft = worst.get("ttft_p50_ms")
    ratio = wp / bp
    ttft_dominated = ttft is not None and ttft > 0.7 * wp
    extra = ""
    if ttft_dominated:
        extra = (
            f", with TTFT p50 {_fmt(ttft, ' ms')} dominating "
            f"({ttft / wp * 100:.0f}% of e2e)"
        )

    symptom = "ttft_dominance" if ttft_dominated else "slow_baseline"
    why, how, ref = _lookup_finding(
        gpu, worst["framework"], worst["model"],
        symptom=symptom, variant=_row_label(worst),
    )
    why_text, how_text = _format_kb_phrases(
        why, how, ref,
        fallback_why="[TBD — cold-start graph capture, kernel fit, or queue contention?]",
        fallback_how=(
            "[TBD — disable warmup-graph capture, change backend, "
            "or file an upstream issue for the specific failure mode]"
        ),
    )

    return (
        f"- **Non-competitive: `{worst['model']}` on **{worst['framework']}** "
        f"`{_row_label(worst)}` — {_fmt(wp, ' ms')} e2e p50 ({ratio:.0f}× the best run)"
        f"{extra}. **Why:** {why_text} **How to improve:** {how_text}"
    )


def _validity_floor_bullet(rows: list[dict]) -> str | None:
    rates = [r.get("grammar_validity_rate") for r in rows]
    rates = [v for v in rates if v is not None]
    if not rates:
        return None
    vmin, vmax = min(rates), max(rates)
    if vmax > 0.5:
        return None
    return (
        f"- **Validity floor: every row sits in {_fmt_pct(vmin)}–{_fmt_pct(vmax)}.** "
        "`grammar_valid` today is `schema_valid AND safe`; with `DryRunExecutor`, "
        "`exec_accept` mirrors it. Switching framework will not move this — "
        "it's content / prompt / validator-rule bound. **How to improve:** "
        "finalize the eval dataset and split JSON-schema validity from "
        "semantic safety, then wire `exec_accept` to a real downstream signal."
    )


def _energy_spread_bullet(rows: list[dict]) -> str | None:
    pairs = [
        (r, r.get("energy_per_request_j"))
        for r in rows
        if r.get("energy_per_request_j") is not None
    ]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda x: x[1])
    cheap_r, cheap_e = pairs[0]
    exp_r, exp_e = pairs[-1]
    if exp_e < 5 * cheap_e:
        return None
    spread = exp_e / cheap_e
    return (
        f"- **Energy/req spans ~{spread:.0f}×:** {_fmt(cheap_e)} J "
        f"({cheap_r['framework']} + `{cheap_r['model']}`) → {_fmt(exp_e)} J "
        f"({exp_r['framework']} + `{exp_r['model']}`). "
        "Formula: `power_avg × wall_time / n_completed` — long-tail TTFT "
        "or low-throughput runs pay more energy even at lower `power_avg`."
    )


def _mem_bw_bullet(rows: list[dict]) -> str | None:
    pairs = [
        (r, r.get("mem_bw_util_pct_peak"))
        for r in rows
        if r.get("mem_bw_util_pct_peak") is not None
    ]
    if not pairs:
        return None
    bound = [(r, p) for r, p in pairs if p >= 70]
    headroom = [(r, p) for r, p in pairs if p < 50]
    if not (bound and headroom):
        return None
    fmt = lambda lst: ", ".join(f"{r['framework']}+`{r['model']}` ({_fmt(p, '%')})" for r, p in lst[:3])
    return (
        f"- **DRAM_ACTIVE saturation split:** bw-/compute-bound on "
        f"({fmt(bound)}); headroom on ({fmt(headroom)}). The headroom "
        "rows can likely cash spare bandwidth into throughput via larger "
        "`max_num_seqs` / batch / TP — [TBD — confirm with a concurrency sweep]."
    )


def _ttft_dominance_bullet(rows: list[dict]) -> str | None:
    """Is TTFT typically a large fraction of e2e? Tells the operator
    whether to optimise prefill (vision encoder, queue) vs decode."""
    pairs = []
    for r in rows:
        e = _e2e(r, "p50")
        t = r.get("ttft_p50_ms")
        if e and t:
            pairs.append(t / e)
    if len(pairs) < 3:
        return None
    avg_share = sum(pairs) / len(pairs)
    if avg_share < 0.3:
        return None
    return (
        f"- **TTFT dominates e2e** — across rows with both metrics, TTFT "
        f"averages **{avg_share * 100:.0f}%** of e2e p50. Optimising prefill "
        "(vision encoder, queue, chunked-prefill) will move e2e p50 more "
        "than decode tweaks at this completion length."
    )


def _section4_status_bullet(rows: list[dict]) -> str | None:
    """If section 4 is empty across all rows, surface why + that it's fixed."""
    has_any = any(
        r.get("kv_cache_usage_pct") is not None
        or r.get("prefix_cache_hit_rate") is not None
        or r.get("chunked_prefill_enabled") is not None
        or r.get("enforce_eager") is not None
        for r in rows
    )
    if has_any:
        return None
    return (
        "- **Section 4 (cache & scheduling) is empty in this dataset.** "
        "Pre-fix runs scraped `/metrics` once *after* the timed loop "
        "(gauges had drained) and the flag detector ignored framework "
        "defaults. Both fixed: `PromPoller` polls `/metrics` every 500 ms "
        "in-run for peak `kv_cache_usage_pct` / `prefix_cache_hit_rate`, "
        "and `_detect_server_flags` falls back to vllm/sglang defaults "
        "(chunked_prefill on, enforce_eager off) plus sglang's "
        "`--chunked-prefill-size` / `--disable-cuda-graph`. Next sweep "
        "will populate this section."
    )


def _core_findings_section(rows: list[dict], gpu: str) -> list[str]:
    """Auto-generate a Core findings section from `rows`.

    Bullet 1 is always the winner (lowest e2e p50). Subsequent bullets
    cover framework gaps, non-competitive outliers, validity floor,
    energy spread, mem-bw saturation, TTFT-vs-decode share, and
    section-4 status — each conditionally emitted based on whether the
    underlying data warrants it. Bullets cap at 10.

    Underperformer bullets pull Why / How-to-improve from
    `docs/findings/knowledge.yaml` via `_lookup_finding(gpu, ...)`.
    Anything not covered by that knowledge file is left as `[TBD]` so
    an operator pass can fill it in.
    """
    if not rows:
        return []
    out: list[str] = ["## Core findings", ""]
    kb_loaded = bool(_load_findings_knowledge())
    kb_note = (
        "Why / How-to-improve are pulled from "
        "`docs/findings/knowledge.yaml` when a `(gpu, framework, model)` "
        "match exists; unmatched items keep `[TBD]` for an operator pass."
        if kb_loaded
        else "`docs/findings/knowledge.yaml` not found — all Why / "
        "How-to-improve fields will show `[TBD]`."
    )
    out.append(
        f"_Auto-generated from this run's data. {kb_note} See "
        "docs/metrics.md and the saved core-findings style guide for the "
        "structure._"
    )
    out.append("")

    e2e_rows = sorted(
        [r for r in rows if _e2e(r, "p50") is not None],
        key=lambda r: _e2e(r, "p50"),
    )
    if not e2e_rows:
        out.append("- _no rows with `e2e_p50_ms` to summarise._")
        out.append("")
        return out

    bullets: list[str] = [_winner_bullet(e2e_rows[0], rows)]
    for fn_args in (
        (_framework_gap_bullet, (rows, gpu)),
        (_outlier_bullet, (e2e_rows[0], e2e_rows, gpu)),
        (_validity_floor_bullet, (rows,)),
        (_ttft_dominance_bullet, (rows,)),
        (_mem_bw_bullet, (rows,)),
        (_energy_spread_bullet, (rows,)),
        (_section4_status_bullet, (rows,)),
    ):
        fn, args = fn_args
        b = fn(*args)
        if b:
            bullets.append(b)
        if len(bullets) >= 10:
            break

    out.extend(bullets)
    out.append("")
    return out


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
                    b=_fmt(_e2e(baseline, "p50"), " ms"),
                    v=_fmt(_e2e(variant, "p50"), " ms"),
                    d=_delta_pct(_e2e(baseline, "p50"), _e2e(variant, "p50")),
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
                bp = _e2e(baseline, "p50")
                vp = _e2e(variant, "p50")
                if bp and vp:
                    sections.append(
                        f"| {fw} | {model} | {comparison} | cuda_graph_speedup | - | - | {vp/bp:.2f}x |"
                    )
            elif vlabel.startswith("tp"):
                try:
                    tp_n = int(vlabel[2:])
                except ValueError:
                    tp_n = None
                bp = _e2e(baseline, "p50")
                vp = _e2e(variant, "p50")
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
        out.extend(_core_findings_section(rows, gpu))

        out.append("## 1. Decision metrics")
        out.append("")
        out.append(
            "`e2e p50/p95/p99` is **end-to-end pipeline latency**: "
            "`vision_encoder + reasoner (TTFT + all decode tokens) + decoder + "
            "validator + executor`. It's the full `Pipeline.run()` wall time, "
            "i.e. the user-experienced latency from image-in to "
            "action-sequence-out — not just the LLM call. Aggregated over all "
            "completed requests (no validity filter)."
        )
        out.append("")
        out.append(
            "> **Note on `grammar_valid` and `exec_accept`:** today "
            "`grammar_valid` = `schema_valid AND safe` (pydantic-parse + "
            "content-safety check) and `exec_accept` = `DryRunExecutor` "
            "acceptance, which collapses to the same number as "
            "`grammar_valid` because there is no real downstream executor "
            "yet (e.g. a game-app feedback channel). Both metrics will be "
            "refined once the evaluation dataset is finalized — likely "
            "splitting pure JSON-schema validity from semantic safety, and "
            "wiring `exec_accept` to a real executor signal. Treat the "
            "current values as placeholders."
        )
        out.append("")
        _emit_per_model(out, rows, _decision_table)

        out.append("## 2. Latency diagnostics")
        out.append("")
        out.append(
            "Server-side `prefill / decode / queue` times come from the framework's "
            "`/metrics` Prometheus endpoint (vllm/sglang). trtllm-serve has not yet "
            "exposed Prometheus-compatible metrics, so these fields are n/a for trtllm."
        )
        out.append("")
        _emit_per_model(out, rows, _latency_diag_table)

        out.append("## 3. Throughput & token counts")
        out.append("")
        out.append(
            "`tok/s_decode` is decode-only tokens-per-second computed from "
            "`(completion_tokens − 1) / (e2e − ttft)` per request, not wall time. "
            "`mean prompt_toks` includes vision tokens for VLMs."
        )
        out.append("")
        _emit_per_model(out, rows, _throughput_table)

        out.append("## 4. Cache & scheduling")
        out.append("")
        _emit_per_model(out, rows, _cache_table)

        out.append("## 5. GPU resource usage")
        out.append("")
        out.append(
            "`mem_bw` requires DCGM (`DCGM_FI_PROF_DRAM_ACTIVE`); falls back to "
            "n/a when only nvidia-smi is available. `energy/req` = "
            "`power_avg_w × wall_time_s / n_completed`."
        )
        out.append("")
        _emit_per_model(out, rows, _gpu_table)

        out.append("## 6. Cross-run deltas")
        out.append("")
        out.extend(_cross_run_section(rows))
        out.append("")

    if grouped:
        out.append("---")
        out.append("")
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
