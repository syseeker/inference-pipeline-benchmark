"""Colocation resolution — extends / rps_sweep / vary, and the fail-safely contract.

`scenario_config` imports typer at module scope. It's a real dependency of the
package, but these tests exercise pure resolution logic that doesn't need it, so
we stub it when absent to keep them runnable in a bare environment.
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - exercised only in bare environments
    import typer  # noqa: F401
except ImportError:  # pragma: no cover
    _t = types.ModuleType("typer")
    _t.Typer = lambda *a, **k: types.SimpleNamespace(
        command=lambda *a, **k: (lambda f: f)
    )
    _t.Option = lambda *a, **k: None
    _t.echo = print
    _t.Exit = type("Exit", (Exception,), {})
    sys.modules["typer"] = _t

import pytest

from benchmarks.scenario_config import (  # noqa: E402
    Colocation,
    LoadSpec,
    _merge_extends,
    _resolve_tenant,
    _solo_key,
    derive_cap,
    iter_colocation,
    load_gpu_config,
)


@pytest.fixture(scope="module")
def cfg():
    return load_gpu_config("rtx_pro6000")


def _coloc_runs(cfg, name) -> list[Colocation]:
    return list(iter_colocation(cfg, name))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_solo_baselines_precede_the_contention_window(cfg):
    """A contention row is uninterpretable without its baseline, so baselines
    are emitted first — a partial run still yields the reference numbers."""
    runs = _coloc_runs(cfg, "mix-llm-cv")
    solos = [r for r in runs if r.is_solo]
    colocs = [r for r in runs if not r.is_solo]

    assert len(solos) == 2, "one baseline per tenant"
    assert len(colocs) == 1
    assert runs.index(colocs[0]) > max(runs.index(s) for s in solos)
    assert all(len(s.tenants) == 1 for s in solos)
    assert len(colocs[0].tenants) == 2


def test_run_label_pairs_with_summary_cross_run_deltas(cfg):
    """summary.py keys its baseline pairing off run_label."""
    runs = _coloc_runs(cfg, "mix-llm-cv")
    assert {r.run_label for r in runs} == {"solo", "coloc:mix-llm-cv"}


def test_identical_baselines_are_deduped_across_a_sweep(cfg):
    """The LLM tenant's load is constant across an rps_sweep of the CV tenant,
    so it needs one baseline, not four."""
    runs = _coloc_runs(cfg, "cross-llm-vs-cv")
    llm_solos = [r for r in runs if r.is_solo and r.tenants[0].name == "llm"]
    cv_solos = [r for r in runs if r.is_solo and r.tenants[0].name == "cv"]

    assert len(llm_solos) == 1, "constant load ⇒ one baseline"
    assert len(cv_solos) == 4, "each swept rate needs its own baseline"
    assert sorted(s.tenants[0].load.rps for s in cv_solos) == [1.0, 10.0, 50.0, 200.0]


def test_baseline_load_matches_the_contention_load(cfg):
    """A ratio against a baseline collected at a different rate is an artifact."""
    runs = _coloc_runs(cfg, "mix-llm-cv")
    solo = {r.tenants[0].name: r.tenants[0] for r in runs if r.is_solo}
    coloc = {t.name: t for r in runs if not r.is_solo for t in r.tenants}

    for name in ("llm", "cv"):
        assert solo[name].load.rps == coloc[name].load.rps
        assert solo[name].load.pattern == coloc[name].load.pattern


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def test_rps_sweep_expands_only_the_named_tenant(cfg):
    colocs = [r for r in _coloc_runs(cfg, "cross-llm-vs-cv") if not r.is_solo]
    assert len(colocs) == 4
    by_name = [{t.name: t.load.rps for t in c.tenants} for c in colocs]
    assert [d["cv"] for d in by_name] == [1.0, 10.0, 50.0, 200.0]
    assert {d["llm"] for d in by_name} == {4.0}, "other tenants held constant"


def test_vary_changes_the_field_it_names(cfg):
    colocs = [r for r in _coloc_runs(cfg, "secondary-backend-llm") if not r.is_solo]
    backends = [t.round.backend for c in colocs for t in c.tenants if t.name == "llm"]
    assert backends == ["vllm", "sglang"]
    cv = {t.round.backend for c in colocs for t in c.tenants if t.name == "cv"}
    assert cv == {"triton"}, "untouched tenant must not drift"


def test_vary_on_model_covers_the_size_ladder(cfg):
    colocs = [r for r in _coloc_runs(cfg, "cross-size-scaling") if not r.is_solo]
    models = [t.round.model_id for c in colocs for t in c.tenants if t.name == "llm"]
    assert models == ["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]


def test_extends_merges_tenants_by_name_not_by_position(cfg):
    """A child overriding one field of one tenant must inherit the rest —
    positional merging would silently drop the parent's settings."""
    coloc = next(r for r in _coloc_runs(cfg, "mix-vlm-cv") if not r.is_solo)
    cv = next(t for t in coloc.tenants if t.name == "cv")

    assert cv.round.model_id == "dinov2-base", "child override applied"
    assert cv.load.rps == 50.0, "parent load inherited"
    assert cv.round.backend == "triton", "parent backend inherited"


def test_extends_rejects_a_cycle():
    colos = {"a": {"extends": "b"}, "b": {"extends": "a"}}
    with pytest.raises(ValueError, match="circular extends"):
        _merge_extends(colos, "a")


def test_unknown_colocation_names_the_alternatives(cfg):
    with pytest.raises(ValueError, match="unknown colocation"):
        list(iter_colocation(cfg, "does-not-exist"))


# --------------------------------------------------------------------------- #
# Fail-safely contract
# --------------------------------------------------------------------------- #


def test_image_only_model_rejects_a_video_workload(cfg):
    """paligemma2 is a customer pick that cannot serve the video dimension it
    was chosen for. It must fail with the reason, not crash mid-sweep."""
    with pytest.raises(ValueError, match="image-only"):
        _resolve_tenant(
            cfg,
            {"name": "vlm", "backend": "vllm", "model": "gemma-vlm-32b",
             "workload": "vlm_video_long"},
            cfg["workloads"],
        )


def test_image_only_model_still_serves_image_workloads(cfg):
    """...and is NOT dropped from the study — it works for what it can do."""
    t = _resolve_tenant(
        cfg,
        {"name": "ilm", "backend": "vllm", "model": "gemma-vlm-32b",
         "workload": "ilm_document"},
        cfg["workloads"],
    )
    assert t.round.model_id == "gemma-vlm-32b"


def test_cv_model_rejects_a_language_backend(cfg):
    with pytest.raises(ValueError, match="not a language model"):
        _resolve_tenant(
            cfg, {"name": "cv", "backend": "vllm", "model": "yolov8-l"}, cfg["workloads"]
        )


def test_aiperf_cannot_be_forced_onto_triton(cfg):
    """AIPerf dropped kserve/dynamic_grpc, so it genuinely cannot drive Triton.
    A yaml typo here would otherwise produce a run with no load on that tenant."""
    with pytest.raises(ValueError, match="cannot drive Triton"):
        _resolve_tenant(
            cfg,
            {"name": "cv", "backend": "triton", "model": "yolov8-l", "driver": "aiperf"},
            cfg["workloads"],
        )


def test_driver_defaults_follow_the_transport(cfg):
    llm = _resolve_tenant(
        cfg, {"name": "llm", "backend": "vllm", "model": "qwen2.5-7b"}, cfg["workloads"]
    )
    cv = _resolve_tenant(
        cfg, {"name": "cv", "backend": "triton", "model": "yolov8-l"}, cfg["workloads"]
    )
    assert llm.driver == "aiperf"
    assert cv.driver == "perf_analyzer"


def test_unknown_workload_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown workload"):
        _resolve_tenant(
            cfg,
            {"name": "llm", "backend": "vllm", "model": "qwen2.5-7b", "workload": "nope"},
            cfg["workloads"],
        )


# --------------------------------------------------------------------------- #
# Load spec
# --------------------------------------------------------------------------- #


def test_open_loop_is_the_default_shape(cfg):
    """Closed-loop would make every degradation ratio describe the harness."""
    t = _resolve_tenant(
        cfg,
        {"name": "llm", "backend": "vllm", "model": "qwen2.5-7b",
         "load": {"rps": 4}},
        cfg["workloads"],
    )
    assert t.load.pattern == "poisson"
    assert t.load.is_open_loop


def test_closed_loop_is_not_open_loop():
    assert not LoadSpec(pattern="closed", rps=4).is_open_loop
    assert not LoadSpec(pattern="poisson", rps=None).is_open_loop


def test_output_tokens_come_from_the_workload(cfg):
    """Output length is a property of what we ask for, not how fast we ask."""
    short = _resolve_tenant(
        cfg, {"name": "l", "backend": "vllm", "model": "qwen2.5-7b",
              "workload": "llm_short"}, cfg["workloads"])
    long_ = _resolve_tenant(
        cfg, {"name": "l", "backend": "vllm", "model": "qwen2.5-7b",
              "workload": "llm_long"}, cfg["workloads"])
    assert short.load.output_tokens == 32
    assert long_.load.output_tokens == 512


# --------------------------------------------------------------------------- #
# Device placement
# --------------------------------------------------------------------------- #


def _llm(cfg, *, kv_budget_gb=None, **extra):
    # kv_budget_gb is colocation-level, not a tenant field — it arrives as an
    # argument, which is exactly how iter_colocation passes it down.
    return _resolve_tenant(
        cfg, {"name": "llm", "backend": "vllm", "model": "qwen2.5-7b", **extra},
        cfg["workloads"], kv_budget_gb=kv_budget_gb,
    )


def test_device_defaults_to_gpu_zero(cfg):
    """Every colocation in the shipped yaml omits `device`; they must keep
    meaning "the one card"."""
    t = _llm(cfg)
    assert t.device is None
    assert t.devices == [0]


def test_single_int_device_places_one_tenant(cfg):
    t = _llm(cfg, device=3)
    assert t.device == 3
    assert t.devices == [3]


def test_list_device_is_tensor_parallel_across_all_of_them(cfg):
    t = _llm(cfg, device=[2, 0])
    assert t.devices == [0, 2], "normalised ascending"


def test_device_out_of_range_names_the_tenant_and_the_range(cfg):
    with pytest.raises(ValueError, match=r"'llm'.*out of range.*0\.\.7"):
        _llm(cfg, device=8)


def test_negative_device_is_rejected(cfg):
    with pytest.raises(ValueError, match="out of range"):
        _llm(cfg, device=-1)


def test_duplicate_devices_are_rejected(cfg):
    with pytest.raises(ValueError, match="duplicate"):
        _llm(cfg, device=[0, 0])


def test_empty_device_list_is_rejected(cfg):
    with pytest.raises(ValueError, match="device list is empty"):
        _llm(cfg, device=[])


def test_non_integer_device_is_rejected(cfg):
    with pytest.raises(ValueError, match="not an integer"):
        _llm(cfg, device=["0"])


def test_device_lands_in_the_run_manifest(cfg):
    d = _llm(cfg, device=[0, 1]).to_dict()
    assert d["device"] == [0, 1]
    assert d["devices"] == [0, 1]


def test_solo_baselines_are_not_shared_across_devices(cfg):
    """Same model, same load, different card ⇒ different baseline."""
    a, b = _llm(cfg, device=0), _llm(cfg, device=1)
    assert _solo_key(a) != _solo_key(b)
    assert _solo_key(_llm(cfg)) == _solo_key(a), "unspecified is GPU 0"


def test_solo_baselines_are_not_shared_across_tp_widths(cfg):
    assert _solo_key(_llm(cfg, device=[0, 1])) != _solo_key(_llm(cfg, device=0))


def test_solo_baselines_are_not_shared_across_vram_caps(cfg):
    """The cap sets the KV cache, so the same model at 0.45 and 0.70 are two
    deployments. Sharing one baseline would compare the 4-tenant window
    against a reference taken at the 2-tenant window's cap."""
    a = _llm(cfg, gpu_memory_utilization=0.45)
    b = _llm(cfg, gpu_memory_utilization=0.70)
    assert _solo_key(a) != _solo_key(b)


# --------------------------------------------------------------------------- #
# VRAM cap derivation (docs/contention.md §2b)
# --------------------------------------------------------------------------- #


def test_derive_cap_is_weights_plus_constant_kv_plus_overhead():
    # (15.2 + 20 + 2) / 96 = 0.3875 → 0.39
    assert derive_cap(15.2, 20.0, 96.0) == 0.39


def test_derive_cap_overhead_is_adjustable():
    assert derive_cap(15.2, 20.0, 96.0, overhead_gb=0.0) == 0.37


def test_derive_cap_rejects_a_model_that_cannot_fit():
    """A cap over 1.0 is not a rounding matter — the model plus that KV
    budget does not exist on the card, and a silent clamp would hand the
    ladder an unequal KV cache."""
    with pytest.raises(ValueError, match=r"qwen2\.5-32b.*does not fit"):
        derive_cap(65.5, 40.0, 96.0, model_id="qwen2.5-32b")


def test_explicit_cap_wins_over_derivation(cfg):
    """The yaml value is the escape hatch; derivation only fills gaps."""
    t = _llm(cfg, gpu_memory_utilization=0.45, kv_budget_gb=20.0)
    assert t.gpu_memory_utilization == 0.45
    assert t.kv_budget_gb is None, "not derived, so nothing was held constant"


def test_missing_cap_is_derived_from_the_colocation_budget(cfg):
    t = _llm(cfg, kv_budget_gb=20.0)
    weights = cfg["models"]["qwen2.5-7b"]["weights_gb"]
    assert t.gpu_memory_utilization == derive_cap(weights, 20.0, cfg["vram_gb"])
    assert t.kv_budget_gb == 20.0


def test_size_ladder_varies_the_cap_but_not_the_kv_cache(cfg):
    """The point of the sizing rule: only the weights move between rungs, so
    a degradation ratio is attributable to the neighbour rather than to a
    KV cache that shrank underneath it."""
    runs = [r for r in _coloc_runs(cfg, "cross-size-scaling") if not r.is_solo]
    llms = [next(t for t in r.tenants if t.name == "llm") for r in runs]
    assert len(llms) == 4

    caps = [t.gpu_memory_utilization for t in llms]
    assert len(set(caps)) == len(caps), "weights differ, so the caps must"

    assert {t.kv_budget_gb for t in llms} == {16.0}, "one budget for the whole ladder"

    vram = cfg["vram_gb"]
    kv = [
        t.gpu_memory_utilization * vram - cfg["models"][t.round.model_id]["weights_gb"]
        for t in llms
    ]
    # The cap is a 2-dp fraction, so the KV cache can only be held constant to
    # within half of that last digit — 0.005 × 96 GB ≈ 0.5 GB, against a 16 GB
    # budget. Anything wider means the sizing, not the rounding.
    assert max(kv) - min(kv) <= 0.005 * vram * 2, f"KV drifted across the ladder: {kv}"


def test_size_ladder_top_rung_can_actually_load_its_weights(cfg):
    """The inherited 0.45 gave 43 GB to a 45 GB checkpoint — it never
    started. Every rung's reservation must clear its own weights."""
    for run in _coloc_runs(cfg, "cross-size-scaling"):
        for t in run.tenants:
            weights = (cfg["models"][t.round.model_id] or {}).get("weights_gb")
            if weights is None or t.gpu_memory_utilization is None:
                continue
            assert t.gpu_memory_utilization * cfg["vram_gb"] > weights, t.round.model_id


def test_solo_baseline_inherits_the_contention_cap(cfg):
    """§2b: a baseline run at a bigger cap has a bigger KV cache, and the
    ratio then reports our own memory allocation as contention."""
    runs = _coloc_runs(cfg, "cross-size-scaling")
    contention = {
        t.round.model_id: t
        for r in runs if not r.is_solo for t in r.tenants if t.name == "llm"
    }
    solos = [
        r.tenants[0] for r in runs
        if r.is_solo and r.tenants[0].round.model_id in contention
    ]
    assert solos
    for s in solos:
        assert s.gpu_memory_utilization == contention[s.round.model_id].gpu_memory_utilization


# --------------------------------------------------------------------------- #
# Serialisation (consumed by the orchestrator over NDJSON)
# --------------------------------------------------------------------------- #


def test_to_dict_is_json_serialisable_and_carries_identity(cfg):
    import json

    coloc = next(r for r in _coloc_runs(cfg, "mix-llm-cv") if not r.is_solo)
    d = json.loads(json.dumps(coloc.to_dict()))

    assert d["id"] == "mix-llm-cv"
    assert d["n_tenants"] == 2
    assert d["run_label"] == "coloc:mix-llm-cv"
    assert d["isolation"] == "mps"
    assert d["phase"] == 3
    assert {t["name"] for t in d["tenants"]} == {"llm", "cv"}
    assert d["tenants"][0]["round"]["hf_id"]


def test_every_defined_colocation_resolves(cfg):
    """Catches typos in model/backend/workload names across the whole block."""
    for name in cfg["colocations"]:
        runs = _coloc_runs(cfg, name)
        assert runs, f"{name} produced no runs"
        assert any(not r.is_solo for r in runs), f"{name} produced no contention window"
