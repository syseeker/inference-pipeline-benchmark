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
    runs = _coloc_runs(cfg, "cross-llm-vs-cv-rps")
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
    colocs = [r for r in _coloc_runs(cfg, "cross-llm-vs-cv-rps") if not r.is_solo]
    assert len(colocs) == 4
    by_name = [{t.name: t.load.rps for t in c.tenants} for c in colocs]
    assert [d["cv"] for d in by_name] == [1.0, 10.0, 50.0, 200.0]
    assert {d["llm"] for d in by_name} == {4.0}, "other tenants held constant"


def test_vary_changes_the_field_it_names(cfg):
    colocs = [r for r in _coloc_runs(cfg, "secondary-backend-llm-a") if not r.is_solo]
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
    positional merging would silently drop the parent's settings.

    cross-memory-pressure-kv13 is the sharpest case in the config: it extends
    -kv03 and overrides nothing but the two caps, so every model, workload and
    rate has to survive the merge untouched. That is also the invariant the
    whole curve rests on — if a model changed between rungs, the curve would
    be measuring the model rather than the KV cache.
    """
    coloc = next(r for r in _coloc_runs(cfg, "cross-memory-pressure-kv13")
                 if not r.is_solo)
    anchor = next(t for t in coloc.tenants if t.name == "anchor")
    neighbour = next(t for t in coloc.tenants if t.name == "neighbour")

    assert anchor.gpu_memory_utilization == 0.58, "child override applied"
    assert anchor.round.model_id == "qwen2.5-72b", "parent model inherited"
    assert anchor.load.rps == 2.0, "parent load inherited"
    assert neighbour.gpu_memory_utilization == 0.22, "child override applied"
    assert neighbour.round.model_id == "qwen2.5-7b", "parent model inherited"
    assert neighbour.workload == "llm_short", "parent workload inherited"


# ── tenant naming ────────────────────────────────────────────────────────────

def test_tenant_names_match_what_they_actually_run(cfg):
    """A tenant's name is its label in the manifest and in every summary row,
    so a VLM called "llm" is a mislabel a reader cannot see through.

    This regressed once: `extends` merges tenants BY NAME, so overriding
    mix-llm-cv's `llm` tenant to hold kosmos-2.5 forced the name "llm" onto a
    document model. Spelling the tenants out instead of inheriting is the fix,
    and this test is what stops the shortcut being taken again.
    """
    CV = {"yolov8-l", "yolov8-n", "dinov2-base", "dinov2-large",
          "rfdetr-medium", "paddleocr"}
    ILM = {"kosmos-2.5"}
    VLM = {"qwen2.5-vl-7b", "qwen2.5-vl-72b", "gemma-4-31b-it-fp8"}
    # Role names that deliberately describe a position in the experiment
    # rather than a model category.
    ROLE_NAMES = {"anchor", "neighbour", "subject"}

    def category(model_id):
        if model_id in CV:
            return "cv"
        if model_id in ILM:
            return "ilm"
        if model_id in VLM:
            return "vlm"
        return "llm"

    bad = []
    for name in cfg["colocations"]:
        for coloc in _coloc_runs(cfg, name):
            if coloc.is_solo:
                continue
            for t in coloc.tenants:
                stem = t.name.split("_")[0].rstrip("0123456789")
                if stem in ROLE_NAMES:
                    continue
                expected = category(t.round.model_id)
                # An image-language WORKLOAD on a video-capable VLM is a
                # legitimate ILM role — judge by the job, not just the weights.
                if expected == "vlm" and stem == "ilm" and t.workload == "ilm_document":
                    continue
                if stem != expected:
                    bad.append(f"{name}: tenant {t.name!r} runs "
                               f"{t.round.model_id} ({expected})")
    assert not bad, "tenant names disagree with their models:\n  " + "\n  ".join(sorted(set(bad)))


def test_rps_sweep_on_star_moves_every_tenant_together(cfg):
    """A same-category colocation asks where the PAIR saturates the card,
    which is a curve in AGGREGATE offered load. Sweeping one tenant against a
    fixed neighbour would silently be a cross-* experiment instead."""
    colocs = [r for r in _coloc_runs(cfg, "same-llm") if not r.is_solo]
    assert len(colocs) == 4
    for c in colocs:
        rates = {t.load.rps for t in c.tenants}
        assert len(rates) == 1, "both tenants move, and move together"
    assert [c.tenants[0].load.rps for c in colocs] == [1.0, 4.0, 16.0, 64.0]


def test_rps_sweep_names_an_unknown_tenant_loudly():
    """A typo'd tenant name would otherwise sweep nothing and produce N
    identical runs that look like a completed experiment."""
    colos = {
        "x": {
            "tenants": [
                {"name": "llm", "backend": "vllm", "model": "qwen2.5-7b"},
            ],
            "rps_sweep": {"tenant": "lmm", "values": [1, 2]},
        }
    }
    with pytest.raises(ValueError, match=r"not in this roster"):
        list(iter_colocation({"vram_gb": 96, "colocations": colos,
                              "models": {"qwen2.5-7b": {"hf_id": "x"}},
                              "backends": {"vllm": {"base_url": "u", "port": 1}}}, "x"))


# --------------------------------------------------------------------------- #
# Repetitions
# --------------------------------------------------------------------------- #


def test_repetitions_emit_one_window_per_repeat(cfg):
    """Near-OOM behaviour is bimodal — the model either fits or thrashes, and
    the mean of those two states describes neither. The spread across repeats
    is the finding, so the runs have to actually exist."""
    colocs = [r for r in _coloc_runs(cfg, "cross-memory-pressure-kv29") if not r.is_solo]
    assert len(colocs) == 3
    assert [c.repetition for c in colocs] == [1, 2, 3]
    assert {c.run_label for c in colocs} == {"coloc:cross-memory-pressure-kv29"}, (
        "repeats are samples of one experiment, not three experiments"
    )


def test_repetitions_do_not_multiply_the_baselines(cfg):
    """Baselines dedup by `_solo_key` here and are cached again per session by
    the orchestrator, so a repeated baseline would be dropped anyway."""
    solos = [r for r in _coloc_runs(cfg, "cross-memory-pressure-kv29") if r.is_solo]
    assert len(solos) == 2, "one per tenant, not one per tenant per repeat"


def test_repetitions_default_to_one(cfg):
    for r in _coloc_runs(cfg, "mix-llm-cv"):
        assert r.repetition == 1


def test_repetitions_below_one_is_rejected():
    colos = {
        "x": {
            "tenants": [{"name": "llm", "backend": "vllm", "model": "qwen2.5-7b"}],
            "repetitions": 0,
        }
    }
    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        list(iter_colocation({"vram_gb": 96, "colocations": colos,
                              "models": {"qwen2.5-7b": {"hf_id": "x"}},
                              "backends": {"vllm": {"base_url": "u", "port": 1}}}, "x"))


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


# --------------------------------------------------------------------------- #
# Experiment-design coverage (docs/contention-phases.md)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,n_tenants,n_points",
    [
        ("same-llm", 2, 4),
        ("same-cv", 2, 4),
        ("same-vlm", 2, 4),
        ("same-ilm", 2, 4),
    ],
)
def test_same_category_family_is_a_saturation_curve(cfg, name, n_tenants, n_points):
    """§3's "two models stressing the same resource fight hardest" has nothing
    to be checked against while every colocation is cross-category."""
    colocs = [r for r in _coloc_runs(cfg, name) if not r.is_solo]
    assert len(colocs) == n_points
    assert all(len(c.tenants) == n_tenants for c in colocs)
    rates = [sorted(t.load.rps for t in c.tenants) for c in colocs]
    assert rates == sorted(rates), "load must climb monotonically along the curve"


def test_same_vlm_pairs_two_models_that_can_actually_serve_video(cfg):
    """The customer's second VLM (paligemma2) is image-only, so a video pair
    cannot be built from their picks. Whatever we substituted must not have
    inherited the same defect."""
    coloc = next(r for r in _coloc_runs(cfg, "same-vlm") if not r.is_solo)
    for t in coloc.tenants:
        assert t.workload == "vlm_video_long"
        unsupported = cfg["models"][t.round.model_id].get("unsupported_workloads") or {}
        assert "vlm_video_long" not in unsupported, t.round.model_id


def test_mix_full_is_the_only_four_category_window(cfg):
    """It is what makes multi-GPU placement a real decision — with four
    tenants and two cards there are three pairings to rank."""
    coloc = next(r for r in _coloc_runs(cfg, "mix-full") if not r.is_solo)
    assert {t.name for t in coloc.tenants} == {"llm", "vlm", "ilm", "cv"}
    assert len(coloc.tenants) == 4


def test_mix_full_tenants_keep_the_two_tenant_kv_budget(cfg):
    """§2b: if the caches shrink when the neighbours arrive, the 4-tenant
    degradation is partly our own memory allocation."""
    coloc = next(r for r in _coloc_runs(cfg, "mix-full") if not r.is_solo)
    derived = [t.kv_budget_gb for t in coloc.tenants if t.kv_budget_gb is not None]
    assert derived and set(derived) == {20.0}
    caps = [t.gpu_memory_utilization or 0 for t in coloc.tenants]
    assert sum(caps) < 1.0, "four tenants must still fit on one card"


@pytest.mark.parametrize("name", ["cross-ilm-vs-cv", "cross-cv-vs-llm-rps"])
def test_cross_sweeps_move_one_tenant_and_hold_the_subject(cfg, name):
    """A cross-* experiment is only attributable if the subject's own load is
    the one thing that did not move."""
    colocs = [r for r in _coloc_runs(cfg, name) if not r.is_solo]
    assert len(colocs) == 4
    swept = "cv" if name == "cross-ilm-vs-cv" else "llm"
    subject = {t.name for c in colocs for t in c.tenants} - {swept}
    for held in subject:
        assert len({
            next(t.load.rps for t in c.tenants if t.name == held) for c in colocs
        }) == 1, f"{held} drifted"
    assert len({
        next(t.load.rps for t in c.tenants if t.name == swept) for c in colocs
    }) == 4


MEMORY_PRESSURE_CURVE = [
    "cross-memory-pressure-kv03",
    "cross-memory-pressure-kv13",
    "cross-memory-pressure-kv22",
    "cross-memory-pressure-kv29",
]


def test_memory_pressure_curve_climbs_towards_the_ceiling(cfg):
    """Four points from comfortable to near-OOM. Non-monotone reservation
    would mean the curve doubles back and the knee is unlocatable."""
    reserved = []
    for name in MEMORY_PRESSURE_CURVE:
        c = next(r for r in _coloc_runs(cfg, name) if not r.is_solo)
        assert len(c.tenants) == 2
        reserved.append(sum(t.gpu_memory_utilization for t in c.tenants))
    assert reserved == sorted(reserved), reserved
    assert reserved[-1] > 0.95, "the top rung has to actually approach the ceiling"
    assert reserved[-1] <= 1.0, "and must still be loadable"


def test_memory_pressure_moves_the_kv_cache_and_nothing_else(cfg):
    """KV is the single independent variable, so it must move monotonically —
    and the models must NOT move with it.

    Note the direction. The customer's ladder swapped in a bigger neighbour at
    each rung, so KV *shrank* as reservation grew. Here the models are fixed
    and only the caps move, so KV *grows* along the curve: kv03 is the starved
    end (where the eviction cliff should be) and kv29 is the committed end
    (where the allocator should misbehave). Asserting the old direction would
    be asserting the old confound.
    """
    kv, models = [], []
    for name in MEMORY_PRESSURE_CURVE:
        c = next(r for r in _coloc_runs(cfg, name) if not r.is_solo)
        t = next(t for t in c.tenants if t.name == "neighbour")
        weights = cfg["models"][t.round.model_id]["weights_gb"]
        kv.append(round(t.gpu_memory_utilization * cfg["vram_gb"] - weights, 1))
        models.append(tuple(sorted(x.round.model_id for x in c.tenants)))
        assert t.kv_budget_gb is None, "explicit cap, not derived"
    assert kv == sorted(kv), f"KV must move monotonically along the curve: {kv}"
    assert len(set(models)) == 1, (
        f"the models must be identical at every rung, else a throughput drop "
        f"could be the model rather than the cache: {set(models)}"
    )


def test_memory_pressure_holds_the_kv_split_between_tenants(cfg):
    """Only the TOTAL KV may move. If the anchor:neighbour split drifted too,
    the curve would have two variables and neither tenant's cliff would be
    attributable."""
    ratios = []
    for name in MEMORY_PRESSURE_CURVE:
        c = next(r for r in _coloc_runs(cfg, name) if not r.is_solo)
        kv = {}
        for t in c.tenants:
            weights = cfg["models"][t.round.model_id]["weights_gb"]
            kv[t.name] = t.gpu_memory_utilization * cfg["vram_gb"] - weights - 2.0
        ratios.append(kv["anchor"] / kv["neighbour"])
    # 2 dp caps cannot hit an exact ratio; hold it to a band rather than a value.
    assert max(ratios) - min(ratios) < 0.5, f"KV split drifts across rungs: {ratios}"


def test_memory_pressure_holds_the_offered_load_constant(cfg):
    """The customer's config halves the rate at the extreme point. Ours loads,
    so halving would confound the cliff with a load change."""
    loads = []
    for name in MEMORY_PRESSURE_CURVE:
        c = next(r for r in _coloc_runs(cfg, name) if not r.is_solo)
        loads.append({t.name: t.load.rps for t in c.tenants})
    assert all(d == loads[0] for d in loads), loads


def test_every_memory_pressure_rung_can_load_its_weights(cfg):
    for name in MEMORY_PRESSURE_CURVE:
        for run in _coloc_runs(cfg, name):
            for t in run.tenants:
                weights = cfg["models"][t.round.model_id]["weights_gb"]
                assert t.gpu_memory_utilization * cfg["vram_gb"] > weights, (
                    f"{name}/{t.round.model_id}"
                )


SECONDARY_DIMENSIONS = [
    ("secondary-backend-llm", 2),
    ("secondary-backend-cv", 3),
    ("secondary-output-length", 2),
    ("secondary-input-size-cv", 2),
    ("secondary-input-size-llm", 2),
    ("secondary-asymmetry", 3),
    ("secondary-arrival", 2),
]


@pytest.mark.parametrize("stem,n_points", SECONDARY_DIMENSIONS)
def test_every_secondary_dimension_runs_against_both_baselines(cfg, stem, n_points):
    """The interaction is the finding — "backend choice matters 3x more under
    memory pressure" is invisible with one baseline. A dimension that exists
    only in `-a` answers half its question."""
    for suffix, base in (("-a", "mix-llm-cv"), ("-b", "mix-memory-bound")):
        colocs = [r for r in _coloc_runs(cfg, stem + suffix) if not r.is_solo]
        assert len(colocs) == n_points, stem + suffix
        assert cfg["colocations"][stem + suffix]["extends"] == base


def test_the_two_baselines_actually_contrast_in_memory_pressure(cfg):
    """A pair of baselines that reserve the same VRAM is one baseline run
    twice."""
    def reserved(name):
        c = next(r for r in _coloc_runs(cfg, name) if not r.is_solo)
        return sum(t.gpu_memory_utilization or 0 for t in c.tenants)

    assert reserved("mix-llm-cv") < 0.6
    assert reserved("mix-memory-bound") > 0.9


def test_baseline_b_still_leaves_room_for_its_triton_tenant(cfg):
    """The CV tenant has no `gpu_memory_utilization` to reserve with — it takes
    whatever the vLLM tenants did not, so `sum < 1.0` is not a formality."""
    c = next(r for r in _coloc_runs(cfg, "mix-memory-bound") if not r.is_solo)
    vllm_caps = sum(t.gpu_memory_utilization for t in c.tenants
                    if t.gpu_memory_utilization is not None)
    assert vllm_caps <= 0.95
    assert (1.0 - vllm_caps) * cfg["vram_gb"] > 4.0, "Triton needs a few GB"


def test_input_size_llm_moves_prefill_without_moving_decode(cfg):
    """A different question from output-length: llm_short → llm_long changes
    both the prompt AND the output, so it can never separate the two."""
    colocs = [r for r in _coloc_runs(cfg, "secondary-input-size-llm-a") if not r.is_solo]
    outs = {next(t.load.output_tokens for t in c.tenants if t.name == "llm")
            for c in colocs}
    assert outs == {32}, "output length must be the constant here"


# --------------------------------------------------------------------------- #
# Phase 5 — placement (docs/contention.md §5)
# --------------------------------------------------------------------------- #


PLACEMENT_PAIRINGS = {
    # colocation -> {tenant: device}
    "place-p1": {"llm": 0, "vlm": 0, "ilm": 1, "cv": 1},
    "place-p2": {"llm": 0, "ilm": 0, "vlm": 1, "cv": 1},
    "place-p3": {"llm": 0, "cv": 0, "vlm": 1, "ilm": 1},
}


def _window(cfg, name) -> Colocation:
    return next(r for r in _coloc_runs(cfg, name) if not r.is_solo)


@pytest.mark.parametrize("name,expected", sorted(PLACEMENT_PAIRINGS.items()))
def test_each_pairing_splits_four_tenants_two_per_card(cfg, name, expected):
    """With four tenants over two cards there are exactly three pairings, and
    each one has to actually be a 2/2 split — a pairing that quietly left
    three tenants on GPU 0 would be `mix-full` wearing a Phase 5 name."""
    c = _window(cfg, name)
    assert len(c.tenants) == 4
    placement = {t.name: t.devices for t in c.tenants}
    assert placement == {n: [d] for n, d in expected.items()}

    per_gpu: dict[int, list[str]] = {0: [], 1: []}
    for t in c.tenants:
        per_gpu[t.devices[0]].append(t.name)
    assert len(per_gpu[0]) == 2 and len(per_gpu[1]) == 2


def test_the_three_pairings_rearrange_mix_full_and_change_nothing_else(cfg):
    """Placement is the only variable. If a model or an offered rate drifted
    between the pairings, the ranking would not be a placement finding."""
    ref = {
        t.name: (t.round.model_id, t.workload, t.load.pattern, t.load.rps)
        for t in _window(cfg, "mix-full").tenants
    }
    for name in PLACEMENT_PAIRINGS:
        got = {
            t.name: (t.round.model_id, t.workload, t.load.pattern, t.load.rps)
            for t in _window(cfg, name).tenants
        }
        assert got == ref, name


def test_the_three_pairings_hold_the_kv_cache_constant(cfg):
    """THE constraint of Phase 5, and the one most likely to be "fixed" away.

    P1 puts both vLLM tenants on GPU 0; P2 and P3 give each of them a card.
    Derive the caps per-GPU and P1's tenants get roughly half the KV cache of
    P2's and P3's — P1 then comes out slowest for a reason that has nothing to
    do with its neighbours (docs/contention.md §2b). So the budget is sized
    for P1, the tightest case, and every vLLM tenant gets the SAME absolute
    cap in all three."""
    caps: dict[str, set] = {}
    budgets = set()
    for name in PLACEMENT_PAIRINGS:
        for t in _window(cfg, name).tenants:
            if t.gpu_memory_utilization is None:
                continue           # Triton tenants reserve no GPU fraction
            caps.setdefault(t.name, set()).add(t.gpu_memory_utilization)
            budgets.add(t.kv_budget_gb)

    assert set(caps) == {"llm", "vlm"}, "both vLLM tenants must be covered"
    for tenant, values in caps.items():
        assert len(values) == 1, f"{tenant} cap varies across the pairings: {values}"
    assert budgets == {20.0}, "one KV budget for the whole comparison set"


def test_the_pairings_keep_mix_fulls_kv_budget(cfg):
    """The 1-GPU `mix-full` is the before/after these are ranked against, so
    its tenants have to be running the same caches too."""
    ref = {
        t.name: (t.gpu_memory_utilization, t.kv_budget_gb)
        for t in _window(cfg, "mix-full").tenants
    }
    for name in PLACEMENT_PAIRINGS:
        got = {
            t.name: (t.gpu_memory_utilization, t.kv_budget_gb)
            for t in _window(cfg, name).tenants
        }
        assert got == ref, name


@pytest.mark.parametrize(
    "name", [*PLACEMENT_PAIRINGS, "place-isolated", "place-vlm-prefill-split"]
)
def test_every_placement_window_fits_on_each_card_it_uses(cfg, name):
    """The `sum <= 1.0` rule is per DEVICE on multi-GPU (§5), and the Triton
    tenants take their footprint out of whatever the vLLM caps left on their
    own card."""
    per_gpu: dict[int, float] = {}
    triton_gpus: set[int] = set()
    for t in _window(cfg, name).tenants:
        if t.round.transport == "triton":
            triton_gpus.update(t.devices)
            continue
        for d in t.devices:
            per_gpu[d] = per_gpu.get(d, 0.0) + t.gpu_memory_utilization
    for dev, total in per_gpu.items():
        assert total <= 1.0, f"{name} GPU {dev}: {total}"
        if dev in triton_gpus:
            assert (1.0 - total) * cfg["vram_gb"] > 4.0, (
                f"{name} GPU {dev} leaves no room for its Triton tenant"
            )


@pytest.mark.parametrize("name", ["place-isolated", "place-vlm-prefill-split"])
def test_the_two_gpu_repeats_put_their_tenants_on_different_cards(cfg, name):
    """No shared SMs, bandwidth or VRAM — so these are the runs whose
    degradation ratios must come back ~1.0. Both tenants on one card would
    make that a re-run of the 1-GPU window under a new name."""
    c = _window(cfg, name)
    assert len(c.tenants) == 2
    devices = [t.devices for t in c.tenants]
    assert devices == [[0], [1]] or devices == [[1], [0]]
    assert len({d[0] for d in devices}) == 2


@pytest.mark.parametrize(
    "name,parent",
    [("place-isolated", "mix-llm-cv"),
     ("place-vlm-prefill-split", "cross-vlm-prefill-vs-llm")],
)
def test_the_two_gpu_repeats_are_a_before_after_of_a_one_gpu_run(cfg, name, parent):
    """The extra card is the only difference — same models, same rates, same
    caps. A different cap would fold a KV-cache change into the answer to
    "what does a second GPU buy me?"."""
    assert cfg["colocations"][name]["extends"] == parent
    ref = {
        t.name: (t.round.model_id, t.workload, t.load.rps, t.gpu_memory_utilization)
        for t in _window(cfg, parent).tenants
    }
    got = {
        t.name: (t.round.model_id, t.workload, t.load.rps, t.gpu_memory_utilization)
        for t in _window(cfg, name).tenants
    }
    assert got == ref


def test_placement_baselines_are_taken_on_the_placed_card(cfg):
    """§5: the baseline must match the topology, not just the cap. A tenant
    pinned to GPU 1 compared against a GPU 0 baseline is comparing cards."""
    for name in [*PLACEMENT_PAIRINGS, "place-isolated", "place-vlm-prefill-split"]:
        runs = _coloc_runs(cfg, name)
        placed = {t.name: t.devices for t in _window(cfg, name).tenants}
        solos = [r.tenants[0] for r in runs if r.is_solo]
        assert len(solos) == len(placed)
        for s in solos:
            assert s.devices == placed[s.name], name


def test_no_tensor_parallel_entries_while_the_interconnect_is_unconfirmed(cfg):
    """`nvlink: false` on this card means TP traffic crosses PCIe every
    forward pass, which would dominate the result. Deferred until
    `nvidia-smi topo -m` says otherwise — not silently added."""
    assert cfg.get("nvlink") is False
    for name in cfg["colocations"]:
        for run in _coloc_runs(cfg, name):
            for t in run.tenants:
                assert len(t.devices) == 1, f"{name}/{t.name} is tensor-parallel"


def test_every_defined_colocation_resolves(cfg):
    """Catches typos in model/backend/workload names across the whole block."""
    for name in cfg["colocations"]:
        runs = _coloc_runs(cfg, name)
        assert runs, f"{name} produced no runs"
        assert any(not r.is_solo for r in runs), f"{name} produced no contention window"


# ── one port per HTTP tenant ────────────────────────────────────────────────
#
# Regression: `backends.<b>.port` is a BACKEND-wide default, so two tenants on
# the same backend inherited the same port. The second server took the endpoint
# and each driver was answered by the other tenant's model — mix-full logged
# 332/700 and 83/178 requests failing with HTTP 404 "The model ... does not
# exist", and the survivors were whichever server owned the port at the time.

def test_two_vllm_tenants_get_different_ports():
    cfg = load_gpu_config("rtx_pro6000")
    win = next(c for c in iter_colocation(cfg, "mix-full") if not c.is_solo)
    http = [t for t in win.tenants if t.round.transport != "triton"]
    ports = [t.round.port for t in http]
    assert len(ports) == len(set(ports)), f"HTTP tenants share a port: {ports}"
    for t in http:
        assert f":{t.round.port}" in t.round.base_url, "base_url must follow the port"


def test_first_tenant_keeps_the_configured_port():
    """Bumped only on collision, so a single-HTTP-tenant colocation is
    unchanged and its baseline stays the same deployment."""
    cfg = load_gpu_config("rtx_pro6000")
    win = next(c for c in iter_colocation(cfg, "mix-llm-cv") if not c.is_solo)
    llm = next(t for t in win.tenants if t.name == "llm")
    assert llm.round.port == 8000


def test_solo_baselines_are_not_re_ported():
    """A baseline is the reference for the window; shifting its port would make
    it a different deployment."""
    cfg = load_gpu_config("rtx_pro6000")
    for c in iter_colocation(cfg, "mix-full"):
        if c.is_solo and c.tenants[0].round.transport != "triton":
            assert c.tenants[0].round.port == 8000


def test_triton_tenants_are_left_alone():
    """Triton is addressed per device, not per tenant — two CV tenants on one
    card share a container by design."""
    cfg = load_gpu_config("rtx_pro6000")
    win = next(c for c in iter_colocation(cfg, "mix-full") if not c.is_solo)
    triton = [t.round.port for t in win.tenants if t.round.transport == "triton"]
    assert triton == [8100, 8100]
