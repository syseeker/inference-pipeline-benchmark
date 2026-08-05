"""`bench coloc` — selection, plan dedup, resume and continue-on-error.

Everything here runs without a GPU: ColocationOrchestrator.run is
monkeypatched, so what is under test is the CLI's plan/loop/accounting, which
is where a 7-hour study is won or lost.

The CLI (unlike scenario_config) genuinely needs typer, so this file skips in a
bare environment rather than stubbing a package deep enough to route a command.
"""

from __future__ import annotations

import json
import subprocess

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from benchmarks import coloc as coloc_mod  # noqa: E402
from benchmarks.cli import app, select_colocations  # noqa: E402
from benchmarks.scenario_config import load_gpu_config  # noqa: E402

GPU = "rtx_pro6000"
runner = CliRunner()


@pytest.fixture(scope="module")
def cfg():
    return load_gpu_config(GPU)


def _invoke(*args, out=None):
    argv = ["coloc", "--gpu", GPU, "--json", *args]
    if out is not None:
        argv += ["--out", str(out)]
    res = runner.invoke(app, argv)
    payload = json.loads(res.stdout.strip().splitlines()[-1]) if res.stdout.strip() else None
    return res, payload


# ── selection ───────────────────────────────────────────────────────────────

def test_all_selects_every_colocation_in_the_yaml(cfg):
    assert set(select_colocations(cfg, all_=True)) == set(cfg["colocations"])


def test_selection_is_ordered_by_phase_then_yaml_position(cfg):
    """A study is compared against a previous run of the same study, so the
    order must not depend on set iteration or on how the flags were typed."""
    names = select_colocations(cfg, all_=True)
    declared = list(cfg["colocations"])
    keys = [((cfg["colocations"][n] or {}).get("phase"), declared.index(n)) for n in names]
    assert keys == sorted(keys, key=lambda k: (k[0] if isinstance(k[0], int) else 10**6, k[1]))
    assert names == select_colocations(cfg, all_=True)          # deterministic


def test_phase_selectors_compose_and_are_order_insensitive(cfg):
    both = select_colocations(cfg, phases=[3, 4])
    assert both == select_colocations(cfg, phases=[4, 3])
    assert both == select_colocations(cfg, phases=[3]) + select_colocations(cfg, phases=[4])
    assert all((cfg["colocations"][n] or {}).get("phase") in (3, 4) for n in both)


def test_repeated_colocation_flags_select_a_set(cfg):
    got = select_colocations(cfg, names=["mix-vlm-cv", "mix-llm-cv"])
    assert got == ["mix-llm-cv", "mix-vlm-cv"]                  # yaml order, not argv order


def test_no_selector_names_the_alternatives(cfg):
    with pytest.raises(ValueError, match="--colocation .*--phase.*--all"):
        select_colocations(cfg)


def test_all_with_another_selector_is_an_error(cfg):
    with pytest.raises(ValueError, match="--all already selects every"):
        select_colocations(cfg, all_=True, names=["mix-llm-cv"])
    with pytest.raises(ValueError, match="--all already selects every"):
        select_colocations(cfg, all_=True, phases=[3])


def test_colocation_and_phase_together_is_an_error(cfg):
    with pytest.raises(ValueError, match="pick one selector"):
        select_colocations(cfg, names=["mix-llm-cv"], phases=[3])


def test_unknown_names_and_phases_are_rejected(cfg):
    with pytest.raises(ValueError, match="unknown colocation"):
        select_colocations(cfg, names=["mix-llm-cv", "nope"])
    with pytest.raises(ValueError, match="defined phases"):
        select_colocations(cfg, phases=[99])


def test_missing_selector_exits_generic_with_a_remediation():
    res, payload = _invoke()
    assert res.exit_code == 1
    assert payload["status"] == "error"
    assert "--all" in payload["error"]["remediation"]


# ── the whole point: one plan, deduped baselines ────────────────────────────

def test_the_whole_study_is_one_plan_with_deduped_baselines():
    """41 colocations run as ONE plan: 154 runs (66 solo + 88 contention).
    Run as 39 separate commands it is 237 — 74 redundant baselines, ~3h of GPU."""
    res, payload = _invoke("--all", "--dry-run")
    assert res.exit_code == 0
    assert payload["data"]["n_runs"] == 154
    assert payload["data"]["n_solo"] == 66
    assert payload["data"]["n_coloc"] == 88


def test_a_single_colocation_still_reports_its_own_shape():
    res, payload = _invoke("--colocation", "mix-llm-cv", "--dry-run")
    assert res.exit_code == 0
    assert payload["data"]["colocation"] == "mix-llm-cv"        # unchanged for agents
    assert (payload["data"]["n_solo"], payload["data"]["n_coloc"]) == (2, 1)


def test_dry_run_reports_preflight_across_the_whole_selection():
    res, payload = _invoke("--all", "--dry-run")
    assert "preflight_issues" in payload["data"]
    assert all("preflight_issues" in p for p in payload["data"]["plan"])
    assert len(payload["data"]["plan"]) == 154


def test_dry_run_run_dirs_are_unique(tmp_path):
    _, payload = _invoke("--all", "--dry-run", out=tmp_path)
    dirs = [p["run_dir"] for p in payload["data"]["plan"]]
    assert len(set(dirs)) == len(dirs) == 154


def test_run_dirs_do_not_depend_on_which_phase_was_selected(tmp_path):
    _, everything = _invoke("--all", "--dry-run", out=tmp_path)
    _, phase3 = _invoke("--phase", "3", "--dry-run", out=tmp_path)
    assert {p["run_dir"] for p in phase3["data"]["plan"]} <= {
        p["run_dir"] for p in everything["data"]["plan"]}


# ── live loop (orchestrator stubbed) ────────────────────────────────────────

@pytest.fixture
def fake_run(monkeypatch):
    """Stand in for the orchestrator: records the runs and writes a manifest,
    so --resume has something real to find. `fail_on` names colocation ids
    whose runs raise."""
    def factory(fail_on=()):
        calls: list[str] = []                                   # fresh per call

        def _run(self, coloc, paths):
            calls.append(coloc.id)
            if coloc.id in fail_on:
                raise RuntimeError(f"server for {coloc.id} never went ready")
            paths.root.mkdir(parents=True, exist_ok=True)
            paths.manifest.write_text(json.dumps({"colocation_id": coloc.id}))
            return {}
        monkeypatch.setattr(coloc_mod.ColocationOrchestrator, "run", _run)
        return calls

    return factory


def test_live_run_executes_every_planned_run(fake_run, tmp_path):
    calls = fake_run()
    res, payload = _invoke("--phase", "3", out=tmp_path)
    assert res.exit_code == 0
    assert payload["data"]["n_succeeded"] == len(calls) == payload["data"]["n_runs"]
    assert payload["data"]["n_failed"] == payload["data"]["n_skipped"] == 0


def test_progress_goes_to_stderr_so_json_stays_parseable(fake_run, tmp_path):
    fake_run()
    res, payload = _invoke("--phase", "3", out=tmp_path)
    assert payload["status"] == "ok"                            # stdout parsed as one line
    assert res.stdout.strip().count("\n") == 0
    assert "/12]" in res.stderr and "mix-llm-cv" in res.stderr


def test_resume_skips_runs_that_already_have_a_manifest(fake_run, tmp_path):
    first = fake_run()
    _invoke("--phase", "3", out=tmp_path)
    n = len(first)
    first.clear()

    res, payload = _invoke("--phase", "3", "--resume", out=tmp_path)
    assert res.exit_code == 0
    assert first == []                                          # nothing re-executed
    assert payload["data"]["n_skipped"] == n
    assert payload["data"]["n_succeeded"] == 0


def test_resume_reruns_only_what_is_missing(fake_run, tmp_path):
    calls = fake_run()
    _invoke("--phase", "3", out=tmp_path)
    victim = sorted(tmp_path.rglob("manifest.json"))[0]
    victim.unlink()
    calls.clear()

    _, payload = _invoke("--phase", "3", "--resume", out=tmp_path)
    assert len(calls) == 1
    assert payload["data"]["n_succeeded"] == 1
    assert payload["data"]["n_skipped"] == payload["data"]["n_runs"] - 1


def test_default_is_fail_fast(fake_run, tmp_path):
    calls = fake_run(fail_on={"mix-llm-cv"})
    res, payload = _invoke("--phase", "3", out=tmp_path)
    assert res.exit_code == 3
    assert payload["status"] == "error"
    assert calls[-1] == "mix-llm-cv"                            # stopped there
    assert len(calls) < 12


def test_continue_on_error_finishes_the_plan_and_accounts_for_it(fake_run, tmp_path):
    calls = fake_run(fail_on={"mix-llm-cv"})
    res, payload = _invoke("--phase", "3", "--continue-on-error", out=tmp_path)
    d = payload["data"]
    assert len(calls) == d["n_runs"]                            # nothing thrown away
    assert d["n_failed"] == sum(1 for c in calls if c == "mix-llm-cv")
    assert d["n_succeeded"] == d["n_runs"] - d["n_failed"]
    assert {f["colocation"] for f in d["failures"]} == {"mix-llm-cv"}
    assert "never went ready" in d["failures"][0]["error"]
    assert res.exit_code == 3                                   # partial success is not success
    assert payload["status"] == "error"


def test_continue_on_error_then_resume_retries_only_the_failures(fake_run, tmp_path):
    fake_run(fail_on={"mix-llm-cv"})
    _, first = _invoke("--phase", "3", "--continue-on-error", out=tmp_path)
    calls = fake_run()                                          # the flake is over
    _, second = _invoke("--phase", "3", "--continue-on-error", "--resume", out=tmp_path)
    assert len(calls) == first["data"]["n_failed"]
    assert second["data"]["n_failed"] == 0
    assert second["data"]["n_skipped"] == second["data"]["n_runs"] - len(calls)


def test_summary_is_regenerated_after_the_plan(fake_run, tmp_path, monkeypatch):
    fake_run()
    seen = {}

    def fake_regen(gpu, *, capture):
        seen["gpu"] = gpu
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("benchmarks.cli._regen_summary", fake_regen)
    _, payload = _invoke("--phase", "3", "--summary", out=tmp_path)
    assert seen["gpu"] == GPU
    assert payload["data"]["summary_path"].endswith("summary.md")
    assert payload["data"]["summary_error"] is None


def test_summary_still_runs_when_some_runs_failed(fake_run, tmp_path, monkeypatch):
    """A partial study is exactly when you want to see what you got."""
    fake_run(fail_on={"mix-llm-cv"})
    seen = {}

    def fake_regen(gpu, *, capture):
        seen["gpu"] = gpu
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("benchmarks.cli._regen_summary", fake_regen)
    res, payload = _invoke("--phase", "3", "--continue-on-error", "--summary", out=tmp_path)
    assert seen["gpu"] == GPU
    assert res.exit_code == 3
    assert payload["data"]["n_failed"] > 0


def test_summary_discovery_still_globs_the_nested_layout(fake_run, tmp_path):
    """summary.py finds runs by rglob'ing <gpu>/coloc/**/manifest.json —
    verified against the real nesting, not assumed."""
    from benchmarks.summary import _load_coloc_runs

    fake_run()
    _invoke("--phase", "3", out=tmp_path / "coloc")
    runs = _load_coloc_runs(tmp_path)
    assert len(runs) == 12
    assert {r["manifest"]["colocation_id"] for r in runs} >= {"mix-llm-cv"}
