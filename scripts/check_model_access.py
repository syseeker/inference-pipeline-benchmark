#!/usr/bin/env python3
"""Can this host actually download every model the contention study needs?

A gated HuggingFace repo does not fail at plan time. The server starts, tries
to fetch config.json, gets a 401 and exits — and the run then sits in
`_wait_ready` until its 600s budget expires before reporting "server not ready
within budget". Once per affected colocation. `gemma2-9b` and `llama3.1-8b`
are both gated, and both sit in Phase 4.

This is a HEAD request per model, so it costs a second and needs no GPU. Run it
with the dry-run, before committing hours to a study.

    python scripts/check_model_access.py --gpu rtx_pro6000
    python scripts/check_model_access.py --gpu rtx_pro6000 --json

Exit 0 = every model reachable. Exit 1 = at least one gated or missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Downloaded from GitHub releases by the `ultralytics` package, not from the HF
# repo, so a 401 on the mirror says nothing about whether the model will load.
NOT_FROM_HF = {"yolov8-n", "yolov8-l"}

HF_URL = "https://huggingface.co/{hf_id}/resolve/main/config.json"


def models_in_play(cfg: dict) -> set[str]:
    """Every model id any colocation names, including the ones only reachable
    through a `vary:` on the model field."""
    used: set[str] = set()
    for c in (cfg.get("colocations") or {}).values():
        for t in c.get("tenants") or []:
            if t.get("model"):
                used.add(t["model"])
        vary = c.get("vary") or {}
        if vary.get("field") == "model":
            used.update(vary.get("values") or [])
    return used


def _token() -> str | None:
    """The token `hf auth login` stored, if any.

    Without it this check answers a different question than the one asked:
    an anonymous HEAD on a gated repo is 401 whether or not the caller has
    been granted access, so every gated model would report blocked even after
    the licence was accepted.
    """
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        try:
            return (Path.home() / ".cache/huggingface/token").read_text().strip() or None
        except OSError:
            return None


def check(hf_id: str, timeout: float = 10.0, token: str | None = None) -> tuple[str, str]:
    """(state, detail) for one repo, as the authenticated user. Never raises."""
    req = urllib.request.Request(HF_URL.format(hf_id=hf_id), method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return "ok", ""
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            if not token:
                return "gated", f"HTTP {e.code} — not logged in"
            return "gated", f"HTTP {e.code} — logged in, licence not accepted"
        if e.code == 404:
            return "missing", "HTTP 404 — repo does not exist"
        return "error", f"HTTP {e.code}"
    except Exception as e:  # offline, DNS, timeout
        return "error", type(e).__name__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpu", required=True, help="GPU profile name, e.g. rtx_pro6000")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    import yaml
    cfg_path = REPO_ROOT / "benchmarks" / "configs" / f"{args.gpu}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    models = cfg.get("models") or {}

    token = _token()
    print(f"[ auth ] {'logged in' if token else 'NOT logged in'}"
          if not args.json else "", end="" if args.json else "\n")

    rows = []
    for mid in sorted(models_in_play(cfg)):
        spec = models.get(mid) or {}
        hf_id = spec.get("hf_id")
        if not hf_id:
            continue
        if mid in NOT_FROM_HF:
            rows.append({"model": mid, "hf_id": hf_id, "state": "skip",
                         "detail": "not fetched from HuggingFace"})
            continue
        state, detail = check(hf_id, token=token)
        rows.append({"model": mid, "hf_id": hf_id, "state": state, "detail": detail})

    blocked = [r for r in rows if r["state"] in ("gated", "missing")]

    if args.json:
        print(json.dumps({"ok": not blocked, "models": rows}, indent=2))
    else:
        for r in rows:
            mark = {"ok": "  ok  ", "skip": " skip ", "gated": " GATED",
                    "missing": "MISSING", "error": " ERROR"}[r["state"]]
            line = f"[{mark}] {r['model']:<22} {r['hf_id']}"
            print(line + (f"   <-- {r['detail']}" if r["detail"] else ""))
        if blocked:
            print("\nBlocked. Each of these costs a 600s timeout per colocation, not a "
                  "fast failure:")
            for r in blocked:
                print(f"  - {r['model']}: https://huggingface.co/{r['hf_id']}")
            print("\n  .venv-vllm/bin/hf auth login   # `hf`, not `huggingface-cli`: "
                  "renamed in huggingface_hub 0.34,")
            print("                                 # and it is in the venv, not on PATH")
            print("  Acceptance is per-account, per-model; logging in is not enough.")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
