"""Scenario-source discovery.

`bench scenarios build --source <name>` looks for `<name>` in the
`pipeline_bench.scenario_sources` entry-point group. The built-in
`nitrogen` source is registered there too — there's no special path
for it. Customers add their own source by declaring an entry-point in
their own pyproject.toml:

    [project.entry-points."pipeline_bench.scenario_sources"]
    my-gameplay = "my_pkg.scenarios:build"

The function signature is intentionally generic — the CLI passes its
non-source flags through as keyword args. Each source declares the
flags it consumes; unknown kwargs should be ignored.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

GROUP = "pipeline_bench.scenario_sources"

# Each source is `(*, n: int, out: Path, **kwargs) -> int` returning the
# number of scenarios successfully written.
ScenarioSource = Callable[..., int]


def discover() -> dict[str, ScenarioSource]:
    """Return all registered sources, keyed by entry-point name.

    Errors loading a single entry-point are surfaced as a stub function
    that raises on call (so `bench scenarios build` can list it in
    --help and the user sees the real error only on use).
    """
    sources: dict[str, ScenarioSource] = {}
    for ep in metadata.entry_points(group=GROUP):
        try:
            sources[ep.name] = ep.load()
        except Exception as exc:  # pragma: no cover - install-error path
            err = exc

            def _broken(**_: Any) -> int:
                raise RuntimeError(f"scenario source {ep.name!r} failed to load: {err}") from err

            sources[ep.name] = _broken
    return sources


def get(name: str) -> ScenarioSource:
    """Return the named source. Raises KeyError with the list if unknown."""
    sources = discover()
    if name not in sources:
        raise KeyError(
            f"unknown scenario source {name!r}; "
            f"registered: {sorted(sources) or '(none)'}. "
            f"Add a source via [project.entry-points.\"{GROUP}\"] in your pyproject.toml."
        )
    return sources[name]
