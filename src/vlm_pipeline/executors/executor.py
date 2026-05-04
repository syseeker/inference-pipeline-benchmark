"""Executor stage.

v0: dry-run. Records the sequence and returns True. Never touches the
host. The benchmark and smoke harness use this exclusively.

v1: real executor that dispatches the validated commands to the host
input layer. Out of scope for this scaffold.
"""

from __future__ import annotations

from typing import Protocol

from vlm_pipeline.schemas import ActionSequence


class Executor(Protocol):
    def execute(self, seq: ActionSequence) -> bool:
        ...


class DryRunExecutor:
    def __init__(self) -> None:
        self.last: ActionSequence | None = None

    def execute(self, seq: ActionSequence) -> bool:
        self.last = seq
        return True
